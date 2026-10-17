"""Isolated forecast runtime, quality, Python-allocation, and process-RSS benchmark.

Run from the repository root with:
    python backend/scripts/benchmark_forecast.py --rows 240 --horizon 24 --repeats 3

Each sample runs in a fresh child process. `--implementation original` reproduces
the pre-optimization recursive feature rebuild for comparison.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import tempfile
import time
import tracemalloc
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

import ml_forecast_pipeline as pipeline  # noqa: E402


def _process_memory(process: Any) -> tuple[int | None, int | None, str]:
    """Return current RSS and OS-reported peak RSS when available."""
    try:
        info = process.memory_info()
        current = int(info.rss)
        peak = getattr(info, 'peak_wset', None) or getattr(info, 'peak_rss', None)
        if peak is not None:
            return current, int(peak), 'psutil peak_wset/peak_rss'
        try:
            import resource
            peak_value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
            if sys.platform != 'darwin':
                peak_value *= 1024
            return current, peak_value, 'resource.ru_maxrss'
        except (ImportError, AttributeError, OSError):
            return current, None, 'current RSS only; peak RSS unavailable'
    except Exception:
        return None, None, 'process memory unavailable'


def _original_recursive_forecast(
    model: Any, frame: pd.DataFrame, features: list[str], target: str,
    date_col: str, horizon: int, frequency: str,
) -> list[dict[str, Any]]:
    """Original implementation retained here as a repeatable benchmark control."""
    config = pipeline.get_config(frequency)
    history = frame[[date_col, target]].copy().sort_values(date_col).reset_index(drop=True)
    forecasts = []
    next_date = pd.Timestamp(history.iloc[-1][date_col]) + config['offset']
    for _ in range(horizon):
        row = pipeline._build_future_row(history, date_col, target, frequency, config, next_date)
        row = row.reindex(columns=features).fillna(float(history[target].mean()))
        prediction = max(0.0, float(model.predict(row)[0]))
        forecasts.append({
            'period': next_date.strftime('%Y-%m-%d'),
            'period_label': pipeline.format_period_label(next_date, frequency),
            'forecast_value': round(prediction, 2),
        })
        history = pd.concat([history, pd.DataFrame([{date_col: next_date, target: prediction}])], ignore_index=True)
        next_date += config['offset']
    return forecasts


def _make_frame(rows: int) -> pd.DataFrame:
    index = np.arange(rows)
    return pd.DataFrame({
        'created': [pd.Timestamp('2020-01-01 08:15') + pd.Timedelta(days=int(i), hours=int(i % 11)) for i in index],
        'sales': 100.0 + (index % 13) + 0.025 * index,
    })


def _seasonal_naive_metrics(feature_frame: pd.DataFrame, target: str, frequency: str) -> dict[str, float]:
    split = max(1, int(len(feature_frame) * 0.8))
    history = feature_frame[target].astype(float).iloc[:split].tolist()
    actual = feature_frame[target].astype(float).iloc[split:].to_numpy()
    season = 52 if frequency == 'weekly' else 12
    if len(history) < season:
        season = 1
    predictions = []
    for _ in actual:
        value = float(history[-season])
        predictions.append(value)
        history.append(value)
    return pipeline.compute_all_metrics(actual, predictions)


def _worker(rows: int, horizon: int, implementation: str) -> None:
    # Production requests run after the FastAPI module is loaded; keep startup
    # imports outside the timed request path while retaining an isolated process.
    import main  # noqa: F401

    try:
        import psutil
        process = psutil.Process()
    except Exception:
        process = None
    memory_before, _, _ = _process_memory(process) if process else (None, None, 'unavailable')
    if implementation == 'original':
        pipeline.forecast_future = _original_recursive_forecast

    captured: dict[str, Any] = {}
    original_writer = pipeline.write_all_outputs

    def capture_metrics(*args: Any, **kwargs: Any) -> dict[str, Any]:
        captured.update(metrics=args[4], feature_frame=args[5].copy(), frequency=args[11])
        return original_writer(*args, **kwargs)

    pipeline.write_all_outputs = capture_metrics
    timings: dict[str, float] = {}
    for name in ('prepare_frame_for_ml_pipeline', 'clean_data', 'engineer_features',
                 'walk_forward_validate', 'forecast_future', 'write_all_outputs'):
        original = getattr(pipeline, name)

        def timed(*args: Any, _name=name, _original=original, **kwargs: Any) -> Any:
            started = time.perf_counter()
            try:
                return _original(*args, **kwargs)
            finally:
                timings[_name] = timings.get(_name, 0.0) + time.perf_counter() - started

        setattr(pipeline, name, timed)

    tracemalloc.start()
    started = time.perf_counter()
    with tempfile.TemporaryDirectory(prefix='forecast-benchmark-') as temp_dir:
        result = pipeline.run_full_pipeline(
            Path(temp_dir), target_col='sales', date_col='created', horizon=horizon,
            frequency='auto', frame=_make_frame(rows), model_type='gradient_boosting',
        )
    elapsed = time.perf_counter() - started
    traced_current, traced_peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    memory_after, peak_rss, peak_source = _process_memory(process) if process else (None, None, 'unavailable')
    feature_frame = captured['feature_frame']
    print('BENCHMARK_JSON=' + json.dumps({
        'implementation': implementation,
        'rows_in': rows,
        'usable_periods': int((result.get('metadata') or {}).get('usable_periods') or 0),
        'horizon': horizon,
        'total_seconds': elapsed,
        'stage_seconds': timings,
        'model_metrics': captured['metrics'],
        'seasonal_naive_metrics': _seasonal_naive_metrics(feature_frame, 'sales', captured['frequency']),
        'frequency': captured['frequency'],
        'rss_before_bytes': memory_before,
        'rss_after_bytes': memory_after,
        'peak_rss_bytes': peak_rss,
        'peak_rss_source': peak_source,
        'tracemalloc_current_bytes': traced_current,
        'tracemalloc_peak_bytes': traced_peak,
    }, default=str))


def _run_isolated(rows: int, horizon: int, implementation: str) -> dict[str, Any]:
    command = [sys.executable, str(Path(__file__).resolve()), '--worker', '--rows', str(rows),
               '--horizon', str(horizon), '--implementation', implementation]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    line = next((item for item in reversed(completed.stdout.splitlines()) if item.startswith('BENCHMARK_JSON=')), None)
    if line is None:
        raise RuntimeError(f'Benchmark worker returned no JSON record. stdout={completed.stdout!r}; stderr={completed.stderr!r}')
    return json.loads(line.removeprefix('BENCHMARK_JSON='))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--rows', type=int, default=240)
    parser.add_argument('--horizon', type=int, default=24)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--implementation', choices=('both', 'original', 'optimized'), default='both')
    parser.add_argument('--worker', action='store_true')
    args = parser.parse_args()
    if args.rows < 48 or args.horizon < 1 or args.repeats < 1:
        parser.error('rows must be >= 48; horizon and repeats must be positive.')
    if args.worker:
        _worker(args.rows, args.horizon, args.implementation)
        return

    versions = ('original', 'optimized') if args.implementation == 'both' else (args.implementation,)
    samples = {name: [_run_isolated(args.rows, args.horizon, name) for _ in range(args.repeats)] for name in versions}
    summary = {}
    for name, records in samples.items():
        def median_of(field: str) -> float | None:
            values = [record[field] for record in records if record[field] is not None]
            return statistics.median(values) if values else None

        summary[name] = {
            'median_total_seconds': statistics.median(record['total_seconds'] for record in records),
            'median_stage_seconds': {
                stage: statistics.median(record['stage_seconds'].get(stage, 0.0) for record in records)
                for stage in records[0]['stage_seconds']
            },
            'median_peak_rss_bytes': median_of('peak_rss_bytes'),
            'median_rss_before_bytes': median_of('rss_before_bytes'),
            'median_rss_after_bytes': median_of('rss_after_bytes'),
            'median_tracemalloc_peak_bytes': median_of('tracemalloc_peak_bytes'),
            'forecast_metrics': records[-1]['model_metrics'],
            'seasonal_naive_metrics': records[-1]['seasonal_naive_metrics'],
            'frequency': records[-1]['frequency'],
            'samples': records,
        }
    print(json.dumps({'command': 'python backend/scripts/benchmark_forecast.py', 'rows': args.rows,
                      'horizon': args.horizon, 'repeats': args.repeats, 'results': summary}, indent=2))


if __name__ == '__main__':
    main()
