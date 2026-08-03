import { describe, expect, it } from 'vitest';
import {
  FORECAST_CHART_DOT_THRESHOLD,
  buildForecastSeriesChartData,
  firstForecastPeriod,
} from '@/components/forecast-series-chart';

describe('buildForecastSeriesChartData', () => {
  it('builds history + future length and keeps segments visually distinct', () => {
    const history = [
      { period: '2024-01', actual: 10 },
      { period: '2024-02', actual: 12 },
      { period: '2024-03', actual: 11 },
    ];
    const testForecast = [{ period: '2024-03', predicted: 11.5 }];
    const futureForecast = [
      { period: '2024-04', predicted: 13, lower: 12, upper: 14 },
      { period: '2024-05', predicted: 14, lower: 13, upper: 15 },
    ];

    const series = buildForecastSeriesChartData(history, testForecast, futureForecast, {
      includeConfidence: true,
    });

    expect(series).toHaveLength(history.length + futureForecast.length);
    expect(firstForecastPeriod(series)).toBe('2024-04');

    const historyOnly = series.filter((point) => point.forecast == null);
    const futureOnly = series.filter((point) => point.forecast != null);
    expect(historyOnly.every((point) => point.actual != null || point.backtest != null)).toBe(true);
    expect(futureOnly.every((point) => point.actual == null && point.backtest == null)).toBe(true);
    expect(series[2].backtest).toBe(11.5);
    expect(series[3].confidenceRange).toBe(2);
  });

  it('keeps large series above the adaptive-dot threshold for readability checks', () => {
    const history = Array.from({ length: 80 }, (_, index) => ({
      period: `p-${index}`,
      actual: index,
    }));
    const futureForecast = Array.from({ length: 6 }, (_, index) => ({
      period: `f-${index}`,
      predicted: 100 + index,
    }));
    const series = buildForecastSeriesChartData(history, [], futureForecast);
    expect(series.length).toBe(86);
    expect(series.length).toBeGreaterThan(FORECAST_CHART_DOT_THRESHOLD);
    expect(firstForecastPeriod(series)).toBe('f-0');
  });
});
