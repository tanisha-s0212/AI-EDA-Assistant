"""Dataset-independent regressions for numeric preservation and ranking."""
from decimal import Decimal
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dtype_inference import infer_universal_dtypes  # noqa: E402
from nl_analytics import answer_question  # noqa: E402


def test_inference_preserves_float64_values_and_nulls():
    original = pd.DataFrame({'measurement': [123456.78, -120.17, 0.0000001, np.nan]})
    inferred, _ = infer_universal_dtypes(original)
    assert inferred['measurement'].dtype == original['measurement'].dtype
    pd.testing.assert_series_equal(inferred['measurement'], original['measurement'])


def test_inference_preserves_existing_decimal_values():
    original = pd.DataFrame({'reading': [Decimal('123456789.123456789'), Decimal('-0.01'), None]})
    inferred, _ = infer_universal_dtypes(original)
    pd.testing.assert_series_equal(inferred['reading'], original['reading'])


def test_rank_direction_and_requested_count_match_independent_grouping():
    frame = pd.DataFrame({'sector': ['A', 'A', 'B', 'C', 'D'], 'units': [4, 1, -2, 8, 3]})
    for phrase, ascending in [('top 2', False), ('bottom 2', True)]:
        result = answer_question(frame, f'Show the {phrase} sectors by total units', top_n=10)
        assert result['status'] == 'success'
        expected = frame.groupby('sector')['units'].sum().sort_values(ascending=ascending).head(2)
        rows = result['tables']['units']['rows']
        assert [(row['group'], row['value']) for row in rows] == list(expected.items())
        assert ('lowest' if ascending else 'highest') in result['answer'].lower()


def test_ranking_honors_explicit_arbitrary_measure():
    frame = pd.DataFrame({'ward': ['A', 'A', 'B', 'C'], 'duration_days': [1.2, 1.3, 8.5, 4.2],
                          'unrelated': [1000, 8000, -9000, 6000]})
    result = answer_question(frame, 'Top 2 ward by total duration_days', top_n=10)
    assert result['status'] == 'success'
    table = next(table for table in result['tables'].values() if table['column'] == 'duration_days')
    expected = frame.groupby('ward')['duration_days'].sum().sort_values(ascending=False).head(2)
    assert [(row['group'], row['value']) for row in table['rows']] == list(expected.items())


def test_ranking_honors_word_limit_and_deterministic_ties():
    frame = pd.DataFrame({'sector': ['C', 'A', 'B', 'D'], 'units': [1, 1, 1, 4]})
    result = answer_question(frame, 'Show the bottom two sectors by total units')
    assert [row['group'] for row in result['tables']['units']['rows']] == ['A', 'B']
    assert all(row['tied'] for row in result['tables']['units']['rows'])


def test_decimal_ranking_exposes_exact_aggregated_value():
    frame = pd.DataFrame({'ward': ['A', 'A', 'B'],
                          'measurement': [Decimal('123456789.123456789'), Decimal('0.000000001'), Decimal('2.1')]})
    result = answer_question(frame, 'Top 2 ward by total measurement')
    assert result['status'] == 'success'
    table = next(t for t in result['tables'].values() if t['column'] == 'measurement')
    assert Decimal(table['rows'][0]['exact_value']) == Decimal('123456789.123456790')


def test_unknown_explicit_filter_does_not_return_unfiltered_ranking():
    frame = pd.DataFrame({'sector': ['A', 'B', 'C'], 'zone': ['east', 'west', 'east'], 'units': [4, 9, 3]})
    result = answer_question(frame, 'Top 2 sectors by total units where zone = nowhere')
    assert result['status'] in ('empty', 'clarify', 'unsupported')
    assert not result.get('tables')


def test_explicit_average_uses_mean_instead_of_sum():
    frame = pd.DataFrame({'ward': ['A', 'A', 'B', 'C'], 'reading': [2., 8., 7., 4.]})
    result = answer_question(frame, 'Top 2 ward by average reading')
    table = next(table for table in result['tables'].values() if table['column'] == 'reading')
    expected = frame.groupby('ward')['reading'].mean().sort_values(ascending=False).head(2)
    assert [(row['group'], row['value']) for row in table['rows']] == list(expected.items())


def test_unmapped_semantic_measure_is_not_guessed_from_variance():
    frame = pd.DataFrame({'team': ['A', 'B', 'C'], 'delay_hours': [1., 2., 100.], 'quality': [5., 7., 8.]})
    result = answer_question(frame, 'Which team has the highest sales?')
    assert result['status'] in ('unsupported', 'clarify')
