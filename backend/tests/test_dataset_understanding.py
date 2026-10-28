from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from dataset_understanding import monthly_coverage  # noqa: E402


def test_monthly_coverage_counts_observed_periods_and_zero_values():
    dates = pd.date_range('2023-01-01', periods=24, freq='MS')
    frame = pd.DataFrame({'month': dates, 'sales': [0.0] + [1.0] * 23, 'item': ['A'] * 24})
    result = monthly_coverage(frame, 'month', 'sales', ['item'])
    assert result['observed_months'] == 24
    assert result['usable_months'] == 24
    assert result['missing_months'] == []
    assert result['meets_minimum'] is True


def test_more_than_24_months_also_meets_the_minimum():
    frame = pd.DataFrame({'month': pd.date_range('2022-01-01', periods=25, freq='MS'), 'sales': [1.0] * 25})
    assert monthly_coverage(frame, 'month', 'sales', [])['meets_minimum'] is True


def test_monthly_coverage_does_not_count_missing_month_as_observed():
    dates = pd.date_range('2023-01-01', periods=24, freq='MS').delete(3)
    frame = pd.DataFrame({'month': dates, 'sales': [10.0] * 23})
    result = monthly_coverage(frame, 'month', 'sales', [])
    assert result['observed_months'] == 23
    assert result['missing_months'] == ['2023-04']
    assert result['meets_minimum'] is False


def test_monthly_coverage_reports_repeated_selected_keys():
    frame = pd.DataFrame({'month': pd.to_datetime(['2024-01-01', '2024-01-01', '2024-02-01']),
                          'sales': [1.0, 2.0, 3.0], 'item': ['A', 'A', 'A']})
    result = monthly_coverage(frame, 'month', 'sales', ['item'])
    assert result['duplicate_key_rows'] == 1
