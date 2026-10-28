"""Financial report output must disclose scenario assumptions."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import main  # noqa: E402


def test_html_report_discloses_unverified_costs_and_currency():
    payload = main.ReportPayload.model_validate({
        'fileName': 'synthetic.csv', 'totalRows': 3, 'columns': [], 'duplicates': 0,
        'memoryUsage': 'N/A', 'cleaningLogs': [], 'cleaningDone': False, 'cleanedRowCount': 3,
        'problemType': 'regression', 'selectedFeatures': [],
        'profitForecast': [{'period': '2024-01-01', 'forecasted_revenue': 10, 'forecasted_cogs': 6,
                            'gross_profit': 4, 'operating_expenses': 1, 'total_losses': 1, 'net_profit': 2}],
        'profitValidationReport': {'validation_reasons': ['Operating expenses use an assumption.'],
                                   'mapped_columns': {'cogs': None, 'operating_expense': None}},
        'displayCurrency': 'USD',
    })
    html = main.build_dynamic_report_html(payload).decode('utf-8')
    assert 'Operating expenses use an assumption.' in html
    assert 'USD' in html
    assert 'not actual accounting results' in html.lower()
