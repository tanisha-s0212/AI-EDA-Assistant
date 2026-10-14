"""Regression coverage for dataset-aware labels on Advanced EDA charts."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

import main  # noqa: E402


@pytest.fixture()
def captured_plotly_figures(monkeypatch: pytest.MonkeyPatch):
    figures = []

    def capture(figure, **_kwargs):
        figures.append(figure)
        return 'data:image/png;base64,test'

    monkeypatch.setattr(main, 'figure_to_base64', capture)
    return figures


def test_field_labels_are_human_readable_for_unusual_names():
    assert main.format_eda_field_label('avg_order_value') == 'Average Order Value'
    assert main.format_eda_field_label('grossMargin_USD') == 'Gross Margin USD'
    assert main.format_eda_field_label(None) == 'Column'


def test_distribution_axes_describe_histogram_and_horizontal_boxplot(captured_plotly_figures):
    frame = pd.DataFrame({'avg_order_value': np.linspace(10, 100, 30)})

    result = main.build_distribution_chart_for_column(frame, 'avg_order_value')

    assert result.startswith('data:image/png')
    figure = captured_plotly_figures[0]
    assert figure.layout.xaxis.title.text == 'Average Order Value'
    assert figure.layout.xaxis2.title.text == 'Average Order Value'
    assert figure.layout.yaxis.title.text == 'Probability Density'
    assert figure.layout.yaxis2.title.text == ''
    assert figure.layout.yaxis2.showticklabels is False


def test_categorical_and_scatter_axes_use_the_plotted_fields(captured_plotly_figures):
    categorical = pd.DataFrame({'Customer Segment': ['Retail', 'Online', 'Retail', 'Partner']})
    categorical_result = main.build_categorical_payload(categorical)
    assert categorical_result['charts']
    category_figure = captured_plotly_figures[-1]
    assert category_figure.layout.xaxis.title.text == 'Count'
    assert category_figure.layout.yaxis.title.text == 'Customer Segment'

    interactions = pd.DataFrame({
        'net_sales': [float(index % 13) for index in range(50)],
        'grossProfit': [float((index * 3) % 17) for index in range(50)],
    })
    interaction_result = main.build_interaction_payload(interactions)
    assert interaction_result['plots']
    interaction_figure = captured_plotly_figures[-1]
    assert interaction_figure.layout.xaxis.title.text in {'Net Sales', 'Gross Profit'}
    assert interaction_figure.layout.yaxis.title.text in {'Net Sales', 'Gross Profit'}
    assert interaction_figure.layout.xaxis.title.text != interaction_figure.layout.yaxis.title.text


def test_missingness_heatmap_labels_fields_rows_and_color_scale(captured_plotly_figures):
    frame = pd.DataFrame({'customerRegion': ['north', None] * 12, 'order_total': [1.0, 2.0] * 12})

    result = main.build_missingness_payload(frame)

    assert result['status'] == 'chart'
    figure = captured_plotly_figures[0]
    assert figure.layout.xaxis.title.text == 'Column'
    assert figure.layout.yaxis.title.text == 'Row Group (% of rows)'
    assert figure.data[0].colorbar.title.text == 'Missing (%)'
    assert list(figure.layout.xaxis.ticktext) == ['Customer Region']


def test_matplotlib_fallbacks_keep_dynamic_axis_labels(monkeypatch: pytest.MonkeyPatch):
    captured_axes = []

    def capture(fig):
        captured_axes.append(fig.axes)
        main.plt.close(fig)
        return 'fallback-image'

    monkeypatch.setattr(main, 'matplotlib_figure_to_base64', capture)

    numeric_frame = pd.DataFrame({'avg_order_value': np.linspace(1, 50, 25)})
    main.build_distribution_chart_matplotlib(numeric_frame, ['avg_order_value'])
    histogram_axes, box_axes = captured_axes.pop()
    assert histogram_axes.get_xlabel() == 'Average Order Value'
    assert histogram_axes.get_ylabel() == 'Probability Density'
    assert box_axes.get_xlabel() == 'Average Order Value'
    assert len(box_axes.get_yticks()) == 0

    main.build_categorical_chart_matplotlib(pd.Series([3, 2], index=['Retail', 'Online']), 'Categories', 'customer_segment')
    category_axes = captured_axes.pop()[0]
    assert category_axes.get_xlabel() == 'Count'
    assert category_axes.get_ylabel() == 'Customer Segment'

    main.build_interaction_chart_matplotlib(
        np.array([1.0, 2.0, 3.0]), np.array([2.0, 5.0, 4.0]), 'Plot', 'net_sales', 'grossProfit',
    )
    interaction_axes = captured_axes.pop()[0]
    assert interaction_axes.get_xlabel() == 'Net Sales'
    assert interaction_axes.get_ylabel() == 'Gross Profit'

    main.build_missingness_chart_matplotlib([[10.0]], ['customerRegion'], ['0-100%'])
    missingness_axes = captured_axes.pop()[0]
    assert missingness_axes.get_xlabel() == 'Column'
    assert missingness_axes.get_ylabel() == 'Row Group (% of rows)'
    assert [tick.get_text() for tick in missingness_axes.get_xticklabels()] == ['Customer Region']
