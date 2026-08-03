"""Sales preset detection used by cleaning tab and Agentic data cleaning."""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from fastapi import HTTPException

BACKEND_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BACKEND_DIR))

from sales_domain import should_enable_sales_preset  # noqa: E402


def test_should_enable_sales_preset_ev_columns_false():
    assert should_enable_sales_preset(['sessionId', 'kwhTotal', 'dollars']) is False


def test_should_enable_sales_preset_sales_columns_true():
    assert should_enable_sales_preset(['total_total_value_sale_free']) is True


def test_execute_data_cleaning_sets_sales_preset_from_cache():
    import agentic.agentic_adapter as adapter

    backend = MagicMock()
    backend.DATASET_CACHE = {
        'ds-sales': {'columns': ['year_month', 'total_total_value_sale_free']},
        'ds-ev': {'columns': ['sessionId', 'kwhTotal', 'dollars']},
    }
    backend.ParquetCleaningRequest = MagicMock(side_effect=lambda **kwargs: kwargs)
    backend.clean_dataset = MagicMock(return_value={'ok': True})
    request = MagicMock()

    with patch.object(adapter, 'get_backend_module', return_value=backend), patch.object(
        adapter, 'session_dataset_id', side_effect=lambda sid: sid.replace('sess-', 'ds-')
    ):
        adapter.execute_data_cleaning('sess-sales', request)
        sales_kwargs = backend.ParquetCleaningRequest.call_args.kwargs
        assert sales_kwargs['sales_preset'] is True
        assert sales_kwargs['protect_forecast_target'] is True

        adapter.execute_data_cleaning('sess-ev', request)
        ev_kwargs = backend.ParquetCleaningRequest.call_args.kwargs
        assert ev_kwargs['sales_preset'] is False
        assert ev_kwargs['protect_forecast_target'] is False


def test_execute_data_cleaning_requires_cached_dataset():
    import agentic.agentic_adapter as adapter

    backend = MagicMock()
    backend.DATASET_CACHE = {}
    backend.logger = MagicMock()
    request = MagicMock()

    with patch.object(adapter, 'get_backend_module', return_value=backend), patch.object(
        adapter, 'session_dataset_id', return_value='missing-id'
    ):
        with pytest.raises(HTTPException) as exc_info:
            adapter.execute_data_cleaning('sess-x', request)
        assert exc_info.value.status_code == 422
