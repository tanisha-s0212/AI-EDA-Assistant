"""Real sample and synthetic input-format checks using the production loader."""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from main import load_dataset  # noqa: E402
from nl_analytics import answer_question, profile_dataset  # noqa: E402


SAMPLES = BACKEND_DIR.parent / "samples" / "sales"


def test_repository_sales_samples_load_profile_and_answer_query():
    expected = (1440, 20)
    for filename in ("sales_monthly.csv", "sales_monthly.tsv", "sales_monthly.xlsx",
                     "sales_monthly.parquet"):
        path = SAMPLES / filename
        if not path.is_file():
            continue
        frame = load_dataset(str(path))
        assert frame.shape == expected, filename
        assert "net_sales" in frame.columns and "year_month" in frame.columns
        assert pd.to_numeric(frame["net_sales"], errors="coerce").notna().all()
        profile = profile_dataset(frame, filename)
        assert profile["rows"] == expected[0]
        result = answer_question(frame, "What is total net sales?", dataset_id=filename)
        assert result["status"] == "success", filename

    invoices = load_dataset(str(SAMPLES / "sales_invoices.csv"))
    assert invoices.shape == (540, 13)
    assert pd.api.types.is_datetime64_any_dtype(invoices["invoice_date"])
    result = answer_question(invoices, "What is total net sales?", dataset_id="invoices")
    assert result["status"] == "success"


def test_json_loader_profiles_nontransactional_data_with_nulls_and_duplicates(tmp_path):
    source = pd.DataFrame({
        "amount ($)": [10.0, 10.0, None, 25.0],
        "region / name": ["north", "north", "south", None],
        "fixed": [1, 1, 1, 1],
    })
    path = tmp_path / "unusual_columns.json"
    source.to_json(path, orient="records")

    loaded = load_dataset(str(path))
    assert loaded.shape == (4, 3)
    assert loaded.columns.tolist() == source.columns.tolist()
    assert loaded["amount ($)"].isna().sum() == 1
    profile = profile_dataset(loaded, "json")
    assert profile["rows"] == 4
    result = answer_question(loaded, "What is the average amount?", dataset_id="json")
    assert result["status"] == "success"
    assert "amount ($)" in str(result)
