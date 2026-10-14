"""Tests for customer next-purchase prediction and recommendation flows.

Uses seeded synthetic transaction fixtures with known behaviour, plus the real
sample datasets for grain/limitation checks. Production logic contains no
hardcoded customers, products, or expected answers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from nl_analytics import answer_question, classify_intent, profile_dataset  # noqa: E402
from purchase_predict import (  # noqa: E402
    build_purchase_histories,
    customers_at_risk,
    customers_for_product,
    discover_purchase_schema,
    predict_next_for_customer,
    predict_timing_for_customer,
    validate_methods_temporal,
)

SAMPLES = BACKEND_DIR.parent / "samples" / "sales"


def make_transactions(seed: int = 11, n_customers: int = 16) -> pd.DataFrame:
    """Two affinity groups: C001-C008 prefer P1/P2, rest prefer P3/P4."""
    rng = np.random.RandomState(seed)
    rows = []
    oid = 0
    for c in range(1, n_customers + 1):
        cust = f"C{c:03d}"
        prods = ["P1", "P2"] if c <= n_customers // 2 else ["P3", "P4"]
        for w in range(rng.randint(4, 8)):
            oid += 1
            rows.append({
                "cust_id": cust,
                "sku": str(rng.choice(prods, p=[0.7, 0.3])),
                "order_no": f"O{oid:05d}",
                "purchased_on": pd.Timestamp("2023-01-05") + pd.Timedelta(days=int(w * 30 + rng.randint(0, 9))),
                "qty": int(rng.randint(1, 4)),
            })
    return pd.DataFrame(rows)


@pytest.fixture()
def tx() -> pd.DataFrame:
    return make_transactions()


@pytest.fixture()
def client():
    import main
    return TestClient(main.app)


# --- Schema discovery ---------------------------------------------------------

def test_transaction_schema_discovered_dynamically(tx):
    schema = discover_purchase_schema(tx, profile_dataset(tx, "t"))
    assert schema["mapping"]["customer_col"] == "cust_id"
    assert schema["mapping"]["product_col"] == "sku"
    assert schema["mapping"]["order_col"] == "order_no"
    assert schema["mapping"]["date_col"] == "purchased_on"
    assert schema["grain"] == "transaction"
    assert schema["sufficient"] is True


def test_renamed_columns_still_mapped():
    df = make_transactions().rename(columns={"cust_id": "client_number",
                                             "sku": "item_code",
                                             "purchased_on": "order_date"})
    schema = discover_purchase_schema(df, profile_dataset(df, "t"))
    assert schema["mapping"]["customer_col"] == "client_number"
    assert schema["mapping"]["product_col"] == "item_code"
    assert schema["mapping"]["date_col"] == "order_date"
    assert schema["sufficient"] is True


def test_monthly_aggregate_refuses_individual_prediction():
    df = pd.read_csv(SAMPLES / "sales_monthly.csv")
    result = answer_question(df, "What product will customer C001 buy next?", dataset_id="m")
    assert result["status"] == "unsupported"
    assert "C001" not in str(result.get("prediction", ""))
    assert result.get("prediction") is None
    # Supported aggregate alternative is offered, nothing fabricated.
    assert "forecast" in result["answer"].lower() or "aggregat" in result["answer"].lower()


def test_missing_customer_column_reports_limitation():
    df = pd.DataFrame({"sku": ["P1", "P2", "P1"],
                       "purchased_on": pd.to_datetime(["2023-01-01", "2023-02-01", "2023-03-01"])})
    result = answer_question(df, "Predict the next purchase of customer C001.", dataset_id="t")
    assert result["status"] == "unsupported"
    assert "customer" in result["answer"].lower()


def test_missing_timestamps_report_limitation():
    df = pd.DataFrame({"cust_id": ["C1", "C1", "C2"], "sku": ["P1", "P2", "P1"]})
    result = answer_question(df, "What will customer C1 buy next?", dataset_id="t")
    assert result["status"] == "unsupported"
    assert "timestamp" in result["answer"].lower() or "chronolog" in result["answer"].lower()


# --- Intent / paraphrase generalisation ----------------------------------------

@pytest.mark.parametrize("q", [
    "Predict the next purchase of this customer.",
    "What product will this customer buy next?",
    "Which product is this customer most likely to purchase?",
    "What are the top three products this customer may buy next?",
    "Forecast the next item for this buyer.",
    "What should this client order next?",
])
def test_paraphrases_resolve_to_purchase_capability(tx, q):
    routing = classify_intent(q, profile_dataset(tx, "t"))
    assert routing["intent"] in {"next_purchase", "recommend", "confidence", "explain"}
    assert routing["needs_ml"] is True


def test_ranking_question_still_deterministic(tx):
    routing = classify_intent("Which products have the highest sales?", profile_dataset(tx, "t"))
    assert routing["intent"] == "ranking" and not routing["needs_ml"]


# --- End-to-end prediction ------------------------------------------------------

def test_next_purchase_differs_per_customer_history(tx):
    a = answer_question(tx, "What product will customer C002 buy next?", dataset_id="t")
    b = answer_question(tx, "What product will customer C012 buy next?", dataset_id="t")
    assert a["status"] == "success" and b["status"] == "success"
    assert a["customer"] == "C002" and b["customer"] == "C012"
    assert a["prediction"]["ranked"][0]["product"] in {"P1", "P2"}
    assert b["prediction"]["ranked"][0]["product"] in {"P3", "P4"}
    assert a["prediction"]["ranked"][0]["evidence"]["times_bought_by_customer"] >= 1


def test_this_customer_with_context(tx):
    result = answer_question(tx, "Predict the next purchase of this customer.",
                             dataset_id="t", context={"customer_id": "C003"})
    assert result["status"] == "success"
    assert result["customer"] == "C003"
    assert result["output_label"] in {"prediction", "recommendation"}


def test_this_customer_without_context_asks_clarification(tx):
    result = answer_question(tx, "Predict the next purchase of this customer.", dataset_id="t")
    assert result["status"] == "clarify"
    assert result["needs"] == ["customer_id"]
    assert "which customer" in result["answer"].lower()


def test_unknown_customer_is_not_invented(tx):
    result = answer_question(tx, "What will customer ZZZ999 buy next?", dataset_id="t")
    assert result["status"] == "clarify"
    assert "prediction" not in result


def test_top_k_ranked_and_grounded(tx):
    result = answer_question(tx, "What are the top three products customer C004 may buy next?",
                             dataset_id="t")
    assert result["status"] == "success"
    ranked = result["prediction"]["ranked"]
    assert 1 <= len(ranked) <= 3
    valid = set(tx["sku"].unique().tolist())
    assert all(r["product"] in valid for r in ranked)
    scores = [r["score"] for r in ranked]
    assert scores == sorted(scores, reverse=True)


def test_timing_estimate_grounded_in_history(tx):
    result = answer_question(tx, "When is customer C005 likely to purchase again?", dataset_id="t")
    assert result["status"] == "success"
    timing = result["timing"]
    assert timing["status"] == "estimate"
    last = pd.to_datetime(tx[tx["cust_id"] == "C005"]["purchased_on"]).max()
    assert pd.Timestamp(timing["predicted_next_date"]) > last


def test_sparse_customer_timing_unavailable(tx):
    df = tx.copy()
    df = pd.concat([df, pd.DataFrame([{"cust_id": "C999", "sku": "P1",
                                       "order_no": "O99999",
                                       "purchased_on": pd.Timestamp("2023-06-01"),
                                       "qty": 1}])], ignore_index=True)
    result = answer_question(df, "When is customer C999 likely to purchase again?", dataset_id="t")
    assert result["status"] == "success"
    assert result["timing"]["status"] == "unavailable"
    assert "unavailable" in result["timing"]["reason"].lower()


def test_explanation_references_real_evidence(tx):
    result = answer_question(tx, "Why did you recommend this product for customer C002?",
                             dataset_id="t")
    assert result["status"] == "success"
    why = result["explanation"]["why"][0]
    assert why["times_bought"] >= 1 and why["last_bought"] is not None
    assert result["explanation"]["method"] == result["provenance"]["method"]


# --- Confidence honesty ---------------------------------------------------------

def test_confidence_labels_scores_not_probabilities(tx):
    result = answer_question(tx, "What is the confidence level of the prediction for customer C002?",
                             dataset_id="t")
    assert result["status"] == "success"
    conf = result["confidence"]
    assert conf["probability"]["value"] is None
    assert "unavailable" in conf["probability"]["reason"].lower()
    rel = conf["method_reliability"]
    assert rel["type"] == "hit_rate"
    assert 0 <= rel["hit_rate_at_3"] <= 1
    assert "held-out" in rel["meaning"]


def test_unvalidated_confidence_says_unavailable():
    df = make_transactions(n_customers=3)
    df = df[df["cust_id"].isin(["C001", "C002"])]
    result = answer_question(df, "What is the confidence for customer C001?", dataset_id="t")
    assert result["status"] == "success"
    rel = result["confidence"]["method_reliability"]
    assert rel["type"] == "unavailable"
    assert "unavailable" in rel["reason"].lower()
    assert result["output_label"] == "recommendation"


# --- Buyers / churn ---------------------------------------------------------------

def test_buyers_of_product_grounded(tx):
    result = answer_question(tx, "Which customers are likely to purchase P1?", dataset_id="t")
    assert result["status"] == "success"
    assert result["buyer_count"] > 0
    actual = set(tx[tx["sku"] == "P1"]["cust_id"].tolist())
    assert all(b["customer"] in actual for b in result["buyers"])


def test_buyers_unknown_product_asks_clarification(tx):
    result = answer_question(tx, "Which customers will buy the MoonWidget?", dataset_id="t")
    assert result["status"] == "clarify"
    assert "needs" in result


def test_churn_list_grounded(tx):
    result = answer_question(tx, "Which customers are at risk of not purchasing again?",
                             dataset_id="t")
    assert result["status"] == "success"
    assert result["intent"] == "churn"
    known = set(tx["cust_id"].unique().tolist())
    for r in result["at_risk"]:
        assert r["customer"] in known
        assert r["days_since_last"] > r["typical_interval_days"]


# --- Data quality: leakage, duplicates, returns -------------------------------------

def test_temporal_validation_has_no_future_leakage():
    # Every customer's last order is a brand-new product: holdout must be unpredictable.
    rows = []
    for c in range(8):
        cust = f"L{c:03d}"
        for w, prod in enumerate(["Q1", "Q1", "Q2", f"ONLY{c}"]):
            rows.append({"cust_id": cust, "sku": prod, "order_no": f"LO{c}{w}",
                         "purchased_on": pd.Timestamp("2023-01-01") + pd.Timedelta(days=w * 30)})
    df = pd.DataFrame(rows)
    schema = discover_purchase_schema(df, profile_dataset(df, "t"))
    built = build_purchase_histories(df, schema)
    validation = validate_methods_temporal(built["customers"])
    assert validation["validated"] is True
    # Unseen holdout items cannot be predicted from strictly-past data.
    assert validation["metrics"]["repeat_preference"]["hit_rate_at_3"] == 0


def test_temporal_validation_excludes_same_day_and_future_evidence(monkeypatch):
    import purchase_predict as pp

    histories = {}
    # Each customer's held-out purchase is deliberately unique. All customers
    # share the same holdout date, so that date and later events must be hidden
    # from every method while evaluating any one customer.
    for c in range(8):
        orders = []
        for day, product in [(1, "COMMON"), (2, "COMMON"), (3, "EARLY"), (4, f"HELD{c}")]:
            orders.append({"date": str(pd.Timestamp("2023-01-01") + pd.Timedelta(days=day)),
                           "products": [product]})
        histories[f"T{c:03d}"] = {"orders": orders, "order_count": len(orders)}

    seen = {"popularity": [], "item_item": [], "collaborative": []}
    old_popularity = pp._score_popularity
    old_item_item = pp._score_item_item
    old_collaborative = pp._score_collaborative

    def check_popularity(counts):
        seen["popularity"].append(set(counts))
        return old_popularity(counts)

    def check_item_item(matrix, recent):
        seen["item_item"].append(set(matrix) | {p for values in matrix.values() for p in values})
        return old_item_item(matrix, recent)

    def check_collaborative(target_items, others):
        seen["collaborative"].append(set().union(*others.values()) if others else set())
        return old_collaborative(target_items, others)

    monkeypatch.setattr(pp, "_score_popularity", check_popularity)
    monkeypatch.setattr(pp, "_score_item_item", check_item_item)
    monkeypatch.setattr(pp, "_score_collaborative", check_collaborative)
    validation = validate_methods_temporal(histories)
    assert validation["validated"] is True
    assert seen["popularity"] and seen["item_item"] and seen["collaborative"]
    assert all(not any(item.startswith("HELD") for item in products)
               for method in seen.values() for products in method)


def test_customer_segment_is_not_accepted_as_individual_customer_id():
    rows = []
    for c in range(5):
        for day in range(4):
            rows.append({"customer_segment": f"Segment {c % 2}", "product": f"P{day % 2}",
                         "order_id": f"O{c}-{day}", "purchase_date": pd.Timestamp("2024-01-01")
                         + pd.Timedelta(days=day)})
    df = pd.DataFrame(rows)
    schema = discover_purchase_schema(df, profile_dataset(df, "t"))
    assert not schema["sufficient"]
    assert any("grouping" in issue for issue in schema["issues"])


def test_duplicates_and_returns_handled_transparently():
    df = make_transactions(seed=5).reset_index(drop=True)
    dup = df.head(10).copy()
    df2 = pd.concat([df, dup], ignore_index=True)
    df2["line_status"] = "sold"
    # Mark 7 original (non-duplicated) rows as returned; dup rows stay intact.
    df2.loc[df2.index[len(df) - 7:len(df)], "line_status"] = "returned"
    schema = discover_purchase_schema(df2, profile_dataset(df2, "t"))
    built = build_purchase_histories(df2, schema)
    assert built["report"]["duplicate_rows"] >= 10
    assert built["report"]["excluded_returns"] == 7
    before = answer_question(df, "What will customer C001 buy next?", dataset_id="t")
    after = answer_question(df2, "What will customer C001 buy next?", dataset_id="t")
    assert after["status"] == "success"
    assert after["prediction"]["ranked"][0]["product"] == before["prediction"]["ranked"][0]["product"]


# --- Lifecycle ------------------------------------------------------------------------

def test_dataset_replacement_invalidates_purchase_mapping(tx):
    sig1 = profile_dataset(tx, "d")["signature"]
    df2 = tx.rename(columns={"cust_id": "member_id"})
    sig2 = profile_dataset(df2, "d")["signature"]
    assert sig1 != sig2
    s2 = discover_purchase_schema(df2, profile_dataset(df2, "d"))
    assert s2["mapping"]["customer_col"] == "member_id"
    result = answer_question(df2, "Predict the next purchase of this customer.",
                             dataset_id="d", context={"customer_id": "C001"})
    assert result["status"] == "success" and result["customer"] == "C001"


# --- API + regression --------------------------------------------------------------------

def _json_rows(df: pd.DataFrame) -> list[dict]:
    out = df.copy()
    for col in out.columns:
        if pd.api.types.is_datetime64_any_dtype(out[col]):
            out[col] = out[col].dt.strftime("%Y-%m-%d")
    return out.to_dict(orient="records")


def test_assistant_ask_endpoint_with_customer_context(client, tx):
    rows = _json_rows(tx)
    response = client.post("/api/assistant/ask", json={
        "question": "Predict the next purchase of this customer.",
        "data": rows, "top_n": 3, "customer_id": "C002"})
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "success"
    assert body["customer"] == "C002"
    assert body["prediction"]["ranked"]
    assert "plan" in body and body["plan"]["status"] == "completed"


def test_assistant_ask_endpoint_backward_compatible(client, tx):
    rows = _json_rows(tx)
    response = client.post("/api/assistant/ask", json={
        "question": "What is the total qty?", "data": rows})
    assert response.status_code == 200
    assert response.json()["status"] == "success"


def test_oem_regression_still_independent():
    df = pd.read_csv(SAMPLES / "sales_monthly.csv")
    result = answer_question(
        df, "Which product lines have the highest sales, units shipped, and order counts "
            "for the OEM customer segment?", dataset_id="monthly")
    assert result["status"] == "success"
    assert result["routing"]["needs_ml"] is False
    oem = df[df["customer_segment"] == "OEM"]
    expected_units = oem.groupby("product_line")["units_shipped"].sum().sort_values(ascending=False)
    assert result["tables"]["units"]["rows"][0]["group"] == str(expected_units.index[0])
