"""Customer next-purchase prediction and recommendation engine.

Executed (not planned-only) by ``nl_analytics.answer_question`` for
recommendation / next-purchase / timing / churn / confidence questions.

Design rules (see task spec):
- Dynamic schema discovery: no hardcoded column names. Candidate customer /
  product / order / date / quantity / value columns are found from roles,
  name tokens, uniqueness and value distributions in the *live* frame.
- Transaction grain required for individual predictions. Monthly/aggregate
  grains are refused honestly (they still support forecasting elsewhere).
- Deterministic counting-based methods with temporal (last-purchase holdout)
  validation; simplest method wins ties. No fabricated probabilities: scores
  are labelled scores, reliability is reported as measured hit-rate@k, timing
  as interval estimates, otherwise ``Confidence unavailable``.
- No ``exec``/``eval``/SQL; only validated columns and closed pandas ops.
- Depends only on pandas/numpy/re so it is unit-testable in isolation.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
import pandas as pd

__all__ = [
    "discover_purchase_schema",
    "build_purchase_histories",
    "validate_methods_temporal",
    "predict_next_for_customer",
    "predict_timing_for_customer",
    "customers_at_risk",
    "customers_for_product",
    "METHOD_ORDER",
]

METHOD_ORDER = ["popularity", "repeat_preference", "item_item", "collaborative"]
MIN_EVAL_CUSTOMERS = 5
RANDOM_SEED = 42

_CUSTOMER_TOKENS = ["customer_id", "customerid", "cust_id", "customer_no", "customer",
                    "client_id", "client", "account_id", "account", "member_id", "member",
                    "user_id", "buyer_id", "buyer", "shopper_id", "shopper", "custid",
                    "cust", "cid"]
_PRODUCT_TOKENS = ["product_id", "productid", "sku", "item_id", "itemid", "item_code",
                   "item", "material_id", "material", "product_code", "product_sku",
                   "goods_id", "product", "prod"]
_PRODUCT_LINE_FALLBACK_TOKENS = ["product_line", "product_category", "product_family",
                                 "product_group", "category"]
_ORDER_TOKENS = ["order_id", "orderid", "order_no", "invoice_id", "invoice_no", "invoice",
                 "sales_order_id", "order", "transaction_id", "txn_id", "receipt_id",
                 "receipt", "basket_id", "shipment_id", "sales_order", "ord"]
_DATE_TOKENS = ["purchase_date", "order_date", "invoice_date", "transaction_date",
                "txn_date", "buy_date", "date", "timestamp", "created", "sold_date"]
_QTY_TOKENS = ["quantity", "qty", "units", "units_shipped", "units_sold", "count"]
_VALUE_TOKENS = ["net_sales", "revenue", "sales", "amount", "total", "unit_price",
                 "price", "value", "extended_price", "line_total"]
_CATEGORY_TOKENS = ["product_family", "product_category", "category", "product_line",
                    "department", "brand", "product_group"]
_STATUS_TOKENS = ["status", "state", "order_status", "return_flag", "is_return",
                  "cancelled", "canceled", "void", "type", "transaction_type"]
_NEGATIVE_HINTS = re.compile(r"return|cancel|void|refund|credit|reject|fail", re.I)


def _token_score(name: str, tokens: list[str]) -> int:
    lowered = re.sub(r"[^a-z0-9]+", "_", str(name).lower()).strip("_")
    best = 0
    for tok in tokens:
        t = tok.strip("_")
        if len(t) <= 3 and lowered != t:
            continue  # short abbreviations match exactly only (avoids "acid" -> "cid")
        if lowered == t:
            best = max(best, 100)
        elif lowered.startswith(t + "_") or lowered.endswith("_" + t):
            best = max(best, 85)
        elif t in lowered:
            best = max(best, 55)
    return best


def _id_like(name: str, tokens: list[str], frame: pd.DataFrame,
             profile: dict[str, Any], role_ok: tuple[str, ...]) -> list[tuple[int, str]]:
    out = []
    for c in profile.get("columns", []):
        cname = c["name"]
        if cname not in frame.columns or c.get("is_empty"):
            continue
        if c.get("role") not in role_ok:
            continue
        s = _token_score(cname, tokens)
        if s > 0:
            out.append((s, cname))
    out.sort(key=lambda t: (-t[0], t[1]))
    return out


def discover_purchase_schema(frame: pd.DataFrame, profile: dict[str, Any]) -> dict[str, Any]:
    """Map purchase roles onto the live schema; assess grain and sufficiency."""
    issues: list[str] = []
    assumptions: list[str] = []
    mapping: dict[str, str | None] = {
        "customer_col": None, "product_col": None, "product_grain": None,
        "order_col": None, "date_col": None, "qty_col": None,
        "value_col": None, "category_col": None, "status_col": None,
    }

    # Customer: identifier/categorical with customer-ish name, not an order id,
    # and with repeat customers (avg rows per customer > 1.2).
    for score, cname in _id_like("c", _CUSTOMER_TOKENS, frame, profile,
                                 ("identifier", "categorical", "text", "numeric")):
        lowered = cname.lower()
        if any(t in lowered for t in ("order", "invoice", "shipment", "transaction", "txn", "receipt", "basket")):
            continue
        try:
            nunique = int(frame[cname].nunique(dropna=True))
        except Exception:
            continue
        if nunique < 2:
            continue
        avg_rows = len(frame) / max(nunique, 1)
        mapping["customer_col"] = cname
        if re.search(r"segment|group|type|category|cohort", lowered):
            # A segment/grouping column is not an individual customer identifier:
            # honest limitation for individual next-product prediction.
            issues.append(f"Column '{cname}' is a customer grouping (segment), not an "
                          f"individual customer identifier: individual next-product "
                          f"prediction is unsupported on this dataset.")
        elif avg_rows < 1.2:
            issues.append(f"Column '{cname}' looks like a customer id but averages "
                          f"{avg_rows:.1f} rows per customer: too sparse for personal prediction.")
        break
    if mapping["customer_col"] is None:
        issues.append("No customer identifier column found: individual next-purchase "
                      "prediction is unsupported on this dataset.")

    # Product: prefer item/SKU-level columns, else product-line-level (documented).
    prod_cands = _id_like("p", _PRODUCT_TOKENS, frame, profile, ("categorical", "identifier", "text", "numeric"))
    prod_cands = [(s, c) for s, c in prod_cands
                  if not any(t in c.lower() for t in ("line", "family", "category", "group", "segment"))]
    if prod_cands:
        mapping["product_col"] = prod_cands[0][1]
        mapping["product_grain"] = "item"
    else:
        for score, cname in _id_like("p", _PRODUCT_LINE_FALLBACK_TOKENS, frame, profile,
                                      ("categorical", "text")):
            mapping["product_col"] = cname
            mapping["product_grain"] = "product_line"
            assumptions.append(f"Predictions use '{cname}' (product-line grain): no SKU-level "
                               f"product column exists, so distinct items within a line cannot be distinguished.")
            break
    if mapping["product_col"] is None:
        issues.append("No product/item column found: the next product cannot be determined.")

    # Order id (optional but used for order-grain histories + dedup evidence).
    order_cands = _id_like("o", _ORDER_TOKENS, frame, profile, ("identifier", "categorical", "text", "numeric"))
    if order_cands:
        mapping["order_col"] = order_cands[0][1]

    # Date: prefer purchase-ish datetime columns.
    date_cands = _id_like("d", _DATE_TOKENS, frame, profile, ("datetime",))
    if date_cands:
        mapping["date_col"] = date_cands[0][1]
    elif profile.get("datetime_columns"):
        mapping["date_col"] = profile["datetime_columns"][0]
        assumptions.append(f"Using '{mapping['date_col']}' as the purchase timestamp "
                           f"(no purchase-specific date column name found).")
    if mapping["date_col"] is None:
        issues.append("No purchase timestamp column found: chronological next-purchase "
                      "prediction is unsupported.")

    # Quantity / value / category (optional enrichments).
    qty_cands = _id_like("q", _QTY_TOKENS, frame, profile, ("numeric",))
    if qty_cands:
        mapping["qty_col"] = qty_cands[0][1]
    val_cands = _id_like("v", _VALUE_TOKENS, frame, profile, ("numeric",))
    if val_cands:
        mapping["value_col"] = val_cands[0][1]
    cat_cands = _id_like("c", _CATEGORY_TOKENS, frame, profile, ("categorical", "text"))
    cat_cands = [(s, c) for s, c in cat_cands if c != mapping["product_col"]]
    if cat_cands:
        mapping["category_col"] = cat_cands[0][1]

    # Status-like column for returns/cancellations.
    for score, cname in _id_like("s", _STATUS_TOKENS, frame, profile, ("categorical", "text", "boolean", "numeric")):
        mapping["status_col"] = cname
        break

    # Grain assessment.
    grain = "unknown"
    if mapping["customer_col"] and mapping["product_col"] and mapping["date_col"]:
        grain = "transaction"
        # Aggregate guard: each (customer, date) has at most one row AND the date
        # has day-precision stripped (e.g. monthly buckets) -> likely aggregated.
        try:
            dc = pd.to_datetime(frame[mapping["date_col"]], errors="coerce")
            if dc.notna().mean() > 0.9:
                is_month_start = (dc.dt.day == 1).mean() > 0.9
                pairs = pd.DataFrame({"c": frame[mapping["customer_col"]].astype(str),
                                      "d": dc.dt.date}).drop_duplicates()
                if is_month_start and len(pairs) == len(frame):
                    grain = "aggregate"
                    issues.append("Data looks aggregated (one row per customer per month-start period): "
                                  "individual next-product prediction is unsupported; aggregate forecasting applies.")
        except Exception:
            pass
    elif mapping["date_col"] and not mapping["customer_col"]:
        grain = "aggregate"

    # Any recorded issue blocks individual prediction (assumptions live separately
    # and never block; they are reported transparently instead).
    sufficient = grain == "transaction" and not issues
    return {"mapping": mapping, "grain": grain, "issues": issues,
            "assumptions": assumptions, "sufficient": sufficient}


def build_purchase_histories(frame: pd.DataFrame, schema: dict[str, Any]) -> dict[str, Any]:
    """Chronological per-customer histories with documented cleaning.

    Returns histories + cleaning report (invalid dates, duplicates, excluded
    return/cancel rows). Training examples always use strictly-past data.
    """
    m = schema["mapping"]
    ccol, pcol, ocol, dcol = m["customer_col"], m["product_col"], m["order_col"], m["date_col"]
    report: dict[str, Any] = {"excluded_returns": 0, "duplicate_rows": 0,
                              "invalid_dates": 0, "rows_used": 0}
    if not (ccol and pcol and dcol):
        return {"customers": {}, "report": report, "clean": pd.DataFrame()}

    work = pd.DataFrame({
        "customer": frame[ccol].astype(str),
        "product": frame[pcol].astype(str),
        "order": frame[ocol].astype(str) if ocol and ocol in frame.columns else "__row__",
        "date": pd.to_datetime(frame[dcol], errors="coerce"),
        "qty": (pd.to_numeric(frame[m["qty_col"]], errors="coerce").fillna(1)
                if m["qty_col"] and m["qty_col"] in frame.columns else 1),
        "category": (frame[m["category_col"]].astype(str)
                     if m["category_col"] and m["category_col"] in frame.columns else ""),
    })
    bad_dates = int(work["date"].isna().sum())
    report["invalid_dates"] = bad_dates
    work = work.dropna(subset=["date"])
    work = work[(work["customer"] != "") & (work["customer"].str.lower() != "nan")
                & (work["product"] != "") & (work["product"].str.lower() != "nan")]
    # Returns/cancellations: excluded from "purchases", counted transparently.
    if m["status_col"] and m["status_col"] in frame.columns:
        neg = frame[m["status_col"]].astype(str).str.contains(_NEGATIVE_HINTS, na=False)
        neg = neg.reindex(work.index, fill_value=False)
        report["excluded_returns"] = int(neg.sum())
        work = work[~neg]
        assumptions_note = (f"Rows flagged as return/cancel/refund via '{m['status_col']}' "
                            f"are excluded from purchase histories.")
    else:
        assumptions_note = ("No return/cancel indicator column detected: every row is "
                            "treated as a purchase; negative quantities (if any) are kept as recorded.")
        try:
            neg_qty = int((pd.to_numeric(work["qty"], errors="coerce") < 0).sum())
            if neg_qty:
                assumptions_note += f" ({neg_qty} negative-quantity rows kept.)"
        except Exception:
            pass
    report["assumption"] = assumptions_note
    before = len(work)
    work = work.drop_duplicates(subset=["customer", "order", "product", "date"])
    report["duplicate_rows"] = int(before - len(work))
    work = work.sort_values(["customer", "date", "order"]).reset_index(drop=True)
    report["rows_used"] = int(len(work))

    customers: dict[str, Any] = {}
    for cust, grp in work.groupby("customer", sort=False):
        grp = grp.sort_values(["date", "order"])
        orders = []
        for (odate, oid), og in grp.groupby(["date", "order"], sort=True):
            items = og["product"].tolist()
            orders.append({"date": str(pd.Timestamp(odate).date()), "order_id": str(oid),
                           "products": items,
                           "quantities": {p: float(og[og["product"] == p]["qty"].sum()) for p in set(items)}})
        customers[str(cust)] = {"orders": orders,
                                "products": grp["product"].tolist(),
                                "order_count": len(orders)}
    return {"customers": customers, "report": report, "clean": work}


# --- Scoring methods ----------------------------------------------------------

def _recency_weight(days: float) -> float:
    return 1.0 / (1.0 + max(days, 0.0) / 30.0)


def _score_repeat(history_products: list[tuple[str, pd.Timestamp]],
                  as_of: pd.Timestamp, depth: int | None = None) -> dict[str, float]:
    """Frequency x recency over strictly-past purchases (leakage-safe)."""
    seq = history_products[:depth] if depth is not None else history_products
    scores: dict[str, float] = {}
    counts: dict[str, int] = {}
    last_seen: dict[str, pd.Timestamp] = {}
    for prod, ts in seq:
        if ts >= as_of:
            continue
        counts[prod] = counts.get(prod, 0) + 1
        if prod not in last_seen or ts > last_seen[prod]:
            last_seen[prod] = ts
    for prod, cnt in counts.items():
        days = (as_of - last_seen[prod]).days
        scores[prod] = cnt * _recency_weight(days)
    return scores


def _score_popularity(global_counts: dict[str, int]) -> dict[str, float]:
    total = sum(global_counts.values()) or 1
    return {p: c / total for p, c in global_counts.items()}


def _score_item_item(co_matrix: dict[str, dict[str, int]],
                     recent_items: list[str]) -> dict[str, float]:
    scores: dict[str, float] = {}
    for item in recent_items:
        for other, cnt in co_matrix.get(item, {}).items():
            if other in recent_items:
                continue
            scores[other] = scores.get(other, 0.0) + cnt
    return scores


def _score_collaborative(target_items: set[str], others: dict[str, set[str]],
                         k: int = 5) -> dict[str, float]:
    sims = []
    for cust, items in others.items():
        union = len(target_items | items)
        sim = len(target_items & items) / union if union else 0.0
        if sim > 0:
            sims.append((sim, items))
    sims.sort(key=lambda t: -t[0])
    scores: dict[str, float] = {}
    for sim, items in sims[:k]:
        for p in items - target_items:
            scores[p] = scores.get(p, 0.0) + sim
    return scores


def _build_aux(clean: pd.DataFrame) -> tuple[dict[str, int], dict[str, dict[str, int]]]:
    global_counts = clean["product"].astype(str).value_counts().to_dict()
    co: dict[str, dict[str, int]] = {}
    for (_, _), og in clean.groupby(["customer", "order"]):
        items = sorted(set(og["product"].astype(str).tolist()))
        for i, a in enumerate(items):
            for b in items[i + 1:]:
                co.setdefault(a, {}).setdefault(b, 0)
                co[a][b] += 1
                co.setdefault(b, {}).setdefault(a, 0)
                co[b][a] += 1
    return global_counts, co


def _rank(scores: dict[str, float], top_k: int) -> list[tuple[str, float]]:
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]


def validate_methods_temporal(histories: dict[str, Any],
                              top_k: int = 3) -> dict[str, Any]:
    """Last-purchase holdout per customer with >=2 orders. No future leakage:
    every method trains only on strictly-earlier orders."""
    eval_custs = {c: h for c, h in histories.items() if h["order_count"] >= 2}
    if len(eval_custs) < MIN_EVAL_CUSTOMERS:
        return {"evaluable_customers": len(eval_custs),
                "required": MIN_EVAL_CUSTOMERS,
                "validated": False,
                "reason": (f"Only {len(eval_custs)} customers have >= 2 orders "
                           f"(need {MIN_EVAL_CUSTOMERS} for temporal validation)."),
                "metrics": {}, "chosen_method": "repeat_preference"}
    hits = {m: {"h1": 0, "h3": 0, "rr": 0.0} for m in METHOD_ORDER}
    n = 0
    for cust, h in sorted(eval_custs.items()):
        orders = h["orders"]
        train_orders, held = orders[:-1], orders[-1]
        held_items = set(held["products"])
        train_products = [(p, pd.Timestamp(o["date"])) for o in train_orders for p in o["products"]]
        as_of = pd.Timestamp(held["date"])
        past_seq = [(p, t) for p, t in train_products if t < as_of]
        if not past_seq or not held_items:
            continue
        n += 1
        # Every validation method must use only events known at this customer's
        # historical prediction time. Other customers' final held-out purchases
        # are future information too and must not enter popularity, co-purchase,
        # or collaborative evidence.
        global_counts: dict[str, int] = {}
        co: dict[str, dict[str, int]] = {}
        other_sets: dict[str, set[str]] = {}
        for c2, h2 in histories.items():
            visible_products: set[str] = set()
            for order in h2["orders"]:
                order_date = pd.Timestamp(order["date"])
                if order_date >= as_of:
                    continue
                items = sorted(set(order["products"]))
                visible_products.update(items)
                for product in items:
                    global_counts[product] = global_counts.get(product, 0) + 1
                for i, a in enumerate(items):
                    for b in items[i + 1:]:
                        co.setdefault(a, {}).setdefault(b, 0)
                        co[a][b] += 1
                        co.setdefault(b, {}).setdefault(a, 0)
                        co[b][a] += 1
            if c2 != cust:
                other_sets[c2] = visible_products
        recent = [p for p, _ in past_seq[-5:]]
        cand = {
            "popularity": _score_popularity(global_counts),
            "repeat_preference": _score_repeat(past_seq, as_of),
            "item_item": _score_item_item(co, recent),
            "collaborative": _score_collaborative(set(p for p, _ in past_seq), other_sets),
        }
        for m, scores in cand.items():
            ranked = [p for p, _ in _rank(scores, 3)]
            if ranked and ranked[0] in held_items:
                hits[m]["h1"] += 1
            if any(p in held_items for p in ranked):
                hits[m]["h3"] += 1
            for rank, p in enumerate(ranked, start=1):
                if p in held_items:
                    hits[m]["rr"] += 1.0 / rank
                    break
    if n < MIN_EVAL_CUSTOMERS:
        return {"evaluable_customers": n, "required": MIN_EVAL_CUSTOMERS,
                "validated": False,
                "reason": (f"Only {n} customers yielded valid holdout examples "
                           f"(need {MIN_EVAL_CUSTOMERS})."),
                "metrics": {}, "chosen_method": "repeat_preference"}
    metrics = {}
    for m in METHOD_ORDER:
        metrics[m] = {"hit_rate_at_1": round(hits[m]["h1"] / n, 3),
                      "hit_rate_at_3": round(hits[m]["h3"] / n, 3),
                      "mrr": round(hits[m]["rr"] / n, 3)}
    # Simplest method wins ties (METHOD_ORDER is simplest-first).
    best = METHOD_ORDER[0]
    for m in METHOD_ORDER[1:]:
        if (metrics[m]["mrr"], metrics[m]["hit_rate_at_3"]) > (metrics[best]["mrr"], metrics[best]["hit_rate_at_3"]):
            best = m
    return {"evaluable_customers": n, "required": MIN_EVAL_CUSTOMERS,
            "validated": True, "metrics": metrics, "chosen_method": best}


def predict_next_for_customer(clean: pd.DataFrame, histories: dict[str, Any],
                              customer: str, method: str, top_k: int = 3) -> dict[str, Any]:
    """Score products from strictly-past history of this customer."""
    h = histories.get(customer)
    if h is None or not h["orders"]:
        # Cold start: explicitly labeled fallback with evidence of what was used
        # and what is missing. Prefer category-aware best-sellers when a category
        # column exists; otherwise global popularity.
        global_counts, _ = _build_aux(clean)
        fallback = {"why": "no purchase history for this customer",
                    "evidence_used": ["global product frequencies"],
                    "missing": ["customer purchase history", "repeat/interval signals"],
                    "is_personalized": False}
        ranked_raw = _rank(_score_popularity(global_counts), top_k)
        category_note = None
        if "category" in clean.columns and str(clean["category"].astype(str).str.strip().ne("").any()):
            cat_pop = (clean.assign(_cat=clean["category"].astype(str))
                       .groupby("_cat")["product"].agg(lambda s: s.value_counts().to_dict()))
            top_cats = clean["category"].astype(str).value_counts().head(3).to_dict()
            category_note = {str(k): int(v) for k, v in top_cats.items()}
            fallback["evidence_used"].append("category popularity distribution")
            fallback["why"] += "; category best-sellers shown alongside global ones"
        ranked = []
        for p, s in ranked_raw:
            try:
                cat = str(clean.loc[clean["product"].astype(str) == str(p), "category"]
                          .mode().iloc[0]) if "category" in clean.columns else ""
            except Exception:
                cat = ""
            ranked.append({"product": p, "score": round(float(s), 4),
                           "evidence": {"times_bought_by_customer": 0,
                                        "last_bought": None,
                                        "category": cat or None,
                                        "reason": "no history for this customer; best-seller fallback"}})
        return {"customer": customer, "cold_start": True, "label": "recommendation",
                "ranked": ranked, "method": "popularity",
                "fallback": fallback, "category_popularity": category_note}
    past_seq = [(p, pd.Timestamp(o["date"])) for o in h["orders"] for p in o["products"]]
    as_of = pd.Timestamp(h["orders"][-1]["date"]) + pd.Timedelta(days=1)
    recent = [p for p, _ in past_seq[-5:]]
    if method == "popularity":
        counts: dict[str, int] = {}
        for p, _ in past_seq:
            counts[p] = counts.get(p, 0) + 1
        scores = _score_popularity(counts)
    elif method == "item_item":
        _, co = _build_aux(clean[clean["customer"] != customer] if "customer" in clean.columns else clean)
        scores = _score_item_item(co, recent)
        if not scores:
            scores = _score_repeat(past_seq, as_of)
    elif method == "collaborative":
        other_sets = {}
        for c2, h2 in histories.items():
            if c2 != customer:
                other_sets[c2] = set(h2["products"])
        scores = _score_collaborative(set(p for p, _ in past_seq), other_sets)
        if not scores:
            scores = _score_repeat(past_seq, as_of)
    else:
        scores = _score_repeat(past_seq, as_of)
    if not scores:
        global_counts, _ = _build_aux(clean)
        scores = _score_popularity(global_counts)
    # Evidence per product from this customer's own history.
    last_seen: dict[str, str] = {}
    counts_c: dict[str, int] = {}
    co_with: dict[str, list[str]] = {}
    for o in h["orders"]:
        for p in o["products"]:
            counts_c[p] = counts_c.get(p, 0) + 1
            last_seen[p] = o["date"]
    ranked = []
    total = sum(scores.values()) or 1.0
    for p, s in _rank(scores, top_k):
        ranked.append({"product": p, "score": round(float(s / total), 4),
                       "evidence": {"times_bought_by_customer": counts_c.get(p, 0),
                                    "last_bought": last_seen.get(p),
                                    "also_bought_with": sorted({q for o in h["orders"]
                                                                if p in o["products"]
                                                                for q in o["products"] if q != p})[:5]}})
    return {"customer": customer, "cold_start": False, "label": "prediction",
            "ranked": ranked, "method": method}


def predict_timing_for_customer(histories: dict[str, Any], customer: str) -> dict[str, Any]:
    """Inter-purchase interval estimate; unavailable when history is too thin."""
    h = histories.get(customer)
    if h is None or len(h["orders"]) < 2:
        return {"status": "unavailable",
                "reason": "Confidence unavailable: fewer than 2 dated orders for this customer, "
                          "so no purchase interval can be estimated."}
    dates = sorted(pd.Timestamp(o["date"]) for o in h["orders"])
    gaps = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1)]
    gaps = [g for g in gaps if g >= 0]
    if not gaps:
        return {"status": "unavailable",
                "reason": "Confidence unavailable: order dates are identical; no interval measurable."}
    median = float(np.median(gaps))
    mean = float(np.mean(gaps))
    std = float(np.std(gaps)) if len(gaps) > 1 else 0.0
    nxt = dates[-1] + pd.Timedelta(days=int(round(median)))
    lo = dates[-1] + pd.Timedelta(days=int(round(max(median - std, 0)))) if std else None
    hi = dates[-1] + pd.Timedelta(days=int(round(median + std))) if std else None
    return {"status": "estimate", "type": "interval_estimate",
            "predicted_next_date": str(nxt.date()),
            "range": {"lower": str(lo.date()) if lo else None,
                      "upper": str(hi.date()) if hi else None},
            "median_interval_days": round(median, 1), "mean_interval_days": round(mean, 1),
            "std_interval_days": round(std, 1), "intervals_observed": len(gaps),
            "note": ("Estimate from this customer's own inter-purchase intervals, not a "
                     "calibrated probability.")}


def customers_at_risk(histories: dict[str, Any], as_of: pd.Timestamp | None = None,
                      limit: int = 20) -> dict[str, Any]:
    """Customers whose silence exceeds their own typical interval (median + 2*std)."""
    rows = []
    now = as_of or pd.Timestamp.now(tz=None)
    for cust, h in histories.items():
        if len(h["orders"]) < 2:
            continue
        dates = sorted(pd.Timestamp(o["date"]) for o in h["orders"])
        gaps = [(dates[i + 1] - dates[i]).days for i in range(len(dates) - 1) if (dates[i + 1] - dates[i]).days >= 0]
        if not gaps:
            continue
        med = float(np.median(gaps))
        std = float(np.std(gaps)) if len(gaps) > 1 else 0.0
        threshold = med + 2 * std if std else 2 * med
        silent = (now - dates[-1]).days
        if silent > threshold and silent > 0:
            rows.append({"customer": cust, "days_since_last": int(silent),
                         "typical_interval_days": round(med, 1),
                         "last_purchase": str(dates[-1].date()),
                         "orders_observed": len(h["orders"])})
    rows.sort(key=lambda r: -r["days_since_last"])
    return {"at_risk": rows[:limit], "evaluated": sum(1 for h in histories.values() if len(h["orders"]) >= 2),
            "rule": "days_since_last > median_interval + 2*std (or > 2x median when std unavailable)"}


def customers_for_product(clean: pd.DataFrame, histories: dict[str, Any],
                          product: str, limit: int = 20) -> dict[str, Any]:
    """Rank known buyers of a product by recency x frequency of that product."""
    rows = []
    for cust, h in histories.items():
        buys = [(pd.Timestamp(o["date"])) for o in h["orders"] if product in o["products"]]
        if not buys:
            continue
        last = max(buys)
        score = len(buys) * _recency_weight((pd.Timestamp(h["orders"][-1]["date"]) - last).days)
        rows.append({"customer": cust, "times_bought": len(buys),
                     "last_bought": str(last.date()), "score": round(float(score), 4)})
    rows.sort(key=lambda r: (-r["score"], r["customer"]))
    return {"product": product, "buyers": rows[:limit], "buyer_count": len(rows)}
