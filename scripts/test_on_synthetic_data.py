#!/usr/bin/env python3
"""Independent Synthetic Data Generator and Reliability Test Runner.

Generates dynamic, randomized datasets across 9 diverse scenarios and tests
the system's detection capabilities via student_api.
"""
from __future__ import annotations

import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

# Ensure UTF-8 output on Windows console
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import student_api


# ==========================================
# 1. INDEPENDENT SYNTHETIC DATA GENERATORS
# ==========================================

def generate_healthy_orders(n_rows: int = 200, seed: int = 42) -> pd.DataFrame:
    """Generate a clean, completely valid orders dataset."""
    random.seed(seed)
    now = datetime.now(timezone.utc)
    rows = []
    statuses = ["completed", "completed", "pending", "refunded"]
    currencies = ["USD", "VND"]

    for i in range(1, n_rows + 1):
        updated = now - timedelta(minutes=random.randint(2, 20))
        created = updated - timedelta(minutes=random.randint(5, 120))
        rows.append({
            "order_id": i,
            "customer_id": f"C{random.randint(1, 50):04d}",
            "amount": round(random.uniform(10.0, 500.0), 2),
            "currency": random.choice(currencies),
            "status": random.choice(statuses),
            "created_at": created.isoformat(),
            "updated_at": updated.isoformat(),
        })
    return pd.DataFrame(rows)


def generate_type_drift_orders(n_rows: int = 100, seed: int = 99) -> pd.DataFrame:
    """Generate a dataset with subtle and severe type drift (strings in integers, negative amounts)."""
    df = generate_healthy_orders(n_rows, seed=seed)
    df.loc[5, "order_id"] = "ORD_005_DIRTY"  # String in integer
    df.loc[12, "order_id"] = 12.75  # Float in integer
    df.loc[20, "amount"] = -99.50  # Negative amount
    df.loc[30, "currency"] = "EUR"  # Unaccepted currency
    df.loc[40, "created_at"] = "invalid-date-format"
    return df


def generate_stale_orders(n_rows: int = 100, delay_hours: float = 3.5) -> pd.DataFrame:
    """Generate orders with timestamps delayed past the freshness SLA."""
    df = generate_healthy_orders(n_rows, seed=123)
    stale_time = datetime.now(timezone.utc) - timedelta(hours=delay_hours)
    df["updated_at"] = [(stale_time - timedelta(minutes=i)).isoformat() for i in range(n_rows)]
    return df


def generate_duplicate_pk_orders(n_rows: int = 100) -> pd.DataFrame:
    """Generate orders with duplicate primary keys."""
    df = generate_healthy_orders(n_rows, seed=777)
    df.loc[10, "order_id"] = df.loc[0, "order_id"]
    df.loc[25, "order_id"] = df.loc[5, "order_id"]
    return df


def generate_distribution_samples(n_samples: int = 300, shift_type: str = "none", seed: int = 55) -> tuple[list[float], list[float]]:
    """Generate baseline and current distribution value streams."""
    rng = np.random.default_rng(seed)
    baseline = rng.normal(loc=100.0, scale=15.0, size=n_samples).tolist()

    if shift_type == "none":
        current = rng.normal(loc=100.0, scale=15.0, size=n_samples).tolist()
    elif shift_type == "mean_shift":
        current = rng.normal(loc=250.0, scale=15.0, size=n_samples).tolist()
    elif shift_type == "bimodal_variance_shift":
        c1 = rng.normal(loc=40.0, scale=5.0, size=n_samples // 2)
        c2 = rng.normal(loc=180.0, scale=5.0, size=n_samples // 2)
        current = np.concatenate([c1, c2]).tolist()
    else:
        current = baseline
    return current, baseline


def generate_complex_graph() -> tuple[dict[str, list[str]], str, list[str]]:
    """Generate a multi-tier diamond dependency graph with potential cycles."""
    graph = {
        "raw_cdc.orders": ["stg_orders", "cdc_audit_log"],
        "stg_orders": ["int_orders_enriched", "fct_orders"],
        "int_orders_enriched": ["fct_orders", "dim_customer_orders"],
        "fct_orders": ["fct_daily_revenue", "ml_fraud_detection"],
        "fct_daily_revenue": ["ceo_dashboard", "finance_ledger"],
        "finance_ledger": ["raw_cdc.orders"],  # Cycle back
    }
    start_node = "raw_cdc.orders"
    expected_downstream = [
        "stg_orders",
        "cdc_audit_log",
        "int_orders_enriched",
        "fct_orders",
        "dim_customer_orders",
        "fct_daily_revenue",
        "ml_fraud_detection",
        "ceo_dashboard",
        "finance_ledger",
    ]
    return graph, start_node, expected_downstream


# ==========================================
# 2. COMPREHENSIVE TEST RUNNER
# ==========================================

def run_synthetic_test_suite() -> bool:
    print("=" * 80)
    print("RUNNING COMPREHENSIVE TEST ON NEW INDEPENDENT SYNTHETIC DATASETS")
    print("=" * 80)

    contract_path = ROOT / "contracts" / "orders_contract.yaml"
    all_passed = True
    results: list[dict[str, Any]] = []

    def log_result(test_name: str, passed: bool, expected: str, actual: str):
        nonlocal all_passed
        if not passed:
            all_passed = False
        status = "[PASS]" if passed else "[FAIL]"
        results.append({
            "Test Scenario": test_name,
            "Status": status,
            "Expected": expected,
            "Actual": actual,
        })

    # Test 1: Clean Synthetic Dataset
    df_healthy = generate_healthy_orders(n_rows=350, seed=101)
    issues_healthy = student_api.validate_orders(df_healthy, contract_path)
    failed_healthy = [i for i in issues_healthy if not i["passed"]]
    log_result(
        "1. Healthy Synthetic Dataset Validation",
        len(failed_healthy) == 0,
        "0 failures",
        f"{len(failed_healthy)} failures",
    )

    # Test 2: Type Drift Synthetic Dataset
    df_typedrift = generate_type_drift_orders(n_rows=150, seed=202)
    issues_typedrift = student_api.validate_orders(df_typedrift, contract_path)
    failed_types = [i for i in issues_typedrift if not i["passed"] and i["check"] == "type"]
    failed_currencies = [i for i in issues_typedrift if not i["passed"] and i["check"] == "accepted_values"]
    log_result(
        "2. Type Drift Detection (string/float in int & currency drift)",
        len(failed_types) >= 1 and len(failed_currencies) >= 1,
        "Bắt được type issue và accepted_values issue",
        f"type_fails={len(failed_types)}, currency_fails={len(failed_currencies)}",
    )

    # Test 3: Freshness SLA Breach Dataset
    df_stale = generate_stale_orders(n_rows=100, delay_hours=3.5)
    issues_stale = student_api.validate_orders(df_stale, contract_path)
    failed_fresh = [i for i in issues_stale if not i["passed"] and i["check"] == "freshness"]
    log_result(
        "3. Freshness SLA Violation (3.5h delay > 30m threshold)",
        len(failed_fresh) >= 1,
        "Bắt được freshness warning check",
        f"freshness_fails={len(failed_fresh)}",
    )

    # Test 4: Duplicate Primary Key Dataset
    df_dup = generate_duplicate_pk_orders(n_rows=120)
    issues_dup = student_api.validate_orders(df_dup, contract_path)
    failed_dup = [i for i in issues_dup if not i["passed"] and i["check"] == "unique"]
    log_result(
        "4. Duplicate Primary Key Detection (order_id uniqueness)",
        len(failed_dup) >= 1 and failed_dup[0]["severity"] == "critical",
        "Bắt được unique critical failure",
        f"dup_fails={len(failed_dup)}",
    )

    # Test 5: Statistical Anomaly (Volume Drop & Spike)
    history_vol = [1200, 1250, 1180, 1220, 1210, 1190, 1230]
    res_drop = student_api.detect_metric(150, history_vol, method="auto")
    res_normal = student_api.detect_metric(1205, history_vol, method="auto")
    log_result(
        "5. Anomaly Detection (Volume Drop 150 vs Normal 1205)",
        res_drop["is_anomaly"] is True and res_normal["is_anomaly"] is False,
        "Drop=Anomaly(True), Normal=Anomaly(False)",
        f"Drop={res_drop['is_anomaly']} (score={res_drop['score']:.2f}), Normal={res_normal['is_anomaly']}",
    )

    # Test 6: Distribution Shift (Two-Sample KS Test on Bimodal Drift)
    cur_dist, base_dist = generate_distribution_samples(n_samples=400, shift_type="bimodal_variance_shift")
    res_dist = student_api.detect_distribution(cur_dist, base_dist)
    log_result(
        "6. Distribution Drift (Bimodal Variance & Shape Shift)",
        res_dist["is_anomaly"] is True,
        "is_anomaly=True with KS test",
        f"is_anomaly={res_dist['is_anomaly']} (score={res_dist['score']:.3f}, method={res_dist['method']})",
    )

    # Test 7: Complex Diamond & Cycle Lineage Graph Traversal
    graph, start, expected_assets = generate_complex_graph()
    actual_assets = student_api.downstream_assets(graph, start)
    log_result(
        "7. Complex Lineage Traversal (Diamond & Cycle-Safe BFS)",
        set(actual_assets) == set(expected_assets) and len(actual_assets) == len(expected_assets),
        f"{len(expected_assets)} unique assets traversed",
        f"{len(actual_assets)} unique assets traversed without infinite loops",
    )

    # Test 8: SLO Multi-window Burn Alerting Policies
    res_page = student_api.multiwindow_burn(short_window_burn=18.0, long_window_burn=18.0)
    res_warn = student_api.multiwindow_burn(short_window_burn=18.0, long_window_burn=0.5)
    res_info = student_api.multiwindow_burn(short_window_burn=0.5, long_window_burn=0.8)
    log_result(
        "8. SLO Multi-window Burn Policy (Sustained vs Transient vs Healthy)",
        res_page["page"] is True and res_warn["page"] is False and res_info["severity"] == "info",
        "Fast Burn=Page(True), Transient Spike=Page(False)/Warn, Healthy=Info",
        f"FastBurn page={res_page['page']}, Transient page={res_warn['page']} (sev={res_warn['severity']})",
    )

    # Test 9: RAG Vector Embedding Collapse & Text Drift
    base_norms = [1.0, 1.02, 0.99, 1.01, 1.0, 0.98, 1.01]
    zero_norms = [0.001, 0.002, 0.001]
    res_rag_norm = student_api.rag_embedding_shift(zero_norms, base_norms)
    res_rag_len = student_api.rag_length_shift(["a b", "c d"], [50.0, 52.0, 48.0, 51.0, 49.0])
    log_result(
        "9. RAG Text Collapse & Zero-Vector Embedding Drift",
        res_rag_norm["is_anomaly"] is True and res_rag_len["is_anomaly"] is True,
        "Both RAG signals trigger anomaly",
        f"Embedding drift={res_rag_norm['is_anomaly']}, Text collapse={res_rag_len['is_anomaly']}",
    )

    # Format output as clean text table
    print("\n{:<65} | {:<8} | {:<40}".format("Test Scenario", "Status", "Actual Detail"))
    print("-" * 120)
    for r in results:
        print("{:<65} | {:<8} | {:<40}".format(r["Test Scenario"], r["Status"], r["Actual"]))
    print("-" * 120)

    if all_passed:
        print("ALL 9 SYNTHETIC DATASET SCENARIOS PASSED WITH 100% ACCURACY!")
    else:
        print("SOME SYNTHETIC TESTS FAILED.")
    print("=" * 80)
    return all_passed


if __name__ == "__main__":
    success = run_synthetic_test_suite()
    sys.exit(0 if success else 1)
