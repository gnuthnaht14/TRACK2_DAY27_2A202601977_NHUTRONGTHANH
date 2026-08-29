from datetime import datetime, timedelta, timezone
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

from student_api import (
    column_downstream,
    detect_distribution,
    detect_metric,
    downstream_assets,
    multiwindow_burn,
    rag_embedding_shift,
    rag_length_shift,
    slo_status,
    validate_orders,
)
from src.contract_validator import determine_action, failed_issues, load_contract, validate_dataframe

ROOT = Path(__file__).resolve().parents[1]
ORDERS_CONTRACT = ROOT / "contracts" / "orders_contract.yaml"
KB_CONTRACT = ROOT / "contracts" / "kb_contract.yaml"


def test_type_drift_is_detected_in_contracts():
    df = pd.DataFrame([
        {
            "order_id": "NOT_AN_INT",  # Invalid int
            "customer_id": "C1",
            "amount": 100.0,
            "currency": "USD",
            "status": "completed",
            "created_at": "2026-08-28T10:00:00Z",
            "updated_at": "2026-08-28T10:05:00Z",
        }
    ])
    issues = validate_orders(df, ORDERS_CONTRACT)
    failed = [i for i in issues if not i["passed"]]
    assert any(i["check"] == "type" and i["column"] == "order_id" for i in failed)


def test_freshness_violation_in_contracts():
    # 2 hours old data against a 30-minute max delay
    old_time = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    df = pd.DataFrame([
        {
            "order_id": 1,
            "customer_id": "C1",
            "amount": 50.0,
            "currency": "USD",
            "status": "completed",
            "created_at": old_time,
            "updated_at": old_time,
        }
    ])
    issues = validate_dataframe(df, load_contract(ORDERS_CONTRACT))
    failed = [i for i in issues if not i["passed"]]
    assert any(i["check"] == "freshness" and i["column"] == "updated_at" for i in failed)


def test_determine_action_policy():
    critical_issues = [{"check": "unique", "severity": "critical", "passed": False}]
    warning_issues = [{"check": "freshness", "severity": "warning", "passed": False}]
    passed_issues = [{"check": "not_null", "severity": "critical", "passed": True}]

    assert determine_action(critical_issues) == "block"
    assert determine_action(warning_issues) == "quarantine"
    assert determine_action(passed_issues) == "pass"


def test_mad_zero_mad_edge_case():
    # Over 50% identical values causing MAD = 0
    history = [100.0, 100.0, 100.0, 100.0, 105.0, 95.0]
    # Extreme outlier
    result = detect_metric(500.0, history, method="mad")
    assert result["is_anomaly"] is True

    # Normal value matching median
    result_normal = detect_metric(100.0, history, method="mad")
    assert result_normal["is_anomaly"] is False


def test_context_aware_auto_anomaly_detection():
    # General history has variance, but same-segment (e.g. Saturday) is low volume
    general_history = [1000, 1050, 980, 1020, 1010]
    saturday_history = [300, 310, 295, 305, 302]

    # Without context, 300 looks like an anomaly against general_history (1000)
    # But WITH same_segment_history, 305 is completely normal on Saturday
    result_with_context = detect_metric(
        305,
        general_history,
        method="auto",
        context={"day_of_week": 5, "same_segment_history": saturday_history},
    )
    assert result_with_context["is_anomaly"] is False


def test_ks_distribution_shift():
    baseline_dist = np.random.normal(loc=50.0, scale=5.0, size=100)
    shifted_dist = np.random.normal(loc=150.0, scale=5.0, size=100)

    result = detect_distribution(shifted_dist, baseline_dist)
    assert result["is_anomaly"] is True
    assert result["method"] == "ks_distribution_test"


def test_transitive_column_lineage():
    column_graph = {
        "raw_orders.order_id": ["stg_orders.order_id"],
        "stg_orders.order_id": ["fct_daily_revenue.order_count", "fct_order_summary.order_id"],
        "fct_daily_revenue.order_count": ["dashboard.kpi_orders"],
    }
    downstream = column_downstream(column_graph, "raw_orders.order_id")
    assert downstream == [
        "stg_orders.order_id",
        "fct_daily_revenue.order_count",
        "fct_order_summary.order_id",
        "dashboard.kpi_orders",
    ]


def test_multiwindow_burn_rate_policies():
    # 1. Sustained Fast Burn -> Must PAGE
    fast_burn = multiwindow_burn(short_window_burn=15.0, long_window_burn=15.0)
    assert fast_burn["page"] is True
    assert fast_burn["severity"] == "critical"

    # 2. Transient Spike (short elevated, long low) -> Do NOT Page, Warn
    transient = multiwindow_burn(short_window_burn=15.0, long_window_burn=1.0)
    assert transient["page"] is False
    assert transient["severity"] == "warning"

    # 3. Healthy / Low Burn -> Info
    healthy = multiwindow_burn(short_window_burn=0.8, long_window_burn=0.9)
    assert healthy["page"] is False
    assert healthy["severity"] == "info"


def test_rag_embedding_norm_drift():
    baseline_norms = [1.0, 1.02, 0.98, 1.01, 0.99, 1.0, 1.01]
    # Collapse in vector norms
    drifted_norms = [0.1, 0.12, 0.09, 0.11]
    res = rag_embedding_shift(drifted_norms, baseline_norms)
    assert res["is_anomaly"] is True
