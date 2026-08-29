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


def test_direct_dict_contract_validation():
    contract_dict = {
        "columns": {
            "order_id": {"type": "integer", "required": True, "unique": True, "severity": "critical"},
            "amount": {"type": "number", "min": 0, "severity": "critical"},
        }
    }
    df = pd.DataFrame([{"order_id": 1, "amount": -10.0}])
    issues = validate_dataframe(df, contract_dict)
    failed = [i for i in issues if not i["passed"]]
    assert any(i["check"] == "range" and i["column"] == "amount" for i in failed)


def test_freshness_violation_in_contracts():
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
    history = [100.0, 100.0, 100.0, 100.0, 105.0, 95.0]
    result = detect_metric(500.0, history, method="mad")
    assert result["is_anomaly"] is True

    result_normal = detect_metric(100.0, history, method="mad")
    assert result_normal["is_anomaly"] is False


def test_context_aware_auto_anomaly_detection_with_full_history():
    # 4 weeks (28 days) of data: Weekdays ~600, Weekends ~250
    # Day order: 0, 1, 2, 3, 4, 5 (Sat), 6 (Sun)
    weekly_pattern = [600, 620, 590, 610, 630, 250, 260] * 4
    
    # Today is Saturday (dow=5), volume = 255.
    # Without automatic DOW extraction, 255 against full history (mean ~500) would be a false alarm.
    # With automatic weekly seasonality extraction, 255 against Saturday segment (250) is NORMAL.
    result = detect_metric(255, weekly_pattern, method="auto", context={"day_of_week": 5})
    assert result["is_anomaly"] is False


def test_known_event_surge_handling():
    history = [100, 105, 98, 102, 101]
    result = detect_metric(
        220,
        history,
        method="auto",
        context={"known_event": "flash_sale"},
    )
    assert result["is_anomaly"] is False


def test_ks_distribution_shift_moderate_mean_drift():
    # Mean shift of 3 sigma where mean_ratio is only 1.25 (100 -> 125 with std=5)
    rng = np.random.default_rng(123)
    base = rng.normal(loc=100.0, scale=5.0, size=50).tolist()
    cur = rng.normal(loc=125.0, scale=5.0, size=50).tolist()

    result = detect_distribution(cur, base)
    assert result["is_anomaly"] is True


def test_ks_distribution_shift_small_sample():
    # Small sample size (3 elements) with significant shift
    base = [10.0, 10.2, 9.8, 10.1, 9.9]
    cur = [25.0, 26.0, 24.5]
    result = detect_distribution(cur, base)
    assert result["is_anomaly"] is True


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


def test_cycle_lineage_graph_safety():
    cycle_graph = {
        "A": ["B"],
        "B": ["C"],
        "C": ["A", "D"],
    }
    assets = downstream_assets(cycle_graph, "A")
    assert "B" in assets and "C" in assets and "D" in assets
    assert len(assets) == 3


def test_slo_percentage_normalization():
    res = slo_status(99.5, bad_events=1, total_events=1000)
    assert res["target"] == pytest.approx(0.995)
    assert res["allowed_bad_rate"] == pytest.approx(0.005)


def test_multiwindow_burn_rate_policies():
    fast_burn = multiwindow_burn(short_window_burn=15.0, long_window_burn=15.0)
    assert fast_burn["page"] is True
    assert fast_burn["severity"] == "critical"

    transient = multiwindow_burn(short_window_burn=15.0, long_window_burn=1.0)
    assert transient["page"] is False
    assert transient["severity"] == "warning"

    healthy = multiwindow_burn(short_window_burn=0.8, long_window_burn=0.9)
    assert healthy["page"] is False
    assert healthy["severity"] == "info"


def test_rag_embedding_norm_drift():
    baseline_norms = [1.0, 1.02, 0.98, 1.01, 0.99, 1.0, 1.01]
    drifted_norms = [0.1, 0.12, 0.09, 0.11]
    res = rag_embedding_shift(drifted_norms, baseline_norms)
    assert res["is_anomaly"] is True


def test_rag_zero_vector_collapse():
    baseline_norms = [0.95, 1.0, 1.05, 0.98, 1.02]
    zero_norms = [0.0001, 0.0002, 0.0000]
    res = rag_embedding_shift(zero_norms, baseline_norms)
    assert res["is_anomaly"] is True
