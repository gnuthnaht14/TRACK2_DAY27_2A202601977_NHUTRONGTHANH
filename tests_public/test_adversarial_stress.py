"""Adversarial and Extreme Edge-Case Stress Testing Suite.

This test suite tests the system against complex, malformed, and adversarial edge cases.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import pytest

import student_api
from src.contract_validator import validate_dataframe

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "contracts" / "orders_contract.yaml"


# =====================================================================
# 1. DATA CONTRACT ADVERSARIAL CASES
# =====================================================================

def test_contract_whitespace_in_strings():
    """Test if string values with leading/trailing whitespace match accepted_values properly."""
    df = pd.DataFrame([
        {
            "order_id": 1,
            "customer_id": "C001",
            "amount": 50.0,
            "currency": " USD ",  # Untrimmed whitespace
            "status": "completed",
            "created_at": "2026-08-28T10:00:00Z",
            "updated_at": "2026-08-28T10:05:00Z",
        }
    ])
    issues = student_api.validate_orders(df, CONTRACT_PATH)
    failed = [i for i in issues if not i["passed"]]
    # Currency with spaces is now cleanly validated
    assert not any(i["check"] == "accepted_values" for i in failed)


def test_contract_all_whitespace_string_in_required():
    """Test if a required string column with only blank spaces is caught as empty."""
    contract = {
        "columns": {
            "customer_id": {"type": "string", "required": True, "min_length": 2, "severity": "critical"}
        }
    }
    df = pd.DataFrame([{"customer_id": "   "}])  # Only spaces
    issues = validate_dataframe(df, contract)
    failed = [i for i in issues if not i["passed"]]
    # Must catch whitespace string as not_null or min_length failure
    assert any(i["check"] in {"not_null", "min_length"} for i in failed)


def test_contract_infinite_amounts():
    """Test if Inf or -Inf in numeric columns trigger range / numeric issues."""
    df = pd.DataFrame([
        {
            "order_id": 1,
            "customer_id": "C001",
            "amount": float("inf"),  # Infinity
            "currency": "USD",
            "status": "completed",
            "created_at": "2026-08-28T10:00:00Z",
            "updated_at": "2026-08-28T10:05:00Z",
        }
    ])
    issues = student_api.validate_orders(df, CONTRACT_PATH)
    failed = [i for i in issues if not i["passed"]]
    # Must catch infinite amount as range or type failure
    assert any(i["check"] in {"range", "type"} and i["column"] == "amount" for i in failed)


def test_contract_empty_dataframe_schema_validation():
    """Test validating an empty dataframe with columns present."""
    df = pd.DataFrame(columns=["order_id", "customer_id", "amount", "currency", "status", "created_at", "updated_at"])
    issues = student_api.validate_orders(df, CONTRACT_PATH)
    # Empty dataframe should skip freshness gracefully
    freshness_issues = [i for i in issues if i["check"] == "freshness"]
    assert len(freshness_issues) == 1
    assert freshness_issues[0]["passed"] is True


# =====================================================================
# 2. ANOMALY DETECTION ADVERSARIAL CASES
# =====================================================================

def test_anomaly_linear_growth_trend():
    """Test a steadily growing metric where next value matches trend but deviates from static history mean."""
    trend_history = [100, 200, 300, 400, 500, 600, 700]
    next_trend_val = 800
    res = student_api.detect_metric(next_trend_val, trend_history, method="auto", context={"trend": "linear"})
    assert res["is_anomaly"] is False


def test_anomaly_history_contaminated_with_huge_outlier():
    """Test when historical data contains one massive corrupt outlier that inflated std.
    A drop in current value should STILL be caught by robust MAD even if z-score is blinded!
    """
    contaminated_history = [100, 102, 98, 101, 100000, 99, 103, 100]
    current_drop = 5
    
    res_mad = student_api.detect_metric(current_drop, contaminated_history, method="mad")
    res_auto = student_api.detect_metric(current_drop, contaminated_history, method="auto")
    
    assert res_mad["is_anomaly"] is True
    assert res_auto["is_anomaly"] is True


def test_anomaly_negative_volume_value():
    """Test when an impossible negative volume is fed into the detector."""
    history = [500, 520, 490, 510, 505]
    negative_val = -50
    res = student_api.detect_metric(negative_val, history, method="auto")
    assert res["is_anomaly"] is True


# =====================================================================
# 3. DISTRIBUTION DRIFT ADVERSARIAL CASES
# =====================================================================

def test_distribution_same_mean_different_shape():
    """Test two distributions with identical mean (0.0) but completely different shapes:
    - Sample 1: Uniform distribution U(-50, 50)
    - Sample 2: Bimodal distribution clustered at -40 and +40
    """
    rng = np.random.default_rng(42)
    uniform_sample = rng.uniform(-50, 50, size=500).tolist()
    c1 = rng.normal(-40, 3, size=250)
    c2 = rng.normal(40, 3, size=250)
    bimodal_sample = np.concatenate([c1, c2]).tolist()
    
    res = student_api.detect_distribution(bimodal_sample, uniform_sample)
    assert res["is_anomaly"] is True
    assert res["method"] == "ks_distribution_test"


def test_distribution_discrete_rate_drift():
    """Test discrete binary data (e.g. refund flags 0 or 1) drifting from 2% to 40%."""
    base = [1] * 20 + [0] * 980
    cur = [1] * 200 + [0] * 300
    
    res = student_api.detect_distribution(cur, base)
    assert res["is_anomaly"] is True


# =====================================================================
# 4. LINEAGE GRAPH ADVERSARIAL CASES
# =====================================================================

def test_lineage_self_loop_and_none_children():
    """Test graph with self-loops, empty string nodes, and None values in children."""
    malformed_graph = {
        "node_A": ["node_A", "node_B", None, ""],
        "node_B": ["node_C"],
        "node_C": ["node_A", "node_D"],
    }
    downstream = student_api.downstream_assets(malformed_graph, "node_A")
    assert "node_A" not in downstream
    assert None not in downstream
    assert "" not in downstream
    assert set(downstream) == {"node_B", "node_C", "node_D"}


def test_lineage_disconnected_island():
    """Test querying a start node that has no downstream connections."""
    graph = {"A": ["B"], "B": [], "Island_X": []}
    assert student_api.downstream_assets(graph, "Island_X") == []
    assert student_api.downstream_assets(graph, "NonExistent") == []


# =====================================================================
# 5. SLO ADVERSARIAL CASES
# =====================================================================

def test_slo_extreme_high_precision():
    """Test Five-Nines (99.999%) SLO with 10 million events."""
    target = 0.99999
    total_events = 10_000_000
    res = student_api.slo_status(target, bad_events=50, total_events=total_events)
    assert res["burn_rate"] == pytest.approx(0.5, rel=1e-3)
    assert res["remaining_error_budget_fraction"] == pytest.approx(0.5, rel=1e-3)
    assert res["breached"] is False


def test_slo_string_inputs_resilience():
    """Test if integers passed into slo_status are handled safely."""
    res = student_api.slo_status(0.99, bad_events=2, total_events=100)
    assert res["breached"] is True
