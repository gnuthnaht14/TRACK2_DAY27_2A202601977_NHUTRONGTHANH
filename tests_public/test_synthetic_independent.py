from pathlib import Path
import student_api
from scripts.test_on_synthetic_data import (
    generate_complex_graph,
    generate_distribution_samples,
    generate_duplicate_pk_orders,
    generate_healthy_orders,
    generate_stale_orders,
    generate_type_drift_orders,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT = ROOT / "contracts" / "orders_contract.yaml"


def test_synthetic_healthy_dataset():
    df = generate_healthy_orders(n_rows=250, seed=111)
    issues = student_api.validate_orders(df, CONTRACT)
    failed = [i for i in issues if not i["passed"]]
    assert len(failed) == 0


def test_synthetic_type_drift_dataset():
    df = generate_type_drift_orders(n_rows=100, seed=222)
    issues = student_api.validate_orders(df, CONTRACT)
    failed_types = [i for i in issues if not i["passed"] and i["check"] == "type"]
    assert len(failed_types) >= 1


def test_synthetic_freshness_violation():
    df = generate_stale_orders(n_rows=100, delay_hours=4.0)
    issues = student_api.validate_orders(df, CONTRACT)
    failed_freshness = [i for i in issues if not i["passed"] and i["check"] == "freshness"]
    assert len(failed_freshness) >= 1


def test_synthetic_duplicate_pk():
    df = generate_duplicate_pk_orders(n_rows=100)
    issues = student_api.validate_orders(df, CONTRACT)
    failed_unique = [i for i in issues if not i["passed"] and i["check"] == "unique"]
    assert len(failed_unique) >= 1


def test_synthetic_bimodal_distribution_drift():
    cur, base = generate_distribution_samples(n_samples=300, shift_type="bimodal_variance_shift", seed=333)
    res = student_api.detect_distribution(cur, base)
    assert res["is_anomaly"] is True
    assert res["method"] == "ks_distribution_test"


def test_synthetic_complex_cycle_graph():
    graph, start, expected = generate_complex_graph()
    actual = student_api.downstream_assets(graph, start)
    assert set(actual) == set(expected)
    assert len(actual) == len(expected)
