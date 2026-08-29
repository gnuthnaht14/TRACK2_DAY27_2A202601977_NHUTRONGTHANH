#!/usr/bin/env python3
from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from observability.anomaly import detect_anomaly
from observability.lineage import get_downstream_assets
from observability.rag_metrics import detect_embedding_norm_shift, detect_text_length_shift
from observability.slo import calculate_slo
from src.contract_validator import failed_issues, load_contract, validate_dataframe
from src.io_utils import load_jsonl


def main() -> None:
    orders = pd.read_csv(ROOT / "data" / "incoming" / "orders.csv")
    history = pd.read_csv(ROOT / "data" / "history" / "metrics_history.csv")
    orders_contract = load_contract(ROOT / "contracts" / "orders_contract.yaml")
    orders_issues = validate_dataframe(orders, orders_contract)

    # Validate KB documents against kb_contract
    kb_docs = load_jsonl(ROOT / "data" / "incoming" / "kb_documents.jsonl")
    kb_df = pd.DataFrame(kb_docs)
    kb_contract = load_contract(ROOT / "contracts" / "kb_contract.yaml")
    kb_issues = validate_dataframe(kb_df, kb_contract)

    all_issues = orders_issues + kb_issues
    failed = failed_issues(all_issues)
    critical_failed = failed_issues(all_issues, min_severity="critical")

    # Segment by weekday before applying the auto detector
    current_dow = datetime.now().weekday()
    segment = history.loc[history["day_of_week"] == current_dow, "row_count"].tail(8).tolist()
    row_history = segment if len(segment) >= 3 else history["row_count"].tail(14).tolist()
    row_result = detect_anomaly(
        len(orders),
        row_history,
        method="auto",
        context={"metric_name": "row_count", "day_of_week": current_dow, "same_segment_history": segment},
    )

    updated = pd.to_datetime(orders["updated_at"], utc=True, errors="coerce")
    freshness_minutes = (
        pd.Timestamp(datetime.now(timezone.utc)) - updated.max()
    ).total_seconds() / 60.0

    # KB signals
    kb_published = pd.to_datetime(kb_df["published_at"], utc=True, errors="coerce")
    kb_freshness_minutes = (
        pd.Timestamp(datetime.now(timezone.utc)) - kb_published.max()
    ).total_seconds() / 60.0

    text_result = detect_text_length_shift(
        [d["content"] for d in kb_docs], history["mean_text_length"].tail(14).tolist()
    )

    # RAG embedding drift proxy signal
    embedding_result = detect_embedding_norm_shift(
        history["embedding_norm_mean"].tail(4).tolist(),
        history["embedding_norm_mean"].head(30).tolist(),
    )

    # SLO calculation: evaluate critical contract passes
    bad = 1 if critical_failed else 0
    contract_slo = calculate_slo(0.999, bad_events=bad, total_events=1)

    with open(ROOT / "data" / "baseline" / "lineage_graph.json", "r", encoding="utf-8") as f:
        lineage = json.load(f)["dataset_lineage"]
    blast_radius = get_downstream_assets(lineage, "stg_orders")

    report = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "orders_rows": int(len(orders)),
        "kb_docs_count": int(len(kb_docs)),
        "failed_contract_checks": len(failed),
        "critical_contract_failures": len(critical_failed),
        "failed_check_details": failed,
        "row_count_anomaly": row_result,
        "orders_freshness_minutes": freshness_minutes,
        "kb_freshness_minutes": kb_freshness_minutes,
        "kb_text_length_signal": text_result,
        "rag_embedding_drift_signal": embedding_result,
        "contract_slo": contract_slo,
        "sample_blast_radius_from_stg_orders": blast_radius,
    }
    out = ROOT / "reports" / "latest_metrics.json"
    out.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")

    print("=== DATA RELIABILITY BASELINE ===")
    print(f"orders rows              : {len(orders)}")
    print(f"kb docs count            : {len(kb_docs)}")
    print(f"contract failed checks   : {len(failed)}")
    print(f"critical contract fails  : {len(critical_failed)}")
    print(f"row-count anomaly        : {row_result['is_anomaly']} ({row_result['method']}, score={row_result['score']:.2f})")
    print(f"orders freshness (min)   : {freshness_minutes:.1f}")
    print(f"kb freshness (min)       : {kb_freshness_minutes:.1f}")
    print(f"KB length anomaly        : {text_result['is_anomaly']}")
    print(f"sample blast radius      : {', '.join(blast_radius)}")
    print(f"report                   : {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
