from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "latest_metrics.json"
HISTORY = ROOT / "data" / "history" / "metrics_history.csv"

st.set_page_config(page_title="Data Reliability Center", layout="wide", page_icon="🛡️")
st.title("🛡️ Data Reliability & Observability Center")
st.caption("E-Commerce Data Pipeline Observability, SLO Monitoring & Incident Blast Radius")

if not REPORT.exists():
    st.warning("Run `make baseline` first to generate `reports/latest_metrics.json`")
    st.stop()

report = json.loads(REPORT.read_text(encoding="utf-8"))

# Top KPIs
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Orders Rows", report.get("orders_rows", 0))
c2.metric("KB Documents", report.get("kb_docs_count", 0))
c3.metric("Orders Freshness", f"{report.get('orders_freshness_minutes', report.get('freshness_minutes', 0.0)):.1f} min")
c4.metric("KB Freshness", f"{report.get('kb_freshness_minutes', 0.0):.1f} min")
c5.metric("Critical Failures", report.get("critical_contract_failures", 0), delta=f"{report.get('failed_contract_checks', 0)} total fails", delta_color="inverse")

st.divider()

# SLO and Signals
col_left, col_right = st.columns(2)

with col_left:
    st.subheader("🎯 SLO & Error Budget Status")
    slo = report.get("contract_slo", {})
    if slo:
        target = slo.get("target", 0.999)
        budget_left = slo.get("remaining_error_budget_fraction", 1.0) * 100
        burn = slo.get("burn_rate", 0.0)
        breached = slo.get("breached", False)

        st.write(f"**Target Availability:** `{target * 100:.2f}%`")
        st.write(f"**Remaining Error Budget:** `{budget_left:.1f}%`")
        st.progress(max(0.0, min(1.0, budget_left / 100.0)))
        st.write(f"**Burn Rate:** `{burn:.2f}x` | **Status:** `{'BREACHED 🚨' if breached else 'HEALTHY ✅'}`")

    st.subheader("🔍 Automated Anomaly & Drift Signals")
    st.json({
        "row_count_anomaly": report.get("row_count_anomaly"),
        "kb_text_length_signal": report.get("kb_text_length_signal"),
        "rag_embedding_drift_signal": report.get("rag_embedding_drift_signal"),
    })

with col_right:
    st.subheader("💥 Incident Blast Radius Lineage")
    blast_radius = report.get("sample_blast_radius_from_stg_orders", [])
    if blast_radius:
        st.info("🚨 **Upstream Source Impacted:** `stg_orders`")
        st.markdown(" ➔ ".join([f"`{node}`" for node in ["stg_orders"] + blast_radius]))
    else:
        st.success("No downstream assets currently impacted.")

    st.subheader("⚠️ Failed Contract Checks Details")
    failed_details = report.get("failed_check_details", [])
    if failed_details:
        st.dataframe(pd.DataFrame(failed_details), use_container_width=True)
    else:
        st.success("All data contract checks passed successfully! ✨")

# Historical Chart
if HISTORY.exists():
    st.divider()
    history = pd.read_csv(HISTORY)
    st.subheader("📈 Historical Ingestion Metrics Trend")
    st.line_chart(history.set_index("date")[["row_count", "mean_text_length"]])
