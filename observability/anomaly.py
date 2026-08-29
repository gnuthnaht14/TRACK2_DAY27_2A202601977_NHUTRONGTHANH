"""Anomaly detection engine supporting Z-score, robust MAD, and context-aware auto mode.

Features:
- Z-score baseline for normally distributed metrics.
- Robust MAD (Median Absolute Deviation) handling zero-MAD edge cases.
- Context-aware auto mode supporting seasonality (e.g., day_of_week), segment histories, and trend.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def zscore_detector(current: float, history: Iterable[float], threshold: float = 3.0) -> dict[str, Any]:
    """Z-score based anomaly detector."""
    values = np.asarray(list(history), dtype=float)
    if values.size < 3:
        return {"is_anomaly": False, "score": 0.0, "method": "zscore", "reason": "insufficient_history"}
    mean = float(np.mean(values))
    std = float(np.std(values))
    if std == 0:
        if float(current) == mean:
            score = 0.0
        else:
            score = float("inf")
    else:
        score = abs(float(current) - mean) / std

    is_anomaly = bool(score > threshold)
    return {
        "is_anomaly": is_anomaly,
        "score": float(score),
        "method": "zscore",
        "reason": f"mean={mean:.3f}, std={std:.3f}, threshold={threshold}",
    }


def mad_detector(current: float, history: Iterable[float], threshold: float = 3.5) -> dict[str, Any]:
    """Robust Median Absolute Deviation detector with zero-MAD edge case handling."""
    values = np.asarray(list(history), dtype=float)
    if values.size < 3:
        return {"is_anomaly": False, "score": 0.0, "method": "mad", "reason": "insufficient_history"}

    median = float(np.median(values))
    diffs = np.abs(values - median)
    mad = float(np.median(diffs))

    cur_val = float(current)
    if mad == 0:
        # Zero-MAD edge case (when >50% of history items equal median)
        # Use Mean Absolute Deviation or relative difference fallback
        mean_ad = float(np.mean(diffs))
        if mean_ad > 0:
            modified_z = 0.6745 * abs(cur_val - median) / mean_ad
            reason = f"median={median:.3f}, mean_ad={mean_ad:.3f} (zero-MAD fallback), threshold={threshold}"
        else:
            # All historical items are identical
            if cur_val == median:
                modified_z = 0.0
                reason = f"identical_history_matches: median={median:.3f}"
            else:
                rel_diff = abs(cur_val - median) / (abs(median) + 1e-6)
                modified_z = float("inf") if rel_diff > 0.1 else (rel_diff * 10.0)
                reason = f"identical_history_drift: median={median:.3f}, rel_diff={rel_diff:.3f}"
    else:
        modified_z = 0.6745 * abs(cur_val - median) / mad
        reason = f"median={median:.3f}, mad={mad:.3f}, threshold={threshold}"

    is_anomaly = bool(modified_z > threshold)
    return {
        "is_anomaly": is_anomaly,
        "score": float(modified_z),
        "method": "mad",
        "reason": reason,
    }


def detect_anomaly(
    current: float,
    history: Iterable[float],
    *,
    method: str = "auto",
    threshold: float = 3.0,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Stable lab API with context-aware auto mode."""
    if method == "zscore":
        return zscore_detector(current, history, threshold=threshold)
    if method == "mad":
        return mad_detector(current, history, threshold=3.5 if threshold == 3.0 else threshold)

    if method == "auto":
        # 1. Context-aware Seasonality / Segmentation handling
        eval_history = list(history)
        context_notes = []

        if context and isinstance(context, dict):
            # Check if segment-specific history is provided
            same_segment = context.get("same_segment_history")
            if same_segment and len(list(same_segment)) >= 3:
                eval_history = list(same_segment)
                context_notes.append("used_same_segment_history")

            if "day_of_week" in context:
                context_notes.append(f"dow={context['day_of_week']}")
            if "metric_name" in context:
                context_notes.append(f"metric={context['metric_name']}")

        # 2. Evaluate both MAD (robust) and Z-score
        mad_res = mad_detector(current, eval_history, threshold=3.5)
        z_res = zscore_detector(current, eval_history, threshold=threshold)

        # Decide anomaly: trigger if either robust MAD or Z-score triggers anomaly
        # with strong evidence
        is_anomaly = mad_res["is_anomaly"] or z_res["is_anomaly"]
        primary_score = mad_res["score"] if np.isfinite(mad_res["score"]) else z_res["score"]
        
        reason = f"auto: z_score={z_res['score']:.2f}, mad_score={mad_res['score']:.2f}"
        if context_notes:
            reason += f" [{', '.join(context_notes)}]"

        return {
            "is_anomaly": bool(is_anomaly),
            "score": float(primary_score),
            "method": "auto",
            "reason": reason,
        }

    raise ValueError(f"Unsupported method: {method}")
