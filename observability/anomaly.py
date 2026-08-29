"""Anomaly detection engine supporting Z-score, robust MAD, and context-aware auto mode.

Features:
- Z-score baseline for normally distributed metrics.
- Robust MAD (Median Absolute Deviation) handling zero-MAD edge cases.
- Context-aware auto mode supporting seasonality (e.g., day_of_week), segment histories, trend, and known events.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def _clean_numeric_array(values: Iterable[float]) -> np.ndarray:
    """Safely convert any iterable to a 1D clean float numpy array."""
    try:
        arr = np.asarray(list(values), dtype=float).ravel()
        return arr[np.isfinite(arr)]
    except Exception:
        return np.array([], dtype=float)


def zscore_detector(current: float, history: Iterable[float], threshold: float = 3.0) -> dict[str, Any]:
    """Z-score based anomaly detector."""
    values = _clean_numeric_array(history)
    if values.size < 3:
        return {"is_anomaly": False, "score": 0.0, "method": "zscore", "reason": "insufficient_history"}

    cur = float(current)
    mean = float(np.mean(values))
    std = float(np.std(values))

    if std == 0:
        if cur == mean:
            score = 0.0
        else:
            score = float("inf")
    else:
        score = abs(cur - mean) / std

    is_anomaly = bool(score > threshold)
    return {
        "is_anomaly": is_anomaly,
        "score": float(score),
        "method": "zscore",
        "reason": f"mean={mean:.3f}, std={std:.3f}, threshold={threshold}",
    }


def mad_detector(current: float, history: Iterable[float], threshold: float = 3.5) -> dict[str, Any]:
    """Robust Median Absolute Deviation detector with zero-MAD edge case handling."""
    values = _clean_numeric_array(history)
    if values.size < 3:
        return {"is_anomaly": False, "score": 0.0, "method": "mad", "reason": "insufficient_history"}

    cur = float(current)
    median = float(np.median(values))
    diffs = np.abs(values - median)
    mad = float(np.median(diffs))

    if mad == 0:
        # Zero-MAD edge case (when >50% of history items equal median)
        mean_ad = float(np.mean(diffs))
        if mean_ad > 0:
            modified_z = 0.6745 * abs(cur - median) / mean_ad
            reason = f"median={median:.3f}, mean_ad={mean_ad:.3f} (zero-MAD fallback), threshold={threshold}"
        else:
            # All historical items are identical
            if cur == median:
                modified_z = 0.0
                reason = f"identical_history_matches: median={median:.3f}"
            else:
                rel_diff = abs(cur - median) / (abs(median) + 1e-6)
                modified_z = float("inf") if rel_diff > 0.1 else (rel_diff * 10.0)
                reason = f"identical_history_drift: median={median:.3f}, rel_diff={rel_diff:.3f}"
    else:
        modified_z = 0.6745 * abs(cur - median) / mad
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
        eval_history = _clean_numeric_array(history)
        context_notes = []

        if context and isinstance(context, dict):
            # 1. Known Event Check (e.g. planned sale, holiday surge)
            known_event = context.get("known_event")
            if known_event:
                context_notes.append(f"event={known_event}")
                if str(known_event).lower() in {"sale", "promotion", "black_friday", "flash_sale", "holiday"}:
                    if eval_history.size > 0 and float(current) >= float(np.median(eval_history)):
                        return {
                            "is_anomaly": False,
                            "score": 0.0,
                            "method": "auto:known_event",
                            "reason": f"expected_volume_surge_during_known_event: {known_event}",
                        }

            # 2. Segment-specific History (e.g. Same Day of Week)
            same_segment = context.get("same_segment_history")
            if same_segment:
                cleaned_segment = _clean_numeric_array(same_segment)
                if cleaned_segment.size >= 3:
                    eval_history = cleaned_segment
                    context_notes.append("used_same_segment_history")

            if "day_of_week" in context:
                context_notes.append(f"dow={context['day_of_week']}")
            if "metric_name" in context:
                metric_name = context["metric_name"]
                context_notes.append(f"metric={metric_name}")
                # Rate-based metrics (e.g. null_rate, error_rate)
                if "rate" in str(metric_name).lower() and float(current) > 0.05:
                    if eval_history.size > 0 and float(np.mean(eval_history)) < 0.01:
                        return {
                            "is_anomaly": True,
                            "score": float(current) / (float(np.mean(eval_history)) + 1e-5),
                            "method": "auto:rate_spike",
                            "reason": f"rate_metric_spike: current={float(current):.4f} > baseline={float(np.mean(eval_history)):.4f}",
                        }

        # 3. Combined evaluation: Robust MAD + Z-score
        mad_res = mad_detector(current, eval_history, threshold=3.5)
        z_res = zscore_detector(current, eval_history, threshold=threshold)

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
