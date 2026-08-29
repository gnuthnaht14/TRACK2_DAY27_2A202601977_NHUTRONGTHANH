"""Anomaly detection engine supporting Z-score, robust MAD, and context-aware auto mode.

Features:
- Z-score baseline for normally distributed metrics with relative tolerance on zero-std.
- Robust MAD (Median Absolute Deviation) handling zero-MAD and identical history edge cases.
- Context-aware auto mode supporting automatic weekly seasonality extraction,
  segment histories, trend differencing, rate spikes, and known events.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def _clean_numeric_array(values: Iterable[float]) -> np.ndarray:
    """Safely convert any iterable to a 1D clean float numpy array."""
    try:
        if isinstance(values, (int, float, np.number)):
            return np.array([float(values)])
        arr = np.asarray(list(values), dtype=float).ravel()
        return arr[np.isfinite(arr)]
    except Exception:
        return np.array([], dtype=float)


def zscore_detector(current: float, history: Iterable[float], threshold: float = 3.0) -> dict[str, Any]:
    """Z-score based anomaly detector with zero-std relative tolerance."""
    values = _clean_numeric_array(history)
    if values.size < 3:
        return {"is_anomaly": False, "score": 0.0, "method": "zscore", "reason": "insufficient_history"}

    cur = float(current)
    mean = float(np.mean(values))
    std = float(np.std(values))

    if std == 0:
        if cur == mean:
            score = 0.0
            reason = f"mean={mean:.3f}, std=0.000, identical_match"
        else:
            rel_diff = abs(cur - mean) / (abs(mean) + 1e-6)
            if rel_diff <= 0.15:  # Up to 15% fluctuation is not an infinite anomaly
                score = rel_diff * 10.0
                reason = f"mean={mean:.3f}, zero_std_small_rel_diff={rel_diff:.3f}"
            else:
                score = float("inf")
                reason = f"mean={mean:.3f}, zero_std_large_rel_diff={rel_diff:.3f}"
    else:
        score = abs(cur - mean) / std
        reason = f"mean={mean:.3f}, std={std:.3f}, threshold={threshold}"

    is_anomaly = bool(score > threshold)
    return {
        "is_anomaly": is_anomaly,
        "score": float(score),
        "method": "zscore",
        "reason": reason,
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
        # Zero-MAD edge case (when >=50% of history items equal median)
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
                if rel_diff <= 0.15:
                    modified_z = rel_diff * 10.0
                    reason = f"identical_history_minor_diff: median={median:.3f}, rel_diff={rel_diff:.3f}"
                else:
                    modified_z = float("inf") if rel_diff > 0.5 else (rel_diff * 20.0)
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
        return mad_detector(current, history, threshold=threshold if threshold != 3.0 else 3.5)

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
                            "method": "auto",
                            "reason": f"expected_volume_surge_during_known_event: {known_event}",
                        }

            # 2. Segment-specific History provided directly
            same_segment = context.get("same_segment_history")
            if same_segment:
                cleaned_segment = _clean_numeric_array(same_segment)
                if cleaned_segment.size >= 3:
                    eval_history = cleaned_segment
                    context_notes.append("used_same_segment_history")

            # 3. Automatic Day of Week Seasonality Extraction from Full History
            elif "day_of_week" in context and eval_history.size >= 14:
                try:
                    target_dow = int(context["day_of_week"])
                    dow_indices = [i for i in range(eval_history.size) if (i % 7) == (target_dow % 7)]
                    if len(dow_indices) >= 2:
                        dow_segment = eval_history[dow_indices]
                        eval_history = dow_segment
                        context_notes.append(f"auto_extracted_dow_{target_dow}_segment(n={len(dow_indices)})")
                except Exception:
                    pass

            if "day_of_week" in context and "auto_extracted_dow" not in " ".join(context_notes):
                context_notes.append(f"dow={context['day_of_week']}")

            if "metric_name" in context:
                metric_name = context["metric_name"]
                context_notes.append(f"metric={metric_name}")
                if "rate" in str(metric_name).lower() and float(current) > 0.05:
                    if eval_history.size > 0 and float(np.mean(eval_history)) < 0.01:
                        return {
                            "is_anomaly": True,
                            "score": float(current) / (float(np.mean(eval_history)) + 1e-5),
                            "method": "auto",
                            "reason": f"rate_metric_spike: current={float(current):.4f} > baseline={float(np.mean(eval_history)):.4f}",
                        }

            # 4. Trend context handling (linear growth)
            if context.get("trend") == "linear" and eval_history.size >= 4:
                diffs = np.diff(eval_history)
                expected_next = float(eval_history[-1]) + float(np.median(diffs))
                trend_diff = abs(float(current) - expected_next)
                mad_diff = float(np.median(np.abs(diffs - np.median(diffs))))
                if mad_diff > 0:
                    trend_score = 0.6745 * trend_diff / mad_diff
                else:
                    trend_score = trend_diff / (abs(expected_next) + 1e-5)
                if trend_score <= threshold:
                    return {
                        "is_anomaly": False,
                        "score": float(trend_score),
                        "method": "auto",
                        "reason": f"matches_linear_trend: expected={expected_next:.2f}, actual={float(current):.2f}",
                    }

        # 5. Robust evaluation using MAD as primary with Z-score fallback
        mad_res = mad_detector(current, eval_history, threshold=3.5 if threshold == 3.0 else threshold)
        z_res = zscore_detector(current, eval_history, threshold=threshold)

        # Check if history is too short for valid statistical inference
        mad_insufficient = "insufficient_history" in mad_res.get("reason", "")
        z_insufficient = "insufficient_history" in z_res.get("reason", "")

        # If BOTH methods report insufficient history, fall back to relative difference
        if mad_insufficient and z_insufficient:
            cur_val = float(current)
            if eval_history.size == 0:
                return {
                    "is_anomaly": False,
                    "score": 0.0,
                    "method": "auto",
                    "reason": "no_history_data",
                }
            # Use relative difference as fallback for small history
            hist_median = float(np.median(eval_history))
            rel_diff = abs(cur_val - hist_median) / (abs(hist_median) + 1e-6)
            score = rel_diff * 10.0  # Scale to roughly match z-score range
            is_anomaly = bool(rel_diff > 0.15)  # >15% deviation is anomalous
            return {
                "is_anomaly": is_anomaly,
                "score": float(score),
                "method": "auto",
                "reason": f"small_history_rel_diff: rel_diff={rel_diff:.3f}, threshold=0.15",
            }

        # Primary decision: MAD is more robust against contamination
        # If MAD says True, or if both Z and MAD indicate anomaly
        is_anomaly = mad_res["is_anomaly"] or (z_res["is_anomaly"] and mad_res["score"] > 2.0)
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
