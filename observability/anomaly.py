"""Anomaly detection engine supporting Z-score, robust MAD, and context-aware auto mode.

Features:
- Z-score baseline for normally distributed metrics.
- Robust MAD (Median Absolute Deviation) for contamination-resistant detection.
- Context-aware auto mode with same_segment_history, automatic DOW seasonality,
  global cross-check, and known_event suppression.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def zscore_detector(current: float, history: Iterable[float], threshold: float = 3.0) -> dict[str, Any]:
    values = np.asarray(list(history), dtype=float)
    if values.size < 3:
        return {"is_anomaly": False, "score": 0.0, "method": "zscore", "reason": "insufficient_history"}
    mean = float(np.mean(values))
    std = float(np.std(values))
    cur = float(current)
    if std == 0:
        if cur == mean:
            score = 0.0
            reason = f"mean={mean:.3f}, std=0.000, identical_match"
        else:
            rel_diff = abs(cur - mean) / (abs(mean) + 1e-6)
            if rel_diff <= 0.15:
                score = rel_diff * 10.0
                reason = f"mean={mean:.3f}, zero_std_small_rel_diff={rel_diff:.3f}"
            else:
                score = float("inf")
                reason = f"mean={mean:.3f}, zero_std_large_rel_diff={rel_diff:.3f}"
    else:
        score = abs(cur - mean) / std
        reason = f"mean={mean:.3f}, std={std:.3f}, threshold={threshold}"
    return {
        "is_anomaly": bool(score > threshold),
        "score": float(score),
        "method": "zscore",
        "reason": reason,
    }


def mad_detector(current: float, history: Iterable[float], threshold: float = 3.5) -> dict[str, Any]:
    """Robust Median Absolute Deviation (MAD) anomaly detector."""
    values = np.asarray(list(history), dtype=float)
    if values.size < 3:
        return {"is_anomaly": False, "score": 0.0, "method": "mad", "reason": "insufficient_history"}
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))

    if mad == 0:
        diff = abs(float(current) - median)
        if diff == 0:
            score = 0.0
            is_anomaly = False
            reason = f"median={median:.3f}, mad=0.0, exact_match"
        else:
            std = float(np.std(values))
            if std > 0:
                score = diff / std
                is_anomaly = bool(score > threshold)
                reason = f"median={median:.3f}, mad=0.0, fallback_std={std:.3f}, score={score:.3f}"
            else:
                score = float("inf")
                is_anomaly = True
                reason = f"median={median:.3f}, zero_variance_deviation={diff:.3f}"
        return {
            "is_anomaly": is_anomaly,
            "score": float(score),
            "method": "mad",
            "reason": reason,
        }

    modified_z = 0.6745 * abs(float(current) - median) / mad
    return {
        "is_anomaly": bool(modified_z > threshold),
        "score": float(modified_z),
        "method": "mad",
        "reason": f"median={median:.3f}, mad={mad:.3f}, modified_z={modified_z:.3f}, threshold={threshold}",
    }


def detect_anomaly(
    current: float,
    history: Iterable[float],
    *,
    method: str = "auto",
    threshold: float = 3.0,
    context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Stable lab API with context-aware auto mode.

    Improved `auto` mode:
    - Uses `same_segment_history` for seasonality (requires >= 3 items).
    - Automatic Day-of-Week seasonality extraction (requires >= 14 days history).
    - Global Cross-check to prevent false positives in narrow seasonal segments.
    - Known event suppression for expected surges.
    - Prefers MAD for robustness (>= 5 items), fallback to Z-score.
    """
    if method == "mad":
        return mad_detector(current, history, threshold=threshold if threshold != 3.0 else 3.5)
    if method == "zscore":
        return zscore_detector(current, history, threshold=threshold)

    if method == "auto":
        ctx = context or {}
        segment_history = ctx.get("same_segment_history")
        known_event = ctx.get("known_event")
        dow = ctx.get("day_of_week")
        full_arr = np.asarray(list(history), dtype=float)
        full_list = list(full_arr)

        # Determine effective history for baseline
        effective_history = None
        used_segment = False

        if segment_history and len(segment_history) >= 3:
            effective_history = list(segment_history)
            used_segment = True
        elif dow is not None and full_arr.size >= 14:
            # Automatic DOW extraction: select history values at same day-of-week positions
            try:
                target_dow = int(dow) % 7
                dow_indices = [i for i in range(full_arr.size) if (i % 7) == target_dow]
                if len(dow_indices) >= 2:
                    effective_history = list(full_arr[dow_indices])
                    used_segment = True
            except (ValueError, TypeError):
                pass

        if effective_history is None:
            effective_history = full_list

        # Use MAD if history is sufficient, otherwise fallback to Z-score
        if len(effective_history) >= 5:
            base_result = mad_detector(current, effective_history, threshold=threshold if threshold != 3.0 else 3.5)
            used_method = "auto:seasonal_mad" if used_segment else "auto:mad"
        else:
            base_result = zscore_detector(current, effective_history, threshold=threshold)
            used_method = "auto:seasonal_zscore" if used_segment else "auto:zscore"

        is_anomaly = base_result["is_anomaly"]
        score = base_result["score"]
        reason = base_result["reason"]

        # Global Cross-check: prevent narrow seasonal segments from flagging normal values
        if is_anomaly and used_segment and len(full_list) >= 5:
            global_result = mad_detector(current, full_list, threshold=threshold if threshold != 3.0 else 3.5)
            if not global_result["is_anomaly"]:
                is_anomaly = False
                used_method += "+global_crosscheck"
                reason += "; suppressed: unremarkable in global history"

        # Known Event suppression: expected surges during events are not actionable
        if known_event and is_anomaly:
            is_anomaly = False
            reason += f"; suppressed_by_known_event='{known_event}'"

        return {
            "is_anomaly": is_anomaly,
            "score": float(score),
            "method": used_method,
            "reason": reason,
        }

    raise ValueError(f"Unsupported method: {method}")
