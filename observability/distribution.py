"""Distribution drift detector using Two-Sample Kolmogorov-Smirnov (KS) test
and mean ratio comparison.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def _two_sample_ks_statistic(sample1: np.ndarray, sample2: np.ndarray) -> float:
    """Compute the two-sample Kolmogorov-Smirnov statistic D = max|F1(x) - F2(x)|."""
    s1 = np.sort(sample1)
    s2 = np.sort(sample2)
    n1 = len(s1)
    n2 = len(s2)

    if n1 == 0 or n2 == 0:
        return 0.0

    all_vals = np.concatenate([s1, s2])
    all_vals.sort()

    cdf1 = np.searchsorted(s1, all_vals, side="right") / n1
    cdf2 = np.searchsorted(s2, all_vals, side="right") / n2

    d_stat = float(np.max(np.abs(cdf1 - cdf2)))
    return d_stat


def detect_distribution_shift(
    current_values: Iterable[float],
    baseline_values: Iterable[float],
    *,
    ks_threshold: float = 0.15,
    ratio_threshold: float = 2.5,
) -> dict[str, Any]:
    """Detect distribution drift between current and baseline value distributions.

    Combines:
    1. Two-Sample Kolmogorov-Smirnov (KS) test for distribution shape / CDF shift.
    2. Mean ratio check.
    """
    try:
        cur = np.asarray(list(current_values), dtype=float).ravel()
        base = np.asarray(list(baseline_values), dtype=float).ravel()
        cur = cur[np.isfinite(cur)]
        base = base[np.isfinite(base)]
    except Exception:
        return {
            "is_anomaly": False,
            "score": 0.0,
            "method": "ks_distribution_test",
            "reason": "invalid_input_data",
        }

    n_cur = cur.size
    n_base = base.size

    if n_cur == 0 or n_base == 0:
        return {
            "is_anomaly": False,
            "score": 0.0,
            "method": "ks_distribution_test",
            "reason": "empty_input",
        }

    # Identical samples check
    if n_cur == n_base and np.array_equal(cur, base):
        return {
            "is_anomaly": False,
            "score": 0.0,
            "method": "ks_distribution_test",
            "reason": "identical_samples",
        }

    # 1. Two-sample KS test
    ks_stat = _two_sample_ks_statistic(cur, base)

    # 2. Mean ratio check
    cur_mean = float(np.mean(cur))
    base_mean = float(np.mean(base))

    if abs(base_mean) > 1e-6:
        mean_ratio = max(abs(cur_mean / base_mean), abs(base_mean / cur_mean)) if abs(cur_mean) > 1e-6 else float("inf")
    else:
        mean_ratio = float("inf") if abs(cur_mean) > 1e-3 else 1.0

    # Anomaly conditions
    ks_drift = bool(ks_stat >= ks_threshold)
    ratio_drift = bool(mean_ratio >= ratio_threshold)

    is_anomaly = bool(ks_drift or ratio_drift)
    primary_score = float(ks_stat)

    return {
        "is_anomaly": is_anomaly,
        "score": primary_score,
        "method": "ks_distribution_test",
        "reason": f"ks_stat={ks_stat:.3f}, mean_ratio={mean_ratio:.2f}",
    }
