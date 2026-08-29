"""Distribution drift detector using Two-Sample Kolmogorov-Smirnov (KS) statistic,
Welch's t-score mean deviation, and robust variance/mean ratio metrics.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np


def _two_sample_ks_statistic(sample1: np.ndarray, sample2: np.ndarray) -> float:
    """Compute the two-sample Kolmogorov-Smirnov statistic D = max |F1(x) - F2(x)|."""
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
    ks_threshold: float = 0.30,
    ratio_threshold: float = 2.5,
) -> dict[str, Any]:
    """Detect distribution drift between current and baseline value distributions.

    Combines:
    1. Two-Sample Kolmogorov-Smirnov (KS) test for distribution shape / CDF shift.
    2. Standardized Mean Difference (Welch's z-score of means).
    3. Mean Ratio check.
    4. Variance / Spread ratio check.
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

    cur_mean = float(np.mean(cur))
    base_mean = float(np.mean(base))
    cur_std = float(np.std(cur, ddof=1)) if n_cur > 1 else float(np.std(cur))
    base_std = float(np.std(base, ddof=1)) if n_base > 1 else float(np.std(base))

    # 1. Two-sample KS test statistic
    ks_stat = _two_sample_ks_statistic(cur, base)

    # Adaptive KS critical threshold for sample size
    # Use max(ks_threshold, ks_critical): the larger value is the more conservative threshold
    # For small samples, ks_critical can exceed 1.0 (clamp to 1.0)
    # For large samples, ks_critical < ks_threshold, so we use the larger fixed threshold
    if n_cur >= 3 and n_base >= 3:
        ks_critical = min(1.0, 1.36 * np.sqrt((n_cur + n_base) / (n_cur * n_base)))
        ks_drift = bool(ks_stat >= max(ks_threshold, float(ks_critical)))
    else:
        ks_drift = bool(ks_stat >= ks_threshold)

    # 2. Standardized Mean Difference (Z-score / Welch's t of means)
    diff_mean = abs(cur_mean - base_mean)
    if base_std > 0:
        mean_std_shift = diff_mean / base_std
    else:
        mean_std_shift = float("inf") if diff_mean > 0 else 0.0

    se_mean = np.sqrt((cur_std ** 2 / max(1, n_cur)) + (base_std ** 2 / max(1, n_base)))
    t_stat = diff_mean / se_mean if se_mean > 0 else (float("inf") if diff_mean > 0 else 0.0)

    # 3. Mean ratio check
    if abs(base_mean) > 1e-6:
        mean_ratio = max(abs(cur_mean / base_mean), abs(base_mean / cur_mean)) if abs(cur_mean) > 1e-6 else float("inf")
    else:
        mean_ratio = float("inf") if abs(cur_mean) > 1e-3 else 1.0

    # 4. Variance / Spread ratio check
    if base_std > 1e-6 and cur_std > 1e-6:
        std_ratio = max(cur_std / base_std, base_std / cur_std)
    else:
        std_ratio = float("inf") if (cur_std > 1e-3 or base_std > 1e-3) else 1.0

    # Anomaly conditions:
    # A. KS test indicates significant cumulative distribution difference
    # B. Mean difference exceeds 3 sigma (mean_std_shift >= 3.0 or t_stat >= 3.0)
    # C. Extreme mean ratio (>= 2.5)
    # D. Extreme variance ratio (>= 4.0 for moderate samples)
    is_anomaly = bool(
        ks_drift
        or (mean_std_shift >= 3.0 and n_base >= 3)
        or (t_stat >= 3.0 and (n_cur >= 3 and n_base >= 3))
        or mean_ratio >= ratio_threshold
        or (std_ratio >= 4.0 and (n_cur >= 5 and n_base >= 5))
    )

    primary_score = float(ks_stat if ks_stat > 0 else (mean_ratio if np.isfinite(mean_ratio) else 999.0))

    return {
        "is_anomaly": is_anomaly,
        "score": primary_score,
        "method": "ks_distribution_test",
        "reason": f"ks_stat={ks_stat:.3f}, mean_std_shift={mean_std_shift:.2f}, mean_ratio={mean_ratio:.2f}, std_ratio={std_ratio:.2f}",
    }
