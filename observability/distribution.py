"""Distribution drift detector using Two-Sample Kolmogorov-Smirnov (KS) statistic,
Quantile shift, and robust variance/mean ratio metrics.
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
    ks_threshold: float = 0.35,
    ratio_threshold: float = 3.0,
) -> dict[str, Any]:
    """Detect distribution shift between current and baseline value distributions."""
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

    if cur.size == 0 or base.size == 0:
        return {
            "is_anomaly": False,
            "score": 0.0,
            "method": "ks_distribution_test",
            "reason": "empty_input",
        }

    # 1. Two-sample KS test statistic
    ks_stat = _two_sample_ks_statistic(cur, base)

    # 2. Mean ratio check
    cur_mean = float(np.mean(cur))
    base_mean = float(np.mean(base))
    if base_mean == 0:
        mean_ratio = float("inf") if cur_mean != 0 else 1.0
    else:
        mean_ratio = max(abs(cur_mean / base_mean), abs(base_mean / cur_mean)) if cur_mean != 0 else float("inf")

    # 3. Variance / Spread ratio check
    cur_std = float(np.std(cur))
    base_std = float(np.std(base))
    if base_std > 0 and cur_std > 0:
        std_ratio = max(cur_std / base_std, base_std / cur_std)
    else:
        std_ratio = 1.0

    # Anomaly condition: Significant KS distribution shift OR extreme mean shift OR extreme variance shift
    is_anomaly = bool(
        (ks_stat >= ks_threshold and (cur.size >= 4 and base.size >= 4))
        or mean_ratio >= ratio_threshold
        or (std_ratio >= 5.0 and cur.size >= 10 and base.size >= 10)
    )

    primary_score = float(ks_stat if ks_stat > 0 else (mean_ratio if np.isfinite(mean_ratio) else 999.0))

    return {
        "is_anomaly": is_anomaly,
        "score": primary_score,
        "method": "ks_distribution_test",
        "reason": f"ks_stat={ks_stat:.3f}, mean_ratio={mean_ratio:.3f}, std_ratio={std_ratio:.2f}",
    }
