"""RAG pipeline observability metrics including text length shift and embedding norm drift.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from observability.anomaly import zscore_detector


def approximate_token_lengths(texts: Iterable[str]) -> list[int]:
    """Token length proxy based on whitespace word counting."""
    return [len(str(t).split()) for t in texts]


def detect_text_length_shift(
    current_texts: Iterable[str],
    baseline_batch_means: Iterable[float],
    *,
    threshold: float = 3.0,
) -> dict[str, Any]:
    """Detect anomalous drop or spike in unstructured text token lengths."""
    lengths = approximate_token_lengths(current_texts)
    current_mean = float(np.mean(lengths)) if lengths else 0.0
    result = zscore_detector(current_mean, baseline_batch_means, threshold=threshold)
    result["metric"] = "mean_text_length"
    result["current_mean"] = current_mean
    return result


def detect_embedding_norm_shift(
    current_norms: Iterable[float],
    baseline_norms: Iterable[float],
    *,
    threshold: float = 3.0,
) -> dict[str, Any]:
    """Detect distribution drift in embedding vector norms or cosine similarities."""
    cur = np.asarray(list(current_norms), dtype=float)
    base = np.asarray(list(baseline_norms), dtype=float)

    cur = cur[np.isfinite(cur)]
    base = base[np.isfinite(base)]

    if cur.size < 3 or base.size < 3:
        return {
            "is_anomaly": False,
            "score": 0.0,
            "method": "embedding_norm_shift",
            "reason": "insufficient_data",
        }

    cur_mean = float(np.mean(cur))
    base_mean = float(np.mean(base))
    base_std = float(np.std(base))

    if base_std == 0:
        score = float("inf") if cur_mean != base_mean else 0.0
    else:
        score = abs(cur_mean - base_mean) / base_std

    is_anomaly = bool(score > threshold)

    return {
        "is_anomaly": is_anomaly,
        "score": float(score),
        "method": "embedding_norm_shift",
        "reason": f"baseline_mean={base_mean:.3f}, baseline_std={base_std:.3f}, current_mean={cur_mean:.3f}, z_score={score:.2f}",
        "current_mean": cur_mean,
        "baseline_mean": base_mean,
    }
