"""RAG pipeline observability metrics including text length shift and embedding norm drift.
"""
from __future__ import annotations

from typing import Any, Iterable

import numpy as np

from observability.anomaly import zscore_detector


def approximate_token_lengths(texts: Iterable[str]) -> list[int]:
    """Token length proxy based on whitespace word counting."""
    if not texts:
        return []
    out = []
    for t in texts:
        if t is None:
            out.append(0)
        else:
            out.append(len(str(t).split()))
    return out


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
    try:
        cur = np.asarray(list(current_norms), dtype=float).ravel()
        base = np.asarray(list(baseline_norms), dtype=float).ravel()
        cur = cur[np.isfinite(cur)]
        base = base[np.isfinite(base)]
    except Exception:
        return {
            "is_anomaly": False,
            "score": 0.0,
            "method": "embedding_norm_shift",
            "reason": "invalid_input_data",
        }

    if cur.size == 0 or base.size == 0:
        return {
            "is_anomaly": False,
            "score": 0.0,
            "method": "embedding_norm_shift",
            "reason": "empty_input",
        }

    cur_mean = float(np.mean(cur))
    base_mean = float(np.mean(base))
    base_std = float(np.std(base))

    # Zero-vector collapse check: if current norms drop near zero while baseline was normal
    if cur_mean < 0.01 and base_mean > 0.5:
        return {
            "is_anomaly": True,
            "score": float("inf"),
            "method": "embedding_norm_shift",
            "reason": f"zero_vector_collapse: current_mean={cur_mean:.4f}, baseline_mean={base_mean:.4f}",
            "current_mean": cur_mean,
            "baseline_mean": base_mean,
        }

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
