"""SLO and Error Budget calculation engine with Multi-window Multi-burn-rate Alerting.

Follows Google SRE Alerting on SLOs best practices:
- Distinguishes sustained fast burn (requiring immediate paging) from transient spikes (non-paging).
- Accurately tracks remaining error budget fraction and burn rate.
"""
from __future__ import annotations

from typing import Any


def calculate_slo(target: float, bad_events: int, total_events: int) -> dict[str, Any]:
    """Calculate SLI/SLO metrics, error budget, and burn rate."""
    if not 0 < target < 1:
        raise ValueError("target must be between 0 and 1 (exclusive)")
    if bad_events < 0 or total_events < 0 or bad_events > total_events:
        raise ValueError("invalid event counts")
    
    allowed_bad_rate = 1.0 - target
    if total_events == 0:
        return {
            "target": target,
            "actual_bad_rate": 0.0,
            "allowed_bad_rate": allowed_bad_rate,
            "burn_rate": 0.0,
            "remaining_error_budget_fraction": 1.0,
            "breached": False,
        }
        
    actual_bad_rate = bad_events / total_events
    burn_rate = actual_bad_rate / allowed_bad_rate
    consumed_fraction = actual_bad_rate / allowed_bad_rate
    
    return {
        "target": target,
        "actual_bad_rate": actual_bad_rate,
        "allowed_bad_rate": allowed_bad_rate,
        "burn_rate": burn_rate,
        "remaining_error_budget_fraction": max(0.0, 1.0 - consumed_fraction),
        "breached": bool(actual_bad_rate > allowed_bad_rate),
    }


def evaluate_multiwindow_burn(
    *,
    short_window_burn: float,
    long_window_burn: float,
    policy: str = "google_sre",
) -> dict[str, Any]:
    """Evaluate multi-window burn rate to decide whether to page or warn.
    
    Google SRE Multiwindow Burn Rate Rules:
    - Paging requires BOTH short-window and long-window to sustain elevated burn rates.
    - Transient spike: High short-window but low long-window -> Do NOT page (ticket/warning).
    - Sustained fast burn: High short-window AND high long-window -> PAGE immediately.
    """
    short = float(short_window_burn)
    long_ = float(long_window_burn)

    # 1. Critical Paging Alert: Sustained Fast Burn (e.g. 14x 1h + 14x 6h or 5x 1h + 2x 6h)
    if (short >= 14.0 and long_ >= 14.0) or (short >= 5.0 and long_ >= 2.0) or (short >= 2.5 and long_ >= 2.5):
        return {
            "page": True,
            "severity": "critical",
            "reason": f"sustained_fast_burn: short={short:.2f}, long={long_:.2f}",
            "short_window_burn": short,
            "long_window_burn": long_,
        }

    # 2. Transient Spike: High short window, but low long window -> Ticket/Warning (NO PAGE)
    if short >= 5.0 and long_ < 2.0:
        return {
            "page": False,
            "severity": "warning",
            "reason": f"transient_spike_no_page: short={short:.2f} elevated but long={long_:.2f} safe",
            "short_window_burn": short,
            "long_window_burn": long_,
        }

    # 3. Slow Burn: Sustained moderate burn (> 1.0) -> Warning
    if long_ > 1.0 or short > 1.0:
        return {
            "page": False,
            "severity": "warning",
            "reason": f"slow_burn_detected: short={short:.2f}, long={long_:.2f}",
            "short_window_burn": short,
            "long_window_burn": long_,
        }

    # 4. Healthy / Normal Burn Rate
    return {
        "page": False,
        "severity": "info",
        "reason": f"healthy_burn_rate: short={short:.2f}, long={long_:.2f}",
        "short_window_burn": short,
        "long_window_burn": long_,
    }
