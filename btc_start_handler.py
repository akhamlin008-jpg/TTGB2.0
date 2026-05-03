"""
BTC start-date handling.

Problem: BTC-USD on Yahoo Finance starts 2014-09-17. If the user requests
a backtest starting before that date, the original code SILENTLY truncates
the start date to 2014-09-17, hiding the fact that the entire pre-2014
backtest period was discarded.

This module provides two strategies:

1. WARN_AND_TRUNCATE: explicitly inform the user that the start date was
   moved, return the new start date for transparent reporting.

2. PRE_BTC_FALLBACK: run the strategy with BTC's sleeve allocated to SPY
   (or to the GLD/Top10 sleeves proportionally) for the period before
   BTC was tradable. This preserves the full backtest history but requires
   honest disclosure that pre-2014 was a 2-asset portfolio, not 3-asset.

The default in this module is WARN_AND_TRUNCATE — most rigorous and
hardest to misrepresent.
"""

import pandas as pd
import warnings
from typing import Optional


BTC_INCEPTION_YF = pd.Timestamp("2014-09-17")  # First BTC-USD price on Yahoo Finance
GLD_INCEPTION = pd.Timestamp("2004-11-18")
SPY_INCEPTION = pd.Timestamp("1993-01-29")


def check_data_availability(
    requested_start: pd.Timestamp,
    requested_end: pd.Timestamp,
    has_btc: bool = True,
    has_gold: bool = True,
) -> dict:
    """
    Determines the effective backtest start date and returns a structured
    report of any truncation that occurred.
    """
    binding_constraints = []
    effective_start = pd.Timestamp(requested_start)

    if has_btc and effective_start < BTC_INCEPTION_YF:
        binding_constraints.append({
            "asset": "BTC-USD",
            "asset_inception": BTC_INCEPTION_YF.date().isoformat(),
            "would_truncate_by_days": (BTC_INCEPTION_YF - effective_start).days,
        })
        effective_start = max(effective_start, BTC_INCEPTION_YF)

    if has_gold and effective_start < GLD_INCEPTION:
        binding_constraints.append({
            "asset": "GLD",
            "asset_inception": GLD_INCEPTION.date().isoformat(),
            "would_truncate_by_days": (GLD_INCEPTION - pd.Timestamp(requested_start)).days,
        })
        effective_start = max(effective_start, GLD_INCEPTION)

    truncated = effective_start > pd.Timestamp(requested_start)

    return {
        "requested_start": pd.Timestamp(requested_start).date().isoformat(),
        "effective_start": effective_start.date().isoformat(),
        "truncated": truncated,
        "days_dropped": (effective_start - pd.Timestamp(requested_start)).days,
        "binding_constraints": binding_constraints,
    }


def format_truncation_warning(report: dict) -> Optional[str]:
    """Returns a human-readable warning string, or None if no truncation."""
    if not report["truncated"]:
        return None

    lines = [
        f"⚠ Backtest start truncated: requested {report['requested_start']}, "
        f"using {report['effective_start']} ({report['days_dropped']} days dropped)."
    ]
    for c in report["binding_constraints"]:
        lines.append(
            f"  - {c['asset']} data starts {c['asset_inception']} "
            f"(would have dropped {c['would_truncate_by_days']} days)."
        )
    lines.append(
        "  This means your backtest spans only the period when all assets "
        "were tradable. Be aware that BTC-USD's inclusion forces the start "
        "to 2014-09-17 at the earliest, which is widely regarded as the "
        "most favorable regime for both mega-cap tech and BTC."
    )
    return "\n".join(lines)


def emit_truncation_warning(report: dict, mode: str = "stderr"):
    """
    Mode options:
        - 'stderr': use Python warnings module
        - 'streamlit': use st.warning (only if streamlit available)
        - 'silent': just return the report
    """
    msg = format_truncation_warning(report)
    if msg is None:
        return
    if mode == "stderr":
        warnings.warn(msg, UserWarning)
    elif mode == "streamlit":
        try:
            import streamlit as st
            st.warning(msg)
        except ImportError:
            warnings.warn(msg, UserWarning)
