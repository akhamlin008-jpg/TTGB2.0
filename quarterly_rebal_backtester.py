"""
Portfolio Backtester – Quarterly Rebalance
Portfolio A: 100 % SPY buy-and-hold
Portfolio B: 50 % historical S&P 500 top-10 (normalised weights) + 40 % GLD + 10 % BTC-USD
Rebalances on the first trading day of each calendar quarter (Jan / Apr / Jul / Oct).
Snapshot-trigger rebalances fire when top-10 composition updates between quarters.
"""

import io
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import streamlit as st
import yfinance as yf

# New modules — transaction costs, BTC start-date handling, parameter sensitivity
from transaction_costs import apply_transaction_costs
from btc_start_handler import check_data_availability, emit_truncation_warning
from sensitivity import run_sensitivity_grid, summarize_robustness, default_weight_grid
# ─────────────────────────────────────────────
# CONSTANTS
# ─────────────────────────────────────────────
BENCHMARK = "SPY"
GOLD_ETF = "GLD"          # GLD inception 2004-11-18 beats IAU 2005-01-21
BTC_TICKER = "BTC-USD"

TOP10_SLEEVE = 0.50
GLD_SLEEVE   = 0.40
BTC_SLEEVE   = 0.10

USE_NEXT_DAY_EXECUTION = True
STRICT_START_WHEN_ALL_COMPONENTS_LIVE = True

# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────
def normalize_ticker_for_yf(t):
    if pd.isna(t):
        return None
    t = str(t).strip().upper().replace(" ", "").replace("/", "-").rstrip("*").replace(".", "-")
    ticker_map = {
        "BRKB": "BRK-B", "BRK-B": "BRK-B", "BRK.B": "BRK-B",
        "FB": "META", "META": "META",
        "GOOG-L": "GOOGL", "GOOGLE": "GOOGL",
    }
    t = ticker_map.get(t, t)
    if t in {"", "N/A", "NA", "NONE", "NULL", "0R01"}:
        return None
    return t


@st.cache_data(show_spinner="Downloading prices from Yahoo Finance …")
def download_close_matrix(tickers, start, end):
    tickers = sorted(set([str(x).upper() for x in tickers]))
    raw = yf.download(tickers=tickers, start=start, end=end,
                      auto_adjust=True, progress=False,
                      group_by="column", threads=True)
    if raw is None or len(raw) == 0:
        raise RuntimeError("yfinance returned no data.")

    if isinstance(raw.columns, pd.MultiIndex):
        lvl0 = [str(x) for x in raw.columns.get_level_values(0)]
        lvl1 = [str(x) for x in raw.columns.get_level_values(1)]
        if "Close" in set(lvl0):
            px = raw["Close"].copy()
        elif "Adj Close" in set(lvl0):
            px = raw["Adj Close"].copy()
        elif "Close" in set(lvl1):
            px = raw.xs("Close", axis=1, level=1).copy()
        elif "Adj Close" in set(lvl1):
            px = raw.xs("Adj Close", axis=1, level=1).copy()
        else:
            raise RuntimeError("Could not extract Close from yfinance output.")
    else:
        if "Close" in raw.columns:
            px = pd.DataFrame(raw["Close"])
            px.columns = [tickers[0]]
        elif "Adj Close" in raw.columns:
            px = pd.DataFrame(raw["Adj Close"])
            px.columns = [tickers[0]]
        else:
            raise RuntimeError(f"Unexpected yfinance columns: {list(raw.columns)}")

    if isinstance(px, pd.Series):
        px = px.to_frame()
    px.columns = [str(c).upper() for c in px.columns]
    rename = {}
    if "BRK.B" in px.columns and "BRK-B" not in px.columns:
        rename["BRK.B"] = "BRK-B"
    if "BF.B" in px.columns and "BF-B" not in px.columns:
        rename["BF.B"] = "BF-B"
    if rename:
        px = px.rename(columns=rename)
    px = px.sort_index().dropna(how="all")
    return px


def next_trading_day(index, dt):
    idx = pd.DatetimeIndex(index)
    pos = idx.searchsorted(pd.Timestamp(dt), side="right")
    return None if pos >= len(idx) else idx[pos]


def same_or_next_trading_day(index, dt):
    idx = pd.DatetimeIndex(index)
    pos = idx.searchsorted(pd.Timestamp(dt), side="left")
    return None if pos >= len(idx) else idx[pos]


def quarterly_rebalance_dates(trade_index):
    """First trading day of each calendar quarter (Jan, Apr, Jul, Oct)."""
    ti = pd.DatetimeIndex(trade_index).sort_values()
    s = pd.Series(1, index=ti)
    quarter_periods = ti.to_period("Q")
    first_of_quarter = s.groupby(quarter_periods).head(1).index
    return pd.DatetimeIndex(first_of_quarter)


def rebalance_dates_by_frequency(trade_index, frequency="quarterly"):
    """
    First trading day of each period, given a frequency.

    Supported frequencies:
        - "monthly"     → first trading day of each month
        - "quarterly"   → first trading day of each calendar quarter
        - "semiannual"  → first trading day of January and July
        - "annual"      → first trading day of each calendar year
    """
    ti = pd.DatetimeIndex(trade_index).sort_values()
    s = pd.Series(1, index=ti)

    if frequency == "monthly":
        periods = ti.to_period("M")
    elif frequency == "quarterly":
        periods = ti.to_period("Q")
    elif frequency == "semiannual":
        # Group by year + half-year (1 = Jan-Jun, 2 = Jul-Dec)
        periods = pd.PeriodIndex.from_fields(
            year=ti.year,
            quarter=((ti.month - 1) // 6) * 2 + 1,  # quarter 1 or 3
            freq="Q",
        )
    elif frequency == "annual":
        periods = ti.to_period("Y")
    else:
        raise ValueError(f"Unknown rebalance frequency: {frequency!r}. "
                         f"Use one of: monthly, quarterly, semiannual, annual")

    first_of_period = s.groupby(periods).head(1).index
    return pd.DatetimeIndex(first_of_period)


# ─── performance helpers ────
def annualized_return_from_daily(daily_ret, periods=252):
    r = pd.Series(daily_ret).dropna().astype(float)
    if len(r) == 0:
        return np.nan
    n = len(r)
    total = float((1.0 + r).prod() - 1.0)
    years = n / periods
    return float((1.0 + total) ** (1.0 / years) - 1.0) if years > 0 else np.nan


def annualized_vol(daily_ret, periods=252):
    r = pd.Series(daily_ret).dropna().astype(float)
    return float(r.std(ddof=1) * np.sqrt(periods)) if len(r) >= 2 else np.nan


def sharpe_ratio(daily_ret, rf_annual=0.0, periods=252):
    r = pd.Series(daily_ret).dropna().astype(float)
    if len(r) < 2:
        return np.nan
    rf_daily = (1.0 + float(rf_annual)) ** (1.0 / periods) - 1.0
    ex = r - rf_daily
    vol = ex.std(ddof=1) * np.sqrt(periods)
    return float((ex.mean() * periods) / vol) if vol > 0 else np.nan


def beta_to_benchmark(port_ret, bench_ret):
    df = pd.DataFrame({"p": port_ret, "b": bench_ret}).dropna()
    if len(df) < 3:
        return np.nan
    vb = float(df["b"].var(ddof=1))
    return float(df["p"].cov(df["b"]) / vb) if vb > 0 else np.nan


def treynor_ratio(port_ret, bench_ret, rf_annual=0.0, periods=252):
    b = beta_to_benchmark(port_ret, bench_ret)
    if b == 0 or np.isnan(b):
        return np.nan
    ann = annualized_return_from_daily(port_ret, periods)
    return float((ann - float(rf_annual)) / b)


def cagr_from_equity(eq):
    eq = pd.Series(eq).dropna()
    if len(eq) < 2 or float(eq.iloc[0]) <= 0:
        return np.nan
    days = (eq.index[-1] - eq.index[0]).days
    years = max(days / 365.25, 1e-9)
    return float((float(eq.iloc[-1]) / float(eq.iloc[0])) ** (1.0 / years) - 1.0)


def max_drawdown(equity_curve):
    eq = pd.Series(equity_curve).astype(float)
    if eq.empty:
        return np.nan
    return float((eq / eq.cummax() - 1.0).min())


# ─── contribution helpers ────
def build_quarterly_contribution_series(
    index,
    annual_amount,
    contrib_month=1,
    contrib_day=2,
    include_first_year=False,
):
    """
    Build a contribution cash-flow series.

    Quarterly contributions are placed on (or after) the configured
    contribution day of months {contrib_month, +3, +6, +9}. If the chosen
    calendar day is not a trading day (weekend/holiday), the contribution
    falls on the next available trading day.

    contrib_month: 1-12, the calendar month of the *first* contribution
                   in each year (e.g., 1=January). Subsequent contributions
                   land 3, 6, and 9 months after that.
    contrib_day:   1-31. Snapped to next trading day if necessary, or to
                   month-end if the day exceeds the month length.
    include_first_year: if False, skip contributions in the very first
                        calendar year of the backtest (common for accounts
                        that fund initial_capital and contribute starting
                        the next year).
    """
    idx = pd.DatetimeIndex(index).sort_values()
    cf = pd.Series(0.0, index=idx, dtype=float)
    if annual_amount == 0 or len(idx) == 0:
        return cf

    quarterly_amount = annual_amount / 4

    # Generate target dates: quarterly anchors at (month, day) and +3, +6, +9 months
    contrib_months = [((contrib_month - 1 + offset) % 12) + 1 for offset in (0, 3, 6, 9)]

    first_year = int(idx.min().year)
    last_year = int(idx.max().year)

    for year in range(first_year, last_year + 1):
        if not include_first_year and year == first_year:
            continue
        for m in contrib_months:
            # Snap the day to the actual month length (handles Feb 30 → Feb 28/29)
            month_end_day = pd.Timestamp(year=year, month=m, day=1).days_in_month
            day = min(contrib_day, month_end_day)
            target = pd.Timestamp(year=year, month=m, day=day)

            # Snap to next trading day in the index (or skip if past the data range)
            actual = same_or_next_trading_day(idx, target)
            if actual is None:
                continue
            cf.loc[actual] += float(quarterly_amount)

    return cf


def apply_cashflows_to_returns(daily_ret, initial_capital, cashflows=None, timing="end_of_day"):
    r = pd.Series(daily_ret).fillna(0.0).astype(float)
    idx = r.index
    cf = pd.Series(0.0, index=idx) if cashflows is None else cashflows.reindex(idx).fillna(0.0)
    vals = []
    v = float(initial_capital)
    for dt in idx:
        rt = float(r.loc[dt])
        ct = float(cf.loc[dt])
        if timing == "start_of_day":
            v = (v + ct) * (1.0 + rt)
        else:
            v = v * (1.0 + rt) + ct
        vals.append(v)
    return pd.Series(vals, index=idx, dtype=float), cf


def xirr(cashflows, dates, guess=0.10, max_iter=100, tol=1e-7):
    cf = np.asarray(cashflows, dtype=float)
    dts = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    if len(cf) != len(dts) or len(cf) == 0:
        return np.nan
    if not (np.any(cf > 0) and np.any(cf < 0)):
        return np.nan
    t0 = dts.iloc[0]
    years = np.array([(d - t0).days / 365.25 for d in dts], dtype=float)
    r = float(guess)
    for _ in range(max_iter):
        denom = (1.0 + r) ** years
        if np.any(denom == 0):
            return np.nan
        f = np.sum(cf / denom)
        fp = np.sum(-years * cf / ((1.0 + r) ** (years + 1.0)))
        if fp == 0 or not np.isfinite(fp):
            return np.nan
        r_new = r - f / fp
        if not np.isfinite(r_new) or r_new <= -0.999999:
            return np.nan
        if abs(r_new - r) < tol:
            return float(r_new)
        r = r_new
    return np.nan


def compute_xirr_for_account(eq_curve, contrib_series, initial_capital):
    eq = pd.Series(eq_curve).dropna()
    cf = pd.Series(contrib_series).reindex(eq.index).fillna(0.0)
    cashflows = [-float(initial_capital)]
    dates = [eq.index[0]]
    c_nonzero = cf[cf != 0]
    for dt, amt in c_nonzero.items():
        cashflows.append(-float(amt))
        dates.append(dt)
    cashflows.append(float(eq.iloc[-1]))
    dates.append(eq.index[-1])
    return xirr(cashflows, dates)


def build_year_table(equity_curve, contrib_series):
    eq = pd.Series(equity_curve).copy()
    cf = pd.Series(contrib_series).reindex(eq.index).fillna(0.0)
    df = pd.DataFrame({"equity": eq, "contrib": cf})
    rows = []
    for y, g in df.groupby(df.index.year):
        g = g.sort_index()
        sv = float(g["equity"].iloc[0])
        ev = float(g["equity"].iloc[-1])
        c = float(g["contrib"].sum())
        base = sv + 0.5 * c
        ret = (ev - sv - c) / base if base > 0 else np.nan
        dd = (g["equity"] / g["equity"].cummax() - 1.0).min()
        rows.append({"Year": int(y), "Start Value": sv, "Contribution": c,
                      "End Value": ev, "Approx Return (flow-adj)": ret,
                      "Max DD (year)": float(dd)})
    return pd.DataFrame(rows)


# ─────────────────────────────────────────────
# CORE BACKTEST
# ─────────────────────────────────────────────
def run_backtest(holdings_df, start_date, end_date, initial_capital,
                 annual_contrib, contrib_timing, rf_annual,
                 cost_bps=10.0, crypto_cost_bps=30.0,
                 top10_sleeve=None, gld_sleeve=None, btc_sleeve=None,
                 rebal_frequency="quarterly",
                 contrib_month=1, contrib_day=2, include_first_year=False):
    """
    If top10_sleeve / gld_sleeve / btc_sleeve are None, the module-level
    defaults (0.50 / 0.40 / 0.10) are used. The sensitivity grid passes
    explicit values to test alternate allocations.
    """
    # Resolve sleeve weights (use module defaults if not specified)
    top10_w = TOP10_SLEEVE if top10_sleeve is None else float(top10_sleeve)
    gld_w   = GLD_SLEEVE   if gld_sleeve   is None else float(gld_sleeve)
    btc_w   = BTC_SLEEVE   if btc_sleeve   is None else float(btc_sleeve)

    # Sanity check — sleeves must sum to ~1.0
    sleeve_sum = top10_w + gld_w + btc_w
    if abs(sleeve_sum - 1.0) > 1e-4:
        raise ValueError(f"Sleeve weights must sum to 1.0, got {sleeve_sum:.4f} "
                         f"(top10={top10_w}, gld={gld_w}, btc={btc_w})")
    """
    Returns dict with all results needed for display.
    """
    # ── parse holdings CSV ──
    holdings = holdings_df.copy()
    holdings["snapshot_date"] = pd.to_datetime(holdings["snapshot_date"], errors="coerce")
    holdings["rank"] = pd.to_numeric(holdings["rank"], errors="coerce")
    holdings["weight_pct"] = pd.to_numeric(holdings["weight_pct"], errors="coerce")
    holdings["ticker"] = holdings["ticker"].astype(str).map(normalize_ticker_for_yf)

    holdings = holdings.dropna(subset=["snapshot_date", "rank", "ticker", "weight_pct"])
    holdings = holdings[(holdings["rank"] >= 1) & (holdings["rank"] <= 10)]
    holdings = holdings[holdings["weight_pct"] > 0]
    if holdings.empty:
        raise ValueError("No valid holdings rows after cleaning.")

    holdings = (holdings
                .sort_values(["snapshot_date", "rank", "weight_pct"], ascending=[True, True, False])
                .drop_duplicates(subset=["snapshot_date", "ticker"], keep="first"))

    snapshot_map = {}
    snapshot_dates = []
    for sd, g in holdings.groupby("snapshot_date"):
        g = g.sort_values(["rank", "weight_pct"], ascending=[True, False]).head(10).copy()
        w = g.groupby("ticker")["weight_pct"].sum().sort_values(ascending=False)
        w = w[w > 0]
        if len(w) >= 2:
            snapshot_map[pd.Timestamp(sd)] = w
            snapshot_dates.append(pd.Timestamp(sd))
    snapshot_dates = sorted(snapshot_dates)
    if not snapshot_dates:
        raise ValueError("No usable top-10 snapshots in the CSV.")

    # ── download prices ──
    all_top10_names = sorted(set(holdings["ticker"].unique()))
    tickers = [BENCHMARK, GOLD_ETF, BTC_TICKER] + all_top10_names
    prices_raw = download_close_matrix(tickers, start_date, end_date)

    for req in [BENCHMARK, GOLD_ETF, BTC_TICKER]:
        if req not in prices_raw.columns:
            raise RuntimeError(f"Required ticker missing from price data: {req}")

    trade_index = prices_raw[BENCHMARK].dropna().index
    if len(trade_index) == 0:
        raise RuntimeError("No SPY trading dates found.")

    prices_sampled_raw = prices_raw.reindex(trade_index)
    prices = prices_sampled_raw.ffill().dropna(how="all")

    if STRICT_START_WHEN_ALL_COMPONENTS_LIVE:
        live_mask = prices_sampled_raw[[BENCHMARK, GOLD_ETF, BTC_TICKER]].notna().all(axis=1)
        if not live_mask.any():
            raise RuntimeError("No overlap where SPY, GLD, and BTC all have data.")

        # Check whether the user's requested start was truncated by data availability
        effective_start_date = live_mask[live_mask].index.min()
        truncation_report = check_data_availability(
            requested_start=pd.Timestamp(start_date),
            requested_end=pd.Timestamp(end_date),
            has_btc=True,
            has_gold=True,
        )
        emit_truncation_warning(truncation_report, mode="streamlit")

        prices = prices.loc[effective_start_date:].copy()
        trade_index = prices.index

    rets = prices.pct_change().fillna(0.0)

    # ── snapshot execution schedule ──
    signal_dates = [d for d in snapshot_dates if prices.index.min() <= d <= prices.index.max()]
    if not signal_dates:
        raise RuntimeError("No snapshot dates overlap the price range.")

    exec_pairs = []
    for sd in signal_dates:
        ex = next_trading_day(prices.index, sd) if USE_NEXT_DAY_EXECUTION \
            else same_or_next_trading_day(prices.index, sd)
        if ex is not None and ex in prices.index:
            exec_pairs.append((pd.Timestamp(sd), pd.Timestamp(ex)))

    if not exec_pairs:
        raise RuntimeError("No execution dates generated from snapshot dates.")

    snapshot_exec_df = (pd.DataFrame(exec_pairs, columns=["signal_date", "exec_date"])
                        .drop_duplicates().sort_values(["exec_date", "signal_date"])
                        .reset_index(drop=True))

    # ── QUARTERLY rebalance schedule ──
# ── Rebalance schedule (frequency configurable for sensitivity analysis) ──
    quarterly_dates = rebalance_dates_by_frequency(prices.index, frequency=rebal_frequency)
    snapshot_rebal_dates = pd.DatetimeIndex(snapshot_exec_df["exec_date"].unique())
    rebal_dates = pd.DatetimeIndex(sorted(set(quarterly_dates) | set(snapshot_rebal_dates)))
    rebal_dates = rebal_dates[(rebal_dates >= prices.index.min()) & (rebal_dates <= prices.index.max())]

    if len(rebal_dates) == 0:
        raise RuntimeError("No rebalance dates generated.")

    snapshot_exec_sorted = snapshot_exec_df.sort_values("exec_date").reset_index(drop=True)

    # ── build target weights ──
    target_weights = pd.DataFrame(0.0, index=prices.index, columns=prices.columns)
    rebal_log = []

    for rd in rebal_dates:
        eligible = snapshot_exec_sorted[snapshot_exec_sorted["exec_date"] <= rd]
        if eligible.empty:
            continue

        active_row = eligible.iloc[-1]
        active_signal_date = pd.Timestamp(active_row["signal_date"])
        active_exec_date = pd.Timestamp(active_row["exec_date"])

        tw = pd.Series(0.0, index=prices.columns)

        w_raw = snapshot_map.get(active_signal_date, None)
        top10_source = "csv"
        top10_used = []
        if w_raw is None or len(w_raw) < 2:
            tw[BENCHMARK] += top10_w
            top10_source = "fallback_spy_missing_snapshot"
        else:
            usable = [(t, float(w)) for t, w in w_raw.items()
                      if (t in prices.columns and pd.notna(prices.at[rd, t]))]
            if len(usable) < 2:
                tw[BENCHMARK] += top10_w
                top10_source = "fallback_spy_missing_prices"
            else:
                wsum = sum(w for _, w in usable)
                for t, w in usable:
                    tw[t] += top10_w * (w / wsum)   # normalise within sleeve
                top10_used = [t for t, _ in usable]

        # gold
        if pd.notna(prices.at[rd, GOLD_ETF]):
            tw[GOLD_ETF] += gld_w
        else:
            tw[BENCHMARK] += gld_w

        # btc — only allocate if BTC sleeve > 0 (sensitivity grid tests no-BTC variants)
        if btc_w > 0:
            if pd.notna(prices.at[rd, BTC_TICKER]):
                tw[BTC_TICKER] += btc_w
            else:
                tw[BENCHMARK] += btc_w

        s = tw.sum()
        if s <= 0:
            tw[BENCHMARK] = 1.0
        else:
            tw = tw / s

        target_weights.loc[rd] = tw

        in_quarterly = rd in set(quarterly_dates)
        in_snapshot = rd in set(snapshot_rebal_dates)
        if in_quarterly and in_snapshot:
            reason = "quarterly+snapshot_update"
        elif in_snapshot:
            reason = "snapshot_update"
        else:
            reason = "quarterly"

        # Build per-holding detail rows for the trade log
        top10_set = set(top10_used)
        for ticker in prices.columns:
            wt = float(tw[ticker])
            if wt > 0:
                if ticker == GOLD_ETF:
                    sleeve = "gold"
                elif ticker == BTC_TICKER:
                    sleeve = "btc"
                elif ticker == BENCHMARK:
                    sleeve = "benchmark_fallback"
                elif ticker in top10_set:
                    sleeve = "top10"
                else:
                    sleeve = "top10"

                rebal_log.append({
                    "exec_date": rd,
                    "signal_date": active_signal_date,
                    "rebalance_reason": reason,
                    "ticker": ticker,
                    "target_weight": round(wt, 8),
                    "sleeve": sleeve,
                    "top10_source": top10_source,
                })

    rebal_log_df = pd.DataFrame(rebal_log) if rebal_log else pd.DataFrame()

    # Also build a summary-level rebal log (one row per rebalance date)
    summary_rebal_log = []
    for rd in rebal_dates:
        eligible = snapshot_exec_sorted[snapshot_exec_sorted["exec_date"] <= rd]
        if eligible.empty:
            continue
        active_row = eligible.iloc[-1]
        w_row = target_weights.loc[rd]
        top10_tickers = [c for c in w_row.index
                         if c not in [BENCHMARK, GOLD_ETF, BTC_TICKER] and w_row[c] > 0]
        in_quarterly = rd in set(quarterly_dates)
        in_snapshot = rd in set(snapshot_rebal_dates)
        if in_quarterly and in_snapshot:
            reason = "quarterly+snapshot_update"
        elif in_snapshot:
            reason = "snapshot_update"
        else:
            reason = "quarterly"
        summary_rebal_log.append({
            "exec_date": rd,
            "signal_date": pd.Timestamp(active_row["signal_date"]),
            "rebalance_reason": reason,
            "top10_sleeve_sum": round(sum(w_row[t] for t in top10_tickers), 6),
            "gld_weight": round(float(w_row.get(GOLD_ETF, 0)), 6),
            "btc_weight": round(float(w_row.get(BTC_TICKER, 0)), 6),
            "spy_fallback_weight": round(float(w_row.get(BENCHMARK, 0)), 6),
            "total_weight": round(float(w_row.sum()), 6),
            "top10_count": len(top10_tickers),
            "top10_names": ", ".join(top10_tickers[:10]),
        })
    summary_rebal_df = pd.DataFrame(summary_rebal_log) if summary_rebal_log else pd.DataFrame()

    # forward-fill weights
    zero_rows = (target_weights.sum(axis=1) == 0)
    target_weights.loc[zero_rows, :] = np.nan
    target_weights = target_weights.ffill().fillna(0.0)

    # ── compute returns ──
    # ── compute returns ──
    portA_ret = rets[BENCHMARK].fillna(0.0)
    w_lag = target_weights.shift(1).reindex(rets.index).fillna(0.0)
    common_cols = [c for c in w_lag.columns if c in rets.columns]
    portB_ret_pre_cost = (w_lag[common_cols] * rets[common_cols]).sum(axis=1)

    # Apply transaction costs to Portfolio B
    # Build pre-cost equity curve for cost calculation (uses same starting capital)
    eqB_pre_cost = initial_capital * (1 + portB_ret_pre_cost).cumprod()
    cost_dollars = apply_transaction_costs(
        target_weights=target_weights,
        equity_curve=eqB_pre_cost,
        rebal_dates=rebal_dates,
        cost_bps=cost_bps,
        crypto_tickers=[BTC_TICKER],
        crypto_cost_bps=crypto_cost_bps,
    )

    # Convert dollar costs to a return drag (cost / portfolio_value at each rebalance)
    cost_drag = pd.Series(0.0, index=portB_ret_pre_cost.index)
    for dt, c in cost_dollars.items():
        if c > 0 and dt in eqB_pre_cost.index:
            pv = float(eqB_pre_cost.loc[dt])
            if pv > 0:
                cost_drag.loc[dt] = c / pv
    portB_ret = portB_ret_pre_cost - cost_drag

    # Save the total cost for reporting
    total_costs_paid = float(cost_dollars.sum())

    first_b = target_weights.sum(axis=1)
    first_b_date = first_b[first_b > 0].index.min()
    if pd.isna(first_b_date):
        raise RuntimeError("Portfolio B never received weights.")

    start_compare = max(rets.index.min(), first_b_date)
    portA_ret = portA_ret.loc[start_compare:]
    portB_ret = portB_ret.loc[start_compare:]

    # ── contribution-aware curves ──
    contrib = build_quarterly_contribution_series(
        portA_ret.index,
        annual_contrib,
        contrib_month=contrib_month,
        contrib_day=contrib_day,
        include_first_year=include_first_year,
    )

    eqA_lump = initial_capital * (1 + portA_ret).cumprod()
    eqB_lump = initial_capital * (1 + portB_ret).cumprod()

    eqA_cf, cf_used = apply_cashflows_to_returns(portA_ret, initial_capital, contrib, contrib_timing)
    eqB_cf, _ = apply_cashflows_to_returns(portB_ret, initial_capital, contrib, contrib_timing)

    total_contributed = float(initial_capital + cf_used.sum())

    # ── metrics ──
    def row_metrics(label, eq_lump, eq_cf, r, bench_r):
        return {
            "Portfolio": label,
            "Start": str(r.index[0].date()),
            "End": str(r.index[-1].date()),
            "Days": int((r.index[-1] - r.index[0]).days),
            "Terminal (Lump Sum)": float(eq_lump.iloc[-1]),
            "Terminal (w/ Contrib)": float(eq_cf.iloc[-1]),
            "Total Contributed": total_contributed,
            "Net Gain on Contrib": float(eq_cf.iloc[-1] - total_contributed),
            "CAGR (Lump Sum)": cagr_from_equity(eq_lump),
            "Ann Return (geom)": annualized_return_from_daily(r),
            "Annual Vol": annualized_vol(r),
            "Sharpe": sharpe_ratio(r, rf_annual),
            "Beta vs SPY": beta_to_benchmark(r, bench_r),
            "Treynor vs SPY": treynor_ratio(r, bench_r, rf_annual),
            "Max DD (Lump Sum)": max_drawdown(eq_lump),
            "Max DD (Contrib)": max_drawdown(eq_cf),
            "XIRR (Contrib)": compute_xirr_for_account(eq_cf, cf_used, initial_capital),
        }

    metrics = pd.DataFrame([
        row_metrics("SPY Buy & Hold", eqA_lump, eqA_cf, portA_ret, portA_ret),
        row_metrics("Top10 50 / GLD 40 / BTC 10 (Qtrly)", eqB_lump, eqB_cf, portB_ret, portA_ret),
    ]).set_index("Portfolio")

    yearly_A = build_year_table(eqA_cf, cf_used)
    yearly_B = build_year_table(eqB_cf, cf_used)

    # ── Koyfin trade ticket ──
    # At each rebalance date back into shares from portfolio value + price.
    # delta shares = new target shares - previously held shares → trade ticket.
    koyfin_trades = []
    prev_shares = {}  # ticker -> float shares held

    for rd in sorted(rebal_dates):
        if rd not in eqB_lump.index:
            continue
        portfolio_value = float(eqB_lump.loc[rd])
        tw = target_weights.loc[rd]

        new_shares = {}
        for ticker in tw.index:
            wt = float(tw[ticker])
            if wt <= 0 or ticker not in prices.columns:
                continue
            price = float(prices.loc[rd, ticker])
            if np.isnan(price) or price <= 0:
                continue
            new_shares[ticker] = (portfolio_value * wt) / price

        for ticker in sorted(set(new_shares.keys()) | set(prev_shares.keys())):
            delta = new_shares.get(ticker, 0.0) - prev_shares.get(ticker, 0.0)
            if abs(delta) < 1e-6:
                continue
            price = float(prices.loc[rd, ticker]) if ticker in prices.columns else np.nan
            koyfin_trades.append({
                "Purchase Date": rd.strftime("%Y-%m-%d"),
                "Symbol/ISIN": ticker,
                "Quantity": round(delta, 6),
                "Cost Per Share": round(price, 4),
            })

        prev_shares = new_shares

    koyfin_df = pd.DataFrame(
        koyfin_trades,
        columns=["Purchase Date", "Symbol/ISIN", "Quantity", "Cost Per Share"]
    )

    return {
        "metrics": metrics,
        "eqA_lump": eqA_lump, "eqB_lump": eqB_lump,
        "eqA_cf": eqA_cf, "eqB_cf": eqB_cf,
        "cf_used": cf_used,
        "total_contributed": total_contributed,
        "rebal_log_detail": rebal_log_df,
        "rebal_log_summary": summary_rebal_df,
        "yearly_A": yearly_A, "yearly_B": yearly_B,
        "portA_ret": portA_ret, "portB_ret": portB_ret,
        "koyfin_trades": koyfin_df,
        "total_costs_paid": total_costs_paid,
        "cost_dollars": cost_dollars,
    }


# ─────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────
st.set_page_config(page_title="Quarterly Rebalance Backtester", layout="wide")
st.title("Portfolio Backtester — Quarterly Rebalance")
st.caption("Port A: 100 % SPY  |  Port B: 50 % Top-10 (normalised) + 40 % GLD + 10 % BTC-USD")

# ── sidebar: CSV upload + params ──
with st.sidebar:
    st.header("Settings")

    csv_file = "sp500_top10_semiannual_2007_2026_ivv_proxy.csv"

    st.subheader("Backtest Period")
    col_s, col_e = st.columns(2)
    with col_s:
        start_input = st.date_input("Start", value=pd.Timestamp("2015-01-01").date())
    with col_e:
        end_input = st.date_input("End", value=pd.Timestamp.today().date())

    st.subheader("Capital & Contributions")
    initial_capital = st.number_input("Initial Capital ($)", value=10_000.0, min_value=0.0, step=1000.0)
    annual_contrib = st.number_input("Annual Contribution ($)", value=10_000.0, min_value=0.0, step=1000.0)
    contrib_month = st.selectbox("Contribution Month", list(range(1, 13)), index=0)
    contrib_day = st.slider("Contribution Day", 1, 31, 2)
    include_first_year = st.checkbox("Include first-year contribution", value=False)
    timing = st.selectbox("Contribution Timing", ["end_of_day", "start_of_day"])
    rf_annual = st.number_input("Risk-Free Rate (annual)", value=0.0, step=0.005, format="%.4f")

    st.subheader("Transaction Costs")
    cost_bps = st.number_input("Equity cost (bps)", value=10.0, min_value=0.0, step=1.0,
                                help="Per-side cost in basis points for SPY/GLD/equities. 10 bps = 0.10%")
    crypto_cost_bps = st.number_input("BTC cost (bps)", value=30.0, min_value=0.0, step=5.0,
                                       help="Per-side cost in basis points for BTC. Historically wider spreads.")

    st.subheader("Display")
    show_lump = st.checkbox("Lump-sum curves", value=True)
    show_contrib = st.checkbox("Contribution curves", value=True)
    show_dd = st.checkbox("Drawdown chart", value=True)
    show_yearly = st.checkbox("Year-by-year table", value=True)

    st.subheader("Sensitivity Analysis")
    run_sensitivity = st.checkbox(
        "Run parameter sensitivity grid",
        value=False,
        help="Tests the strategy across multiple weight combinations and "
             "rebalance frequencies to check whether results are robust or "
             "depend on a single specific parameter point (overfitting check). "
             "Adds significant runtime — disable for quick iterations."
    )

    run = st.button("Run Backtest", type="primary")

# ── main area ──
if csv_file is None:
    st.info("Upload the top-10 holdings CSV in the sidebar to begin.")
    st.stop()

if not run:
    st.info("Adjust settings in the sidebar, then press **Run Backtest**.")
    st.stop()

# read CSV
holdings_df = pd.read_csv(csv_file)
required_cols = {"snapshot_date", "rank", "ticker", "weight_pct"}
missing = required_cols - set(holdings_df.columns)
if missing:
    st.error(f"CSV is missing required columns: {missing}")
    st.stop()

with st.spinner("Running backtest …"):
    try:
        res = run_backtest(
    holdings_df,
    start_date=str(start_input),
    end_date=str(end_input),
    initial_capital=initial_capital,
    annual_contrib=annual_contrib,
    contrib_timing=timing,
    rf_annual=rf_annual,
    cost_bps=cost_bps,
    crypto_cost_bps=crypto_cost_bps,
    contrib_month=contrib_month,
    contrib_day=contrib_day,
    include_first_year=include_first_year,
)
    except Exception as e:
        st.error(f"Backtest failed: {e}")
        st.stop()

# ── display results ──
st.subheader("Performance Summary")

fmt = res["metrics"].copy()
pct_cols = ["CAGR (Lump Sum)", "Ann Return (geom)", "Annual Vol",
            "Max DD (Lump Sum)", "Max DD (Contrib)", "XIRR (Contrib)"]
for c in pct_cols:
    if c in fmt.columns:
        fmt[c] = fmt[c].map(lambda x: "" if pd.isna(x) else f"{x:.2%}")
for c in ["Sharpe", "Beta vs SPY", "Treynor vs SPY"]:
    if c in fmt.columns:
        fmt[c] = fmt[c].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
for c in ["Terminal (Lump Sum)", "Terminal (w/ Contrib)", "Total Contributed", "Net Gain on Contrib"]:
    if c in fmt.columns:
        fmt[c] = fmt[c].map(lambda x: "" if pd.isna(x) else f"${x:,.2f}")

st.dataframe(fmt, use_container_width=True)

st.metric(
    label="Total Transaction Costs Paid (Portfolio B)",
    value=f"${res['total_costs_paid']:,.2f}",
    help=f"Cumulative cost drag from {len(res['cost_dollars'][res['cost_dollars'] > 0])} rebalance events. "
         f"Already deducted from Portfolio B's returns."
)
# ── equity curves ──
if show_lump or show_contrib:
    st.subheader("Equity Curves")
    fig, ax = plt.subplots(figsize=(12, 5))
    if show_lump:
        ax.plot(res["eqA_lump"].index, res["eqA_lump"], label="SPY Buy & Hold (Lump)")
        ax.plot(res["eqB_lump"].index, res["eqB_lump"], label="Top10/GLD/BTC (Lump)")
    if show_contrib:
        ax.plot(res["eqA_cf"].index, res["eqA_cf"], ls="--", label="SPY (+ Contrib)")
        ax.plot(res["eqB_cf"].index, res["eqB_cf"], ls="--", label="Top10/GLD/BTC (+ Contrib)")
    marks = res["cf_used"][res["cf_used"] != 0]
    for dt in marks.index:
        ax.axvline(dt, alpha=0.06)
    ax.set_ylabel("Portfolio Value ($)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    st.pyplot(fig)

# ── drawdowns ──
if show_dd:
    st.subheader("Drawdowns (Lump Sum)")
    fig2, ax2 = plt.subplots(figsize=(12, 4))
    ddA = res["eqA_lump"] / res["eqA_lump"].cummax() - 1
    ddB = res["eqB_lump"] / res["eqB_lump"].cummax() - 1
    ax2.plot(ddA.index, ddA, label="SPY")
    ax2.plot(ddB.index, ddB, label="Top10/GLD/BTC")
    ax2.set_ylabel("Drawdown")
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    st.pyplot(fig2)

# ── year-by-year ──
if show_yearly:
    st.subheader("Year-by-Year — SPY Buy & Hold")
    st.dataframe(res["yearly_A"], use_container_width=True, hide_index=True)
    st.subheader("Year-by-Year — Top10/GLD/BTC (Quarterly)")
    st.dataframe(res["yearly_B"], use_container_width=True, hide_index=True)

# ── contribution ledger ──
non_zero = res["cf_used"][res["cf_used"] != 0]
if len(non_zero):
    st.subheader("Contribution Ledger")
    st.dataframe(non_zero.to_frame("Contribution ($)"), use_container_width=True)

# ── rebalance log (summary) ──
if not res["rebal_log_summary"].empty:
    st.subheader(f"Rebalance Log — Summary ({len(res['rebal_log_summary'])} events)")
    st.dataframe(res["rebal_log_summary"], use_container_width=True, hide_index=True)

    # weight sanity check
    sums = res["rebal_log_summary"]["total_weight"]
    if sums.nunique() == 1 and np.isclose(sums.iloc[0], 1.0):
        st.success("All rebalance events sum to 100 % — weight integrity verified.")
    else:
        st.warning("Some rebalance events have weight sums != 1.0. Inspect the log.")

# ── Koyfin trade ticket ──
st.subheader("Koyfin Trade Ticket")
st.caption("Symbol/ISIN, Quantity (negative = sell), Cost Per Share, Purchase Date — ready to upload directly to Koyfin.")

koyfin = res["koyfin_trades"]
if not koyfin.empty:
    st.dataframe(koyfin, use_container_width=True, hide_index=True)
    buf = io.StringIO()
    koyfin.to_csv(buf, index=False)
    st.download_button(
        label="Download Koyfin Trade CSV",
        data=buf.getvalue(),
        file_name="koyfin_trades.csv",
        mime="text/csv",
    )
else:
    st.info("No trades generated.")


# ── Sensitivity Analysis ──
if run_sensitivity:
    st.markdown("---")
    st.subheader("Sensitivity Analysis")
    st.caption(
        "Tests the strategy across multiple weight allocations and rebalance "
        "frequencies. A robust strategy should perform reasonably across the "
        "grid; a strategy that only works at one specific point is likely "
        "overfit."
    )

    weight_grid = default_weight_grid()
    rebal_freqs = ["monthly", "quarterly", "semiannual", "annual"]

    progress_text = st.empty()
    progress_text.info(
        f"Running {len(weight_grid)} weights × {len(rebal_freqs)} frequencies "
        f"= {len(weight_grid) * len(rebal_freqs)} backtests. This will take a minute…"
    )

    with st.spinner("Running sensitivity grid..."):
        try:
            sens_df = run_sensitivity_grid(
                run_backtest_fn=run_backtest,
                holdings_df=holdings_df,
                start_date=str(start_input),
                end_date=str(end_input),
                initial_capital=initial_capital,
                weight_grid=weight_grid,
                rebal_freqs=rebal_freqs,
                annual_contrib=annual_contrib,
                contrib_timing=timing,
                rf_annual=rf_annual,
                cost_bps=cost_bps,
            )
            progress_text.empty()
        except Exception as e:
            progress_text.empty()
            st.error(f"Sensitivity analysis failed: {e}")
            sens_df = pd.DataFrame()

    if not sens_df.empty:
        # Summary statistics
        summary = summarize_robustness(sens_df)
        if "error" not in summary:
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Combinations tested",
                    f"{summary['n_combinations_tested']}",
                )
            with col2:
                st.metric(
                    "% beating SPY (Sharpe)",
                    f"{summary['pct_combinations_beating_spy_sharpe']:.0%}",
                    help="What fraction of weight/frequency combinations had a higher "
                         "Sharpe ratio than SPY buy-and-hold."
                )
            with col3:
                st.metric(
                    "Sharpe range across grid",
                    f"{summary['sharpe_min']:.2f} — {summary['sharpe_max']:.2f}",
                    help="Narrower range = more robust. Wide range with the baseline "
                         "near the top suggests the chosen point may be optimized."
                )

            st.write("**Best configuration:**", summary["best_config"])
            st.write("**Worst configuration:**", summary["worst_config"])

        # Format the results table for display
        display_df = sens_df.copy()
        for col in ["Strategy CAGR", "Strategy Vol", "Strategy Max DD",
                    "SPY CAGR", "Excess CAGR vs SPY"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].map(
                    lambda x: "" if pd.isna(x) else f"{x:.2%}"
                )
        for col in ["Strategy Sharpe", "SPY Sharpe", "Sharpe Diff vs SPY"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].map(
                    lambda x: "" if pd.isna(x) else f"{x:.3f}"
                )
        for col in ["Strategy Terminal"]:
            if col in display_df.columns:
                display_df[col] = display_df[col].map(
                    lambda x: "" if pd.isna(x) else f"${x:,.0f}"
                )

        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # Download as CSV for offline analysis
        csv_buf = io.StringIO()
        sens_df.to_csv(csv_buf, index=False)
        st.download_button(
            label="Download Sensitivity Results CSV",
            data=csv_buf.getvalue(),
            file_name="sensitivity_results.csv",
            mime="text/csv",
        )
