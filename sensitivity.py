"""
Sensitivity / parameter-sweep analysis.

Runs the backtest across a grid of allocation weights and rebalance
frequencies. Reports Sharpe, CAGR, max DD, and final value for each
combination so you can see whether the strategy works at one specific
point in parameter space (overfit) or across a robust region.

USAGE:
    from sensitivity import run_sensitivity_grid

    results_df = run_sensitivity_grid(
        run_backtest_fn=run_backtest,
        holdings_df=holdings_df,
        start_date="2015-01-01",
        end_date="2024-12-31",
        initial_capital=10000,
        weight_grid=[
            (0.50, 0.40, 0.10),  # baseline
            (0.40, 0.40, 0.20),
            (0.60, 0.30, 0.10),
            (0.70, 0.25, 0.05),
            (0.30, 0.50, 0.20),
        ],
        rebal_freqs=["quarterly", "monthly", "annual"],
    )

This function expects you to have refactored run_backtest to accept
sleeve weights and rebal frequency as parameters. If your current
run_backtest hardcodes 50/40/10 and quarterly, you'll need a small
patch (shown in README).
"""

import numpy as np
import pandas as pd


def annualized_return_from_daily(daily_ret, periods=252):
    """Local copy to avoid circular imports."""
    r = pd.Series(daily_ret).dropna().astype(float)
    if len(r) == 0:
        return np.nan
    n = len(r)
    total = float((1.0 + r).prod() - 1.0)
    years = n / periods
    return float((1.0 + total) ** (1.0 / years) - 1.0) if years > 0 else np.nan


def run_sensitivity_grid(
    run_backtest_fn,
    holdings_df: pd.DataFrame,
    start_date: str,
    end_date: str,
    initial_capital: float,
    weight_grid: list,
    rebal_freqs: list,
    annual_contrib: float = 0.0,
    contrib_timing: str = "end_of_day",
    rf_annual: float = 0.0,
    cost_bps: float = 10.0,
) -> pd.DataFrame:
    """
    Cross every weight combination with every rebalance frequency.

    weight_grid: list of (top10_pct, gld_pct, btc_pct) tuples that sum to 1.0
    rebal_freqs: list of strings; supported values depend on your refactor.
                 Suggested: ["monthly", "quarterly", "semiannual", "annual"]

    Returns a DataFrame with one row per parameter combination.
    """
    rows = []
    for w in weight_grid:
        if abs(sum(w) - 1.0) > 1e-6:
            print(f"Skipping {w}: weights don't sum to 1.0")
            continue
        top10_w, gld_w, btc_w = w
        for freq in rebal_freqs:
            try:
                res = run_backtest_fn(
                    holdings_df=holdings_df,
                    start_date=start_date,
                    end_date=end_date,
                    initial_capital=initial_capital,
                    annual_contrib=annual_contrib,
                    contrib_timing=contrib_timing,
                    rf_annual=rf_annual,
                    top10_sleeve=top10_w,
                    gld_sleeve=gld_w,
                    btc_sleeve=btc_w,
                    rebal_frequency=freq,
                    cost_bps=cost_bps,
                )
                # Extract Portfolio B (the strategy) from the metrics
                metrics = res["metrics"]
                strategy_row = metrics.iloc[1]  # row 1 = strategy
                spy_row = metrics.iloc[0]       # row 0 = SPY benchmark

                rows.append({
                    "top10_pct": top10_w,
                    "gld_pct": gld_w,
                    "btc_pct": btc_w,
                    "rebal_freq": freq,
                    "Strategy CAGR": strategy_row["CAGR (Lump Sum)"],
                    "Strategy Vol": strategy_row["Annual Vol"],
                    "Strategy Sharpe": strategy_row["Sharpe"],
                    "Strategy Max DD": strategy_row["Max DD (Lump Sum)"],
                    "Strategy Terminal": strategy_row["Terminal (Lump Sum)"],
                    "SPY CAGR": spy_row["CAGR (Lump Sum)"],
                    "SPY Sharpe": spy_row["Sharpe"],
                    "Excess CAGR vs SPY": (
                        strategy_row["CAGR (Lump Sum)"] - spy_row["CAGR (Lump Sum)"]
                    ),
                    "Sharpe Diff vs SPY": (
                        strategy_row["Sharpe"] - spy_row["Sharpe"]
                    ),
                })
            except Exception as e:
                rows.append({
                    "top10_pct": top10_w,
                    "gld_pct": gld_w,
                    "btc_pct": btc_w,
                    "rebal_freq": freq,
                    "error": str(e),
                })

    return pd.DataFrame(rows)


def summarize_robustness(sensitivity_df: pd.DataFrame) -> dict:
    """
    Summarize how robust the strategy is across the parameter grid.

    A strategy that's only good at one point is overfit. A strategy that's
    good across a wide region is more credible.
    """
    df = sensitivity_df.dropna(subset=["Strategy Sharpe"])
    if df.empty:
        return {"error": "No successful runs"}

    sharpe_vals = df["Strategy Sharpe"]
    excess_vals = df["Excess CAGR vs SPY"]

    return {
        "n_combinations_tested": len(df),
        "sharpe_median": float(sharpe_vals.median()),
        "sharpe_std": float(sharpe_vals.std()),
        "sharpe_min": float(sharpe_vals.min()),
        "sharpe_max": float(sharpe_vals.max()),
        "pct_combinations_beating_spy_sharpe": float((df["Sharpe Diff vs SPY"] > 0).mean()),
        "pct_combinations_beating_spy_cagr": float((excess_vals > 0).mean()),
        "best_config": df.loc[sharpe_vals.idxmax(), [
            "top10_pct", "gld_pct", "btc_pct", "rebal_freq", "Strategy Sharpe"
        ]].to_dict(),
        "worst_config": df.loc[sharpe_vals.idxmin(), [
            "top10_pct", "gld_pct", "btc_pct", "rebal_freq", "Strategy Sharpe"
        ]].to_dict(),
    }


def default_weight_grid() -> list:
    """
    A reasonable default grid: vary each sleeve by ~10% increments
    while keeping the three-sleeve structure.
    """
    grid = []
    # Equity-heavy
    grid.append((0.70, 0.20, 0.10))
    grid.append((0.60, 0.30, 0.10))
    grid.append((0.60, 0.25, 0.15))
    # Baseline
    grid.append((0.50, 0.40, 0.10))
    grid.append((0.50, 0.35, 0.15))
    grid.append((0.50, 0.30, 0.20))
    # Gold-heavy
    grid.append((0.40, 0.50, 0.10))
    grid.append((0.40, 0.40, 0.20))
    # Crypto-heavy (test the limits)
    grid.append((0.40, 0.30, 0.30))
    grid.append((0.30, 0.40, 0.30))
    # No-crypto control
    grid.append((0.55, 0.45, 0.00))
    grid.append((0.60, 0.40, 0.00))
    return grid
