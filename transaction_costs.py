"""
Transaction cost model.

Applies a flat basis-points cost to portfolio turnover at each rebalance.

Cost is computed as:  cost_dollars = portfolio_value * turnover * (cost_bps / 10000)

where turnover = 0.5 * sum(|new_weight - old_weight|) — i.e. the fraction
of the portfolio that changed hands. We use 0.5x because each "trade" is
a sell on one side and a buy on the other; we don't want to double-count.

The cost is deducted from the portfolio value on the rebalance date as a
negative cashflow, which then flows through the existing equity-curve math.

USAGE:
    from transaction_costs import apply_transaction_costs

    cost_series = apply_transaction_costs(
        target_weights=target_weights,   # DataFrame of weights, indexed by date
        equity_curve=eqB_lump,            # before-cost equity curve
        rebal_dates=rebal_dates,
        cost_bps=10,                      # 10 bps = 0.10% per dollar traded
        crypto_tickers=["BTC-USD"],       # optional: extra cost for high-spread assets
        crypto_cost_bps=30,               # default 30 bps for BTC (retail-realistic)
    )

DEFAULTS — these are deliberately conservative (slightly pessimistic):
    - 10 bps for liquid US equity ETFs and mega-cap stocks (commission-free
      brokerage, but small spread + market impact)
    - 30 bps for BTC-USD on retail venues. Spreads have tightened over time
      but were materially wider pre-2020. A flat 30 bps averages over the
      backtest period.

These are user-tunable; a quant interviewer may push back on the specific
numbers, but having ANY cost model is the point.
"""

import numpy as np
import pandas as pd


def compute_turnover(prev_weights: pd.Series, new_weights: pd.Series) -> float:
    """
    Returns one-way turnover: the fraction of the portfolio that was traded.

    For a deployed-to-deployed rotation (both sides sum to ~1.0), this is
    0.5 * sum(|delta|) because every dollar bought has an offsetting dollar
    sold, and we only want to count one side.

    For initial deployment from cash (prev sums to 0), this is sum(|delta|)
    because there are no offsetting sells.

    The general formula is: 0.5 * (turnover_buy_side + turnover_sell_side)
    where each side is summed separately. When both sides match (rotation),
    this equals the simple 0.5 * sum(|delta|). When deploying from cash,
    sell side = 0 and buy side = 1.0, so result = 0.5? No — we want the
    DOLLAR cost of trading, which is just the sum of buys (or equivalently
    the sum of sells, when matched).

    So the correct formula is: turnover = max(buys, sells) where
    buys = sum of positive deltas, sells = sum of |negative deltas|.

    For matched rotations buys == sells. For initial deployment buys = 1.0,
    sells = 0.0, so turnover = 1.0 (correct).
    """
    aligned = pd.DataFrame({"prev": prev_weights, "new": new_weights}).fillna(0.0)
    deltas = aligned["new"] - aligned["prev"]
    buys = float(deltas[deltas > 0].sum())
    sells = float(-deltas[deltas < 0].sum())
    # The cost is on whichever side is larger (the binding constraint on trading)
    return max(buys, sells)


def compute_per_asset_cost_bps(
    weight_deltas: pd.Series,
    crypto_tickers: list,
    crypto_cost_bps: float,
    default_cost_bps: float,
) -> float:
    """
    Returns the dollar-weighted average cost in bps across all assets traded.

    weight_deltas: Series of |new_weight - old_weight| per ticker.
    """
    total_traded = weight_deltas.sum()
    if total_traded <= 0:
        return 0.0
    crypto_set = set(crypto_tickers)
    weighted_bps = 0.0
    for ticker, delta in weight_deltas.items():
        if delta <= 0:
            continue
        bps = crypto_cost_bps if ticker in crypto_set else default_cost_bps
        weighted_bps += bps * (delta / total_traded)
    return weighted_bps


def apply_transaction_costs(
    target_weights: pd.DataFrame,
    equity_curve: pd.Series,
    rebal_dates: pd.DatetimeIndex,
    cost_bps: float = 10.0,
    crypto_tickers: list = None,
    crypto_cost_bps: float = 30.0,
) -> pd.Series:
    """
    Returns a Series of dollar costs per rebalance date.

    The caller is responsible for subtracting these from the equity curve.
    """
    if crypto_tickers is None:
        crypto_tickers = []

    costs = pd.Series(0.0, index=rebal_dates, dtype=float)
    prev_weights = pd.Series(0.0, index=target_weights.columns)

    for rd in rebal_dates:
        if rd not in target_weights.index or rd not in equity_curve.index:
            continue

        new_weights = target_weights.loc[rd]
        weight_deltas = (new_weights - prev_weights).abs()

        # Dollar-weighted blended cost (handles BTC differently than equities)
        blended_bps = compute_per_asset_cost_bps(
            weight_deltas=weight_deltas,
            crypto_tickers=crypto_tickers,
            crypto_cost_bps=crypto_cost_bps,
            default_cost_bps=cost_bps,
        )

        # Total turnover (correctly handles initial deployment vs rotation)
        turnover = compute_turnover(prev_weights, new_weights)
        portfolio_value = float(equity_curve.loc[rd])
        costs.loc[rd] = portfolio_value * turnover * (blended_bps / 10000.0)

        prev_weights = new_weights.copy()

    return costs


def apply_costs_to_returns(
    daily_ret: pd.Series,
    cost_series: pd.Series,
    equity_curve_pre_cost: pd.Series,
) -> pd.Series:
    """
    Convert dollar costs at rebalance dates into a return drag, applied
    to the daily return series.

    On a rebalance date, return is reduced by (cost / portfolio_value).
    This keeps everything in return-space so it composes with the existing
    cashflow math.
    """
    adjusted = daily_ret.copy()
    for dt, cost in cost_series.items():
        if cost > 0 and dt in adjusted.index and dt in equity_curve_pre_cost.index:
            pv = float(equity_curve_pre_cost.loc[dt])
            if pv > 0:
                adjusted.loc[dt] -= cost / pv
    return adjusted
