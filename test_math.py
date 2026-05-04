"""
Unit tests for backtester math functions.

Run with: pytest test_math.py -v

These tests validate the financial math against known-answer cases.
Each test states the expected value and why.
"""

import math
import numpy as np
import pandas as pd
import pytest

# Import the functions under test. Adjust the import path to match your project.
# If your main file is `backtester.py`, this works; otherwise change it.
from quarterly_rebal_backtester import (
    annualized_return_from_daily,
    annualized_vol,
    sharpe_ratio,
    beta_to_benchmark,
    treynor_ratio,
    cagr_from_equity,
    max_drawdown,
    xirr,
    apply_cashflows_to_returns,
    build_year_table,
)


# ─────────────────────────────────────────────
# CAGR
# ─────────────────────────────────────────────
class TestCAGR:
    def test_doubling_in_one_year(self):
        """A portfolio that doubles in exactly 1 year has 100% CAGR."""
        idx = pd.date_range("2020-01-01", "2020-12-31", freq="D")
        eq = pd.Series(np.linspace(100, 200, len(idx)), index=idx)
        # ~365 days, so CAGR should be ~100%
        result = cagr_from_equity(eq)
        # Use 365.25 day-year, so very slight deviation from exactly 1.0
        assert abs(result - 1.0) < 0.01, f"Expected ~1.0, got {result}"

    def test_flat_returns_zero(self):
        """No change in equity over any period → 0% CAGR."""
        idx = pd.date_range("2020-01-01", "2024-12-31", freq="D")
        eq = pd.Series(100.0, index=idx)
        assert abs(cagr_from_equity(eq)) < 1e-9

    def test_known_three_year_cagr(self):
        """100 → 133.1 over 3 years = exactly 10% CAGR."""
        idx = pd.DatetimeIndex(["2020-01-01", "2023-01-01"])
        eq = pd.Series([100.0, 133.1], index=idx)
        result = cagr_from_equity(eq)
        assert abs(result - 0.10) < 0.001, f"Expected 0.10, got {result}"


# ─────────────────────────────────────────────
# Max Drawdown
# ─────────────────────────────────────────────
class TestMaxDrawdown:
    def test_monotonic_increase_no_drawdown(self):
        eq = pd.Series([100, 110, 120, 130, 140])
        assert max_drawdown(eq) == 0.0

    def test_known_drawdown_50pct(self):
        """Peak 200, trough 100 → -50% drawdown."""
        eq = pd.Series([100, 150, 200, 150, 100, 120])
        assert abs(max_drawdown(eq) - (-0.5)) < 1e-9

    def test_recovery_doesnt_erase_drawdown(self):
        """Drawdown is the WORST point, even if we recover."""
        eq = pd.Series([100, 200, 100, 300])  # 50% DD then full recovery
        assert abs(max_drawdown(eq) - (-0.5)) < 1e-9


# ─────────────────────────────────────────────
# Annualized Return / Vol / Sharpe
# ─────────────────────────────────────────────
class TestAnnualizedMetrics:
    def test_constant_daily_return_annualizes_correctly(self):
        """A constant daily return r_d annualizes to (1+r_d)^252 - 1."""
        r_d = 0.0004  # 4 bps/day
        rets = pd.Series([r_d] * 252)
        expected = (1 + r_d) ** 252 - 1
        result = annualized_return_from_daily(rets, periods=252)
        assert abs(result - expected) < 1e-6

    def test_zero_returns_zero_vol(self):
        rets = pd.Series([0.0] * 252)
        assert annualized_vol(rets) == 0.0

    def test_known_volatility(self):
        """If daily std=0.01, annualized vol = 0.01 * sqrt(252)."""
        np.random.seed(42)
        rets = pd.Series(np.random.normal(0, 0.01, 10000))
        result = annualized_vol(rets, periods=252)
        expected = 0.01 * np.sqrt(252)
        assert abs(result - expected) < 0.005

    def test_sharpe_zero_when_returns_equal_rf(self):
        """If portfolio returns exactly match risk-free, Sharpe should be ~0."""
        rf = 0.05
        rf_daily = (1 + rf) ** (1 / 252) - 1
        # Add tiny noise so vol > 0
        np.random.seed(0)
        rets = pd.Series([rf_daily] * 252) + pd.Series(np.random.normal(0, 1e-6, 252))
        result = sharpe_ratio(rets, rf_annual=rf)
        assert abs(result) < 1.0  # Should be very close to 0

    def test_sharpe_positive_when_outperforming_rf(self):
        rets = pd.Series([0.001] * 252)  # ~28% annual return
        result = sharpe_ratio(rets, rf_annual=0.0)
        assert result > 0


# ─────────────────────────────────────────────
# Beta / Treynor
# ─────────────────────────────────────────────
class TestBeta:
    def test_beta_one_for_identical_series(self):
        """A series regressed on itself has beta = 1."""
        np.random.seed(42)
        r = pd.Series(np.random.normal(0, 0.01, 500))
        assert abs(beta_to_benchmark(r, r) - 1.0) < 1e-9

    def test_beta_two_for_2x_leveraged(self):
        """If port_ret = 2 * bench_ret, beta = 2."""
        np.random.seed(42)
        b = pd.Series(np.random.normal(0, 0.01, 500))
        p = 2 * b
        assert abs(beta_to_benchmark(p, b) - 2.0) < 1e-9

    def test_beta_zero_for_uncorrelated(self):
        """Truly independent series → beta near 0."""
        np.random.seed(42)
        b = pd.Series(np.random.normal(0, 0.01, 5000))
        p = pd.Series(np.random.normal(0, 0.01, 5000))
        assert abs(beta_to_benchmark(p, b)) < 0.05


# ─────────────────────────────────────────────
# XIRR — most important to test, this is the function most likely to have bugs
# ─────────────────────────────────────────────
class TestXIRR:
    def test_simple_one_year_double(self):
        """
        Invest $100, get back $200 over 365.25 days → exactly 100% IRR
        under this code's day-year convention.

        NOTE: This code uses a 365.25-day year (averaging over leap years),
        which differs from Excel's 365.0 convention. Both are defensible.
        Tests are written against this code's convention.
        """
        # 365.25 days from start
        start = pd.Timestamp("2020-01-01")
        end = start + pd.Timedelta(days=365)  # use 365 days exactly
        cf = [-100, 200]
        dates = pd.DatetimeIndex([start, end])
        result = xirr(cf, dates)
        # 365 days under 365.25 convention → years = 0.99932
        # so r = 2^(1/0.99932) - 1 ≈ 1.00069
        expected = 2 ** (365.25 / 365.0) - 1
        assert abs(result - expected) < 1e-4, \
            f"Expected ~{expected:.4f}, got {result:.4f}"

    def test_zero_return(self):
        """Invest $100, get back $100 a year later → 0% IRR."""
        cf = [-100, 100]
        dates = pd.to_datetime(["2020-01-01", "2021-01-01"])
        result = xirr(cf, dates)
        assert abs(result) < 1e-4

    def test_xirr_self_consistency(self):
        """
        Verify XIRR is self-consistent: if we invest $X, then withdraw
        $X * (1+r)^t after t years (using the code's 365.25 convention),
        XIRR should return r.
        """
        r = 0.15  # 15% target IRR
        start = pd.Timestamp("2020-01-01")
        end = start + pd.Timedelta(days=int(365.25 * 3))  # ~3 years
        years = (end - start).days / 365.25
        future_value = 100 * (1 + r) ** years
        cf = [-100, future_value]
        dates = pd.DatetimeIndex([start, end])
        result = xirr(cf, dates)
        assert abs(result - r) < 1e-4, f"Expected {r}, got {result}"

    def test_xirr_multiple_cashflows_self_consistent(self):
        """
        Multi-cashflow self-consistency check.
        Construct cashflows such that NPV = 0 at a known rate, then
        verify XIRR recovers that rate.
        """
        target_r = 0.20
        start = pd.Timestamp("2020-01-01")
        # Investment, then 3 contributions, then a final withdrawal
        dates = [
            start,
            start + pd.Timedelta(days=180),
            start + pd.Timedelta(days=365),
            start + pd.Timedelta(days=730),
        ]
        # Pick first three cashflows arbitrarily, solve for the last
        cf_known = [-1000, -500, -500, None]
        years = [(d - start).days / 365.25 for d in dates]
        # NPV = sum(cf_i / (1+r)^t_i) = 0
        # Solve for cf[3]: cf[3] = -sum(cf_known[:3] / (1+r)^t_i) * (1+r)^t_3
        npv_partial = sum(c / (1 + target_r) ** t
                          for c, t in zip(cf_known[:3], years[:3]))
        cf_known[3] = -npv_partial * (1 + target_r) ** years[3]
        result = xirr(cf_known, pd.DatetimeIndex(dates))
        assert abs(result - target_r) < 1e-4, \
            f"Expected {target_r}, got {result}"

    def test_returns_nan_when_all_positive(self):
        """If there's no negative cashflow, XIRR is undefined."""
        cf = [100, 200, 300]
        dates = pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01"])
        assert math.isnan(xirr(cf, dates))

    def test_returns_nan_when_all_negative(self):
        """If there's no positive cashflow, XIRR is undefined."""
        cf = [-100, -200, -300]
        dates = pd.to_datetime(["2020-01-01", "2021-01-01", "2022-01-01"])
        assert math.isnan(xirr(cf, dates))


# ─────────────────────────────────────────────
# Cashflow Application — critical for contribution accounting
# ─────────────────────────────────────────────
class TestCashflowApplication:
    def test_no_cashflows_matches_compounding(self):
        """With zero cashflows, equity curve is just compounded returns."""
        idx = pd.date_range("2020-01-01", periods=10, freq="D")
        rets = pd.Series([0.01] * 10, index=idx)
        eq, cf = apply_cashflows_to_returns(rets, initial_capital=100, cashflows=None)
        expected = 100 * (1.01 ** 10)
        assert abs(eq.iloc[-1] - expected) < 1e-9

    def test_end_of_day_contribution_doesnt_earn_today(self):
        """A contribution added end-of-day shouldn't earn that day's return."""
        idx = pd.date_range("2020-01-01", periods=2, freq="D")
        rets = pd.Series([0.10, 0.0], index=idx)  # 10% day 1, 0% day 2
        cf = pd.Series([100.0, 0.0], index=idx)   # $100 contributed end of day 1
        eq, _ = apply_cashflows_to_returns(rets, 100, cf, timing="end_of_day")
        # Day 1: 100 * 1.10 + 100 = 210
        # Day 2: 210 * 1.00 = 210
        assert abs(eq.iloc[-1] - 210) < 1e-9

    def test_start_of_day_contribution_earns_today(self):
        """A contribution added start-of-day SHOULD earn that day's return."""
        idx = pd.date_range("2020-01-01", periods=2, freq="D")
        rets = pd.Series([0.10, 0.0], index=idx)
        cf = pd.Series([100.0, 0.0], index=idx)
        eq, _ = apply_cashflows_to_returns(rets, 100, cf, timing="start_of_day")
        # Day 1: (100 + 100) * 1.10 = 220
        # Day 2: 220 * 1.00 = 220
        assert abs(eq.iloc[-1] - 220) < 1e-9


# ─────────────────────────────────────────────
# Modified Dietz (build_year_table)
# ─────────────────────────────────────────────
class TestModifiedDietz:
    def test_no_contribution_simple_return(self):
        """Without contributions, Modified Dietz reduces to (end-start)/start."""
        idx = pd.date_range("2020-01-02", "2020-12-31", freq="B")
        eq = pd.Series(np.linspace(100, 120, len(idx)), index=idx)
        cf = pd.Series(0.0, index=idx)
        df = build_year_table(eq, cf)
        assert len(df) == 1
        assert abs(df["Approx Return (flow-adj)"].iloc[0] - 0.20) < 0.01

    def test_with_contribution_uses_half_weight(self):
        """
        Verify the 0.5 * contribution denominator:
        Start = 100, End = 220, Contribution = 100 mid-year.
        Net gain = 220 - 100 - 100 = 20
        Base = 100 + 0.5*100 = 150
        Return ≈ 20/150 = 13.33%
        """
        idx = pd.date_range("2020-01-02", "2020-12-31", freq="B")
        eq = pd.Series(np.linspace(100, 220, len(idx)), index=idx)
        cf = pd.Series(0.0, index=idx)
        cf.iloc[len(idx) // 2] = 100  # contribution mid-year
        df = build_year_table(eq, cf)
        expected = 20 / 150
        assert abs(df["Approx Return (flow-adj)"].iloc[0] - expected) < 0.01


# ─────────────────────────────────────────────
# Transaction Costs
# ─────────────────────────────────────────────
class TestTransactionCosts:
    """
    Test the transaction cost module.

    Imports inline to keep main backtester import-clean if cost module
    isn't installed.
    """

    def test_initial_deployment_full_turnover(self):
        from transaction_costs import compute_turnover
        prev = pd.Series({"AAPL": 0.0, "MSFT": 0.0, "GLD": 0.0})
        new = pd.Series({"AAPL": 0.5, "MSFT": 0.3, "GLD": 0.2})
        assert abs(compute_turnover(prev, new) - 1.0) < 1e-9

    def test_no_change_zero_turnover(self):
        from transaction_costs import compute_turnover
        w = pd.Series({"AAPL": 0.5, "MSFT": 0.5})
        assert compute_turnover(w, w) == 0.0

    def test_full_rotation(self):
        """Rotating 100% from AAPL to MSFT = 100% turnover (one-way)."""
        from transaction_costs import compute_turnover
        prev = pd.Series({"AAPL": 1.0, "MSFT": 0.0})
        new = pd.Series({"AAPL": 0.0, "MSFT": 1.0})
        assert abs(compute_turnover(prev, new) - 1.0) < 1e-9

    def test_partial_rotation(self):
        from transaction_costs import compute_turnover
        prev = pd.Series({"AAPL": 1.0, "MSFT": 0.0})
        new = pd.Series({"AAPL": 0.5, "MSFT": 0.5})
        assert abs(compute_turnover(prev, new) - 0.5) < 1e-9

    def test_cost_dollar_amount(self):
        """10000 portfolio * 100% turnover * 10bps = $10."""
        from transaction_costs import apply_transaction_costs
        dates = pd.DatetimeIndex(["2020-01-01"])
        weights = pd.DataFrame({"AAPL": [1.0]}, index=dates)
        equity = pd.Series([10000.0], index=dates)
        costs = apply_transaction_costs(weights, equity, dates, cost_bps=10)
        assert abs(costs.iloc[0] - 10.0) < 1e-9

    def test_crypto_higher_cost(self):
        """BTC trade should cost more than equity trade at the same turnover."""
        from transaction_costs import apply_transaction_costs
        dates = pd.DatetimeIndex(["2020-01-01"])
        # All-BTC initial deployment
        w_btc = pd.DataFrame({"BTC-USD": [1.0]}, index=dates)
        # All-equity initial deployment
        w_eq = pd.DataFrame({"AAPL": [1.0]}, index=dates)
        equity = pd.Series([10000.0], index=dates)
        cost_btc = apply_transaction_costs(
            w_btc, equity, dates, cost_bps=10,
            crypto_tickers=["BTC-USD"], crypto_cost_bps=30
        ).iloc[0]
        cost_eq = apply_transaction_costs(
            w_eq, equity, dates, cost_bps=10,
            crypto_tickers=["BTC-USD"], crypto_cost_bps=30
        ).iloc[0]
        assert cost_btc > cost_eq
        assert abs(cost_btc - 30.0) < 1e-9  # 10000 * 1.0 * 30bps
        assert abs(cost_eq - 10.0) < 1e-9   # 10000 * 1.0 * 10bps


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
