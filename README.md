# Quarterly Rebalance Portfolio Backtester

A Streamlit-based backtester comparing two portfolios:

- **Portfolio A**: 100% SPY buy-and-hold (benchmark)
- **Portfolio B**: 50% S&P 500 top-10 (cap-weighted within sleeve) / 40% GLD / 10% BTC-USD, rebalanced quarterly with event-driven rebalances on top-10 composition changes

The project is intentionally a **methodology demonstration** — the allocation itself uses well-known building blocks (concentrated mega-cap exposure, gold as a hard-asset hedge, a small BTC satellite). The contribution is the testing infrastructure: lagged weight application, dual lump-sum/DCA accounting, XIRR for irregular cash flows, transaction-cost modeling, and a parameter sensitivity analysis to test for overfitting.

## What's implemented

- yfinance data ingestion with robust MultiIndex/single-ticker handling and historical ticker normalization (BRK.B, FB→META, etc.)
- Quarterly rebalancing on the first trading day of each calendar quarter, augmented by event-triggered rebalances when the top-10 composition changes between quarters
- Next-day execution after signal date to prevent intraday look-ahead
- Lagged weight application (`weights.shift(1)` against returns) to prevent same-day look-ahead
- Dual accounting: lump-sum and dollar-cost-averaged equity curves computed in parallel, with explicit start-of-day vs end-of-day contribution timing
- XIRR via Newton-Raphson with sign-change requirement and divergence guards
- Modified Dietz approximation for year-by-year flow-adjusted returns
- Transaction cost model with separate equity/crypto bps assumptions and proper turnover calculation that distinguishes initial deployment from rotations
- Parameter sensitivity grid across allocation weights and rebalance frequencies
- Explicit warnings when the requested backtest start date is truncated by data availability
- Koyfin-formatted trade ticket export

## Running

```bash
pip install -r requirements.txt
quarterly_rebal_backtester.py```

Tests:

```bash
pytest test_math.py -v
```

31 tests covering CAGR, max drawdown, annualized return/vol, Sharpe, beta, XIRR, cashflow application, Modified Dietz, and transaction cost computation. All pass.

## Known Limitations

This section is here on purpose. If you read this code and find an issue I haven't acknowledged below, I'd genuinely like to hear about it.

### 1. Top-10 holdings data provenance

The historical top-10 holdings CSV is reconstructed from iShares' IVV (iShares Core S&P 500 ETF) historical disclosures, used as a proxy for the S&P 500 index itself. IVV tracks the index with very tight weights, so the top-10 are essentially identical to the index's top-10 minus negligible tracking differences.

**iShares' public archive only reaches reliably back to ~2010.** Snapshots before that date should be treated with skepticism — if your CSV claims to have 2007-2009 data, verify the source. The cleanest defensible path is to start the backtest at 2010-01-01 or later.

The `snapshot_date` field is treated as the *publication* date (the date the holdings list became publicly observable). Combined with `USE_NEXT_DAY_EXECUTION=True`, this prevents look-ahead. If the CSV's snapshot dates are *as-of* dates with publication lag, look-ahead bias may be present.

### 2. BTC start-date constrains the backtest

BTC-USD on Yahoo Finance starts **2014-09-17**. With `STRICT_START_WHEN_ALL_COMPONENTS_LIVE=True`, the effective backtest cannot begin before this date regardless of the user's requested start. The code now emits a visible warning when this truncation occurs.

This is a meaningful limitation for interpretation: any backtest of Portfolio B is implicitly a post-2014 backtest, which coincides with the most favorable historical regime for mega-cap tech and BTC. **A favorable result is not necessarily strategy alpha — it may be regime selection.** The sensitivity analysis (no-BTC variants) helps isolate this.

### 3. Transaction costs are conservative estimates, not market-microstructure-accurate

The cost model uses flat basis-points assumptions (default 10 bps for equities, 30 bps for BTC). These are deliberately on the pessimistic side of likely retail execution but do not reflect:
- Time-varying spreads (BTC was much wider pre-2018)
- Market impact for large rebalances
- Bid-ask spread asymmetry on illiquid days
- Tax drag (this is a pre-tax backtest)

A more rigorous treatment would use historical effective spreads. The current model is sufficient to demonstrate that costs are a non-trivial drag (~30-60 bps/year on this strategy), not to produce execution-quality numbers.

### 4. The 50/40/10 allocation is chosen, not derived

The baseline weights were selected based on common practitioner heuristics (~50% concentrated equity, meaningful hard-asset hedge, small risk-on satellite) rather than derived from a formal optimization (risk parity, equal risk contribution, mean-variance). The sensitivity analysis tests robustness across nearby allocations — if the strategy only works at exactly 50/40/10, that's a red flag.

### 5. Forward-fill on prices

`prices = prices_sampled_raw.ffill()` handles holidays and missing data, but combined with `pct_change().fillna(0.0)` it means a delisted security would show zero return forever. This is unlikely to bite for the mega-cap top-10 universe but is a known code pattern that would need fixing before extending the strategy to a broader universe (e.g., top-50 or top-100).

### 6. BTC trades 365 days/year; SPY/GLD trade ~252

The Sharpe / volatility annualization uses 252 periods/year. Portfolio B's blended return series mixes a 365-day asset (BTC) with 252-day assets (SPY, GLD), so the "correct" annualization factor depends on weights. With BTC at 10% and weekend BTC moves filled as zeros for non-BTC assets, the impact is small (estimated <5% on the Sharpe) but not zero. A more careful treatment would use trading-day returns only, or weight-adjust the annualization factor.

### 7. XIRR uses 365.25-day year; Excel uses 365.0

The `xirr()` function uses a 365.25-day year (averaging over leap years), which differs from Excel's `XIRR` function (365.0 days). Both are defensible conventions. If you reconcile XIRR numbers from this backtester against Excel, expect small differences (~20-30 bps on annual rates).

### 8. No tax modeling

This is a pre-tax backtest. Quarterly rebalancing of appreciated positions in a taxable account would generate short-term capital gains in many cases. For a tax-deferred account (IRA, 401k) this is irrelevant; for a taxable account the realized return would be materially lower.

## What I'd build next, in priority order

1. **Replace the `weight_pct` data source** with a verified iShares historical pull using the `etf-scraper` package, starting 2010-01-01. Document each snapshot's actual publication date.
2. **Add a sub-period robustness check** — does the strategy work in 2014-2018 separately from 2018-2024? A strategy that only works in one sub-period is not a strategy.
3. **Replace forward-fill with explicit mask handling** so positions in untradeable assets contribute zero return without that being implicit.
4. **Migrate from yfinance to a paid data vendor** (Polygon, Tiingo, or similar) for production use. yfinance has known issues with historical adjustments around corporate actions.
5. **Refactor into modules** — `data.py`, `metrics.py`, `backtest.py`, `app.py`, `tests/` — currently everything is in one Streamlit file.

## Methodology references

- Modified Dietz return: industry-standard approximation for time-weighted returns under flows. CFA Institute, *GIPS Standards*.
- Newton-Raphson for IRR: see e.g. Press et al., *Numerical Recipes*, ch. 9.
- Cap-weighting within concentrated sleeves: a common technique in factor and thematic ETF construction; see e.g. MSCI methodology papers on top-N concentration indices.
