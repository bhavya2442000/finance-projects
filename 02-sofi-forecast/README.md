# 02 - SoFi Driver-Based Revenue Forecast (Strategic Finance)

**Question:** Can a driver-based model of SoFi, built from members, products, originations and the revenue each one earns, forecast revenue a year ahead? What does it say about SoFi's FY2026 guidance?

**Short answer:**
- **The holdout test:** a forecast made at Q2 2025 came within **6.0%** of the next four quarters' actual adjusted net revenue ($4,511mm forecast vs $4,255mm actual). The best statistical method from the earlier test window missed by **14.6%**.
- **FY2026 guidance:** the model puts FY2026 adjusted net revenue at **$5.16bn** against guidance of $4.75-4.85bn. The revenue guide looks conservative. The EBITDA guide ($1.6bn) is the harder part: it needs a **36% margin in H2 vs 30.4% in H1**.

## Data

- **Source:** 21 SoFi earnings releases (EX-99.1 of each 8-K Item 2.02, Aug 2021 - Jul 2026) from SEC EDGAR, plus SEC XBRL companyfacts.
- **Parsing:** every quarterly table in the releases was parsed, 12,583 facts in total. Where a later release restated a number, the newest value was kept, which left 6,044 unique facts.
- **Driver panel:** 29 quarters, Q2 2019 - Q2 2026. The model starts at Q4 2020, the first quarter with originations by product (23 quarters).
- **Checks run on every build:**
  - The four segments sum to consolidated revenue in all 26 quarters where all four are reported.
  - Net income matches SEC XBRL in 19 quarters and deposits in 18, with 0.0% difference.
  - Renamed line items (e.g. "Contribution profit" → "Contribution profit – Lending") were stitched, with no conflicts above 0.5%.

## Driver tree

| Line | Driver | How it is forecast |
|---|---|---|
| Lending revenue | total originations (personal + student + home) × revenue per $ originated | originations by product: forecast method; ratio: held at last quarter or trailing-4Q average |
| Financial Services revenue | FS products × revenue per FS product | products: forecast method; ratio: as above |
| Technology Platform revenue | forecast directly | forecast method |
| Corporate/Other | trailing 4-quarter average | assumption |
| Adjusted net revenue | total net revenue × adjusted/GAAP ratio | ratio as above |
| Adjusted EBITDA | adjusted net revenue × EBITDA margin | ratio as above |
| Contribution profit, per segment | segment revenue × contribution margin | ratio as above |

**Forecast methods compared.** All forecast four quarters ahead:
- **Naive:** hold the last value.
- **Run-rate:** continue the average growth of the last four quarters.
- **Damped Holt trend** on logs (statsmodels).
- **Pooled gradient boosting** (scikit-learn): one direct model per horizon, trained on the growth history of all 12 SoFi KPI series at once, so it has more than 23 points to learn from.

**Benchmark.** The same methods forecast revenue **top-down**, with no drivers. That gives 12 approaches in total: 4 volume methods × 2 ratio methods for the driver model, plus 4 top-down.

## Test design (no look-ahead)

- **Rolling origin:** at each quarter-end, only data up to that quarter is used and the next four quarters are forecast.
- **Selection window:** forecasts made at Q2 2023 - Q2 2024. Both champions (best driver approach, best top-down approach) were picked here, on adjusted net revenue MAPE.
- **Holdout:** the forecast made at Q2 2025 for Q3 2025 - Q2 2026. It was not used for any choice.

## Results

### Backtest accuracy (MAPE, adjusted net revenue, quarterly)

| Approach | Selection window | Holdout | All 9 origins |
|---|---|---|---|
| Top-down, damped Holt **(top-down champion)** | **5.5%** | 14.3% | 8.5% |
| Top-down, run-rate | 7.2% | 2.2% | 5.8% |
| Driver, run-rate + last-quarter ratios **(driver champion)** | 7.7% | **5.8%** | 7.0% |
| Top-down, pooled gradient boosting | 8.0% | 1.4% | 7.5% |
| Driver, damped Holt + 4Q-average ratios | 9.5% | 9.5% | 10.0% |
| Top-down, naive | 15.0% | 18.7% | 16.3% |

All 12 approaches are in `output/sofi_forecast.xlsx`.

- **No method won both windows.**
  - Damped Holt was best in 2023-24, when growth was slowing. It badly under-called 2025-26, when growth sped up again.
  - The ML model (pooled gradient boosting) was best on the holdout but only fourth in the selection window. I did not switch to it after seeing the holdout.
- **The driver champion's volume drivers held up on the holdout** (MAPE): members 0.4%, FS products 2.8%, originations 3.5%, adjusted EBITDA 2.5%. The misses came from rates and one known event.

### Holdout variance bridge: where the 6.0% over-forecast came from

Forecast made at Q2 2025, compared with actuals for Q3 2025 - Q2 2026. Adjusted net revenue, $mm:

| Item | $mm | What happened |
|---|---|---|
| Forecast | 4,511 | |
| Lending: origination volume | -50 | originations came in 2.0% below the run-rate over the year |
| Lending: revenue per $ originated | -40 | held at 5.06% in the forecast; actual ran 4.75-5.28% |
| Fin. Services: product count | +54 | more FS products than forecast |
| Fin. Services: revenue per product | -40 | revenue per product fell as the product count grew |
| Technology Platform | **-84** | a large client left the platform before Dec 31, 2025 (disclosed in the Q1 2026 release) |
| Corporate/Other | -29 | |
| Adjusted vs GAAP revenue ratio | -67 | non-GAAP adjustments swung |
| **Actual** | **4,255** | |

Volume effects are priced at the forecast rate; rate effects use actual volume, so the bridge ties exactly.

### Forward view vs FY2026 guidance

SoFi's FY2026 guidance comes from its Q2 2026 release (July 29, 2026): adjusted net revenue $4.75-4.85bn, adjusted EBITDA about $1.6bn, members up at least 30%. The model's FY2026 figure is H1 2026 actual plus its H2 2026 forecast:

| Metric | Guidance | Model FY2026 | Range (best/worst backtest error) |
|---|---|---|---|
| Adjusted net revenue | $4,750-4,850mm | **$5,162mm** | $4,861-5,434mm |
| Adjusted EBITDA | ~$1,600mm | $1,549mm (margin held at Q2 level) | |
| Members, Dec 2026 | ≥ 17.75mm (+30%) | 18.35mm | |

**What the guidance implies:**
- **Revenue growth:** the $4.8bn midpoint needs only **2.6% q/q revenue growth** in Q3-Q4 2026. Actual q/q growth over the last four quarters ran **6.7-10.9%**.
- **Methods above the midpoint:** 10 of the 12 approaches put FY2026 above $4.8bn. A zero-growth forecast (naive) gives $4.70bn.
- **EBITDA margin:** $1.6bn needs an **H2 EBITDA margin of 36.0%**. H1 2026 was 30.4%, and Q2 was 29.7%.

**Sensitivity:** next-4-quarter adjusted EBITDA, base $1,935mm:
- ±10% originations: about ±$123mm.
- ±50bp lending revenue per $ originated: about ±$125mm.
- The full grid is in the workbook.

## Memo: FY2026 outlook and forecasting process

1. **Revenue guidance looks beatable.** Even at the worst error seen in 9 backtests, FY2026 lands at $4.86bn, above the top of the range. The main risk is a slowdown in originations, since lending is about 60% of revenue: a 10% shortfall costs about $123mm of EBITDA over four quarters.
2. **The EBITDA guidance carries the risk.** Hitting $1.6bn needs about 5.6 points of H2 margin expansion. Holding the margin at the H1 level gives $1.55bn. Track the monthly margin against a 36% H2 target, not revenue.
3. **Fix the two weak links in the driver model:**
   - **Forecast Technology Platform by client.** One client's exit caused the biggest miss (-$84mm), and a trend model cannot see that.
   - **Model revenue per FS product and per $ originated as declining trends, not flat.** Together they cost $80mm.
4. **Run a statistical top-down forecast next to the driver model as a cross-check.** No single method won both test windows. When the two disagree by more than the backtest error range, something in the drivers needs explaining.

## Assumptions and limitations

- **Ratios are held flat.** Revenue per $ originated, revenue per FS product, the adjusted/GAAP ratio and all margins are held at their last-quarter value. The trailing-4Q average version was tested and did worse in the selection window.
- **Corporate/Other** is held at its trailing 4-quarter average.
- **Adjusted EBITDA = adjusted net revenue × a constant margin.** This ignores operating leverage, which is why it comes in under the $1.6bn guidance.
- **Originations include the Loan Platform Business** (loans originated for third parties), as SoFi reports them.
- **Small sample:** 23 quarters, and the range comes from only 9 backtest origins. Treat it as indicative, not a confidence interval.
- **Data source:** figures are SoFi's own, as reported (adjusted net revenue and adjusted EBITDA are non-GAAP), from the newest release that shows each quarter. The model does not predict one-offs such as client exits, fair-value marks or restatements.

## How to rerun

```bash
cd finance-projects
python3 -m venv .venv && .venv/bin/pip install -r 02-sofi-forecast/requirements.txt
cd 02-sofi-forecast
export SEC_USER_AGENT="Your Name you@example.com"   # SEC requires a contact
../.venv/bin/python download_filings.py   # 21 earnings releases + XBRL
../.venv/bin/python parse_releases.py     # release tables -> data/release_facts.csv
../.venv/bin/python build_panel.py        # driver panel + tie-out checks
../.venv/bin/python forecast.py           # backtest, bridge, forward view, workbook
```

## Outputs

- **`output/sofi_forecast.xlsx`:** driver panel, MAPE tables, holdout detail, variance bridge, forecast, guidance check, sensitivity grids.
- **Charts:** `holdout_forecast.png`, `variance_bridge.png`, `forward_forecast.png`.
- **`backtest_forecasts.csv`:** every forecast vs actual.
