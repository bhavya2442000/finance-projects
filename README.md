# Finance Projects

Real, reproducible portfolio projects on public data, aimed at 1-3 year fintech and finance analyst roles: strategic finance / FP&A, credit risk and risk analytics.

## Why this folder exists

The first plan was to rebuild three textbook projects: a DCF, capital budgeting, and cash-flow forecasting. Job research in Sept 2026 (Robinhood, Ramp, Affirm, Brex, Stripe, Chime, Intuit postings) pointed elsewhere. Those roles ask for SQL and Python, driver-based forecasting, cohort and vintage analysis, loss forecasting, unit economics and ML/predictive modeling. So the projects were redesigned around that.

These are **new** projects, built in 2026. They go on the resume as portfolio projects, separate from the three older project bullets already in `../resume-builder/master.json`.

## Rules

- **Never make up numbers.** Every figure comes from a public source (SEC filings, public loan data, market prices) or is a clearly labeled assumption with a reason.
- **Test honestly.** Use out-of-time or holdout tests. Never tune on the test set, and report misses as well as wins.
- **End with a recommendation.** Every project ends with a business decision and the dollars behind it, not just charts.
- **Keep it explainable in an interview.** Label every assumption and give its source.
- **Update the resume bank when a project is done.** Add its bullets to `../resume-builder/master.json` using its real numbers. The resume tailor only allows numbers that exist in the bank.

## Projects

### 01 - Consumer credit risk: default model, expected loss, approval policy ✅
LendingClub's public loan data, 590k loans. The work:
- DuckDB SQL pipeline
- vintage loss curves
- logistic scorecard vs XGBoost, tested on the 2015 vintage (out-of-time)
- SHAP explanations and calibration/stability checks
- PD × LGD × EAD expected loss
- approval-cutoff simulation measured in real dollars

See `01-credit-risk/README.md`.

### 02 - SoFi driver-based revenue forecast (strategic finance) ✅
SoFi's own reported numbers, parsed from 21 SEC earnings releases. The work:
- 29-quarter KPI panel, tied to consolidated revenue and SEC XBRL
- driver model: originations × revenue per $ originated, products × revenue per product, segment margins
- 12 forecast approaches (naive, run-rate, damped Holt, pooled gradient boosting; driver-based vs top-down) in a rolling-origin backtest
- holdout year forecast within 6.0% of actual, with a volume/rate variance bridge
- FY2026 guidance check and sensitivities

See `02-sofi-forecast/README.md`.

### 03 - Optional: valuation of the same fintech
DCF, comps and scenarios on SoFi, using project 02's forecast. Build it only if equity research roles are a target.

## Environment

- Project code runs in `./.venv`, which holds pandas, scikit-learn, xgboost, shap and duckdb. Don't install these into the base Anaconda Python: shap upgrades numpy and pandas and breaks it.
- Numbers.app. No Excel, so openpyxl can't recalculate formulas here. Check any Excel formulas against the Python results.

## Layout, per project

```
NN-name/
  README.md         what was done, key results, assumptions, how to rerun
  requirements.txt
  sql/              SQL steps (where used)
  data/             raw downloads + intermediate files (large; re-download instead of sharing)
  *.py              pipeline scripts, run in order
  output/           workbook (.xlsx), charts, tables
```

## Status

- [x] 01 Credit risk: done
- [x] 02 SoFi driver-based forecast: done
- [ ] 03 Valuation (optional)
