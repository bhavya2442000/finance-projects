# 01 - Consumer Credit Risk: Default Model, Expected Loss and Approval Policy

**Question:** If a lender had scored LendingClub's 2015 loans with a machine-learning default model, trained only on 2012-2014 loans, would it have made better approval decisions than LendingClub's own grades? How much money would that have been worth?

**Short answer:** Yes. The model ranks risk better than LendingClub's sub-grade (AUC 0.703 vs 0.679). Declining the riskiest 20% of 2015 loans by model score would have kept **$13.6mm more net profit** than declining the same number of loans by sub-grade. The realized net return on kept loans would have been **7.48% vs 7.03%**.

## Data

- **Source:** LendingClub accepted loans, 2007-2018Q4, 2.26M loans and 151 columns. This is the public LendingClub loan data, [mirrored on Hugging Face](https://huggingface.co/datasets/codesignal/lending-club-loan-accepted); loan status is as of early 2019.
- **Sample:** 36-month loans issued 2012-2015 with a final outcome, 589,549 loans and $7.5bn funded. All of these loans reached maturity before the data snapshot, so outcomes are observed, not guessed.
- **Target:** "bad" means Charged Off, Default, or 31-120 days late. The 2015 loans still marked Current are excluded (72 loans).
- **Split:** out-of-time. Train on 2012-2014 (306,462 loans, 13.25% bad) and test on 2015 (283,087 loans, 14.90% bad). This is how a lender validates a model: fit on older loans, test on a newer vintage the model has never seen.
- **Leakage control:** features use only information known at application time: bureau and application data, plus LendingClub's grade and rate. Payment, recovery, hardship and settlement fields are used only to measure outcomes.

## What was built

| Step | File | What it does |
|---|---|---|
| 1 | `sql/01_loans_clean.sql` | DuckDB SQL: selects the sample, casts types, builds features (payment-to-income, credit history length, etc.) and outcome fields (exposure at default, net return) |
| 1 | `sql/02_vintage_curves.sql` | Cumulative net charge-off curves by vintage quarter and grade by months on book; loss given default (LGD) by grade |
| 2 | `train_models.py` | Logistic-regression scorecard and XGBoost, each with and without LendingClub's grade and rate; out-of-time AUC, Gini, KS, Brier; calibration by decile; population stability index (PSI); SHAP explanations |
| 3 | `policy_analysis.py` | Expected loss = PD × LGD × EAD vs actual loss; approval-cutoff simulation (model vs grade); realized return by risk decile; Excel workbook |

## Results (2015 out-of-time test)

### Model performance

| Model | AUC | Gini | KS |
|---|---|---|---|
| LendingClub sub-grade (benchmark) | 0.679 | 0.357 | 0.262 |
| Logistic scorecard, bureau + application data | 0.694 | 0.387 | 0.282 |
| XGBoost, bureau + application data | 0.694 | 0.387 | 0.282 |
| Logistic scorecard, + LC grade and rate | 0.697 | 0.395 | 0.288 |
| **XGBoost, + LC grade and rate (champion)** | **0.703** | **0.406** | **0.296** |

- **Without LendingClub's grade,** the model's bureau and application data alone beat the grade (0.694 vs 0.679).
- **The gain from XGBoost is small.** Over logistic regression it's +0.006 AUC, which is typical for tabular credit data. In a regulated lender, the more explainable scorecard might win.
- **Top SHAP drivers:** sub-grade, accounts opened in the past 24 months, annual income, interest rate, DTI, payment-to-income, FICO.
- **Score stability:** PSI between the 2012-2014 and 2015 scores is 0.02, which means stable (below 0.1).

### Calibration: the model under-predicted 2015 losses

- **Default rate:** mean predicted PD was 12.9% against an actual bad rate of 14.9%. The model under-predicted in every decile.
- **Loss dollars:** expected loss was **$229.8mm vs $274.0mm actual**, about 16% under, with most of the gap in grades C-G.
- **Why:** the applicant mix hardly changed (PSI 0.02), yet 2015 loans went bad more often than earlier vintages. The 2015 vintage lost 7.6% of funded by month 36, vs 6.0-7.2% for 2012-2014.
- **What a lender would do:** recalibrate PDs on the most recent vintage that has outcomes, or add a management overlay to the loss reserve. The model ranks risk well; its level needs adjusting when conditions shift.

### Business impact: approval cutoff simulation

Decline the riskiest X% of 2015 loans, ranked by model PD or by LC sub-grade. The same number of loans is declined either way. Results are *realized* outcomes of the loans kept:

| % declined | Net return, model | Net return, LC grade | Profit, model | Profit, LC grade | Bad rate, model | Bad rate, LC grade |
|---|---|---|---|---|---|---|
| 0% | 6.55% | 6.55% | $237.6mm | $237.6mm | 14.9% | 14.9% |
| 10% | 7.21% | 6.90% | $236.0mm | $225.6mm | 12.8% | 13.2% |
| 20% | 7.48% | 7.03% | $219.8mm | $206.2mm | 11.3% | 11.9% |
| 30% | 7.71% | 7.11% | $200.8mm | $183.8mm | 9.9% | 10.7% |

- **Profit peaks near a 5% decline:** $239.5mm vs $237.6mm when all loans are approved.
- **Beyond that, return per dollar keeps rising but total dollars fall.** That is the volume-vs-quality trade-off a credit policy team has to choose on.

### Pricing check: return by model risk decile

| PD decile | Actual bad rate | Avg interest rate | Realized net return |
|---|---|---|---|
| 1 (safest) | 2.8% | 6.4% | 7.2% |
| 3-6 | 7.5-14.5% | 8.8-11.8% | 7.9-8.1% |
| 9 | 24.9% | 14.8% | 4.9% |
| 10 (riskiest) | 33.7% | 16.5% | **0.4%** |

The riskiest decile was underpriced. Its 16.5% average rate did not cover a 33.7% bad rate, so those loans roughly broke even. Middle-risk loans earned the most.

## Recommendation

1. **Add the model score to the approval decision** alongside grade, and decline the top 5-10% of model risk. At 5-10%, total profit held roughly flat ($236-240mm vs $237.6mm) while net return rose from 6.55% to 6.96-7.21% and the bad rate fell from 14.9% to 12.8-13.7%.
2. **Reprice or decline the riskiest decile.** At a 0.4% realized return it carries most of the credit risk and earns almost nothing.
3. **Recalibrate PDs every quarter** on the latest vintage with outcomes, and hold a reserve overlay. The 2015 miss shows losses can shift even when the applicant mix doesn't.

## Assumptions and limitations

- **Net return = (total payments received − funded amount) / funded amount.** It ignores LendingClub's ~1% investor servicing fee and the time value of money, so it overstates true yield. It is used consistently across comparisons.
- **Charge-off month isn't in the file.** It is estimated as last payment month + 5, because LendingClub charged off at about 150 days past due. This only affects the *shape* of the vintage curves, not the totals.
- **LGD and EAD come from 2012-2014 defaults.** LGD is by grade (87-89%). EAD is 57.4% of the original loan, the average balance still owed at default.
- **Selection bias:** only loans LendingClub approved are observed. The simulation can test *tightening* the policy, not approving applicants LendingClub rejected, which would need reject inference.
- **XGBoost settings:** chosen by hand, with early stopping on the last 15% of the training period (by issue date). There was no hyperparameter search, which kept runtime low. The 2015 test set was never used for tuning.

## How to rerun

```bash
cd finance-projects
python3 -m venv .venv && .venv/bin/pip install -r 01-credit-risk/requirements.txt
cd 01-credit-risk
curl -L -o data/accepted_2007_to_2018Q4.csv \
  https://huggingface.co/datasets/codesignal/lending-club-loan-accepted/resolve/main/accepted_2007_to_2018Q4.csv
../.venv/bin/python -c "import duckdb; duckdb.execute(\"COPY (SELECT * FROM read_csv('data/accepted_2007_to_2018Q4.csv', all_varchar=true, sample_size=-1)) TO 'data/loans_raw.parquet' (FORMAT parquet)\")"
../.venv/bin/python build_data.py       # ~10 s
../.venv/bin/python train_models.py     # ~4 min
../.venv/bin/python policy_analysis.py  # ~10 s
```

## Outputs

`output/`: `credit_risk_results.xlsx` (all tables), `model_comparison.csv`, `calibration_by_decile.csv`, `feature_importance_shap.csv`, `roc_2015.png`, `shap_summary.png`, `vintage_curves.png`, `cutoff_simulation.png`.
