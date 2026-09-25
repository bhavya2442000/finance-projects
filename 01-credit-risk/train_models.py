"""Step 2: probability-of-default (PD) models.

Train on 2012-2014 vintages, test out-of-time on the 2015 vintage.
Compares a logistic-regression scorecard with gradient-boosted trees (XGBoost),
each with and without LendingClub's own grade / interest rate, so we can see
how much the model adds on top of LendingClub's pricing.
"""
import json
from pathlib import Path

import duckdb
import matplotlib
import numpy as np
import pandas as pd
import shap
import xgboost as xgb
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).parent
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)
SEED = 42

NUMERIC = [
    "loan_amnt", "annual_inc", "dti", "fico", "emp_length_yrs", "delinq_2yrs",
    "credit_history_months", "inq_last_6mths", "mths_since_last_delinq", "open_acc",
    "pub_rec", "revol_bal", "revol_util", "total_acc", "mort_acc", "pub_rec_bankruptcies",
    "tot_cur_bal", "total_rev_hi_lim", "acc_open_past_24mths", "bc_util", "num_actv_rev_tl",
    "percent_bc_gt_75", "mths_since_recent_inq", "pct_tl_nvr_dlq", "pmt_to_income",
    "loan_to_income",
]
CATEGORICAL = ["home_ownership", "verification_status", "purpose"]
LC_PRICING = ["int_rate", "sub_grade_num"]  # LendingClub's own risk assessment


def load() -> pd.DataFrame:
    con = duckdb.connect(str(ROOT / "data" / "credit.duckdb"), read_only=True)
    df = con.execute("SELECT * FROM loans").df()
    con.close()
    # A1 = 1 ... G5 = 35
    df["sub_grade_num"] = (df["sub_grade"].str[0].map(ord) - ord("A")) * 5 + df["sub_grade"].str[1].astype(int)
    return df


def ks_stat(y, p) -> float:
    fpr, tpr, _ = roc_curve(y, p)
    return float(np.max(tpr - fpr))


def psi(expected, actual, bins=10) -> float:
    """Population stability index of a score between two samples."""
    cuts = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    cuts[0], cuts[-1] = -np.inf, np.inf
    e = np.histogram(expected, cuts)[0] / len(expected)
    a = np.histogram(actual, cuts)[0] / len(actual)
    e, a = np.clip(e, 1e-6, None), np.clip(a, 1e-6, None)
    return float(np.sum((a - e) * np.log(a / e)))


def metrics(name, y, p) -> dict:
    auc = roc_auc_score(y, p)
    return {"model": name, "auc": round(auc, 4), "gini": round(2 * auc - 1, 4),
            "ks": round(ks_stat(y, p), 4), "brier": round(brier_score_loss(y, p), 4),
            "mean_pd": round(float(p.mean()), 4), "actual_bad_rate": round(float(y.mean()), 4)}


def logit_pipeline(numeric, categorical):
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                          ("log", _SignedLog()), ("scale", StandardScaler())]), numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=500), categorical),
    ])
    return Pipeline([("pre", pre), ("clf", LogisticRegression(C=0.1, max_iter=2000))])


class _SignedLog:
    """log1p on heavy-tailed columns so the linear model isn't dominated by outliers."""
    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self.skewed_ = np.nanmax(X, axis=0) > 1000
        return self

    def transform(self, X):
        X = np.asarray(X, dtype=float).copy()
        X[:, self.skewed_] = np.sign(X[:, self.skewed_]) * np.log1p(np.abs(X[:, self.skewed_]))
        return X

    def get_params(self, deep=True):
        return {}

    def set_params(self, **params):
        return self


CATEGORY_LEVELS: dict[str, list] = {}  # fixed from the training set; unseen levels become missing


def xgb_frame(df, cols):
    X = df[cols].copy()
    for c in CATEGORICAL:
        if c in X:
            known = X[c].where(X[c].isin(CATEGORY_LEVELS[c]))
            X[c] = pd.Categorical(known, categories=CATEGORY_LEVELS[c])
    return X


def fit_xgb(train, cols):
    X, y = xgb_frame(train, cols), train["is_bad"]
    # last 15% of the training period (by issue date) is the early-stopping set
    cutoff = train["issue_date"].quantile(0.85)
    fit_mask = train["issue_date"] < cutoff
    model = xgb.XGBClassifier(
        n_estimators=3000, learning_rate=0.03, max_depth=4, min_child_weight=50,
        subsample=0.8, colsample_bytree=0.7, reg_lambda=5, eval_metric="auc",
        early_stopping_rounds=100, enable_categorical=True, tree_method="hist",
        random_state=SEED, n_jobs=-1)
    model.fit(X[fit_mask], y[fit_mask], eval_set=[(X[~fit_mask], y[~fit_mask])], verbose=False)
    return model


def main():
    df = load()
    train = df[df.vintage_year <= 2014].reset_index(drop=True)
    test = df[df.vintage_year == 2015].reset_index(drop=True)
    y_tr, y_te = train["is_bad"], test["is_bad"]
    CATEGORY_LEVELS.update({c: sorted(train[c].dropna().unique()) for c in CATEGORICAL})
    print(f"train {len(train):,} loans ({y_tr.mean():.2%} bad) | test {len(test):,} ({y_te.mean():.2%} bad)")

    results, preds = [], pd.DataFrame({"id": test["id"]})

    # Benchmark: LendingClub's own sub-grade, mapped to its 2012-2014 bad rate
    grade_pd = train.groupby("sub_grade")["is_bad"].mean()
    preds["pd_lc_grade"] = test["sub_grade"].map(grade_pd).fillna(y_tr.mean()).to_numpy()
    results.append(metrics("LendingClub sub-grade (benchmark)", y_te, preds["pd_lc_grade"]))

    feature_sets = {"bureau+app": (NUMERIC, CATEGORICAL),
                    "bureau+app+LC pricing": (NUMERIC + LC_PRICING, CATEGORICAL)}
    models = {}
    for fs_name, (num, cat) in feature_sets.items():
        lr = logit_pipeline(num, cat).fit(train[num + cat], y_tr)
        p = lr.predict_proba(test[num + cat])[:, 1]
        results.append(metrics(f"Logistic scorecard [{fs_name}]", y_te, p))
        preds[f"pd_logit_{'full' if 'LC' in fs_name else 'base'}"] = p

        xm = fit_xgb(train, num + cat)
        p = xm.predict_proba(xgb_frame(test, num + cat))[:, 1]
        results.append(metrics(f"XGBoost [{fs_name}]", y_te, p) | {"trees": xm.best_iteration + 1})
        preds[f"pd_xgb_{'full' if 'LC' in fs_name else 'base'}"] = p
        models[fs_name] = (xm, num + cat)

    res = pd.DataFrame(results)
    print(res.to_string(index=False))
    res.to_csv(OUT / "model_comparison.csv", index=False)

    # Champion = XGBoost with all features; check stability and calibration.
    champ, champ_cols = models["bureau+app+LC pricing"]
    p_train = champ.predict_proba(xgb_frame(train, champ_cols))[:, 1]
    p_test = preds["pd_xgb_full"].to_numpy()
    stability = {"score_psi_train_vs_2015": round(psi(p_train, p_test), 4)}
    print(stability)

    calib = (pd.DataFrame({"pd": p_test, "bad": y_te})
             .assign(decile=lambda d: pd.qcut(d["pd"], 10, labels=range(1, 11)))
             .groupby("decile", observed=True)
             .agg(loans=("bad", "size"), predicted_pd=("pd", "mean"), actual_bad_rate=("bad", "mean")))
    print(calib.round(4).to_string())
    calib.to_csv(OUT / "calibration_by_decile.csv")

    # SHAP: which features drive the champion model
    sample = xgb_frame(test, champ_cols).sample(20000, random_state=SEED)
    sv = shap.TreeExplainer(champ).shap_values(sample)
    importance = (pd.Series(np.abs(sv).mean(axis=0), index=champ_cols)
                  .sort_values(ascending=False).rename("mean_abs_shap"))
    print(importance.head(12).round(4).to_string())
    importance.to_csv(OUT / "feature_importance_shap.csv")
    plt.figure()
    shap.summary_plot(sv, sample, max_display=15, show=False)
    plt.tight_layout()
    plt.savefig(OUT / "shap_summary.png", dpi=150)
    plt.close("all")

    # ROC curves
    fig, ax = plt.subplots(figsize=(6, 5))
    for col, label in [("pd_xgb_full", "XGBoost (all features)"), ("pd_logit_full", "Logistic scorecard"),
                       ("pd_xgb_base", "XGBoost (no LC grade/rate)")]:
        fpr, tpr, _ = roc_curve(y_te, preds[col])
        ax.plot(fpr, tpr, label=f"{label}  AUC {roc_auc_score(y_te, preds[col]):.3f}")
    fpr, tpr, _ = roc_curve(y_te, preds["pd_lc_grade"])
    ax.plot(fpr, tpr, "--", label=f"LC sub-grade  AUC {roc_auc_score(y_te, preds['pd_lc_grade']):.3f}")
    ax.plot([0, 1], [0, 1], color="grey", lw=0.5)
    ax.set(xlabel="False positive rate", ylabel="True positive rate",
           title="Out-of-time ROC, 2015 vintage")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "roc_2015.png", dpi=150)

    preds = preds.merge(test[["id", "grade", "sub_grade", "sub_grade_num", "int_rate", "funded_amnt",
                              "is_bad", "out_ead", "out_net_return", "out_total_pymnt"]], on="id")
    preds.to_parquet(ROOT / "data" / "test_2015_predictions.parquet", index=False)
    (OUT / "stability.json").write_text(json.dumps(stability, indent=2))


if __name__ == "__main__":
    main()
