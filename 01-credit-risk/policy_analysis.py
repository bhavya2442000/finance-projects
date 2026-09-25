"""Step 3: turn model scores into business decisions.

Uses the 2015 out-of-time predictions from train_models.py:
  1. vintage loss curves (from SQL)
  2. expected loss = PD x LGD x EAD vs actual losses
  3. approval-cutoff simulation: decline the riskiest X% by model vs by LendingClub grade,
     and compare the realized return of the loans that would have been kept
  4. return by risk decile (is the interest rate enough to cover the losses?)
Writes charts and an Excel workbook to output/.
"""
from pathlib import Path

import duckdb
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).parent
OUT = ROOT / "output"
PD_COL = "pd_xgb_full"  # champion model
DECLINE_SHARES = [0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]


def load():
    con = duckdb.connect(str(ROOT / "data" / "credit.duckdb"), read_only=True)
    curves = con.execute("SELECT * FROM vintage_curves").df()
    lgd = con.execute("SELECT * FROM lgd_by_grade").df()
    # exposure at default as a share of the original loan, 2012-2014 defaults
    ead_share = con.execute("""SELECT SUM(out_ead) / SUM(funded_amnt) FROM loans
                               WHERE is_bad = 1 AND vintage_year <= 2014""").fetchone()[0]
    recov = con.execute("""SELECT id, out_recoveries - out_recovery_fee AS net_recoveries
                           FROM loans WHERE vintage_year = 2015""").df()
    con.close()
    preds = (pd.read_parquet(ROOT / "data" / "test_2015_predictions.parquet")
             .merge(recov, on="id", how="left"))
    return curves, lgd, ead_share, preds


def plot_vintages(curves):
    by_year = (curves.assign(year=pd.to_datetime(curves["vintage_q"]).dt.year)
               .groupby(["year", "mob"])[["cum_net_loss", "funded"]].sum()
               .assign(rate=lambda d: d["cum_net_loss"] / d["funded"]).reset_index())
    by_grade = (curves.groupby(["grade", "mob"])[["cum_net_loss", "funded"]].sum()
                .assign(rate=lambda d: d["cum_net_loss"] / d["funded"]).reset_index())
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for yr, g in by_year.groupby("year"):
        axes[0].plot(g["mob"], g["rate"] * 100, label=str(yr))
    axes[0].set(title="Cumulative net loss by vintage year", xlabel="Months on book", ylabel="% of funded")
    for gr, g in by_grade.groupby("grade"):
        axes[1].plot(g["mob"], g["rate"] * 100, label=gr)
    axes[1].set(title="Cumulative net loss by grade, 2012-2015", xlabel="Months on book")
    for ax in axes:
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "vintage_curves.png", dpi=150)
    return by_year.pivot(index="mob", columns="year", values="rate")


def expected_loss(preds, lgd, ead_share):
    df = preds.merge(lgd[["grade", "lgd"]], on="grade", how="left")
    df["expected_loss"] = df[PD_COL] * df["lgd"] * ead_share * df["funded_amnt"]
    # principal lost net of recoveries, on the same basis as expected loss
    df["actual_loss"] = np.where(df["is_bad"] == 1, df["out_ead"] - df["net_recoveries"], 0)
    by_grade = df.groupby("grade").agg(loans=("id", "size"), funded=("funded_amnt", "sum"),
                                       expected_loss=("expected_loss", "sum"),
                                       actual_loss=("actual_loss", "sum"),
                                       predicted_pd=(PD_COL, "mean"), actual_bad_rate=("is_bad", "mean"))
    by_grade["expected_loss_rate"] = by_grade["expected_loss"] / by_grade["funded"]
    by_grade["actual_loss_rate"] = by_grade["actual_loss"] / by_grade["funded"]
    return df, by_grade


def cutoff_simulation(df):
    """Decline the riskiest share of 2015 loans, ranked by model PD or by LC sub-grade."""
    rows = []
    rng = np.random.default_rng(0)
    # break sub-grade ties randomly so both rankings decline exactly the same count
    lc_rank = df["sub_grade_num"] + rng.random(len(df)) * 0.01
    for share in DECLINE_SHARES:
        n_keep = int(round(len(df) * (1 - share)))
        for method, score in [("Model PD", df[PD_COL]), ("LC sub-grade", lc_rank)]:
            kept = df.loc[score.rank(method="first") <= n_keep]
            funded = kept["funded_amnt"].sum()
            rows.append({
                "decline_share": share, "method": method, "loans_kept": len(kept),
                "funded_kept_mm": funded / 1e6,
                "bad_rate": kept["is_bad"].mean(),
                "net_profit_mm": (kept["out_total_pymnt"].sum() - funded) / 1e6,
                "net_return_pct": (kept["out_total_pymnt"].sum() - funded) / funded * 100,
                "avg_int_rate": np.average(kept["int_rate"], weights=kept["funded_amnt"]),
            })
    return pd.DataFrame(rows)


def return_by_decile(df):
    out = (df.assign(pd_decile=pd.qcut(df[PD_COL], 10, labels=range(1, 11)))
           .groupby("pd_decile", observed=True)
           .apply(lambda g: pd.Series({
               "loans": len(g),
               "predicted_pd": g[PD_COL].mean(),
               "actual_bad_rate": g["is_bad"].mean(),
               "avg_int_rate": np.average(g["int_rate"], weights=g["funded_amnt"]),
               "net_return_pct": (g["out_total_pymnt"].sum() - g["funded_amnt"].sum())
                                 / g["funded_amnt"].sum() * 100,
           }), include_groups=False))
    return out


def plot_cutoffs(sim):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for method, g in sim.groupby("method"):
        axes[0].plot(g["decline_share"] * 100, g["net_return_pct"], marker="o", label=method)
        axes[1].plot(g["decline_share"] * 100, g["bad_rate"] * 100, marker="o", label=method)
    axes[0].set(title="Realized net return on kept loans (2015)", xlabel="% of loans declined",
                ylabel="Net return, % of funded")
    axes[1].set(title="Bad rate on kept loans (2015)", xlabel="% of loans declined", ylabel="Bad rate %")
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "cutoff_simulation.png", dpi=150)


def main():
    curves, lgd, ead_share, preds = load()
    vint = plot_vintages(curves)
    print("cumulative net loss at month 36 by vintage:\n", vint.loc[36].round(4).to_string())

    df, el_grade = expected_loss(preds, lgd, ead_share)
    print(f"\nEAD share of original loan at default: {ead_share:.3f}")
    print(f"2015 expected loss ${df['expected_loss'].sum()/1e6:,.1f}mm vs actual ${df['actual_loss'].sum()/1e6:,.1f}mm")
    print(el_grade.round(4).to_string())

    sim = cutoff_simulation(df)
    print("\n", sim.round(3).to_string(index=False))
    plot_cutoffs(sim)

    dec = return_by_decile(df)
    print("\n", dec.round(4).to_string())

    with pd.ExcelWriter(OUT / "credit_risk_results.xlsx") as xl:
        pd.read_csv(OUT / "model_comparison.csv").to_excel(xl, sheet_name="Model comparison", index=False)
        pd.read_csv(OUT / "calibration_by_decile.csv").to_excel(xl, sheet_name="Calibration 2015", index=False)
        pd.read_csv(OUT / "feature_importance_shap.csv").to_excel(xl, sheet_name="Feature importance", index=False)
        sim.to_excel(xl, sheet_name="Cutoff simulation", index=False)
        dec.to_excel(xl, sheet_name="Return by risk decile")
        el_grade.to_excel(xl, sheet_name="Expected loss by grade")
        lgd.to_excel(xl, sheet_name="LGD by grade", index=False)
        vint.to_excel(xl, sheet_name="Vintage curves")


if __name__ == "__main__":
    main()
