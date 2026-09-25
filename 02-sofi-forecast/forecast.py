"""Step 3: driver-based revenue forecast for SoFi, backtested on held-out quarters.

Driver tree (quarterly, $mm):
  Lending revenue            = total originations  x lending revenue per $ originated
  Financial Services revenue = FS products          x revenue per FS product
  Technology Platform rev.   = forecast directly
  Corporate/Other            = trailing 4-quarter average (assumption)
  Adjusted net revenue       = total net revenue    x adjusted / GAAP ratio
  Adjusted EBITDA            = adjusted net revenue x adjusted EBITDA margin
  Contribution profit        = segment revenue      x segment contribution margin

Volume drivers are forecast four quarters ahead with four methods: naive (last value),
run-rate (last 4 quarters' average growth continued), damped Holt trend on logs, and a
gradient-boosting model pooled across SoFi's KPI series. Ratio drivers use either the
last quarter or the trailing 4-quarter average. A "top-down" forecast of revenue with
the same methods (no drivers) is the benchmark.

Test design (rolling origin, no look-ahead):
  selection window: forecasts made at Q2 2023 .. Q2 2024 -> pick the champion method
  holdout:          forecast made at Q2 2025 for Q3 2025 .. Q2 2026 -> reported result
"""
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from statsmodels.tsa.holtwinters import ExponentialSmoothing

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
OUT = ROOT / "output"
OUT.mkdir(exist_ok=True)

H = 4
START = pd.Timestamp("2020-12-31")          # first quarter with originations by product
SELECTION_ORIGINS = pd.date_range("2023-06-30", "2024-06-30", freq="QE")
HOLDOUT_ORIGIN = pd.Timestamp("2025-06-30")
ALL_ORIGINS = pd.date_range("2023-06-30", "2025-06-30", freq="QE")

ORIGINATIONS = ["orig_personal", "orig_student", "orig_home"]
VOLUME = ["members", "products_fs", "rev_tp"] + ORIGINATIONS
TOPDOWN = ["rev_total", "adj_net_revenue"]
POOL = VOLUME + TOPDOWN + ["products_total", "products_lending", "tp_accounts", "deposits"]
RATIOS = {
    "lend_rev_per_orig": ("rev_lending", "orig_total"),
    "fs_rev_per_product": ("rev_fs", "products_fs"),
    "adj_to_gaap": ("adj_net_revenue", "rev_total"),
    "ebitda_margin": ("adj_ebitda", "adj_net_revenue"),
    "cm_lending": ("cp_lending", "rev_lending"),
    "cm_tp": ("cp_tp", "rev_tp"),
    "cm_fs": ("cp_fs", "rev_fs"),
}
VOLUME_METHODS = ["naive", "run_rate", "holt_damped", "gbm_pooled"]
RATIO_METHODS = ["last", "avg4"]
REPORT = ["adj_net_revenue", "rev_total", "rev_lending", "rev_fs", "rev_tp", "adj_ebitda",
          "orig_total", "products_fs", "members"]


def load_panel():
    p = pd.read_csv(ROOT / "data" / "driver_panel.csv", index_col=0, parse_dates=True)
    p = p.loc[START:].copy()
    for name, (num, den) in RATIOS.items():
        p[name] = p[num] / p[den]
    return p


# ---------- volume forecasts ----------

def f_naive(y):
    return np.repeat(y.iloc[-1], H)


def f_run_rate(y):
    g = np.log(y).diff().tail(4).mean()
    return y.iloc[-1] * np.exp(g * np.arange(1, H + 1))


def f_holt(y):
    fit = ExponentialSmoothing(np.log(y.to_numpy()), trend="add", damped_trend=True).fit()
    return np.exp(fit.forecast(H))


def gbm_features(logy, t):
    g = logy.diff()
    return [g.iloc[t], g.iloc[t - 1], g.iloc[t - 2], g.iloc[t - 3], g.iloc[t - 3:t + 1].mean(),
            logy.index[t].quarter]


def f_gbm_pooled(hist):
    """Direct multi-horizon model: predict log growth over h quarters from recent growth,
    trained on every KPI series in the panel up to the forecast origin."""
    out = {}
    logs = {s: np.log(hist[s].dropna()) for s in POOL if hist[s].notna().sum() >= 6}
    for h in range(1, H + 1):
        X, y = [], []
        for logy in logs.values():
            for t in range(4, len(logy) - h):
                X.append(gbm_features(logy, t))
                y.append(logy.iloc[t + h] - logy.iloc[t])
        model = GradientBoostingRegressor(n_estimators=200, max_depth=2, learning_rate=0.05,
                                          subsample=0.8, loss="huber", random_state=0)
        model.fit(np.array(X), np.array(y))
        for s, logy in logs.items():
            pred = model.predict(np.array([gbm_features(logy, len(logy) - 1)]))[0]
            out.setdefault(s, []).append(float(np.exp(logy.iloc[-1] + pred)))
    return {s: np.array(v) for s, v in out.items()}


def volume_forecasts(hist, series):
    """{method: {series: array of H forecasts}}"""
    fc = {m: {} for m in VOLUME_METHODS}
    gbm = f_gbm_pooled(hist)
    for s in series:
        y = hist[s].dropna()
        fc["naive"][s] = f_naive(y)
        fc["run_rate"][s] = f_run_rate(y)
        fc["holt_damped"][s] = f_holt(y)
        fc["gbm_pooled"][s] = gbm[s]
    return fc


def ratio_forecasts(hist, method):
    r = {}
    for name in RATIOS:
        y = hist[name].dropna()
        r[name] = np.repeat(y.iloc[-1] if method == "last" else y.tail(4).mean(), H)
    return r


# ---------- driver tree ----------

def build_pnl(vol, rat, corp):
    p = {k: np.asarray(v, dtype=float) for k, v in {**vol, **rat}.items()}
    p["orig_total"] = sum(p[o] for o in ORIGINATIONS)
    p["rev_lending"] = p["orig_total"] * p["lend_rev_per_orig"]
    p["rev_fs"] = p["products_fs"] * p["fs_rev_per_product"]
    p["rev_corp"] = np.asarray(corp, dtype=float)
    p["rev_total"] = p["rev_lending"] + p["rev_fs"] + p["rev_tp"] + p["rev_corp"]
    p["adj_net_revenue"] = p["rev_total"] * p["adj_to_gaap"]
    p["adj_ebitda"] = p["adj_net_revenue"] * p["ebitda_margin"]
    for seg in ["lending", "tp", "fs"]:
        p[f"cp_{seg}"] = p[f"rev_{seg}"] * p[f"cm_{seg}"]
    return p


def forecast_at(panel, origin):
    """All candidate forecasts made with data up to `origin`. Returns tidy rows."""
    hist = panel.loc[:origin]
    quarters = pd.date_range(origin + pd.offsets.QuarterEnd(1), periods=H, freq="QE")
    vol = volume_forecasts(hist, VOLUME + TOPDOWN)
    corp = np.repeat(hist["rev_corp"].tail(4).mean(), H)
    rows = []
    for vm in VOLUME_METHODS:
        for rm in RATIO_METHODS:
            pnl = build_pnl({s: vol[vm][s] for s in VOLUME}, ratio_forecasts(hist, rm), corp)
            for k, v in pnl.items():
                rows += [{"origin": origin, "approach": f"driver: {vm} + ratios {rm}", "driver": k,
                          "h": h + 1, "quarter": q, "forecast": v[h]} for h, q in enumerate(quarters)]
        for s in TOPDOWN:  # top-down benchmark: forecast revenue itself
            rows += [{"origin": origin, "approach": f"top-down: {vm}", "driver": s,
                      "h": h + 1, "quarter": q, "forecast": vol[vm][s][h]} for h, q in enumerate(quarters)]
    return pd.DataFrame(rows)


def attach_actuals(fc, panel):
    full = panel.copy()
    full["orig_total"] = full[ORIGINATIONS].sum(axis=1, min_count=3)
    act = full.stack().rename("actual").reset_index()
    act.columns = ["quarter", "driver", "actual"]
    fc = fc.merge(act, on=["quarter", "driver"], how="left")
    fc["ape"] = (fc["forecast"] / fc["actual"] - 1).abs()
    return fc


def score(fc, origins, drivers=REPORT):
    sub = fc[fc["origin"].isin(origins) & fc["driver"].isin(drivers)]
    return (sub.groupby(["approach", "driver"])["ape"].mean().unstack("driver")
            .reindex(columns=[d for d in drivers if d in sub["driver"].unique()]) * 100)


# ---------- variance analysis ----------

def variance_bridge(panel, fc_hold):
    """Explain the holdout miss on adjusted net revenue, summed over the 4 holdout quarters.
    Each segment is split into volume effect (at forecast rate) and rate effect (at actual volume)."""
    f = fc_hold.pivot(index="quarter", columns="driver", values="forecast")
    a = panel.loc[f.index].copy()
    a["orig_total"] = a[ORIGINATIONS].sum(axis=1)
    items = [
        ("Lending: origination volume", (a["orig_total"] - f["orig_total"]) * f["lend_rev_per_orig"]),
        ("Lending: revenue per $ originated", (a["lend_rev_per_orig"] - f["lend_rev_per_orig"]) * a["orig_total"]),
        ("Fin. Services: product count", (a["products_fs"] - f["products_fs"]) * f["fs_rev_per_product"]),
        ("Fin. Services: revenue per product", (a["fs_rev_per_product"] - f["fs_rev_per_product"]) * a["products_fs"]),
        ("Technology Platform", a["rev_tp"] - f["rev_tp"]),
        ("Corporate/Other", a["rev_corp"] - f["rev_corp"]),
    ]
    # GAAP -> adjusted: forecast ratio applied to each GAAP variance, plus the ratio's own variance
    rows = [(k, (v * f["adj_to_gaap"]).sum()) for k, v in items]
    rows.append(("Adjusted vs GAAP revenue ratio", ((a["adj_to_gaap"] - f["adj_to_gaap"]) * a["rev_total"]).sum()))
    bridge = pd.DataFrame(rows, columns=["item", "variance_mm"])
    fcst, actual = f["adj_net_revenue"].sum(), a["adj_net_revenue"].sum()
    assert abs(fcst + bridge["variance_mm"].sum() - actual) < 1e-6, "bridge does not tie"
    head = pd.DataFrame([("Forecast (made at Q2 2025)", fcst)], columns=bridge.columns)
    tail = pd.DataFrame([("Actual, Q3 2025 - Q2 2026", actual)], columns=bridge.columns)
    return pd.concat([head, bridge, tail], ignore_index=True)


# ---------- forward view ----------

def forward(panel, approach, errors):
    """Forecast Q3 2026 .. Q2 2027 from the latest quarter with every approach.
    Base = champion. Range = champion's smallest / largest backtest error at each horizon."""
    fc = forecast_at(panel, panel.index.max())
    base = fc[fc["approach"] == approach].pivot(index="quarter", columns="driver", values="forecast")
    h = np.arange(1, H + 1)
    # errors are log(actual / forecast) by horizon
    base["adj_net_revenue_low"] = base["adj_net_revenue"] * np.exp(errors.min().reindex(h).to_numpy())
    base["adj_net_revenue_high"] = base["adj_net_revenue"] * np.exp(errors.max().reindex(h).to_numpy())
    return base, fc


def guidance_check(panel, base, fc_fwd):
    """FY2026 = H1 2026 actual + H2 2026 forecast, vs the guidance in SoFi's Q2 2026 release
    (July 29, 2026): adjusted net revenue $4.75-4.85bn, adjusted EBITDA ~$1.6bn,
    members up at least 30% y/y."""
    h1 = panel.loc["2026-01-01":"2026-06-30"]
    h2 = base.loc["2026-07-01":"2026-12-31"]
    h1_rev, h1_ebitda = h1["adj_net_revenue"].sum(), h1["adj_ebitda"].sum()
    check = pd.DataFrame([
        {"metric": "Adjusted net revenue ($mm)", "guidance": 4800.0,
         "fy2026_base": h1_rev + h2["adj_net_revenue"].sum(),
         "fy2026_low": h1_rev + h2["adj_net_revenue_low"].sum(),
         "fy2026_high": h1_rev + h2["adj_net_revenue_high"].sum()},
        {"metric": "Adjusted EBITDA ($mm)", "guidance": 1600.0,
         "fy2026_base": h1_ebitda + h2["adj_ebitda"].sum()},
        {"metric": "Members, Dec 2026 (mm)", "guidance": panel.loc["2025-12-31", "members"] * 1.30 / 1e6,
         "fy2026_base": h2["members"].iloc[-1] / 1e6},
    ])
    # what guidance implies for H2: quarterly growth g with Q3 = Q2 x (1+g), Q4 = Q2 x (1+g)^2
    h2_rev = 4800.0 - h1_rev
    q2 = h1["adj_net_revenue"].iloc[-1]
    g = np.roots([q2, 3 * q2, 2 * q2 - h2_rev]).real.max()   # q2(1+g) + q2(1+g)^2 = h2_rev
    recent_growth = panel["adj_net_revenue"].pct_change().tail(4)
    implied = pd.DataFrame([
        {"item": "H1 2026 actual adjusted net revenue ($mm)", "value": h1_rev},
        {"item": "H2 2026 revenue implied by guidance midpoint ($mm)", "value": h2_rev},
        {"item": "Implied q/q revenue growth in Q3-Q4 2026", "value": g},
        {"item": "Actual q/q revenue growth, last 4 quarters (min)", "value": recent_growth.min()},
        {"item": "Actual q/q revenue growth, last 4 quarters (max)", "value": recent_growth.max()},
        {"item": "H1 2026 actual adjusted EBITDA margin", "value": h1_ebitda / h1_rev},
        {"item": "H2 2026 EBITDA margin implied by guidance", "value": (1600.0 - h1_ebitda) / h2_rev},
    ])
    by_approach = (fc_fwd[(fc_fwd["driver"] == "adj_net_revenue") & (fc_fwd["quarter"] <= "2026-12-31")]
                   .groupby("approach")["forecast"].sum().add(h1_rev).rename("fy2026_adj_net_revenue")
                   .sort_values().to_frame())
    return check, implied, by_approach


def sensitivity(panel, approach):
    """Next-4-quarter adjusted net revenue and adjusted EBITDA ($mm) when originations and
    lending revenue per $ originated move away from the base case. Other drivers held at base."""
    origin = panel.index.max()
    hist = panel.loc[:origin]
    vm, rm = approach.replace("driver: ", "").split(" + ratios ")
    vol = volume_forecasts(hist, VOLUME)[vm]
    rat = ratio_forecasts(hist, rm)
    corp = np.repeat(hist["rev_corp"].tail(4).mean(), H)
    rows = []
    for o_shift in [-0.20, -0.10, 0.0, 0.10, 0.20]:
        for bp in [-100, -50, 0, 50]:
            v = {**vol, **{o: vol[o] * (1 + o_shift) for o in ORIGINATIONS}}
            r = {**rat, "lend_rev_per_orig": rat["lend_rev_per_orig"] + bp / 1e4}
            p = build_pnl(v, r, corp)
            rows.append({"originations_vs_base": o_shift, "lend_rev_per_orig_bp": bp,
                         "adj_net_revenue": p["adj_net_revenue"].sum(), "adj_ebitda": p["adj_ebitda"].sum()})
    return pd.DataFrame(rows)


# ---------- charts ----------

def charts(panel, fc, champ_driver, champ_top, bridge, base, guide_growth):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    hist = panel["adj_net_revenue"]
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(hist.index, hist, color="black", marker="o", ms=3, label="Actual")
    for a, style in [(champ_driver, "-"), (champ_top, "--"), ("top-down: naive", ":")]:
        x = fc[(fc.origin == HOLDOUT_ORIGIN) & (fc.approach == a) & (fc.driver == "adj_net_revenue")]
        ax.plot(x["quarter"], x["forecast"], style, marker="o", ms=3, label=a)
    ax.axvline(HOLDOUT_ORIGIN, color="grey", lw=0.8)
    ax.set(title="Holdout: forecasts made at Q2 2025 vs actual", ylabel="Adjusted net revenue, $mm")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "holdout_forecast.png", dpi=150)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    vals = bridge["variance_mm"].to_numpy()
    running, bottoms, colors = vals[0], [0], ["#4c72b0"]
    for v in vals[1:-1]:
        bottoms.append(running if v >= 0 else running + v)
        colors.append("#55a868" if v >= 0 else "#c44e52")
        running += v
    bottoms.append(0)
    colors.append("#4c72b0")
    heights = np.abs(vals)
    heights[0], heights[-1] = vals[0], vals[-1]
    ax.bar(range(len(vals)), heights, bottom=bottoms, color=colors)
    for i, v in enumerate(vals):
        ax.text(i, bottoms[i] + heights[i] + 15, f"{v:,.0f}", ha="center", fontsize=8)
    ax.set_xticks(range(len(vals)), bridge["item"], rotation=30, ha="right", fontsize=8)
    ax.set_ylim(vals[-1] * 0.9, vals[0] * 1.03)
    ax.set(title="Holdout variance bridge: adjusted net revenue, Q3 2025 - Q2 2026 ($mm)")
    fig.tight_layout()
    fig.savefig(OUT / "variance_bridge.png", dpi=150)

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(hist.index, hist, color="black", marker="o", ms=3, label="Actual")
    ax.plot(base.index, base["adj_net_revenue"], marker="o", ms=3, label="Base forecast (driver model)")
    ax.fill_between(base.index, base["adj_net_revenue_low"], base["adj_net_revenue_high"], alpha=0.2,
                    label="Range of backtest errors")
    last = hist.iloc[-1] * (1 + guide_growth) ** np.arange(1, 3)
    ax.plot(base.index[:2], last, "s--", color="darkred", ms=4,
            label=f"Path implied by FY2026 guidance midpoint ({guide_growth:.1%} q/q)")
    ax.set(title="SoFi adjusted net revenue: forecast Q3 2026 - Q2 2027", ylabel="$mm")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "forward_forecast.png", dpi=150)
    plt.close("all")


# ---------- main ----------

def main():
    panel = load_panel()
    fc = attach_actuals(pd.concat([forecast_at(panel, o) for o in ALL_ORIGINS]), panel)

    sel = score(fc, SELECTION_ORIGINS)
    hold = score(fc, [HOLDOUT_ORIGIN])
    allo = score(fc, ALL_ORIGINS)
    key = "adj_net_revenue"
    champ_driver = sel.loc[sel.index.str.startswith("driver"), key].idxmin()
    champ_top = sel.loc[sel.index.str.startswith("top-down"), key].idxmin()
    summary = pd.DataFrame({"selection_mape": sel[key], "holdout_mape": hold[key], "all_origins_mape": allo[key]})
    print("MAPE on adjusted net revenue, %:\n", summary.round(1).sort_values("selection_mape").to_string())
    print(f"\nchampions picked on the selection window: {champ_driver} | {champ_top}")

    fh = fc[(fc.origin == HOLDOUT_ORIGIN) & (fc.approach == champ_driver)]
    bridge = variance_bridge(panel, fh)
    print("\n", bridge.round(1).to_string(index=False))
    holdout_totals = (fc[(fc.origin == HOLDOUT_ORIGIN) & (fc.driver == key)]
                      .groupby("approach")[["forecast", "actual"]].sum()
                      .assign(miss_pct=lambda d: (d["forecast"] / d["actual"] - 1) * 100))
    print("\nholdout 4-quarter total, $mm:\n", holdout_totals.round(1).to_string())

    errors = (fc[(fc.approach == champ_driver) & (fc.driver == key)]
              .assign(log_err=lambda d: np.log(d["actual"] / d["forecast"]))
              .groupby("h")["log_err"])
    base, fc_fwd = forward(panel, champ_driver, errors)
    guide, implied, fy_by_approach = guidance_check(panel, base, fc_fwd)
    sens = sensitivity(panel, champ_driver)
    print("\nforward base:\n", base[["orig_total", "members", "rev_lending", "rev_fs", "rev_tp",
                                     "adj_net_revenue", "adj_net_revenue_low", "adj_net_revenue_high",
                                     "adj_ebitda"]].round(1).to_string())
    print("\n", guide.round(2).to_string(index=False))
    print("\n", implied.round(4).to_string(index=False))
    print("\n", fy_by_approach.round(1).to_string())

    guide_growth = implied.set_index("item").loc["Implied q/q revenue growth in Q3-Q4 2026", "value"]
    charts(panel, fc, champ_driver, champ_top, bridge, base, guide_growth)
    fc.to_csv(OUT / "backtest_forecasts.csv", index=False)
    with pd.ExcelWriter(OUT / "sofi_forecast.xlsx") as xl:
        panel.to_excel(xl, sheet_name="Driver panel")
        summary.to_excel(xl, sheet_name="Backtest MAPE (adj rev)")
        sel.to_excel(xl, sheet_name="MAPE selection window")
        hold.to_excel(xl, sheet_name="MAPE holdout")
        holdout_totals.to_excel(xl, sheet_name="Holdout totals")
        fh.pivot(index="driver", columns="quarter", values=["forecast", "actual"]).to_excel(
            xl, sheet_name="Holdout detail")
        bridge.to_excel(xl, sheet_name="Variance bridge", index=False)
        base.T.to_excel(xl, sheet_name="Forecast Q3 26 - Q2 27")
        guide.to_excel(xl, sheet_name="FY2026 vs guidance", index=False)
        implied.to_excel(xl, sheet_name="What guidance implies", index=False)
        fy_by_approach.to_excel(xl, sheet_name="FY2026 by method")
        sens.pivot(index="originations_vs_base", columns="lend_rev_per_orig_bp",
                   values="adj_ebitda").to_excel(xl, sheet_name="Sensitivity adj EBITDA")
        sens.pivot(index="originations_vs_base", columns="lend_rev_per_orig_bp",
                   values="adj_net_revenue").to_excel(xl, sheet_name="Sensitivity adj revenue")


if __name__ == "__main__":
    main()
