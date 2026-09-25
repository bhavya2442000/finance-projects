"""Step 2: build a clean quarterly driver panel (one row per quarter) from release_facts.csv.

SoFi renamed some line items over the years (e.g. "Contribution profit" became
"Contribution profit – Lending", "Net income (loss)" became "Net income"). Each driver
lists its labels in priority order; for each quarter the first label with a value wins,
and any quarter where two labels disagree by more than 0.5% is reported.
Dollar amounts are converted from $ thousands to $ millions.
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).parent
TREND = ""  # the untitled 8-quarter trend tables at the back of each release
LEND_TBL = "Lending – Originations and Average Balances"
ORIG = "Origination volume ($ in thousands, during period)"

# driver -> list of (table, section, metric) candidates, highest priority first
SOURCES = {
    "members":           [(TREND, "", "Members")],
    "products_total":    [(TREND, "", "Total Products")],
    "products_lending":  [(TREND, "", "Total Products — Lending segment")],
    "products_fs":       [(TREND, "", "Total Products — Financial Services segment")],
    "tp_accounts":       [(TREND, "", "Total Accounts — Technology Platform segment")],
    "orig_personal":     [(LEND_TBL, ORIG, "Personal loans")],
    "orig_student":      [(LEND_TBL, ORIG, "Student loans")],
    "orig_home":         [(LEND_TBL, ORIG, "Home loans")],
    "deposits":          [(TREND, "Deposits", "Total deposits")],
    "rev_total":         [(TREND, "Consolidated", "Total net revenue")],
    "rev_lending":       [(TREND, "Lending", "Total net revenue")],
    "rev_tp":            [(TREND, "Technology Platform", "Total net revenue")],
    "rev_fs":            [(TREND, "Financial Services", "Total net revenue")],
    "rev_corp":          [(TREND, "Corporate/Other", "Total net revenue (loss)")],
    "adj_net_revenue":   [(TREND, "Consolidated", "Adjusted net revenue")],
    "adj_ebitda":        [(TREND, "Consolidated", "Adjusted EBITDA")],
    "nii_lending":       [(TREND, "Lending", "Net interest income")],
    "nii_fs":            [(TREND, "Financial Services", "Net interest income")],
    "cp_lending":        [(TREND, "Lending", "Contribution profit – Lending"),
                          (TREND, "Lending", "Contribution profit"),
                          (TREND, "Lending", "Contribution profit (loss)")],
    "cp_tp":             [(TREND, "Technology Platform", "Contribution profit – Technology Platform"),
                          (TREND, "Technology Platform", "Contribution profit")],
    "cp_fs":             [(TREND, "Financial Services", "Contribution profit – Financial Services"),
                          (TREND, "Financial Services", "Contribution profit (loss) – Financial Services"),
                          (TREND, "Financial Services", "Contribution profit (loss)"),
                          (TREND, "Financial Services", "Contribution loss")],
    "net_income":        [(TREND, "Consolidated", "Net income"),
                          (TREND, "Consolidated", "Net income (loss)")],
}
COUNTS = {"members", "products_total", "products_lending", "products_fs", "tp_accounts"}


def stitch(facts, driver, candidates):
    series, conflicts = [], []
    for table, section, metric in candidates:
        s = facts[(facts.table == table) & (facts.section == section) & (facts.metric == metric)]
        series.append(s.set_index("quarter_end")["value"])
    out = series[0].copy()
    for s in series[1:]:
        both = out.index.intersection(s.index)
        diff = (out[both] - s[both]).abs() / out[both].abs().clip(lower=1)
        conflicts += [(driver, q) for q in diff[diff > 0.005].index]
        out = out.combine_first(s)
    return out.rename(driver), conflicts


def xbrl_quarterly(tag):
    """Quarterly values ($mm) of one us-gaap tag from SEC companyfacts; newest filing wins.
    Flow items keep only 3-month periods (SEC reports Q4 only inside the full year)."""
    facts = json.loads((ROOT / "data" / "sofi_companyfacts.json").read_text())["facts"]["us-gaap"]
    d = pd.DataFrame(facts[tag]["units"]["USD"])
    d["end"] = pd.to_datetime(d["end"])
    if "start" in d:
        d = d[(d["end"] - pd.to_datetime(d["start"])).dt.days < 100]
    d = d.sort_values("filed").drop_duplicates("end", keep="last")
    return d.set_index("end")["val"] / 1e6


def main():
    facts = pd.read_csv(ROOT / "data" / "release_facts.csv", parse_dates=["quarter_end"],
                        keep_default_na=False)
    cols, conflicts = [], []
    for driver, cands in SOURCES.items():
        s, c = stitch(facts, driver, cands)
        cols.append(s)
        conflicts += c
    panel = pd.concat(cols, axis=1, sort=True).sort_index()
    panel.index.name = "quarter_end"

    money = [c for c in panel.columns if c not in COUNTS]
    panel[money] = panel[money] / 1000  # $ thousands -> $ millions
    panel["orig_total"] = panel[["orig_personal", "orig_student", "orig_home"]].sum(axis=1, min_count=3)
    panel["dae_lending"] = panel["rev_lending"] - panel["cp_lending"]  # directly attributable costs incl. provision

    # sanity checks
    seg = panel[["rev_lending", "rev_tp", "rev_fs", "rev_corp"]].sum(axis=1, min_count=4)
    gap = (seg - panel["rev_total"]).abs().dropna()
    print(f"segment revenue sums to consolidated: max gap ${gap.max():.2f}mm over {len(gap)} quarters")
    print(f"label conflicts > 0.5%: {conflicts or 'none'}")
    for tag, col in [("NetIncomeLoss", "net_income"), ("Deposits", "deposits")]:
        both = pd.concat([xbrl_quarterly(tag), panel[col]], axis=1, sort=True).dropna()
        diff = (both.iloc[:, 1] / both.iloc[:, 0] - 1).abs().max()
        print(f"{col} vs SEC XBRL {tag}: {len(both)} quarters, max difference {diff:.3%}")
    print(panel.notna().sum().to_string())

    panel.to_csv(ROOT / "data" / "driver_panel.csv")
    print(f"\n{len(panel)} quarters, {panel.index.min():%Y-%m} to {panel.index.max():%Y-%m}")


if __name__ == "__main__":
    main()
