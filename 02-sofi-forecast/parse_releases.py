"""Step 1: parse SoFi's quarterly earnings releases (SEC 8-K Item 2.02) into a tidy KPI table.

Every release has tables whose columns are quarters ("June 30, 2026" or
"Three Months Ended June 30," + "2026"). This turns each table into rows of
(section, metric, quarter_end, value), then keeps the value from the newest
release for each (section, metric, quarter) so later restatements win.
"""
import io
import re
import warnings
from pathlib import Path

import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(__file__).parent
RELEASES = ROOT / "data" / "earnings_releases"

MONTHS = "January|February|March|April|May|June|July|August|September|October|November|December"
FULL_DATE = re.compile(rf"({MONTHS})\s+(\d{{1,2}}),?\s+(\d{{4}})")
MONTH_DAY = re.compile(rf"({MONTHS})\s+(\d{{1,2}})")
YEAR = re.compile(r"^(19|20)\d{2}$")
NUMBER = re.compile(r"^\(?-?[\d,]*\.?\d+\)?$")


def to_number(cell):
    s = str(cell).replace("$", "").replace("%", "").replace(",", "").strip()
    if not s or s in {"—", "–", "-"} or not NUMBER.match(s):
        return None
    neg = s.startswith("(") and s.endswith(")")
    v = float(s.strip("()"))
    return -v if neg else v


def clean_label(text):
    text = re.sub(r"\(\d+\)", "", str(text))            # footnote markers like (1)
    text = text.replace(" ", " ").replace("\xa0", " ")
    return re.sub(r"\s+", " ", text).strip(" :")


def column_dates(t):
    """Map each column to a quarter-end date, using only three-month / point-in-time headers."""
    dates = {}
    header = t.head(6)
    for c in t.columns[1:]:
        cells = [str(x) for x in header[c] if pd.notna(x)]
        joined = " ".join(cells)
        if re.search(r"Six Months|Nine Months|Twelve Months|Year Ended|% Change|Change", joined, re.I):
            continue
        m = FULL_DATE.search(joined)
        if m:
            dates[c] = pd.Timestamp(f"{m.group(1)} {m.group(2)} {m.group(3)}")
            continue
        md, yr = MONTH_DAY.search(joined), [x for x in cells if YEAR.match(x.strip())]
        if md and yr:
            dates[c] = pd.Timestamp(f"{md.group(1)} {md.group(2)} {yr[-1].strip()}")
    return dates


def parse_table(t, source):
    t = t.dropna(how="all", axis=1).dropna(how="all").reset_index(drop=True)
    t.columns = range(t.shape[1])  # read_html can return duplicate / multi-level headers
    if t.shape[1] < 3:
        return []
    dates = column_dates(t)
    if not dates:
        return []
    title = clean_label(t.iloc[0, 0]) if str(t.iloc[0, 0]) != "nan" else ""
    section, rows = title, []
    for _, r in t.iterrows():
        label = clean_label(r.iloc[0])
        if not label or label == "nan":
            continue
        vals = {}
        for c, d in dates.items():
            v = to_number(r[c])
            if v is not None and d not in vals:
                vals[d] = v
        if not vals:
            section = label if len(label) < 60 else section
            continue
        rows += [{"table": title, "section": section, "metric": label, "quarter_end": d,
                  "value": v, "source": source} for d, v in vals.items()]
    return rows


def main():
    out = []
    for f in sorted(RELEASES.glob("*.htm")):
        html = f.read_text(encoding="utf-8", errors="ignore")
        for t in pd.read_html(io.StringIO(html)):
            out += parse_table(t, f.name)
    df = pd.DataFrame(out)
    df["filed"] = pd.to_datetime(df["source"].str[:10])
    # newest release wins for each metric/quarter (captures restatements)
    tidy = (df.sort_values("filed")
              .drop_duplicates(["table", "section", "metric", "quarter_end"], keep="last"))
    tidy.to_csv(ROOT / "data" / "release_facts.csv", index=False)
    print(f"{len(df):,} raw facts -> {len(tidy):,} unique; "
          f"{tidy['quarter_end'].nunique()} quarters, {tidy['metric'].nunique()} metrics")


if __name__ == "__main__":
    main()
