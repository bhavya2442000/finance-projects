"""Step 0: download SoFi's SEC data.

- companyfacts (XBRL financial data) and submissions (filing list) JSON
- the earnings-release exhibit (EX-99.1) of every 8-K with Item 2.02 (results of operations)

SEC requires a contact in the User-Agent header. Set it first, e.g.
    export SEC_USER_AGENT="Your Name you@example.com"
"""
import json
import os
import sys
import time
import urllib.request
from pathlib import Path

CIK = "0001818874"  # SoFi Technologies
ROOT = Path(__file__).parent
DATA = ROOT / "data"
RELEASES = DATA / "earnings_releases"


def get(url, ua):
    time.sleep(0.2)  # stay well under SEC's 10 requests/second limit
    req = urllib.request.Request(url, headers={"User-Agent": ua})
    with urllib.request.urlopen(req, timeout=60) as r:
        return r.read()


def main():
    ua = os.environ.get("SEC_USER_AGENT")
    if not ua:
        sys.exit("Set SEC_USER_AGENT to 'Your Name your@email' (SEC requires a contact).")
    RELEASES.mkdir(parents=True, exist_ok=True)

    for name, url in [("sofi_companyfacts.json", f"https://data.sec.gov/api/xbrl/companyfacts/CIK{CIK}.json"),
                      ("sofi_submissions.json", f"https://data.sec.gov/submissions/CIK{CIK}.json")]:
        (DATA / name).write_bytes(get(url, ua))

    recent = json.loads((DATA / "sofi_submissions.json").read_text())["filings"]["recent"]
    for form, items, filed, acc in zip(recent["form"], recent["items"], recent["filingDate"],
                                       recent["accessionNumber"]):
        if form != "8-K" or "2.02" not in items:
            continue
        folder = f"https://www.sec.gov/Archives/edgar/data/{int(CIK)}/{acc.replace('-', '')}"
        index = json.loads(get(f"{folder}/index.json", ua))
        files = [f["name"] for f in index["directory"]["item"]]
        release = [f for f in files if f.endswith("earningsrelease.htm")]
        if not release:
            print(f"{filed}: no earnings-release exhibit found, skipped")
            continue
        out = RELEASES / f"{filed}_{release[0]}"
        if not out.exists():
            out.write_bytes(get(f"{folder}/{release[0]}", ua))
        print(f"{filed}: {out.name}")


if __name__ == "__main__":
    main()
