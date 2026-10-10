#!/usr/bin/env python3
"""Federal contract actions from USAspending, at TRANSACTION level.

    python3 python/ingest_contracts.py
    python3 python/ingest_contracts.py --start 2014-01 --end 2026-09 --min-usd 10e6

WHY TRANSACTION LEVEL AND NOT AWARD LEVEL

The award search endpoint returns `Award Amount`, which is the CUMULATIVE total
obligated over an award's entire life. A 2015 query returned a Lockheed award
with a 1993 start date and a $48bn total. Keying an event study on that number
would import every future modification into a 2015 decision. The transaction
endpoint gives one row per contract action, each with its own `Action Date` and
its own obligation, which is point-in-time by construction. `Mod` separates a
base award ("0") from a later modification ("P00103").

PAGINATION, AND THE BUG THIS IS WRITTEN TO AVOID

A previous ingest in this project walked an `offset` parameter forward until the
API stopped returning rows. The API capped offsets, so it stopped early and
silently at 2021-03, and that truncation got written into the README as a fact
about the data source. The sample was half its true size and every robustness
check ran inside the truncated window, so none of them could see it.

So this pulls MONTH BY MONTH, sorted by amount descending, and stops when the
rows drop below the threshold. That keeps paging shallow (about 8 pages for a
$10M floor) and far from any offset cap. More importantly each month ASSERTS
that it actually crossed below the threshold. A month that ends while still
above it is incomplete by definition, and is reported as a failure rather than
accepted as "no more data".

WHAT IS NOT FIXED HERE, AND MUST BE HANDLED DOWNSTREAM

Recipient names are retroactively MODERNIZED. "RTX CORPORATION" appears on 2015
rows although that entity did not exist until the 2020 Raytheon/United
Technologies merger and was not named RTX until 2023. The same company also
appears as both "LOCKHEED MARTIN CORPORATION" and "LOCKHEED MARTIN CORP". Any
name-to-ticker map therefore encodes TODAY's corporate structure, and attaching
2015 prices to it would attach the wrong company's history. Entity resolution
has to be point-in-time and hand-verified against merger dates; it is
deliberately not done in this file.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "contracts.parquet"
API = "https://api.usaspending.gov/api/v2/search/spending_by_transaction/"
# The codes do NOT mean what their letters suggest, which was checked against
# the API rather than assumed: A is a BPA call, B a purchase order, C a delivery
# order, D a definitive contract. The distinction matters economically, because a
# delivery order merely draws on an existing vehicle while a definitive contract
# is new money, so the type is kept on every row rather than collapsed.
AWARD_TYPES = ["A", "B", "C", "D"]
FIELDS = ["Transaction Amount", "Recipient Name", "Action Date", "Award ID",
          "Mod", "Awarding Agency", "Award Type"]
MAX_PAGES = 60          # a month needs ~8 at a $10M floor; this is a guard, not a target


def post(payload: dict, tries: int = 5, timeout: int = 120) -> dict:
    for a in range(tries):
        req = urllib.request.Request(
            API, data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json",
                     "User-Agent": "edge-audit-research/1.0"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return json.loads(r.read())
        except Exception as e:
            if a == tries - 1:
                raise
            wait = 3.0 * 2 ** a
            print(f"      retry {a + 1}/{tries - 1} in {wait:.0f}s ({type(e).__name__})")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def month_edges(start: str, end: str) -> list[tuple[str, str]]:
    s = pd.Period(start, "M")
    e = pd.Period(end, "M")
    out = []
    while s <= e:
        out.append((s.start_time.date().isoformat(), s.end_time.date().isoformat()))
        s += 1
    return out


def pull_month(lo: str, hi: str, min_usd: float) -> tuple[list[dict], bool]:
    """Rows for one month above min_usd, plus whether the month is COMPLETE.

    Complete means paging ran until the amounts fell below the threshold, which
    proves nothing above it was left behind.
    """
    rows: list[dict] = []
    crossed = False
    for page in range(1, MAX_PAGES + 1):
        r = post({
            "filters": {"award_type_codes": AWARD_TYPES,
                        "time_period": [{"start_date": lo, "end_date": hi}]},
            "fields": FIELDS, "page": page, "limit": 100,
            "sort": "Transaction Amount", "order": "desc",
        })
        res = r.get("results", [])
        if not res:
            crossed = True       # ran out of rows entirely: nothing was skipped
            break
        rows += res
        if res[-1]["Transaction Amount"] < min_usd:
            crossed = True
            break
        if not r.get("page_metadata", {}).get("hasNext"):
            crossed = True
            break
        time.sleep(0.35)
    return [x for x in rows if x["Transaction Amount"] >= min_usd], crossed


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--start", default="2014-01")
    ap.add_argument("--end", default="2026-09")
    ap.add_argument("--min-usd", type=float, default=10e6)
    args = ap.parse_args()

    months = month_edges(args.start, args.end)
    print(f"USAspending contract actions >= ${args.min_usd / 1e6:,.0f}M, "
          f"{len(months)} months from {args.start} to {args.end}\n")

    all_rows, incomplete = [], []
    for i, (lo, hi) in enumerate(months, 1):
        try:
            rows, ok = pull_month(lo, hi, args.min_usd)
        except Exception as e:
            print(f"  {lo[:7]}  FAILED: {type(e).__name__} {str(e)[:90]}")
            incomplete.append(lo[:7])
            continue
        if not ok:
            incomplete.append(lo[:7])
        all_rows += rows
        if i % 12 == 0 or i == len(months):
            print(f"  {lo[:7]}  {len(rows):>4} rows this month, "
                  f"{len(all_rows):>7,} total{'' if ok else '   INCOMPLETE'}")

    d = pd.DataFrame(all_rows)
    if d.empty:
        print("no rows")
        return 1
    d = d.rename(columns={"Transaction Amount": "amount_usd",
                          "Recipient Name": "recipient",
                          "Action Date": "action_date",
                          "Award ID": "award_id", "Mod": "mod",
                          "Awarding Agency": "agency"})
    d["action_date"] = pd.to_datetime(d["action_date"]).dt.date.astype(str)
    # The same action can come back on more than one page near a page boundary.
    before = len(d)
    key = ["award_id", "action_date", "amount_usd", "recipient", "mod"]
    d = d.drop_duplicates(subset=key).sort_values(["action_date", "amount_usd"])
    d["is_base"] = d["mod"].astype(str).str.strip().isin(["0", "", "None", "nan"])

    # SOURCE DATA ERRORS EXIST AND THEY ARE ENORMOUS. A 2015 type-D row credits
    # HENSEL PHELPS CONSTRUCTION CO with $92.49bn across six actions, against
    # roughly $440bn of total federal contract obligations that year, for a
    # private builder. KEPA-TCI JV LLC shows $10.68bn in a single action. These
    # are mis-keyed obligations in the feed, not cash flows. They are FLAGGED
    # rather than deleted, so the count is auditable and a later stage can
    # decide; silently dropping rows is how a sample stops matching its own
    # description.
    d["implausible"] = d["amount_usd"] > 5e9

    OUT.parent.mkdir(exist_ok=True)
    d.to_parquet(OUT, index=False)

    print(f"\n{len(d):,} unique actions ({before - len(d):,} duplicate rows dropped)")
    print(f"  {d['action_date'].min()} to {d['action_date'].max()}")
    print(f"  {d['recipient'].nunique():,} distinct recipient strings")
    print(f"  base awards {d['is_base'].sum():,}, modifications {(~d['is_base']).sum():,}")
    print(f"  total obligated ${d['amount_usd'].sum() / 1e9:,.1f}bn")
    if "Award Type" in d.columns:
        d = d.rename(columns={"Award Type": "award_type"})
        d.to_parquet(OUT, index=False)
    if "award_type" in d.columns:
        print(f"\n  by award type:")
        for t, n in d["award_type"].value_counts().head(10).items():
            print(f"    {str(t)[:28]:<30}{n:>7,}")
    n_imp = int(d["implausible"].sum())
    print(f"\n  flagged implausible (> $5bn single action): {n_imp:,}")
    if n_imp:
        top = d[d["implausible"]].nlargest(5, "amount_usd")
        for _, r in top.iterrows():
            print(f"    ${r['amount_usd'] / 1e9:>7,.1f}bn  {r['action_date']}  "
                  f"{str(r['recipient'])[:40]}")
    print(f"\nrows per year:")
    yr = d.groupby(d["action_date"].str[:4]).size()
    for y, n in yr.items():
        print(f"  {y}  {n:>7,}")
    if incomplete:
        print(f"\n*** {len(incomplete)} INCOMPLETE MONTHS: {incomplete}")
        print("These were NOT proven to page past the threshold. Do not treat this")
        print("sample as complete until they are refetched.")
        return 2
    print(f"\nevery month paged past the threshold; sample is complete")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
