#!/usr/bin/env python3
"""Point-in-time map from USAspending recipient strings to tradeable tickers.

    python3 python/contract_entities.py          # coverage report

WHY THIS FILE IS HAND-WRITTEN AND NOT FUZZY-MATCHED

Three separate traps make automated name matching wrong here, and each one
would corrupt the study silently rather than loudly.

1. RECIPIENT NAMES ARE RETROACTIVELY MODERNIZED. "RTX CORPORATION" appears on
   2015 rows. RTX did not exist until the 2020 Raytheon/United Technologies
   merger and was not named RTX until 2023. Checking the award numbers settles
   which company it actually was: the 2015 "RTX" awards are N00019 (NAVAIR) and
   FA8611 (Air Force) aircraft-engine contracts, which is Pratt & Whitney, so
   those rows are UNITED TECHNOLOGIES. The 2015 "RAYTHEON COMPANY" rows are
   HQ0276 (Missile Defense Agency) and N00024 (Naval Sea Systems), which is
   missiles, so those are the real Raytheon.

2. PRICE SERIES FOLLOW THE ACQUIRER, NOT THE TARGET. Verified directly: RTX
   closed at $81.94 in mid-2019, which is neither UTX (~$130) nor RTN (~$175).
   It is UTX scaled down by the Carrier/Otis spinoff ratio, and $49.93 on
   2020-04-03 is merger-completion day. LHX at $144.54 in June 2018 matches
   Harris Corp exactly. So RTX bars ARE United Technologies and LHX bars ARE
   Harris, which makes UTC and Harris rows priceable and makes Raytheon and
   L3 Technologies rows unpriceable.

3. SURVIVORSHIP RUNS ONE WAY. Every contractor acquired during the window is
   absent from the price database, because that database was built from CURRENT
   index constituents: RTN, UTX, COL, HRS, AJRD, ORB, ESL. Dropping them removes
   exactly the companies that became acquisition targets, which is not random
   with respect to performance. This is a limitation of the sample, not
   something a better join fixes, and the coverage report below states how much
   of the contract dollar volume it costs.

Each entry is (ticker, valid_from, valid_to). A row is priceable only if its
action date falls inside the window, so a subsidiary that changed parents
mid-sample maps to whichever parent held it at the time.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
FAR = "2099-12-31"

# Matched on the UPPERCASED recipient string containing the key. Ordered: the
# first key that matches wins, so put specific subsidiaries before their parents.
MAP: list[tuple[str, str, str, str]] = [
    # key fragment                      ticker  valid_from   valid_to
    # Subsidiaries that bill the government under their own name. These were
    # added AFTER the first run returned a null, which is worth stating: the
    # decision depends only on who owns whom, never on a return, and widening
    # coverage on a null can only make the test more powerful. If adding them
    # had flipped the sign, that would itself have been a reason for suspicion.
    ("ELECTRIC BOAT",                    "GD",  "2014-01-01", FAR),
    ("BATH IRON WORKS",                  "GD",  "2014-01-01", FAR),
    ("OPTUM",                            "UNH", "2014-01-01", FAR),
    ("PFIZER",                           "PFE", "2014-01-01", FAR),
    ("SIKORSKY",                         "RTX", "2014-01-01", "2015-11-05"),
    ("SIKORSKY",                         "LMT", "2015-11-06", FAR),
    ("LOCKHEED MARTIN",                  "LMT", "2014-01-01", FAR),
    ("BOEING",                           "BA",  "2014-01-01", FAR),
    ("NORTHROP GRUMMAN",                 "NOC", "2014-01-01", FAR),
    ("GENERAL DYNAMICS",                 "GD",  "2014-01-01", FAR),
    ("HUNTINGTON INGALLS",               "HII", "2014-01-01", FAR),
    # "RTX CORPORATION" on pre-merger rows is United Technologies, and the RTX
    # price series is UTC's, so the ticker is right for the whole window.
    ("RTX CORPORATION",                  "RTX", "2014-01-01", FAR),
    ("UNITED TECHNOLOGIES",              "RTX", "2014-01-01", FAR),
    ("PRATT & WHITNEY",                  "RTX", "2014-01-01", FAR),
    ("PRATT AND WHITNEY",                "RTX", "2014-01-01", FAR),
    # LHX bars are Harris Corp, so Harris rows are priceable across the window.
    ("HARRIS CORP",                      "LHX", "2014-01-01", FAR),
    ("L3HARRIS",                         "LHX", "2019-06-29", FAR),
    ("LEIDOS",                           "LDOS", "2014-01-01", FAR),
    ("BOOZ ALLEN",                       "BAH", "2014-01-01", FAR),
    ("CACI",                             "CACI", "2014-01-01", FAR),
    ("SCIENCE APPLICATIONS INTERNATIONAL", "SAIC", "2014-01-01", FAR),
    ("TEXTRON",                          "TXT", "2014-01-01", FAR),
    ("HONEYWELL",                        "HON", "2014-01-01", FAR),
    ("OSHKOSH",                          "OSK", "2014-01-01", FAR),
    ("CURTISS",                          "CW",  "2014-01-01", FAR),
    ("AEROVIRONMENT",                    "AVAV", "2014-01-01", FAR),
    ("KRATOS",                           "KTOS", "2014-01-01", FAR),
    ("MERCURY SYSTEMS",                  "MRCY", "2014-01-01", FAR),
    ("TRANSDIGM",                        "TDG", "2014-01-01", FAR),
    ("MCKESSON",                         "MCK", "2014-01-01", FAR),
    ("HUMANA",                           "HUM", "2014-01-01", FAR),
    ("UNITEDHEALTH",                     "UNH", "2014-01-01", FAR),
    ("CENTENE",                          "CNC", "2014-01-01", FAR),
    ("GENERAL ELECTRIC",                 "GE",  "2014-01-01", FAR),
    ("IBM",                              "IBM", "2014-01-01", FAR),
    ("MICROSOFT",                        "MSFT", "2014-01-01", FAR),
    ("AMAZON",                           "AMZN", "2014-01-01", FAR),
    ("ACCENTURE",                        "ACN", "2014-01-01", FAR),
    ("DELOITTE",                         None,  "2014-01-01", FAR),   # private
]

# Recipients deliberately marked unpriceable, with the reason. Being explicit
# beats letting them fall through to "unmatched", because the reason is the
# thing a reader needs in order to judge the survivorship cost.
UNPRICEABLE: dict[str, str] = {
    # SpaceX IPO'd on 2026-06-12 under SPCX, so it is NOT private, but the price
    # database holds no SPCX bars and every contract action here predates the
    # listing. Checked rather than assumed.
    "SPACE EXPLORATION": "public since 2026-06-12 (SPCX) but all actions predate the IPO and no bars exist",
    "FLUOR MARINE":      "government-owned contractor-operated lab; operator fee, not revenue",
    "BAE SYSTEMS":       "BAE Systems plc is LSE-listed; no US price series here",
    "GENERAL ATOMICS":   "private",
    "TRIWEST":           "private",
    "HENSEL PHELPS":     "private",
    "FISHER SAND":       "private",
    "CONSOLIDATED NUCLEAR SECURITY": "FFRDC consortium LLC, not a listed entity",
    "TRIAD NATIONAL":    "FFRDC consortium LLC, not a listed entity",
    "SAVANNAH RIVER":    "FFRDC consortium LLC, not a listed entity",
    "RAYTHEON COMPANY":  "RTN delisted into the 2020 RTX merger; no price series",
    "RAYTHEON MISSILE":  "RTN delisted; no price series",
    "ROCKWELL COLLINS":  "COL acquired by UTC 2018; no price series",
    "ORBITAL":           "acquired by Northrop 2018; no price series",
    "AEROJET":           "acquired by L3Harris 2023; no price series",
    "L3 TECHNOLOGIES":   "merged into L3Harris 2019; LHX bars are Harris, not L3",
    "ESTERLINE":         "acquired by TransDigm 2019; no price series",
    "HEALTH NET":        "acquired by Centene 2016; no price series",
    "UNITED LAUNCH":     "Boeing/Lockheed joint venture, never separately listed",
    "BECHTEL":           "private",
    "AEROSPACE CORPORATION": "non-profit FFRDC",
    "NATIONAL TECHNOLOGY & ENGINEERING": "Sandia FFRDC; operator fee, not revenue",
    "BATTELLE":          "non-profit FFRDC operator",
    "JACOBS":            "ticker changed and business split mid-sample; excluded",
    "UNIVERSITY":        "not a listed company",
    "TRUSTEES":          "not a listed company",
    "INSTITUTE":         "not a listed company",
}


def resolve(recipient: str, action_date: str) -> tuple[str | None, str]:
    """Return (ticker, reason). ticker is None when the row is not priceable."""
    s = (recipient or "").upper()
    for frag, why in UNPRICEABLE.items():
        if frag in s:
            return None, why
    for frag, tkr, lo, hi in MAP:
        if frag in s:
            if tkr is None:
                return None, "private company"
            if lo <= action_date <= hi:
                return tkr, "mapped"
            return None, f"{tkr} outside its valid window {lo}..{hi}"
    return None, "unmatched"


def main() -> int:
    path = ROOT / "data" / "contracts.parquet"
    if not path.exists():
        print(f"missing {path}; run python/ingest_contracts.py first")
        return 1
    d = pd.read_parquet(path)
    res = [resolve(r, a) for r, a in zip(d["recipient"], d["action_date"])]
    d["ticker"] = [t for t, _ in res]
    d["why"] = [w for _, w in res]

    tot = d["amount_usd"].sum()
    ok = d[d["ticker"].notna()]
    print(f"{len(d):,} contract actions, ${tot / 1e9:,.1f}bn obligated")
    print(f"priceable: {len(ok):,} actions ({len(ok) / len(d):.1%}), "
          f"${ok['amount_usd'].sum() / 1e9:,.1f}bn "
          f"({ok['amount_usd'].sum() / tot:.1%} of dollars)")
    print(f"distinct tickers: {ok['ticker'].nunique()}")

    print(f"\nwhy rows were dropped, by dollar volume:")
    g = (d[d["ticker"].isna()].groupby("why")["amount_usd"]
         .agg(["size", "sum"]).sort_values("sum", ascending=False))
    for why, r in g.head(12).iterrows():
        print(f"  ${r['sum'] / 1e9:>7,.1f}bn  {int(r['size']):>6,} rows  {why[:62]}")

    print(f"\nlargest UNMATCHED recipients, which are the next ones worth resolving:")
    u = (d[d["why"] == "unmatched"].groupby("recipient")["amount_usd"]
         .agg(["size", "sum"]).sort_values("sum", ascending=False))
    for nm, r in u.head(15).iterrows():
        print(f"  ${r['sum'] / 1e9:>7,.2f}bn  {int(r['size']):>5,}  {nm[:58]}")

    out = ROOT / "data" / "contracts_mapped.parquet"
    d.to_parquet(out, index=False)
    print(f"\nwrote {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
