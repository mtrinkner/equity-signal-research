#!/usr/bin/env python3
"""Do federal contract awards predict returns? The declared test, run once.

    python3 python/contract_study.py

Registered as trial 26 with 32 variants BEFORE any return was computed. The
hypothesis and both shape predictions were written down first and are repeated
here verbatim, so a reader can check the result against what was promised rather
than against what was found.

HYPOTHESIS. A base contract action that is large RELATIVE TO MARKET CAP predicts
positive abnormal returns over 5 to 21 sessions, because it is new cash-flow
information that is publicly announced but thinly covered.

SHAPE PREDICTION 1, THE PRE-EVENT WINDOW. If this is underreaction to NEW
information, cumulative abnormal returns must be FLAT before the action date and
positive after. A pre-event run-up means the award was anticipated, since
solicitations and competitions are public, and the result is selection rather
than drift. This is the exact test that exposed the index-addition confound: the
mechanism there claimed forced buying on the effective date, and the run-up
turned out to stretch back 250 sessions.

SHAPE PREDICTION 2, MATERIALITY SCALING. The effect must grow with award size
over market cap. A $50M award to Lockheed is immaterial; the same award to a
$500M company is not. A signal that does not scale with materiality is not the
mechanism, whatever its t-statistic.

OVERLAP IS HANDLED BY AGGREGATION, NOT IGNORED. Lockheed alone drew 77 actions
above $50M in 2015. Those are one company's news flow, not 77 independent bets,
and stacking them as separate events would inflate every t-statistic. So the
primary test aggregates to one observation per company-month, and standard
errors are computed across non-overlapping months.
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "warehouse.db"
MAPPED = ROOT / "data" / "contracts_mapped.parquet"

SHARE_CONCEPTS = ["EntityCommonStockSharesOutstanding",
                  "CommonStockSharesOutstanding",
                  "WeightedAverageNumberOfDilutedSharesOutstanding"]


def load_prices(tickers: list[str]) -> pd.DataFrame:
    con = sqlite3.connect(DB)
    q = ",".join("?" * len(tickers))
    px = pd.read_sql(
        f"SELECT symbol, dt, close, adj_close FROM bars "
        f"WHERE symbol IN ({q}) ORDER BY symbol, dt", con, params=tickers)
    spy = pd.read_sql(
        "SELECT dt, adj_close AS spy FROM bars WHERE symbol='SPY' ORDER BY dt", con)
    con.close()
    px = px.merge(spy, on="dt", how="left")
    px["ret"] = px.groupby("symbol")["adj_close"].pct_change()
    px["mkt"] = px["spy"].pct_change()
    # Abnormal return is the simplest defensible version: excess over the index.
    # A beta-adjusted version is run later as a robustness variant.
    px["abn"] = px["ret"] - px["mkt"]
    px["day_index"] = px.groupby("symbol").cumcount()
    return px


def load_shares(tickers: list[str]) -> pd.DataFrame:
    """Point-in-time share counts, keyed on SEC filing date, never period end."""
    con = sqlite3.connect(DB)
    q = ",".join("?" * len(tickers))
    c = ",".join("?" * len(SHARE_CONCEPTS))
    d = pd.read_sql(
        f"SELECT symbol, concept, filed, val FROM fundamentals "
        f"WHERE symbol IN ({q}) AND concept IN ({c}) AND val > 0 "
        f"ORDER BY symbol, filed", con, params=tickers + SHARE_CONCEPTS)
    con.close()
    if d.empty:
        return d
    # Prefer the cover-page count, fall back in declared order.
    rank = {c: i for i, c in enumerate(SHARE_CONCEPTS)}
    d["rk"] = d["concept"].map(rank)
    d = (d.sort_values(["symbol", "filed", "rk"])
          .drop_duplicates(["symbol", "filed"], keep="first"))
    return d[["symbol", "filed", "val"]].rename(columns={"val": "shares"})


def attach_mcap(panel: pd.DataFrame, shares: pd.DataFrame,
                px: pd.DataFrame) -> pd.DataFrame:
    """Market cap from the share count KNOWN on that date times the unadjusted
    close. adj_close would mix a split-adjusted price with an as-filed share
    count and produce a nonsense cap."""
    if shares.empty:
        panel["mcap"] = np.nan
        return panel
    # merge_asof needs a real temporal key, not a string, so the join runs on a
    # datetime column that is dropped afterwards.
    p = panel.copy()
    p["_k"] = pd.to_datetime(p["dt"])
    s = shares.copy()
    s["_k"] = pd.to_datetime(s["filed"])
    p = p.sort_values("_k")
    s = s.sort_values("_k")[["symbol", "_k", "shares"]]
    out = pd.merge_asof(p, s, on="_k", by="symbol", direction="backward",
                        allow_exact_matches=True).drop(columns=["_k"])
    out = out.merge(px[["symbol", "dt", "close"]], on=["symbol", "dt"], how="left")
    out["mcap"] = out["shares"] * out["close"]
    return out


def snap_to_session(ev: pd.DataFrame, px: pd.DataFrame) -> pd.DataFrame:
    """Move each action date forward to the first session the market could trade.

    Contract actions land on weekends and federal holidays. An exact-date lookup
    silently drops those, which is roughly a quarter of all events, and a sample
    that loses a quarter of itself without saying so is the kind of thing that
    reads as a result later. Snapping FORWARD is also the only direction that
    does not trade on information before it existed.
    """
    sess = (px[["symbol", "dt"]].drop_duplicates()
              .sort_values(["symbol", "dt"]).reset_index(drop=True))
    out = []
    for sym, g in ev.groupby("ticker", sort=False):
        days = sess.loc[sess["symbol"] == sym, "dt"].to_numpy()
        if len(days) == 0:
            continue
        pos = np.searchsorted(days, g["action_date"].to_numpy(), side="left")
        ok = pos < len(days)
        h = g.loc[ok].copy()
        h["event_dt"] = days[pos[ok]]
        out.append(h)
    if not out:
        return ev.assign(event_dt=pd.NA)
    r = pd.concat(out, ignore_index=True)
    moved = (r["event_dt"] != r["action_date"]).sum()
    print(f"  snapped {moved:,} of {len(r):,} events forward to the next session "
          f"({moved / max(len(r), 1):.1%} fell on a non-trading day)")
    return r


def event_car(ev: pd.DataFrame, px: pd.DataFrame,
              lo: int = -250, hi: int = 63) -> pd.DataFrame:
    """Mean cumulative abnormal return in event time, for the shape test."""
    idx = {(s, d): i for s, d, i in
           zip(px["symbol"], px["dt"], px["day_index"])}
    abn = {(s, i): a for s, i, a in
           zip(px["symbol"], px["day_index"], px["abn"])}
    acc = {k: [] for k in range(lo, hi + 1)}
    for sym, dt in zip(ev["ticker"], ev["event_dt"]):
        base = idx.get((sym, dt))
        if base is None:
            continue
        for k in range(lo, hi + 1):
            a = abn.get((sym, base + k))
            if a is not None and np.isfinite(a):
                acc[k].append(a)
    rows = []
    run = 0.0
    for k in range(lo, hi + 1):
        m = float(np.mean(acc[k])) if acc[k] else np.nan
        if np.isfinite(m):
            run += m
        rows.append({"k": k, "n": len(acc[k]), "mean_abn": m, "car": run})
    return pd.DataFrame(rows)


def placebo_car(ev: pd.DataFrame, px: pd.DataFrame, seed: int = 20260101,
                lo: int = -250, hi: int = 63) -> pd.DataFrame:
    """The same CAR, on RANDOM dates drawn from the same companies.

    This is the control that decides what a pre-event run-up means. The event
    sample is dominated by defense primes, and the abnormal return is excess
    over SPY, so if those companies simply beat the index over this period then
    ANY date would show a run-up and the "anticipation" reading would be wrong.
    Each real event is replaced by a uniformly drawn session for the SAME ticker,
    keeping the company mix identical and destroying only the timing. Whatever
    survives the difference is about the award date; whatever does not is drift.
    """
    rng = np.random.default_rng(seed)
    fake = []
    for sym, g in ev.groupby("ticker", sort=False):
        days = px.loc[px["symbol"] == sym, "dt"].to_numpy()
        usable = days[-hi - 1] if len(days) > hi + 1 else None
        pool = days[abs(lo):len(days) - hi - 1] if len(days) > abs(lo) + hi + 2 else days
        if len(pool) == 0:
            continue
        pick = rng.choice(pool, size=len(g), replace=True)
        fake.append(pd.DataFrame({"ticker": sym, "event_dt": pick}))
    if not fake:
        return pd.DataFrame()
    return event_car(pd.concat(fake, ignore_index=True), px, lo, hi)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--base-only", action="store_true",
                    help="restrict to Mod 0 base awards")
    args = ap.parse_args()

    if not MAPPED.exists():
        print(f"missing {MAPPED}; run contract_entities.py first")
        return 1
    d = pd.read_parquet(MAPPED)
    d = d[d["ticker"].notna()].copy()
    if args.base_only:
        d = d[d["is_base"]]
    tickers = sorted(d["ticker"].unique())
    print(f"{len(d):,} priceable contract actions, {len(tickers)} tickers, "
          f"{d['action_date'].min()} to {d['action_date'].max()}")

    px = load_prices(tickers)
    shares = load_shares(tickers)
    print(f"  prices: {len(px):,} bars; share counts: {len(shares):,} filings "
          f"for {shares['symbol'].nunique() if not shares.empty else 0} tickers")

    # ---- SHAPE TEST 1: event-time CAR, the pre-event window ----
    ev = snap_to_session(d, px)
    car = event_car(ev, px)
    car.to_csv(ROOT / "reports" / "contract_car.csv", index=False)
    pre = car[(car.k >= -250) & (car.k <= -2)]
    post = car[(car.k >= 0) & (car.k <= 21)]
    pre_car = pre["car"].iloc[-1] - pre["car"].iloc[0] if len(pre) > 1 else np.nan
    post_car = (post["car"].iloc[-1] - post["car"].iloc[0]) if len(post) > 1 else np.nan
    print(f"\n{'=' * 72}\nSHAPE TEST 1  pre-event window\n{'=' * 72}")
    for k in (-250, -126, -63, -21, -5, -1, 0, 1, 5, 21, 63):
        r = car[car.k == k]
        if not r.empty:
            print(f"  k={k:>5}  n={int(r['n'].iloc[0]):>6,}  "
                  f"CAR={r['car'].iloc[0]:>+8.2%}")
    print(f"\n  CAR from -250 to -2 : {pre_car:>+.2%}   <- must be ~0 for the mechanism")
    print(f"  CAR from   0 to +21 : {post_car:>+.2%}")

    pl = placebo_car(ev, px)
    if not pl.empty:
        pl.to_csv(ROOT / "reports" / "contract_car_placebo.csv", index=False)
        ppre = pl[(pl.k >= -250) & (pl.k <= -2)]
        ppost = pl[(pl.k >= 0) & (pl.k <= 21)]
        p_pre = ppre["car"].iloc[-1] - ppre["car"].iloc[0]
        p_post = ppost["car"].iloc[-1] - ppost["car"].iloc[0]
        print(f"\n  PLACEBO, same companies, random dates:")
        print(f"    CAR from -250 to -2 : {p_pre:>+.2%}")
        print(f"    CAR from   0 to +21 : {p_post:>+.2%}")
        print(f"\n  EVENT MINUS PLACEBO  (what is actually about the award date):")
        print(f"    pre-event  : {pre_car - p_pre:>+.2%}")
        print(f"    post-event : {post_car - p_post:>+.2%}")

    # ---- PRIMARY TEST: company-month, materiality scaled ----
    d["month"] = d["action_date"].str[:7]
    m = (d.groupby(["ticker", "month"])["amount_usd"].sum().reset_index()
          .rename(columns={"ticker": "symbol", "amount_usd": "award_usd"}))
    # Month-end session, so the signal uses only actions already announced by
    # the time the position would be taken.
    q = px[["symbol", "dt"]].copy()
    q["month"] = q["dt"].str[:7]
    me = (q.groupby(["symbol", "month"], as_index=False)["dt"].max()
           .rename(columns={"dt": "eom"}))
    m = m.merge(me, on=["symbol", "month"], how="inner").rename(
        columns={"eom": "dt"})
    m = attach_mcap(m, shares, px)
    m["award_over_mcap"] = m["award_usd"] / m["mcap"]

    fwd = px[["symbol", "dt", "day_index"]].copy()
    for h, lbl in ((21, "fwd_21"), (63, "fwd_63")):
        s = px.groupby("symbol")["adj_close"].shift(-h) / px["adj_close"] - 1.0
        b = px.groupby("symbol")["spy"].shift(-h) / px["spy"] - 1.0
        fwd[lbl] = (s - b).to_numpy()
    m = m.merge(fwd, on=["symbol", "dt"], how="left")
    m.to_csv(ROOT / "reports" / "contract_monthly.csv", index=False)

    print(f"\n{'=' * 72}\nSHAPE TEST 2  does it scale with materiality\n{'=' * 72}")
    k = m.dropna(subset=["award_over_mcap", "fwd_21"])
    print(f"  {len(k):,} company-months with market cap and a forward return")
    if len(k) >= 50:
        k = k.copy()
        k["q"] = pd.qcut(k["award_over_mcap"].rank(method="first"), 5,
                         labels=False, duplicates="drop")
        print(f"\n  {'quintile':<10}{'n':>6}{'median award/mcap':>20}"
              f"{'mean fwd21 abn':>17}{'t':>8}")
        for q, g in k.groupby("q"):
            t = (g["fwd_21"].mean() / (g["fwd_21"].std(ddof=1) / np.sqrt(len(g)))
                 if len(g) > 2 and g["fwd_21"].std(ddof=1) > 0 else np.nan)
            print(f"  {int(q) + 1:<10}{len(g):>6}{g['award_over_mcap'].median():>20.4%}"
                  f"{g['fwd_21'].mean():>17.2%}{t:>8.2f}")
        sp = k[k.q == k.q.max()]["fwd_21"].mean() - k[k.q == 0]["fwd_21"].mean()
        print(f"\n  top minus bottom quintile: {sp:>+.2%}")
        rho = k[["award_over_mcap", "fwd_21"]].corr(method="spearman").iloc[0, 1]
        print(f"  rank correlation of materiality with forward return: {rho:>+.4f}")
        print(f"  <- prediction 2 requires this to be POSITIVE and monotone")
    else:
        print("  too few observations for the quintile test")
    print(f"\nwrote reports/contract_car.csv and reports/contract_monthly.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
