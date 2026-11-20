#!/usr/bin/env python3
"""
Nightly order writer for the AMZN wheel (FINTECH 535).

An algorithm is a function that accepts inputs and outputs a set of open
orders. This script is that function, run every night by GitHub Actions:

  inputs   price_history.json   daily AMZN bars, written by Sanji
           positions.json       the book, written by Sanji (cash if missing)
           LSEG option chain    bid / ask for this week's AMZN options
  output   open_orders.json     Sanji orders for the next NYSE session
           (plus logs/latest.json and logs/decisions.csv for the website)

Rules (same as the 535Wheel backtest):
  - Only act on the first session of a trading week.
  - Cash only            -> sell 1 put,  nearest 30-delta, this week's expiry.
  - Long >= 100 shares   -> sell 1 call per 100 shares, nearest 30-delta.
  - Short option open    -> do nothing; Sanji handles expiry and assignment.

Usage:
  python nightly_orders.py                 normal nightly run
  python nightly_orders.py --diagnose      LSEG connectivity report, writes nothing
  python nightly_orders.py --dry-run       run everything, print, write nothing
  python nightly_orders.py --now 2026-11-20T22:00 --chain-csv fixture.csv
                                           offline test with a saved chain
"""

import argparse
import csv
import json
import math
import os
import re
import sys
import time
from datetime import date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd

# ---------------------------------------------------------------- settings
TICKER = "AMZN"
CONID = 3691937                 # IBKR conid for AMZN (also in Sanji's seed list)
LSEG_RIC = "AMZN.O"             # underlying RIC on LSEG
TARGET_DELTA = 0.30             # same as the backtest
CONTRACTS_PER_PUT = 1           # one cash-secured put
RISK_FREE = 0.0375              # course risk-free rate
DIV_YIELD = 0.0                 # AMZN pays no dividend
MIN_LIMIT = 0.05                # do not write an option for less than $0.05
PUT_WINDOW = (0.85, 1.02)       # strikes to quote, as a fraction of spot
CALL_WINDOW = (0.98, 1.15)
QUOTE_BATCH = 40                # LSEG batch size; guidelines say chunk and pace
PACE_SECONDS = 0.4              # <= ~3 requests per second

ROOT = Path(__file__).resolve().parent
PRICE_FILE = ROOT / "price_history.json"
POSITION_FILES = [ROOT / "mkt" / "positions.json", ROOT / "positions.json"]
ORDER_FILES = [ROOT / "open_orders.json", ROOT / "mkt" / "open_orders.json"]
LOG_DIR = ROOT / "logs"
LATEST_FILE = LOG_DIR / "latest.json"
DECISIONS_FILE = LOG_DIR / "decisions.csv"

ET = ZoneInfo("America/New_York")
CAL = xcals.get_calendar("XNYS")


class StopRun(Exception):
    """Something is wrong enough that no orders should be written."""


# ================================================================ calendar
def plan_dates(now_et: datetime):
    """
    for_session:   the session these orders are for: the first NYSE session strictly
                   after today (ET). A manual run during market hours never plans for a
                   session that has already opened.
    last_session:  the most recent session whose close has passed (the data we can see).
    first_of_week: True if for_session is the first session of its trading week.
    """
    today = pd.Timestamp(now_et.date())
    nxt = CAL.next_session(today) if CAL.is_session(today) else CAL.date_to_session(today, direction="next")
    if CAL.is_session(today) and now_et >= CAL.session_close(today).tz_convert(ET).to_pydatetime():
        last = today
    else:
        last = CAL.previous_session(today) if CAL.is_session(today) else CAL.date_to_session(today, direction="previous")
    prev_of_next = CAL.previous_session(nxt)
    first_of_week = nxt.isocalendar()[:2] != prev_of_next.isocalendar()[:2]
    return last.date(), nxt.date(), first_of_week


# ================================================================ inputs
def load_price_history():
    """AMZN bars from Sanji's file. Returns (bars dict, warning or None)."""
    if not PRICE_FILE.exists():
        return {}, "price_history.json not found"
    data = json.loads(PRICE_FILE.read_text(encoding="utf-8"))
    entry = data.get(str(CONID))
    if entry is None:
        return {}, f"conid {CONID} ({TICKER}) is not in price_history.json yet"
    bars = entry.get("price_history", {}) or {}
    if not bars:
        return {}, f"conid {CONID} ({TICKER}) is tracked but has no bars yet"
    return bars, None


def _num(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def parse_positions(raw):
    """
    Sanji's spec names positions.json but does not publish its schema yet,
    so this reads it defensively. Anything it cannot interpret stops the run:
    planning off a book we misread is worse than not trading.

    Returns dict: shares, short_puts, short_calls (lists of {expiry, strike, qty}),
    other (count of entries for other underlyings), cash (or None).
    """
    book = {"shares": 0, "short_puts": [], "short_calls": [], "long_options": 0,
            "other": 0, "cash": None}
    if isinstance(raw, dict):
        if "cash" in raw:
            book["cash"] = _num(raw.get("cash"))
        items = raw.get("positions", raw.get("book"))
        if items is None:
            rest = {k: v for k, v in raw.items() if k != "cash"}
            if not all(isinstance(v, dict) for v in rest.values()):
                raise StopRun(f"positions.json: unrecognised shape, keys {sorted(raw)}")
            items = list(rest.values())
    else:
        items = raw
    if not isinstance(items, list):
        raise StopRun(f"positions.json: cannot find a list of positions in {type(raw).__name__}")

    for p in items:
        if not isinstance(p, dict):
            raise StopRun(f"positions.json: entry is not an object: {p!r}")
        kind = str(p.get("contract", p.get("sec_type", p.get("type", "")))).lower()
        if kind in ("cash", "usd"):
            book["cash"] = _num(p.get("qty", p.get("amount", p.get("value"))))
            continue
        und = p.get("underlying", p.get("conid"))
        qty = _num(p.get("qty", p.get("quantity", p.get("position"))))
        if qty is None or und is None or kind not in ("underlying", "stk", "stock", "put", "call", "p", "c", "opt"):
            raise StopRun(f"positions.json: cannot interpret entry {p!r}")
        side = str(p.get("side", "")).lower()
        if side in ("short", "sht", "sell"):
            qty = -abs(qty)
        if int(und) != CONID:
            book["other"] += 1
            continue
        if kind in ("underlying", "stk", "stock"):
            book["shares"] += int(qty)
            continue
        right = p.get("right", kind)
        right = "put" if str(right).lower() in ("put", "p") else "call"
        leg = {"expiry": str(p.get("expiry", "")), "strike": _num(p.get("strike")), "qty": int(abs(qty))}
        if qty < 0:
            book["short_puts" if right == "put" else "short_calls"].append(leg)
        else:
            book["long_options"] += 1
    return book


def load_positions():
    for path in POSITION_FILES:
        if path.exists():
            raw = json.loads(path.read_text(encoding="utf-8"))
            return parse_positions(raw), str(path.relative_to(ROOT))
    return parse_positions([]), None


# ================================================================ decision
def decide(book, first_of_week, for_session):
    """
    The state machine. Returns (state, action, qty, reason).
    action is 'put', 'call' or None.
    """
    def live(legs):
        out = []
        for leg in legs:
            try:
                exp = date.fromisoformat(leg["expiry"][:10])
            except ValueError:
                exp = None
            if exp is None or exp >= for_session:     # unknown expiry counts as open
                out.append(leg)
        return out

    open_puts, open_calls = live(book["short_puts"]), live(book["short_calls"])
    shares = book["shares"]

    if shares < 0:
        raise StopRun("book is short AMZN shares; the wheel never does that, so stop and look")
    if open_puts or open_calls:
        state = "SHORT_PUT" if open_puts else "SHORT_CALL"
    elif shares >= 100:
        state = "SHARES"
    else:
        state = "CASH"

    if not first_of_week:
        return state, None, 0, "not the first session of the week; the wheel only opens options on the week's first session"
    if state in ("SHORT_PUT", "SHORT_CALL"):
        return state, None, 0, "a short option is still open; wait for Sanji to expire or assign it"
    if state == "SHARES":
        return state, "call", shares // 100, f"long {shares} shares; write {shares // 100} covered call(s)"
    return state, "put", CONTRACTS_PER_PUT, "book is cash; write a cash-secured put"


# ================================================================ Black-Scholes
def _ncdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_price(s, k, t, vol, cp, r=RISK_FREE, q=DIV_YIELD):
    if t <= 0 or vol <= 0:
        return max(0.0, (s - k) if cp == "call" else (k - s))
    d1 = (math.log(s / k) + (r - q + 0.5 * vol * vol) * t) / (vol * math.sqrt(t))
    d2 = d1 - vol * math.sqrt(t)
    if cp == "call":
        return s * math.exp(-q * t) * _ncdf(d1) - k * math.exp(-r * t) * _ncdf(d2)
    return k * math.exp(-r * t) * _ncdf(-d2) - s * math.exp(-q * t) * _ncdf(-d1)


def bs_delta(s, k, t, vol, cp, r=RISK_FREE, q=DIV_YIELD):
    d1 = (math.log(s / k) + (r - q + 0.5 * vol * vol) * t) / (vol * math.sqrt(t))
    return math.exp(-q * t) * (_ncdf(d1) if cp == "call" else _ncdf(d1) - 1.0)


def implied_vol(price, s, k, t, cp):
    """Bisection. Returns None when the price is outside the no-arbitrage range."""
    lo, hi = 1e-4, 5.0
    if t <= 0 or price <= bs_price(s, k, t, lo, cp) or price >= bs_price(s, k, t, hi, cp):
        return None
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if bs_price(s, k, t, mid, cp) > price:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


# ================================================================ LSEG
def open_lseg_session():
    """Same platform session as the class nightly_refresh.py; creds come from GitHub secrets."""
    import lseg.data as ld
    app_key = os.environ.get("LSEG_APP_KEY")
    username = os.environ.get("LSEG_USERNAME")
    password = os.environ.get("LSEG_PASSWORD")
    if not all([app_key, username, password]):
        raise StopRun("Missing LSEG_APP_KEY, LSEG_USERNAME, or LSEG_PASSWORD")
    session = ld.session.platform.Definition(
        app_key=app_key,
        grant=ld.session.platform.GrantPassword(username=username, password=password),
        signon_control=True,
    ).get_session()
    session.open()
    ld.session.set_default(session)
    print("LSEG platform session opened")
    return ld, session


def search_chain(ld, lo: date, hi: date):
    """Listed AMZN options expiring between lo and hi. Tries a date filter, then a sorted search."""
    base = f"SearchAllCategoryv2 eq 'Options' and UnderlyingQuoteRIC eq '{LSEG_RIC}'"
    select = "RIC,StrikePrice,ExpiryDate,CallPutOption"
    attempts = [
        dict(filter=f"{base} and ExpiryDate ge {lo.isoformat()} and ExpiryDate le {hi.isoformat()}", top=3000),
        dict(filter=base, order_by="ExpiryDate", top=3000),
    ]
    last_err = None
    for i, kw in enumerate(attempts, 1):
        try:
            raw = ld.discovery.search(view=ld.discovery.Views.SEARCH_ALL, select=select, **kw)
        except Exception as e:                      # try the next form
            last_err = e
            print(f"  chain search attempt {i} failed: {e}")
            continue
        df = clean_chain(raw)
        df = df[(df["expiry"] >= lo) & (df["expiry"] <= hi)]
        print(f"  chain search attempt {i}: {len(df)} contracts in window")
        if not df.empty:
            return df
    raise StopRun(f"no {TICKER} options found expiring {lo} to {hi} (last error: {last_err})")


def clean_chain(raw):
    df = raw.dropna(subset=["RIC", "StrikePrice", "ExpiryDate", "CallPutOption"]).copy()
    df["RIC"] = df["RIC"].astype(str)
    df = df[~df["RIC"].str.startswith("0#")]
    df["expiry"] = pd.to_datetime(df["ExpiryDate"]).dt.date
    df["strike"] = pd.to_numeric(df["StrikePrice"], errors="coerce")
    df["cp"] = df["CallPutOption"].astype(str).str.lower().map(lambda x: "call" if x.startswith("c") else "put")
    df = df.dropna(subset=["strike"])
    # Prefer the OPRA composite listing (.U) when the search returns several venues.
    if df["RIC"].str.endswith(".U").any():
        df = df[df["RIC"].str.endswith(".U")]
    df = df.drop_duplicates(subset=["expiry", "strike", "cp"])
    return df[["RIC", "expiry", "strike", "cp"]].rename(columns={"RIC": "ric"}).reset_index(drop=True)


def _pick_col(cols, word):
    for c in cols:
        if str(c).strip().lower() == word:
            return c
    for c in cols:
        if word in str(c).lower():
            return c
    return None


def _quotes_history(ld, rics, quote_date):
    """Daily BID/ASK bars. Returns list of (ric, bid, ask, bar_date)."""
    out = []
    df = ld.get_history(universe=rics, fields=["BID", "ASK"], interval="daily",
                        start=(quote_date - timedelta(days=6)).isoformat(),
                        end=(quote_date + timedelta(days=1)).isoformat())
    if df is None or df.empty:
        return out
    df = df[df.index.map(lambda x: pd.Timestamp(x).date() <= quote_date)]
    if df.empty:
        return out
    if isinstance(df.columns, pd.MultiIndex):
        for ric in rics:
            if ric not in df.columns.get_level_values(0):
                continue
            sub = df[ric]
            b, a = _pick_col(sub.columns, "bid"), _pick_col(sub.columns, "ask")
            if b is None or a is None:
                continue
            sub = sub[[b, a]].apply(pd.to_numeric, errors="coerce").dropna()
            if not sub.empty:
                out.append((ric, float(sub[b].iloc[-1]), float(sub[a].iloc[-1]), pd.Timestamp(sub.index[-1]).date()))
    else:
        b, a = _pick_col(df.columns, "bid"), _pick_col(df.columns, "ask")
        if b is not None and a is not None and len(rics) == 1:
            sub = df[[b, a]].apply(pd.to_numeric, errors="coerce").dropna()
            if not sub.empty:
                out.append((rics[0], float(sub[b].iloc[-1]), float(sub[a].iloc[-1]), pd.Timestamp(sub.index[-1]).date()))
    return out


def _quotes_tr(ld, rics, quote_date):
    """Reference-data bid/ask for the quote date."""
    out = []
    d = quote_date.isoformat()
    df = ld.get_data(universe=rics, fields=["TR.BIDPRICE", "TR.ASKPRICE", "TR.BIDPRICE.date"],
                     parameters={"SDate": d, "EDate": d})
    if df is None or df.empty:
        return out
    b, a = _pick_col(df.columns, "bid price"), _pick_col(df.columns, "ask price")
    dcol = _pick_col(df.columns, "date")
    if b is None or a is None:
        return out
    for _, row in df.iterrows():
        bid, ask = _num(row[b]), _num(row[a])
        if bid is None or ask is None or math.isnan(bid) or math.isnan(ask):
            continue
        bar = pd.Timestamp(row[dcol]).date() if dcol is not None and pd.notna(row[dcol]) else quote_date
        out.append((row["Instrument"], bid, ask, bar))
    return out


def _quotes_snapshot(ld, rics, quote_date):
    """Last snapshot BID/ASK. After the close this is the closing quote."""
    out = []
    df = ld.get_data(universe=rics, fields=["BID", "ASK"])
    if df is None or df.empty:
        return out
    for _, row in df.iterrows():
        bid, ask = _num(row.get("BID")), _num(row.get("ASK"))
        if bid is None or ask is None or math.isnan(bid) or math.isnan(ask):
            continue
        out.append((row["Instrument"], bid, ask, quote_date))
    return out


QUOTE_METHODS = [("history", _quotes_history), ("tr_fields", _quotes_tr), ("snapshot", _quotes_snapshot)]


def fetch_quotes(ld, rics, quote_date):
    """Try each method in order; keep the first that returns quotes for most contracts."""
    for name, fn in QUOTE_METHODS:
        got = []
        try:
            for i in range(0, len(rics), QUOTE_BATCH):
                got += fn(ld, rics[i:i + QUOTE_BATCH], quote_date)
                time.sleep(PACE_SECONDS)
        except Exception as e:
            print(f"  quotes via {name} failed: {e}")
            continue
        print(f"  quotes via {name}: {len(got)} of {len(rics)} contracts")
        if len(got) >= max(3, len(rics) // 2):
            q = pd.DataFrame(got, columns=["ric", "bid", "ask", "bar_date"]).drop_duplicates("ric", keep="last")
            q["method"] = name
            return q
    raise StopRun("no LSEG quote method returned bid/ask for this week's chain")


def lseg_spot(ld, quote_date):
    df = ld.get_history(universe=LSEG_RIC, fields=["TRDPRC_1"], interval="daily",
                        start=(quote_date - timedelta(days=6)).isoformat(),
                        end=(quote_date + timedelta(days=1)).isoformat())
    df = df[df.index.map(lambda x: pd.Timestamp(x).date() <= quote_date)]
    col = _pick_col(df.columns, "trdprc_1") or df.columns[0]
    s = pd.to_numeric(df[col], errors="coerce").dropna()
    if s.empty:
        raise StopRun(f"LSEG returned no close for {LSEG_RIC}")
    return float(s.iloc[-1]), pd.Timestamp(s.index[-1]).date()


def get_market(quote_date, for_session, bars, chain_csv=None):
    """
    Returns spot, spot_date, and a chain DataFrame for the target expiry with
    columns ric, expiry, strike, cp, bid, ask, bar_date, method.
    """
    week_end = for_session + timedelta(days=6 - for_session.weekday())     # Sunday of that week
    if chain_csv:
        chain = pd.read_csv(chain_csv)
        chain["expiry"] = pd.to_datetime(chain["expiry"]).dt.date
        chain["bar_date"] = pd.to_datetime(chain["bar_date"]).dt.date
        day = max(d for d in bars if d <= quote_date.isoformat())
        return float(bars[day]["close"]), date.fromisoformat(day), pick_expiry_rows(chain, for_session, week_end), "fixture"

    ld, session = open_lseg_session()
    try:
        if quote_date.isoformat() in bars:
            spot, spot_date = float(bars[quote_date.isoformat()]["close"]), quote_date
            print(f"Spot from price_history.json: {spot:.2f} on {spot_date}")
        else:
            spot, spot_date = lseg_spot(ld, quote_date)
            print(f"Spot from LSEG: {spot:.2f} on {spot_date}")
        listed = search_chain(ld, for_session + timedelta(days=1), for_session + timedelta(days=14))
        rows = pick_expiry_rows(listed, for_session, week_end)
        puts = rows[(rows.cp == "put") & rows.strike.between(spot * PUT_WINDOW[0], spot * PUT_WINDOW[1])]
        calls = rows[(rows.cp == "call") & rows.strike.between(spot * CALL_WINDOW[0], spot * CALL_WINDOW[1])]
        rows = pd.concat([puts, calls], ignore_index=True)
        if rows.empty:
            raise StopRun(f"no strikes inside the quote window around spot {spot:.2f}")
        print(f"Quoting {len(rows)} contracts expiring {rows.expiry.iloc[0]}")
        quotes = fetch_quotes(ld, rows.ric.tolist(), quote_date)
        chain = rows.merge(quotes, on="ric", how="inner")
        return spot, spot_date, chain, quotes["method"].iloc[0]
    finally:
        try:
            session.close()
            print("LSEG session closed")
        except Exception as e:
            print(f"Warning: failed to close LSEG session: {e}")


def pick_expiry_rows(chain, for_session, week_end):
    """
    This week's expiry: the last listed expiry after for_session within its week
    (Friday, or Thursday when Friday is a holiday). On a night that plans the
    expiry day itself (Thursday night plans Friday) there is no time left on this
    week's options, so the shadow candidates use next week's last expiry instead.
    Order nights plan the first session of a week, so they always get this week's.
    """
    exps = sorted(e for e in chain["expiry"].unique() if for_session < e <= week_end)
    if not exps:
        nxt_end = week_end + timedelta(days=7)
        exps = sorted(e for e in chain["expiry"].unique() if week_end < e <= nxt_end)
    if not exps:
        raise StopRun(f"no listed {TICKER} expiry after {for_session} in this week or the next")
    return chain[chain["expiry"] == exps[-1]].copy()


# ================================================================ strike selection
def score_chain(chain, spot, quote_date, for_session):
    """
    For every contract: implied vol from the prior-close mid (time from the quote
    date), then delta and a model price re-valued at the order session (time from
    for_session). That removes the weekend's time decay from both numbers, the
    same way the backtest measured delta on the entry day.
    """
    rows = []
    for r in chain.itertuples(index=False):
        if not (r.bid > 0 and r.ask > r.bid):
            continue
        mid = 0.5 * (r.bid + r.ask)
        t_quote = (r.expiry - r.bar_date).days / 365.0
        t_trade = (r.expiry - for_session).days / 365.0
        if t_trade <= 0:
            continue
        iv = implied_vol(mid, spot, r.strike, t_quote, r.cp)
        if iv is None:
            continue
        rows.append({
            "ric": r.ric, "cp": r.cp, "strike": float(r.strike), "expiry": r.expiry.isoformat(),
            "bid": round(r.bid, 4), "ask": round(r.ask, 4), "mid": round(mid, 4),
            "bar_date": r.bar_date.isoformat(), "iv": round(iv, 4),
            "delta": round(bs_delta(spot, r.strike, t_trade, iv, r.cp), 4),
            "model_px": round(bs_price(spot, r.strike, t_trade, iv, r.cp), 4),
        })
    return pd.DataFrame(rows)


def pick_strike(scored, cp, spot):
    """Nearest absolute delta to TARGET_DELTA, out of the money only. Ties go further OTM (Sanji's rule)."""
    side = scored[scored.cp == cp].copy()
    side = side[side.strike <= spot] if cp == "put" else side[side.strike >= spot]
    if side.empty:
        return None
    side["gap"] = (side.delta.abs() - TARGET_DELTA).abs().round(6)
    side["otm"] = -side.strike if cp == "put" else side.strike
    best = side.sort_values(["gap", "otm"], ascending=[True, False]).iloc[0]
    limit = math.floor(best.model_px * 100) / 100           # round down: a sell limit we can reach
    out = best.drop(["gap", "otm"]).to_dict()
    out["limit"] = limit
    return out


# ================================================================ orders
def build_order(cand, cp, qty, for_session):
    tag = "csp" if cp == "put" else "cc"
    return {
        "id": f"{TICKER.lower()}-{tag}-{for_session.isoformat()}",
        "for": for_session.isoformat(),
        "order_type": "lmt",
        "action": "SHT",
        "qty": int(qty),
        "underlying": CONID,
        "contract": cp,
        "expiry": cand["expiry"],
        "strike": _clean_strike(cand["strike"]),
        "ref": "open",
        "lmt_prc": cand["limit"],
    }


def _clean_strike(k):
    return int(k) if float(k).is_integer() else float(k)


# ---- Sanji spec validator (https://jakevestal.github.io/server-logic-533/order.html)
ALLOWED = {"id", "for", "order_type", "action", "qty", "underlying", "contract",
           "expiry", "strike", "ref", "lmt_prc"}
REQUIRED = {"id", "for", "order_type", "action", "qty", "underlying", "contract"}
RE_DTE = re.compile(r"^\d+DTE[+-]?$")
RE_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")
RE_STRIKE_TOKEN = re.compile(r"^(atm|(otm|itm):[1-9]\d*|pct:-?\d+(\.\d+)?|delta:([1-9]\d?)|premium:\d+(\.\d+)?)$")
RE_QUOTE = re.compile(r"^(mid|bid|ask)([+-]\d+(\.\d+)?)?$")


def _is_num(v):
    return isinstance(v, (int, float)) and not isinstance(v, bool)


def _valid_date(s):
    if not isinstance(s, str) or not RE_DATE.match(s):
        return False
    try:
        date.fromisoformat(s)
        return True
    except ValueError:
        return False


def validate_order(o, book=None):
    """Returns a list of problems. Empty list means Sanji should accept it."""
    errs = []
    if not isinstance(o, dict):
        return ["order is not an object"]
    extra = set(o) - ALLOWED
    if extra:
        errs.append(f"unknown field(s) {sorted(extra)}")
    missing = REQUIRED - set(o)
    if missing:
        errs.append(f"missing field(s) {sorted(missing)}")
    if not isinstance(o.get("id"), str) or not o.get("id"):
        errs.append("id must be a non-empty string")
    f = o.get("for")
    if not _valid_date(f):
        errs.append("for must be YYYY-MM-DD")
    elif not CAL.is_session(pd.Timestamp(f)):
        errs.append(f"for {f} is not an NYSE session")
    if o.get("order_type") not in ("mkt", "lmt"):
        errs.append("order_type must be mkt or lmt")
    if o.get("action") not in ("BUY", "SEL", "SHT", "BTC"):
        errs.append("action must be BUY, SEL, SHT or BTC")
    q = o.get("qty")
    if not (isinstance(q, int) and not isinstance(q, bool) and q >= 1):
        errs.append("qty must be an integer >= 1")
    if not (isinstance(o.get("underlying"), int) and not isinstance(o.get("underlying"), bool)):
        errs.append("underlying must be an integer conid")
    c = o.get("contract")
    if c not in ("put", "call", "underlying"):
        errs.append("contract must be put, call or underlying")
    if c in ("put", "call"):
        e = o.get("expiry")
        if not (isinstance(e, str) and (RE_DTE.match(e) or e in ("W+", "M+") or _valid_date(e))):
            errs.append(f"bad expiry {e!r}")
        k = o.get("strike")
        if not ((_is_num(k) and k > 0) or (isinstance(k, str) and RE_STRIKE_TOKEN.match(k))):
            errs.append(f"bad strike {k!r}")
    elif c == "underlying":
        for fld in ("expiry", "strike"):
            if fld in o:
                errs.append(f"{fld} must be omitted for the underlying")
    if "ref" in o and o["ref"] not in ("open", "prev_close"):
        errs.append("ref must be open or prev_close")
    if o.get("order_type") == "lmt":
        lp = o.get("lmt_prc")
        if not ((_is_num(lp) and lp > 0) or (isinstance(lp, str) and RE_QUOTE.match(lp))):
            errs.append(f"bad lmt_prc {lp!r}")
    elif "lmt_prc" in o:
        errs.append("lmt_prc is forbidden on a mkt order")

    # Book checks. Sanji rejects SEL/BTC with nothing to close; the wheel adds: never a naked call.
    if book is not None and not errs:
        a = o["action"]
        if c == "underlying" and a == "SEL" and book["shares"] < q:
            errs.append("SEL more shares than held")
        if c == "put" and a == "BTC" and sum(l["qty"] for l in book["short_puts"]) < q:
            errs.append("BTC a put that is not short")
        if c == "call" and a == "BTC" and sum(l["qty"] for l in book["short_calls"]) < q:
            errs.append("BTC a call that is not short")
        if c == "call" and a == "SHT" and q * 100 > book["shares"]:
            errs.append("SHT call not covered by shares (wheel rule)")
    return errs


def validate_file(orders):
    errs = []
    if not isinstance(orders, list):
        return ["open_orders.json must be a JSON array"]
    ids = [o.get("id") for o in orders if isinstance(o, dict)]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        errs.append(f"duplicate ids {dupes}")
    for o in orders:
        errs += [f"{o.get('id', '?')}: {e}" for e in validate_order(o)]
    return errs


def merge_orders(existing, new):
    """Keep every old order (Sanji ignores ids already filled and dates already past);
    replace an order only when tonight's run writes the same id again."""
    new_ids = {o["id"] for o in new}
    kept = [o for o in existing if o.get("id") not in new_ids]
    return sorted(kept + new, key=lambda o: (o.get("for", ""), o.get("id", "")))


def read_existing_orders():
    path = ORDER_FILES[0]
    if not path.exists():
        return []
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise StopRun("existing open_orders.json is not a JSON array")
    return data


# ================================================================ outputs
def _json_default(v):
    if hasattr(v, "item"):          # numpy scalars
        return v.item()
    if isinstance(v, (date, datetime)):
        return v.isoformat()
    raise TypeError(f"cannot serialise {type(v).__name__}")


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=_json_default) + "\n", encoding="utf-8")


DECISION_COLS = ["for", "run_at_et", "last_session", "first_of_week", "state", "action", "spot",
                 "expiry", "put_strike", "put_delta", "put_limit", "call_strike", "call_delta",
                 "call_limit", "order_id", "reason"]


def append_decision(row):
    """One row per planned session. A rerun for the same session replaces its row."""
    rows = []
    if DECISIONS_FILE.exists():
        with DECISIONS_FILE.open(newline="", encoding="utf-8") as fh:
            rows = [r for r in csv.DictReader(fh) if r.get("for") != row["for"]]
    rows.append({k: row.get(k, "") for k in DECISION_COLS})
    rows.sort(key=lambda r: r["for"])
    DECISIONS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with DECISIONS_FILE.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=DECISION_COLS)
        w.writeheader()
        w.writerows(rows)


# ================================================================ main
def run(now_et, chain_csv=None, dry_run=False):
    last_session, for_session, first_of_week = plan_dates(now_et)
    print(f"Run at {now_et:%Y-%m-%d %H:%M} ET | last session {last_session} | "
          f"planning {for_session} | first of week: {first_of_week}")

    bars, bars_warning = load_price_history()
    if bars_warning:
        print(f"Warning: {bars_warning}")
    book, pos_source = load_positions()
    print(f"Book ({pos_source or 'no positions.json yet, assuming cash'}): "
          f"shares={book['shares']} short_puts={book['short_puts']} short_calls={book['short_calls']}")

    state, action, qty, reason = decide(book, first_of_week, for_session)
    print(f"State {state} -> action {action} x{qty}: {reason}")

    spot, spot_date, chain, quote_method = get_market(last_session, for_session, bars, chain_csv)
    scored = score_chain(chain, spot, last_session, for_session)
    if scored.empty:
        raise StopRun("no contract had a usable bid/ask/implied vol")
    stale = sorted(set(scored.bar_date) - {last_session.isoformat()})
    if stale:
        print(f"Warning: some quotes are from {stale}, not {last_session}")
    put_c, call_c = pick_strike(scored, "put", spot), pick_strike(scored, "call", spot)
    for name, c in (("put", put_c), ("call", call_c)):
        if c:
            print(f"30-delta {name}: {c['strike']} exp {c['expiry']} delta {c['delta']:+.3f} "
                  f"mid {c['mid']:.2f} limit {c['limit']:.2f} ({c['ric']})")

    orders = []
    if action:
        cand = put_c if action == "put" else call_c
        if cand is None:
            raise StopRun(f"no out-of-the-money {action} to sell")
        if cand["limit"] < MIN_LIMIT:
            reason += f"; skipped, limit {cand['limit']:.2f} is below {MIN_LIMIT:.2f}"
            action = None
        else:
            orders.append(build_order(cand, action, qty, for_session))

    for o in orders:
        problems = validate_order(o, book)
        if problems:
            raise StopRun(f"order {o['id']} failed validation: {problems}")
    merged = merge_orders(read_existing_orders(), orders)
    problems = validate_file(merged)
    if problems:
        raise StopRun(f"open_orders.json would be invalid: {problems}")

    run_at = now_et.strftime("%Y-%m-%d %H:%M")
    latest = {
        "run_at_et": run_at, "last_session": last_session.isoformat(), "for": for_session.isoformat(),
        "first_of_week": first_of_week, "positions_source": pos_source, "book": book,
        "state": state, "action": action, "qty": qty, "reason": reason,
        "ticker": TICKER, "conid": CONID, "spot": spot, "spot_date": spot_date.isoformat(),
        "target_delta": TARGET_DELTA, "risk_free": RISK_FREE, "quote_method": quote_method,
        "price_history_warning": bars_warning, "stale_quote_dates": stale,
        "candidates": {"put": put_c, "call": call_c},
        "orders_tonight": orders, "open_orders_count": len(merged),
        "chain": scored.sort_values(["cp", "strike"]).to_dict(orient="records"),
    }
    decision = {
        "for": for_session.isoformat(), "run_at_et": run_at, "last_session": last_session.isoformat(),
        "first_of_week": first_of_week, "state": state, "action": action or "none",
        "spot": round(spot, 2), "expiry": (put_c or call_c or {}).get("expiry", ""),
        "put_strike": put_c and put_c["strike"], "put_delta": put_c and put_c["delta"],
        "put_limit": put_c and put_c["limit"], "call_strike": call_c and call_c["strike"],
        "call_delta": call_c and call_c["delta"], "call_limit": call_c and call_c["limit"],
        "order_id": ";".join(o["id"] for o in orders), "reason": reason,
    }
    if dry_run:
        print(json.dumps(merged, indent=2, default=_json_default))
        print("Dry run: nothing written")
        return latest
    for path in ORDER_FILES:
        write_json(path, merged)
    write_json(LATEST_FILE, latest)
    append_decision(decision)
    print(f"Wrote {len(orders)} new order(s); open_orders.json holds {len(merged)}")
    return latest


def diagnose():
    """Connectivity report for the first workflow run. Writes nothing."""
    now_et = datetime.now(ET)
    last_session, for_session, first_of_week = plan_dates(now_et)
    print(f"Now {now_et:%Y-%m-%d %H:%M} ET | last session {last_session} | next {for_session} | first of week {first_of_week}")
    bars, warn = load_price_history()
    print(f"price_history.json: {warn or f'{len(bars)} AMZN bars, last {max(bars)}'}")
    ld, session = open_lseg_session()
    try:
        try:
            print("AMZN close from LSEG:", lseg_spot(ld, last_session))
        except Exception as e:
            print("AMZN close from LSEG FAILED:", e)
        listed = search_chain(ld, for_session + timedelta(days=1), for_session + timedelta(days=14))
        print(f"Expiries found: {sorted(set(listed.expiry))}")
        week_end = for_session + timedelta(days=6 - for_session.weekday())
        rows = pick_expiry_rows(listed, for_session, week_end)
        spot = float(bars[max(bars)]["close"]) if bars else float(rows.strike.median())
        rows["dist"] = (rows.strike - spot).abs()
        sample = rows.sort_values("dist").head(4).ric.tolist()
        print(f"Sample contracts: {sample}")
        for name, fn in QUOTE_METHODS:
            try:
                got = fn(ld, sample, last_session)
                print(f"  {name}: {len(got)} quotes -> {got}")
            except Exception as e:
                print(f"  {name}: FAILED {e}")
    finally:
        session.close()
        print("LSEG session closed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--diagnose", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--now", help="pretend the run happens at this ET time, e.g. 2026-11-20T22:00")
    ap.add_argument("--chain-csv", help="offline chain fixture instead of LSEG")
    args = ap.parse_args()
    try:
        if args.diagnose:
            diagnose()
            return 0
        now_et = datetime.fromisoformat(args.now).replace(tzinfo=ET) if args.now else datetime.now(ET)
        run(now_et, chain_csv=args.chain_csv, dry_run=args.dry_run)
        return 0
    except StopRun as e:
        print(f"STOP: {e}. No files written.")
        return 1


if __name__ == "__main__":
    sys.exit(main())
