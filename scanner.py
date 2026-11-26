#!/usr/bin/env python3
"""
Daily last-arrow + 15m same-direction arrow scanner (free, no TradingView alerts).

Logic (ported from the Order Blocks indicator):
  1. Daily chart: find the LAST arrow (up = bullish OB retest, down = bearish OB retest).
  2. 15m chart: if ARROWS_IN_ROW arrows of the SAME direction appear in a row
     (no opposite arrow between) AND that direction equals the daily last arrow -> email.

Usage:
  python scanner.py            # scan and send emails
  python scanner.py --dry      # scan and print only (no email, no state change)
  python scanner.py --test     # send a test email
"""
import argparse
import json
import os
import smtplib
import sys
import time
from email.message import EmailMessage

import numpy as np
import pandas as pd

# ----------------------------- SETTINGS ------------------------------------
# Yahoo Finance symbols. Forex pairs end in =X (EURUSD=X, GBPUSD=X, USDJPY=X ...)
# Gold spot is XAUUSD=X; if Yahoo has no data for it, the scanner falls back to GC=F (gold futures).
SYMBOLS = ["XAUUSD=X", "EURUSD=X", "GBPUSD=X", "USDJPY=X"]
FALLBACKS = {"XAUUSD=X": "GC=F"}

# The list of symbols is read from the file symbols.txt (one per line). SYMBOLS above is only
# used if symbols.txt is missing or empty.

# How the DAILY candle is built. TradingView forex daily candles close at 17:00 New York
# (most brokers). Yahoo's own daily candles use a different cut-off, so by default we build
# the daily candle from 1-hour data with a 17:00 New York cut-off.
# "auto" = forex/gold (symbols ending in =X or =F) use the 17:00 New York cut-off,
#          everything else (NSE stocks, crypto) uses Yahoo's own daily candle  (recommended)
# "ny17" = always 17:00 New York cut-off    "yahoo" = always Yahoo's own daily candle
DAILY_MODE = "auto"
DAILY_ROLLOVER_HOUR_NY = 17

SWING_LEN = 5            # same as indicator "Swing Length"
INVALIDATION = "Wick"    # "Wick" or "Close", same as indicator
ARROWS_IN_ROW = 2        # same-direction arrows needed
USE_DAILY_FILTER = True  # require the daily last arrow to match
LOOKBACK_BARS = 3        # only alert for arrows from the last N completed 15m bars
STATE_FILE = "state.json"

BINS = 40
MAX_OBS = 50
RETEST_GAP = 4
# ---------------------------------------------------------------------------


def pretty(sym):
    """Readable name for emails, e.g. XAUUSD=X -> Gold (XAUUSD), RELIANCE.NS -> RELIANCE."""
    special = {"XAUUSD=X": "Gold (XAUUSD)", "GC=F": "Gold Futures (GC)",
               "XAGUSD=X": "Silver (XAGUSD)", "^NSEI": "Nifty 50", "^NSEBANK": "Bank Nifty"}
    if sym in special:
        return special[sym]
    for suffix in ("=X", ".NS", ".BO"):
        if sym.endswith(suffix):
            return sym[: -len(suffix)]
    return sym


def find_poc(h, l, frm, to):
    """Most-touched price level between bars frm..to (same as the indicator's POC engine)."""
    if frm is None or to is None or to - frm <= 0:
        return None
    seg_l = l[frm:to + 1]
    seg_h = h[frm:to + 1]
    min_p, max_p = seg_l.min(), seg_h.max()
    if not (min_p < max_p):
        return None
    step = (max_p - min_p) / BINS
    if step <= 0:
        return None
    diff = np.zeros(BINS)
    for lo, hi in zip(seg_l, seg_h):
        s = int(np.floor((lo - min_p) / step))
        e = int(np.floor((hi - min_p) / step))
        s = min(max(s, 0), BINS - 1)
        e = min(max(e, 0), BINS - 1)
        diff[s] += 1
        if e + 1 < BINS:
            diff[e + 1] -= 1
    run, best, best_bin = 0.0, 0.0, 0
    for k in range(BINS):
        run += diff[k]
        if run > best:
            best, best_bin = run, k
    return min_p + (best_bin + 0.5) * step


def has_gap(h, l, a, b, is_bull):
    lo_i, hi_i = min(a, b), max(a, b)
    for x in range(lo_i + 1, hi_i + 1):
        if is_bull:
            if l[x] > h[x - 1]:
                return True
        else:
            if h[x] < l[x - 1]:
                return True
    return False


def make_zone(h, l, frm, to, is_bull):
    poc = find_poc(h, l, frm, to)
    if poc is None:
        return None
    if not np.any((l[frm:to + 1] <= poc) & (h[frm:to + 1] >= poc)):
        return None
    best, run = None, None
    for idx in range(to, frm - 1, -1):
        hi, lo = h[idx], l[idx]
        if is_bull:
            run = lo if run is None else min(run, lo)
            if lo <= poc <= hi and lo == run:
                best = idx
        else:
            run = hi if run is None else max(run, hi)
            if lo <= poc <= hi and hi == run:
                best = idx
    if best is None or has_gap(h, l, best, to, is_bull):
        return None
    return float(h[best]), float(l[best])


def overlaps_active(obs, top, bottom):
    for ob in obs:
        if ob["active"] and top >= ob["bottom"] and bottom <= ob["top"]:
            return True
    return False


def prune(obs):
    while len(obs) > MAX_OBS:
        for j in range(len(obs) - 1, -1, -1):
            if not obs[j]["active"]:
                obs.pop(j)
                break
        else:
            obs.pop()


def run_engine(o, h, l, c, swing_len=SWING_LEN, inv=INVALIDATION):
    """Returns (ev, last): ev[i] = arrow direction confirmed on bar i (1 up, -1 down, 0 none);
    last[i] = last arrow direction up to bar i. The arrow itself is drawn on bar i-1."""
    n = len(c)
    L = swing_len
    obs = []
    sh_p = sh_i = sl_p = sl_i = None
    last_bull_rt = last_bear_rt = None
    prev_bear_now = prev_bull_now = False
    prev_sl_i = prev_sh_i = None
    ev = np.zeros(n, dtype=int)
    last = np.zeros(n, dtype=int)
    last_dir = 0

    for i in range(n):
        if i >= 2 * L:
            ctr = i - L
            if h[ctr] == h[ctr - L:i + 1].max():
                sh_p, sh_i = h[ctr], ctr
            if l[ctr] == l[ctr - L:i + 1].min():
                sl_p, sl_i = l[ctr], ctr

        bear_now = (sl_p is not None and i > sl_i and i >= 1
                    and c[i] < sl_p and c[i - 1] >= sl_p)
        bull_now = (sh_p is not None and i > sh_i and i >= 1
                    and c[i] > sh_p and c[i - 1] <= sh_p)

        to_i = i - 1
        got_bull = got_bear = False

        if prev_bear_now:
            if prev_sl_i is not None:
                z = make_zone(h, l, prev_sl_i, to_i, False)
                if z and not overlaps_active(obs, z[0], z[1]):
                    obs.insert(0, dict(top=z[0], bottom=z[1], bull=False,
                                       active=True, created=to_i))
                    prune(obs)
            sl_p = sl_i = None

        if prev_bull_now:
            if prev_sh_i is not None:
                z = make_zone(h, l, prev_sh_i, to_i, True)
                if z and not overlaps_active(obs, z[0], z[1]):
                    obs.insert(0, dict(top=z[0], bottom=z[1], bull=True,
                                       active=True, created=to_i))
                    prune(obs)
            sh_p = sh_i = None

        for ob in obs:
            if not ob["active"]:
                continue
            # invalidation
            if ob["bull"]:
                if inv == "Wick":
                    invalid = l[i] < ob["bottom"]
                else:
                    invalid = i > 0 and c[i - 1] < ob["bottom"]
            else:
                if inv == "Wick":
                    invalid = h[i] > ob["top"]
                else:
                    invalid = i > 0 and c[i - 1] > ob["top"]
            if invalid:
                ob["active"] = False
            # retest of previous candle (checked even on the bar that invalidates, like the indicator)
            if i > 0:
                p = i - 1
                if ob["bull"]:
                    rt = (o[p] > ob["top"] and c[p] > ob["top"]
                          and ob["bottom"] <= l[p] <= ob["top"])
                else:
                    rt = (o[p] < ob["bottom"] and c[p] < ob["bottom"]
                          and ob["bottom"] <= h[p] <= ob["top"])
                if rt and p > ob["created"]:
                    last_side = last_bull_rt if ob["bull"] else last_bear_rt
                    if last_side is None or p - last_side >= RETEST_GAP:
                        if ob["bull"]:
                            got_bull = True
                            last_bull_rt = p
                        else:
                            got_bear = True
                            last_bear_rt = p

        if got_bull and not got_bear:
            ev[i] = 1
        elif got_bear and not got_bull:
            ev[i] = -1
        if ev[i] != 0:
            last_dir = int(ev[i])
        last[i] = last_dir

        prev_bear_now, prev_bull_now = bear_now, bull_now
        prev_sl_i, prev_sh_i = sl_i, sh_i

    return ev, last


# ----------------------------- DATA ----------------------------------------
def load(sym, interval, period):
    import yfinance as yf
    df = yf.Ticker(sym).history(period=period, interval=interval, auto_adjust=False)
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    return df


def build_daily(sym):
    """Daily candles with a 17:00 New York cut-off, built from 1-hour data."""
    uses_ny17 = DAILY_MODE == "ny17" or (
        DAILY_MODE == "auto" and (sym.endswith("=X") or sym.endswith("=F")))
    if not uses_ny17:
        return load(sym, "1d", "3y")
    h1 = load(sym, "1h", "730d")
    if h1.empty:
        return h1
    # wall-clock New York time; shift so the 17:00 cut-off becomes midnight, then group by date
    local = h1.index.tz_convert("America/New_York").tz_localize(None)
    h1 = h1.copy()
    h1.index = local
    key = (h1.index + pd.Timedelta(hours=24 - DAILY_ROLLOVER_HOUR_NY)).floor("D")
    g = h1.groupby(key)
    d = g.agg({"Open": "first", "High": "max", "Low": "min", "Close": "last"})
    d = d[g["Open"].count() >= 3]  # ignore stray 1-2 hour fragments around the weekend
    return d.dropna(subset=["Open", "High", "Low", "Close"])


def analyze(sym, dry=False):
    daily = build_daily(sym)
    m15 = load(sym, "15m", "60d")
    if (len(daily) < 50 or len(m15) < 50) and sym in FALLBACKS:
        print(f"[{sym}] no usable data, trying {FALLBACKS[sym]}")
        alt = FALLBACKS[sym]
        daily = build_daily(alt)
        m15 = load(alt, "15m", "60d")
        sym = alt
    if len(daily) < 50 or len(m15) < 50:
        print(f"[{sym}] not enough data (daily={len(daily)}, 15m={len(m15)})")
        return []

    # keep only completed 15m candles
    now = pd.Timestamp.now(tz=m15.index.tz)
    m15 = m15[m15.index + pd.Timedelta(minutes=15) <= now]

    arr = lambda d, k: d[k].to_numpy(dtype=float)
    _, d_last = run_engine(arr(daily, "Open"), arr(daily, "High"), arr(daily, "Low"), arr(daily, "Close"))
    daily_dir = int(d_last[-1])

    ev, _ = run_engine(arr(m15, "Open"), arr(m15, "High"), arr(m15, "Low"), arr(m15, "Close"))

    # streak of same-direction arrows (an opposite arrow resets it)
    n = len(ev)
    streak, prev = 0, 0
    hits = []
    for i in range(n):
        if ev[i] == 0:
            continue
        streak = streak + 1 if ev[i] == prev else 1
        prev = int(ev[i])
        if i >= n - LOOKBACK_BARS and streak >= ARROWS_IN_ROW:
            if (not USE_DAILY_FILTER) or daily_dir == ev[i]:
                hits.append((i, int(ev[i]), streak))

    name = {1: "UP", -1: "DOWN", 0: "NONE"}
    recent = [(str(m15.index[i - 1])[:16], name[int(ev[i])]) for i in range(n) if ev[i] != 0][-5:]
    print(f"[{sym}] daily last arrow: {name[daily_dir]} | last 15m arrows: {recent}")

    alerts = []
    for i, d, s in hits:
        alerts.append(dict(
            key=f"{sym}|{m15.index[i]}|{d}",
            name=pretty(sym),
            dir_txt="UP" if d == 1 else "DOWN",
            body=(f"{pretty(sym)}  ({sym})\n"
                  f"  Signal: {s} {name[d]} arrows in a row on 15m\n"
                  f"  Daily last arrow: {name[daily_dir]}\n"
                  f"  Arrow candle: {m15.index[i - 1]}\n"
                  f"  Price (close): {m15['Close'].iloc[i]:.5g}\n")))
    return alerts


# ----------------------------- EMAIL ---------------------------------------
def send_email(subject, body):
    user = os.environ.get("GMAIL_USER")
    pwd = os.environ.get("GMAIL_APP_PASSWORD")
    if not user or not pwd:
        sys.exit("Set GMAIL_USER and GMAIL_APP_PASSWORD (Gmail app password).")
    to = os.environ.get("EMAIL_TO", user)
    msg = EmailMessage()
    msg["Subject"] = subject
    msg["From"] = user
    msg["To"] = to
    msg.set_content(body)
    with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
        s.login(user, pwd.replace(" ", ""))
        s.send_message(msg)


def load_symbols():
    try:
        with open("symbols.txt") as f:
            syms = [ln.split("#")[0].strip() for ln in f]
        syms = [x for x in syms if x]
        if syms:
            return syms
    except FileNotFoundError:
        pass
    return SYMBOLS


def load_state():
    try:
        with open(STATE_FILE) as f:
            return json.load(f)
    except Exception:
        return {"sent": []}


def save_state(state):
    state["sent"] = state["sent"][-300:]
    with open(STATE_FILE, "w") as f:
        json.dump(state, f, indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry", action="store_true", help="print only, no email")
    ap.add_argument("--test", action="store_true", help="send a test email")
    a = ap.parse_args()

    if a.test:
        send_email("Arrow scanner test", "Your scanner can send email.")
        print("Test email sent.")
        return

    state = load_state()
    new_alerts = []
    for sym in load_symbols():
        time.sleep(1)  # be gentle with Yahoo
        try:
            alerts = analyze(sym, a.dry)
        except Exception as e:
            print(f"[{sym}] error: {e}")
            continue
        for al in alerts:
            if al["key"] not in state["sent"]:
                new_alerts.append(al)

    if new_alerts:
        names = ", ".join(f"{x['name']} {x['dir_txt']}" for x in new_alerts)
        subject = f"Arrow alert: {names}"
        body = "\n".join(x["body"] for x in new_alerts)
        print(f"ALERT -> {subject}")
        if not a.dry:
            send_email(subject, body)
            state["sent"] += [x["key"] for x in new_alerts]
    else:
        print("No new alerts this run.")
    if not a.dry:
        save_state(state)


if __name__ == "__main__":
    main()
