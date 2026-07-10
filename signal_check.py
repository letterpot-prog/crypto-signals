#!/usr/bin/env python3
"""
Single-run signal check for GitHub Actions (or any cron), with confirmation
filters to reduce whipsaw ("negative") signals.

Base signal: EMA fast/slow crossover on the last CLOSED candle.
Filters (each toggleable) that a crossover must PASS to alert:
  - Trend filter: BUY only if price is above a long EMA (uptrend);
                  SELL only if below it. Blocks counter-trend crossovers.
  - ADX filter:   only if ADX >= ADX_MIN, i.e. the market is actually
                  trending, not chopping sideways.
More filters = fewer but higher-quality signals (and later entries). It reduces
false signals; it does NOT guarantee winners. Validate with a backtest.

Data source resilience: tries several exchanges; uses the first that responds
(GitHub's datacenter IPs are blocked by some exchanges).

Config via environment variables (TELEGRAM_* as GitHub Actions Secrets):
  TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, SYMBOL, TIMEFRAME, CRON_INTERVAL_MIN,
  USE_TREND_FILTER (1/0), TREND_EMA, USE_ADX_FILTER (1/0), ADX_PERIOD, ADX_MIN

Requirements: pip install ccxt pandas requests
This is a tool, not financial advice.
"""

import os
import time

import ccxt
import pandas as pd
import requests


def envflag(name, default="1"):
    return os.getenv(name, default).strip().lower() in ("1", "true", "yes", "on")


SYMBOL = os.getenv("SYMBOL", "BTC/USDC")
TIMEFRAME = os.getenv("TIMEFRAME", "1h")
EMA_FAST = int(os.getenv("EMA_FAST", "9"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "21"))
RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30
CRON_INTERVAL_MIN = int(os.getenv("CRON_INTERVAL_MIN", "15"))

USE_TREND_FILTER = envflag("USE_TREND_FILTER", "1")
TREND_EMA = int(os.getenv("TREND_EMA", "200"))
USE_ADX_FILTER = envflag("USE_ADX_FILTER", "1")
ADX_PERIOD = int(os.getenv("ADX_PERIOD", "14"))
ADX_MIN = float(os.getenv("ADX_MIN", "20"))

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

TF_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30,
              "1h": 60, "2h": 120, "4h": 240, "1d": 1440}


def base_of(symbol):
    return symbol.split("/")[0].upper()


def candidate_sources(symbol):
    b = base_of(symbol)
    return [
        ("kraken", f"{b}/USD"),
        ("coinbase", f"{b}/USDC"),
        ("coinbase", f"{b}/USD"),
        ("kraken", f"{b}/USDT"),
        ("bybit", f"{b}/USDT"),
        ("binance", f"{b}/USDT"),
    ]


def ema(s, span):
    return s.ewm(span=span, adjust=False).mean()


def rsi(s, period):
    d = s.diff()
    gain = d.clip(lower=0)
    loss = -d.clip(upper=0)
    ag = gain.ewm(alpha=1 / period, adjust=False).mean()
    al = loss.ewm(alpha=1 / period, adjust=False).mean()
    return 100 - (100 / (1 + ag / al))


def adx(high, low, close, period):
    """Wilder-style ADX (trend strength, 0-100)."""
    up = high.diff()
    down = -low.diff()
    plus_dm = ((up > down) & (up > 0)) * up
    minus_dm = ((down > up) & (down > 0)) * down
    tr = pd.concat([(high - low),
                    (high - close.shift()).abs(),
                    (low - close.shift()).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, adjust=False).mean()
    plus_di = 100 * plus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    minus_di = 100 * minus_dm.ewm(alpha=1 / period, adjust=False).mean() / atr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, pd.NA)
    return dx.ewm(alpha=1 / period, adjust=False).mean()


def crossover(d_prev, d_last):
    if d_prev <= 0 < d_last:
        return "BUY"
    if d_prev >= 0 > d_last:
        return "SELL"
    return None


def fetch_from_any():
    need = max(250, TREND_EMA * 2) if USE_TREND_FILTER else 200
    need = min(need, 1000)
    for ex_id, sym in candidate_sources(SYMBOL):
        try:
            ex = getattr(ccxt, ex_id)({"enableRateLimit": True})
            ohlcv = ex.fetch_ohlcv(sym, timeframe=TIMEFRAME, limit=need)
            if ohlcv and len(ohlcv) >= 60:
                df = pd.DataFrame(
                    ohlcv,
                    columns=["timestamp", "open", "high", "low", "close", "volume"],
                )
                print(f"Data source: {ex_id} {sym} ({len(df)} candles)", flush=True)
                return df, f"{ex_id}:{sym}"
        except Exception as e:
            print(f"  {ex_id} {sym} unavailable: {str(e)[:120]}", flush=True)
    return None, None


def send_telegram(text):
    print(text, flush=True)
    if not (TELEGRAM_TOKEN and TELEGRAM_CHAT_ID):
        print("(no Telegram creds set -- printed only)", flush=True)
        return
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage",
            data={"chat_id": TELEGRAM_CHAT_ID, "text": text},
            timeout=20,
        )
        print(f"telegram status: {r.status_code}", flush=True)
    except Exception as e:
        print(f"telegram send failed: {e}", flush=True)


def main():
    df, label = fetch_from_any()
    if df is None:
        print("Could not reach any data source this run. Will retry next run.",
              flush=True)
        return

    df["ef"] = ema(df["close"], EMA_FAST)
    df["es"] = ema(df["close"], EMA_SLOW)
    df["rsi"] = rsi(df["close"], RSI_PERIOD)
    df["diff"] = df["ef"] - df["es"]
    df["trend"] = ema(df["close"], TREND_EMA)
    df["adx"] = adx(df["high"], df["low"], df["close"], ADX_PERIOD)

    prev, last = df.iloc[-3], df.iloc[-2]      # last CLOSED candle = -2
    signal = crossover(prev["diff"], last["diff"])
    if not signal:
        print("No new crossover on last closed candle.", flush=True)
        return

    close = float(last["close"])
    trend = float(last["trend"])
    adx_val = last["adx"]

    # Apply confirmation filters.
    reasons = []
    if USE_TREND_FILTER:
        if signal == "BUY" and close <= trend:
            reasons.append(f"price below EMA{TREND_EMA} (not an uptrend)")
        if signal == "SELL" and close >= trend:
            reasons.append(f"price above EMA{TREND_EMA} (not a downtrend)")
    if USE_ADX_FILTER and pd.notna(adx_val):
        if float(adx_val) < ADX_MIN:
            reasons.append(f"ADX {float(adx_val):.0f} < {ADX_MIN:.0f} (weak/choppy)")

    if reasons:
        print(f"{signal} crossover FILTERED OUT: " + "; ".join(reasons), flush=True)
        return

    # Freshness dedupe: only alert if the candle just closed.
    tf_ms = TF_MINUTES.get(TIMEFRAME, 60) * 60_000
    age_min = (time.time() * 1000 - (int(last["timestamp"]) + tf_ms)) / 60_000
    if age_min > CRON_INTERVAL_MIN:
        print(f"Passed filters, but candle closed {age_min:.0f} min ago; "
              f"likely already alerted. Skipping.", flush=True)
        return

    r = float(last["rsi"])
    note = ""
    if signal == "BUY" and r >= RSI_OVERBOUGHT:
        note = "  (RSI overbought -- weaker buy)"
    elif signal == "SELL" and r <= RSI_OVERSOLD:
        note = "  (RSI oversold -- weaker sell)"
    adx_str = f"{float(adx_val):.0f}" if pd.notna(adx_val) else "n/a"
    trend_pos = "above" if close > trend else "below"

    send_telegram(
        f"[{signal}] {base_of(SYMBOL)} @ {close:.2f} | RSI {r:.0f} | "
        f"ADX {adx_str} | {trend_pos} EMA{TREND_EMA} | "
        f"EMA{EMA_FAST}/{EMA_SLOW} crossover{note}  (src {label})"
    )


if __name__ == "__main__":
    main()
