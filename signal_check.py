#!/usr/bin/env python3
"""
Single-run signal check for GitHub Actions (or any cron).

Fetches recent candles, checks the last CLOSED candle for an EMA-crossover
signal, and if that candle JUST closed (within CRON_INTERVAL_MIN), sends a
Telegram alert. Then exits. Designed to be triggered on a schedule, so it does
NOT loop -- the scheduler runs it repeatedly.

Secrets/config come from environment variables (set TELEGRAM_* as GitHub Actions
Secrets; the rest are set in the workflow file):
    TELEGRAM_TOKEN, TELEGRAM_CHAT_ID, SYMBOL, TIMEFRAME, CRON_INTERVAL_MIN

Requirements: pip install ccxt pandas requests
This is a tool, not financial advice.
"""

import os
import time

import ccxt
import pandas as pd
import requests

DATA_EXCHANGE = os.getenv("DATA_EXCHANGE", "bybit")
SYMBOL = os.getenv("SYMBOL", "BTC/USDC")
TIMEFRAME = os.getenv("TIMEFRAME", "1h")
EMA_FAST = int(os.getenv("EMA_FAST", "9"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "21"))
RSI_PERIOD = int(os.getenv("RSI_PERIOD", "14"))
RSI_OVERBOUGHT = 70
RSI_OVERSOLD = 30

# Must match your workflow's cron spacing. Only alert if the last closed candle
# closed within this many minutes -- prevents duplicate alerts when several
# scheduled runs happen during the same candle.
CRON_INTERVAL_MIN = int(os.getenv("CRON_INTERVAL_MIN", "15"))

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "")

TF_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30,
              "1h": 60, "2h": 120, "4h": 240, "1d": 1440}


def ema(s, span):
    return s.ewm(span=span, adjust=False).mean()


def rsi(s, period):
    d = s.diff()
    gain = d.clip(lower=0)
    loss = -d.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, adjust=False).mean()
    return 100 - (100 / (1 + avg_gain / avg_loss))


def crossover(d_prev, d_last):
    if d_prev <= 0 < d_last:
        return "BUY"
    if d_prev >= 0 > d_last:
        return "SELL"
    return None


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
    ex = getattr(ccxt, DATA_EXCHANGE)({"enableRateLimit": True})
    ohlcv = ex.fetch_ohlcv(SYMBOL, timeframe=TIMEFRAME, limit=200)
    df = pd.DataFrame(
        ohlcv, columns=["timestamp", "open", "high", "low", "close", "volume"]
    )
    df["ef"] = ema(df["close"], EMA_FAST)
    df["es"] = ema(df["close"], EMA_SLOW)
    df["rsi"] = rsi(df["close"], RSI_PERIOD)
    df["diff"] = df["ef"] - df["es"]

    # -1 is the still-forming candle; -2 is the last CLOSED candle.
    prev, last = df.iloc[-3], df.iloc[-2]
    signal = crossover(prev["diff"], last["diff"])
    if not signal:
        print("No new crossover on last closed candle.", flush=True)
        return

    # Only alert if that candle closed recently (freshness-based dedupe).
    tf_ms = TF_MINUTES.get(TIMEFRAME, 60) * 60_000
    close_time_ms = int(last["timestamp"]) + tf_ms
    age_min = (time.time() * 1000 - close_time_ms) / 60_000
    if age_min > CRON_INTERVAL_MIN:
        print(f"Crossover found, but candle closed {age_min:.0f} min ago "
              f"(> {CRON_INTERVAL_MIN}); likely already alerted. Skipping.",
              flush=True)
        return

    price = float(last["close"])
    r = float(last["rsi"])
    note = ""
    if signal == "BUY" and r >= RSI_OVERBOUGHT:
        note = "  (RSI overbought -- weaker buy)"
    elif signal == "SELL" and r <= RSI_OVERSOLD:
        note = "  (RSI oversold -- weaker sell)"

    send_telegram(
        f"[{signal}] {SYMBOL} @ {price:.2f} | RSI {r:.0f} | "
        f"EMA{EMA_FAST}/{EMA_SLOW} crossover{note}"
    )


if __name__ == "__main__":
    main()
