#!/usr/bin/env python3
"""
Backtest + filter comparison, made to run in GitHub Actions (manual trigger).

Pulls real candles and reports four setups side by side so you can see whether
the trend / ADX filters actually help:
    No filters | Trend only | ADX only | Trend + ADX (your live config)

For each: number of signals, closed trades, win rate, total return
(unleveraged), and max drawdown -- plus buy & hold for reference.

Position model: a signal sets the target (BUY->long, SELL->short); the position
flips only when direction changes, so filtered/non-alternating signals are
handled correctly. Fees applied per side. Simplified and unleveraged -- leverage
would multiply the returns AND the drawdowns. Past results don't predict future.

Config via env (set in the workflow, all optional):
    SYMBOL, TIMEFRAME, EMA_FAST, EMA_SLOW, TREND_EMA, ADX_PERIOD, ADX_MIN,
    BACKTEST_CANDLES, FEE_RATE

Requirements: pip install ccxt pandas
"""

import os

import ccxt
import pandas as pd

SYMBOL = os.getenv("SYMBOL", "BTC/USDC")
TIMEFRAME = os.getenv("TIMEFRAME", "1h")
EMA_FAST = int(os.getenv("EMA_FAST", "9"))
EMA_SLOW = int(os.getenv("EMA_SLOW", "21"))
TREND_EMA = int(os.getenv("TREND_EMA", "200"))
ADX_PERIOD = int(os.getenv("ADX_PERIOD", "14"))
ADX_MIN = float(os.getenv("ADX_MIN", "20"))
BACKTEST_CANDLES = int(os.getenv("BACKTEST_CANDLES", "1000"))
FEE_RATE = float(os.getenv("FEE_RATE", "0.0006"))

TF_MINUTES = {"1m": 1, "5m": 5, "15m": 15, "30m": 30,
              "1h": 60, "2h": 120, "4h": 240, "1d": 1440}


def base_of(symbol):
    return symbol.split("/")[0].upper()


def candidate_sources(symbol):
    b = base_of(symbol)
    return [("kraken", f"{b}/USD"), ("coinbase", f"{b}/USDC"),
            ("coinbase", f"{b}/USD"), ("kraken", f"{b}/USDT"),
            ("bybit", f"{b}/USDT"), ("binance", f"{b}/USDT")]


def ema(s, span):
    return s.ewm(span=span, adjust=False).mean()


def adx(high, low, close, period):
    up = high.diff()
    down = -low.diff()
    plus_dm = ((up > down) & (up > 0)) * up
    minus_dm = ((down > up) & (down > 0)) * down
    tr = pd.concat([(high - low), (high - close.shift()).abs(),
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


def build_indicators(df):
    df["ef"] = ema(df["close"], EMA_FAST)
    df["es"] = ema(df["close"], EMA_SLOW)
    df["diff"] = df["ef"] - df["es"]
    df["trend"] = ema(df["close"], TREND_EMA)
    df["adx"] = adx(df["high"], df["low"], df["close"], ADX_PERIOD)
    return df


def signals_for(df, use_trend, use_adx):
    warmup = max(EMA_SLOW, TREND_EMA if use_trend else 0,
                 ADX_PERIOD if use_adx else 0) + 1
    out = []
    for i in range(warmup, len(df)):
        s = crossover(df["diff"].iloc[i - 1], df["diff"].iloc[i])
        if not s:
            continue
        close = float(df["close"].iloc[i])
        if use_trend:
            trend = float(df["trend"].iloc[i])
            if s == "BUY" and close <= trend:
                continue
            if s == "SELL" and close >= trend:
                continue
        if use_adx:
            a = df["adx"].iloc[i]
            if pd.notna(a) and float(a) < ADX_MIN:
                continue
        out.append((s, close))
    return out


def simulate(signals):
    position, entry = None, None
    equity, curve, returns = 1.0, [1.0], []
    for side, price in signals:
        target = "long" if side == "BUY" else "short"
        if position is None:
            position, entry = target, price
        elif target != position:
            ret = (price / entry - 1) if position == "long" else (entry / price - 1)
            ret -= 2 * FEE_RATE
            equity *= (1 + ret)
            returns.append(ret)
            curve.append(equity)
            position, entry = target, price
    wins = sum(1 for r in returns if r > 0)
    n = len(returns)
    peak, max_dd = curve[0], 0.0
    for v in curve:
        peak = max(peak, v)
        max_dd = max(max_dd, (peak - v) / peak)
    return {"signals": len(signals), "trades": n,
            "win": (wins / n * 100 if n else 0.0),
            "ret": (equity - 1) * 100, "dd": max_dd * 100}


def fetch():
    for ex_id, sym in candidate_sources(SYMBOL):
        try:
            ex = getattr(ccxt, ex_id)({"enableRateLimit": True})
            ohlcv = ex.fetch_ohlcv(sym, timeframe=TIMEFRAME,
                                   limit=min(BACKTEST_CANDLES, 1000))
            if ohlcv and len(ohlcv) >= 100:
                df = pd.DataFrame(ohlcv, columns=["timestamp", "open", "high",
                                                  "low", "close", "volume"])
                return df, f"{ex_id}:{sym}"
        except Exception as e:
            print(f"  {ex_id} {sym} unavailable: {str(e)[:100]}", flush=True)
    return None, None


def main():
    df, label = fetch()
    if df is None:
        print("Could not reach any data source.", flush=True)
        return
    df = build_indicators(df)

    days = len(df) * TF_MINUTES.get(TIMEFRAME, 60) / 1440
    bh = (df["close"].iloc[-1] / df["close"].iloc[0] - 1) * 100

    print(f"\nBacktest {SYMBOL}  {TIMEFRAME}  | {len(df)} candles (~{days:.0f} days) "
          f"| source {label}")
    print(f"EMA {EMA_FAST}/{EMA_SLOW} | trend EMA{TREND_EMA} | ADX>={ADX_MIN:.0f} "
          f"| fee {FEE_RATE * 100:.2f}%/side")
    print(f"Buy & hold over window: {bh:+.1f}%\n")

    configs = [
        ("No filters", False, False),
        ("Trend only", True, False),
        ("ADX only", False, True),
        ("Trend + ADX (live)", True, True),
    ]
    print(f"{'Config':<20}{'signals':>8}{'trades':>7}{'win%':>7}"
          f"{'return':>9}{'maxDD':>8}")
    print("-" * 59)
    for name, ut, ua in configs:
        m = simulate(signals_for(df, ut, ua))
        print(f"{name:<20}{m['signals']:>8}{m['trades']:>7}{m['win']:>6.0f}%"
              f"{m['ret']:>+8.1f}%{m['dd']:>7.1f}%")
    print("\nFewer signals is expected with filters. Compare return vs drawdown,\n"
          "not just return. Best-on-past != best-in-future; leverage multiplies DD.")


if __name__ == "__main__":
    main()
