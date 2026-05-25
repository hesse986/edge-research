#!/usr/bin/env python3
"""Testowanie pairs trading dla różnych par i interwałów."""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '.')
import data as datamod
from sklearn.linear_model import LinearRegression

START = "2023-01-01"
END   = "2025-06-01"
LOOKBACK = 30
ENTRY_Z = 2.0
EXIT_Z = 0.5
MIN_TRADES = 20
COST = 0.0002

PAIRS = [("LTC/USDT", "XRP/USDT"), ("LTC/USDT", "ADA/USDT"), ("XRP/USDT", "ADA/USDT")]
TIMEFRAMES = ["1h", "4h", "1d"]

def load_asset(symbol, start, end, tf):
    df1h = datamod.load_binance_ohlcv(symbol, "1h", start, end, use_cache=True)
    if tf == "1h":
        return df1h
    elif tf == "4h":
        return datamod.resample_ohlcv(df1h, "4h")
    else:
        return datamod.resample_ohlcv(df1h, "1D")

def simulate_trades(zscore):
    trades = []
    position = 0
    entry_idx = None
    entry_z = None
    z_vals = zscore.values
    for i in range(LOOKBACK+5, len(z_vals)):
        z = z_vals[i]
        if position == 0:
            if z > ENTRY_Z:
                position = -1
                entry_idx = i
                entry_z = z
            elif z < -ENTRY_Z:
                position = 1
                entry_idx = i
                entry_z = z
        else:
            if (position == 1 and z > -EXIT_Z) or (position == -1 and z < EXIT_Z):
                trades.append((entry_idx, i, position, entry_z, z))
                position = 0
    return trades

def run_test(asset1, asset2, tf):
    print(f"\n=== Pair: {asset1} / {asset2} | TF: {tf} ===")
    df1 = load_asset(asset1, START, END, tf)
    df2 = load_asset(asset2, START, END, tf)
    common = df1.index.intersection(df2.index)
    if len(common) < LOOKBACK + 10:
        print("ZA MAŁO DANYCH")
        return
    close1 = df1.loc[common, "close"]
    close2 = df2.loc[common, "close"]
    model = LinearRegression().fit(close2.values.reshape(-1,1), close1.values)
    hedge = model.coef_[0]
    spread = close1 - hedge * close2
    rolling_mean = spread.rolling(LOOKBACK).mean()
    rolling_std = spread.rolling(LOOKBACK).std()
    zscore = (spread - rolling_mean) / rolling_std
    trades = simulate_trades(zscore)
    n = len(trades)
    print(f"Liczba transakcji: {n}")
    if n < MIN_TRADES:
        print("ZA MAŁO TRADOW")
        return
    results = []
    for entry, exit_, pos, z_entry, z_exit in trades:
        ret = (z_exit - z_entry) * pos
        results.append(ret)
    exp_R = np.mean(results)
    print(f"exp_R: {exp_R:.4f}")
    rng = np.random.default_rng(0)
    rand_returns = []
    for _ in range(5000):
        rand_pos = rng.choice([-1,1], size=len(trades))
        rand_ret = np.mean([(zscore.values[exit_] - zscore.values[entry]) * p for (entry, exit_, _, _, _), p in zip(trades, rand_pos)])
        rand_returns.append(rand_ret)
    pct_mr = 100 * np.mean(np.array(rand_returns) < exp_R)
    print(f"pctMR: {pct_mr:.1f}%")
    verdict = "PRZESZEDL" if pct_mr >= 95 else "odrzucony"
    print(f"WERDYKT: {verdict}")

if __name__ == "__main__":
    for a1, a2 in PAIRS:
        for tf in TIMEFRAMES:
            run_test(a1, a2, tf)
