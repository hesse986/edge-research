#!/usr/bin/env python3
"""Pairs trading LTC/XRP – mean reversion spreadu z kointegracją."""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '.')

import data as datamod
from sklearn.linear_model import LinearRegression

START = "2023-01-01"
END   = "2025-06-01"
TF    = "4h"
LOOKBACK = 30
ENTRY_Z = 2.0
EXIT_Z = 0.5
MIN_TRADES = 20
COST = 0.0002

def load_asset(symbol):
    df1h = datamod.load_binance_ohlcv(symbol, "1h", START, END, use_cache=True)
    df = datamod.resample_ohlcv(df1h, TF)
    return df

def simulate_spread_trades(zscore, hedge_ratio):
    """Symulacja mean reversion: wejście gdy |z| > ENTRY_Z, wyjście gdy |z| < EXIT_Z."""
    trades = []
    position = 0
    entry_idx = None
    entry_z = None
    # użyj pozycyjnego indeksu
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
                entry_idx = None
    return trades

def main():
    print("=== Pairs Trading LTC/XRP ===")
    print(f"Okres: {START} -> {END}, TF={TF}")
    df_ltc = load_asset("LTC/USDT")
    df_xrp = load_asset("XRP/USDT")
    common_idx = df_ltc.index.intersection(df_xrp.index)
    ltc = df_ltc.loc[common_idx, "close"]
    xrp = df_xrp.loc[common_idx, "close"]
    # hedge ratio
    model = LinearRegression().fit(xrp.values.reshape(-1,1), ltc.values)
    hedge = model.coef_[0]
    spread = ltc - hedge * xrp
    rolling_mean = spread.rolling(LOOKBACK).mean()
    rolling_std = spread.rolling(LOOKBACK).std()
    zscore = (spread - rolling_mean) / rolling_std
    # symulacja
    trades = simulate_spread_trades(zscore, hedge)
    print(f"Liczba transakcji: {len(trades)}")
    if len(trades) < MIN_TRADES:
        print("ZA MAŁO TRADOW – nie można wyciągnąć wniosków.")
        return
    # uproszczone R = zmiana z-score
    results = []
    for entry, exit_, pos, z_entry, z_exit in trades:
        ret = (z_exit - z_entry) * pos
        results.append(ret)
    exp_R = np.mean(results)
    print(f"exp_R (uproszczone): {exp_R:.4f}")
    # benchmark losowy
    rng = np.random.default_rng(0)
    rand_returns = []
    for _ in range(5000):
        rand_pos = rng.choice([-1,1], size=len(trades))
        rand_ret = np.mean([(zscore.values[exit_] - zscore.values[entry]) * p for (entry, exit_, _, _, _), p in zip(trades, rand_pos)])
        rand_returns.append(rand_ret)
    pct_mr = 100 * np.mean(np.array(rand_returns) < exp_R)
    print(f"pctMR (benchmark losowy): {pct_mr:.1f}%")
    if pct_mr >= 95:
        print("✅ PRZESZEDŁ – edge statystycznie istotny")
    else:
        print("❌ ODRZUCONY – brak przewagi")

if __name__ == "__main__":
    main()
