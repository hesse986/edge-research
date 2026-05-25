#!/usr/bin/env python3
"""Realistyczny backtest pairs trading LTC/XRP z SL/TP, kosztami i forward testem."""

import numpy as np
import pandas as pd
import sys
sys.path.insert(0, '.')

import data as datamod
from sklearn.linear_model import LinearRegression

# ==================================================
# KONFIGURACJA
# ==================================================
CALIB_START = "2023-01-01"
CALIB_END   = "2024-01-01"
VAL_START   = "2024-01-01"
VAL_END     = "2025-01-01"
FORWARD_START = "2025-01-01"
FORWARD_END   = "2026-05-01"
TF = "4h"
LOOKBACK = 30
ENTRY_Z = 2.0
EXIT_Z = 0.5
SL_MULT = 1.5
RR = 2.0
COST = 0.0002  # round-trip
MIN_TRADES = 20

def load_asset(symbol, start, end):
    df1h = datamod.load_binance_ohlcv(symbol, "1h", start, end, use_cache=True)
    return datamod.resample_ohlcv(df1h, TF)

def get_spread_and_zscore(df_ltc, df_xrp, lookback):
    common = df_ltc.index.intersection(df_xrp.index)
    ltc = df_ltc.loc[common, "close"]
    xrp = df_xrp.loc[common, "close"]
    model = LinearRegression().fit(xrp.values.reshape(-1,1), ltc.values)
    hedge = model.coef_[0]
    spread = ltc - hedge * xrp
    rolling_mean = spread.rolling(lookback).mean()
    rolling_std = spread.rolling(lookback).std()
    zscore = (spread - rolling_mean) / rolling_std
    return spread, zscore, hedge

def simulate_trades(spread, zscore, hedge, start_idx=0, cost=COST, sl_mult=SL_MULT, rr=RR):
    trades = []
    position = 0
    entry_bar = None
    entry_spread = None
    entry_z = None
    for i in range(start_idx+LOOKBACK+5, len(spread)):
        z = zscore.iloc[i]
        if position == 0:
            if z > ENTRY_Z:
                position = -1
                entry_bar = i
                entry_z = z
                entry_spread = spread.iloc[i]
            elif z < -ENTRY_Z:
                position = 1
                entry_bar = i
                entry_z = z
                entry_spread = spread.iloc[i]
        else:
            current_spread = spread.iloc[i]
            spread_change = (current_spread - entry_spread) / entry_spread if entry_spread != 0 else 0
            # ATR spreadu
            atr_spread = (spread.rolling(14).max() - spread.rolling(14).min()) / 14
            atr_val = atr_spread.iloc[i] if not np.isnan(atr_spread.iloc[i]) else atr_spread.iloc[i-1]
            sl_dist = sl_mult * atr_val
            tp_dist = rr * sl_dist
            if position == 1:
                hit_sl = spread_change < -sl_dist / entry_spread if entry_spread != 0 else False
                hit_tp = spread_change > tp_dist / entry_spread if entry_spread != 0 else False
            else:
                hit_sl = spread_change > sl_dist / entry_spread if entry_spread != 0 else False
                hit_tp = spread_change < -tp_dist / entry_spread if entry_spread != 0 else False
            exit_condition = (position == 1 and z > -EXIT_Z) or (position == -1 and z < EXIT_Z)
            if exit_condition or hit_sl or hit_tp:
                if hit_sl:
                    r = -1.0 - cost
                elif hit_tp:
                    r = rr - cost
                else:
                    ret = (current_spread - entry_spread) / entry_spread if entry_spread != 0 else 0
                    r = ret / (sl_dist / entry_spread) - cost
                trades.append((entry_bar, i, position, entry_z, z, r))
                position = 0
    return trades

def run_backtest(start, end, label):
    print(f"\n=== {label} ===")
    df_ltc = load_asset("LTC/USDT", start, end)
    df_xrp = load_asset("XRP/USDT", start, end)
    spread, zscore, hedge = get_spread_and_zscore(df_ltc, df_xrp, LOOKBACK)
    trades = simulate_trades(spread, zscore, hedge, start_idx=0)
    n = len(trades)
    print(f"Liczba transakcji: {n}")
    if n < MIN_TRADES:
        print("ZA MAŁO TRADOW")
        return None
    results = [t[-1] for t in trades]
    exp_R = np.mean(results)
    print(f"exp_R: {exp_R:.4f}")
    rng = np.random.default_rng(0)
    rand_returns = []
    for _ in range(5000):
        rand_results = [r * rng.choice([-1,1]) for r in results]
        rand_returns.append(np.mean(rand_results))
    pct_mr = 100 * np.mean(np.array(rand_returns) < exp_R)
    print(f"pctMR: {pct_mr:.1f}%")
    return trades

if __name__ == "__main__":
    run_backtest(CALIB_START, CALIB_END, "Kalibracja")
    run_backtest(VAL_START, VAL_END, "Walidacja")
    run_backtest(FORWARD_START, FORWARD_END, "Forward test")

def export_trades_to_csv(trades, label, filename="backtest_trades.csv"):
    import csv
    from pathlib import Path
    rows = []
    for t in trades:
        rows.append({
            "period": label,
            "entry_bar": t[0],
            "exit_bar": t[1],
            "position": t[2],
            "entry_z": t[3],
            "exit_z": t[4],
            "R": t[5] if len(t) > 5 else None
        })
    if rows:
        file_exists = Path(filename).exists()
        with open(filename, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["period","entry_bar","exit_bar","position","entry_z","exit_z","R"])
            if not file_exists:
                writer.writeheader()
            writer.writerows(rows)

# Modyfikujemy funkcję run_backtest, aby zwracała trades
# Oryginalna funkcja już zwraca trades (na końcu jest return trades). W main() trzeba je przechwycić.
