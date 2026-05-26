#!/usr/bin/env python3
"""Paper trading monitor dla par z zapisem R i symulacją kapitału."""

import time
import csv
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys
sys.path.insert(0, '.')

import data as datamod
from sklearn.linear_model import LinearRegression
import sys
sys.path.insert(0, '.')
from trading_system.risk.risk_agent import position_size_pairs, CAPITAL

# ==================================================
# KONFIGURACJA
# ==================================================
HISTORY_START = "2023-01-01"
LOOKBACK = 30
ENTRY_Z = 2.0
EXIT_Z = 0.5
SL_MULT = 1.5
RR = 2.0
COST = 0.0002
CHECK_EVERY = 60 * 60 * 4
BASE_LOG = Path("paper_trades_pairs_advanced_R.csv")

PAIRS = [
    ("LTC/USDT",  "ADA/USDT",  "ltc_ada"),
    ("ADA/USDT",  "LINK/USDT", "ada_link"),
    ("BNB/USDT",  "SOL/USDT",  "bnb_sol"),
]

HEDGE_RATIOS = {}

def compute_hedge_ratio(asset1, asset2, start, end):
    df1 = datamod.load_binance_ohlcv(asset1, "1h", start, end, use_cache=True)
    df2 = datamod.load_binance_ohlcv(asset2, "1h", start, end, use_cache=True)
    df1_4h = datamod.resample_ohlcv(df1, "4h")
    df2_4h = datamod.resample_ohlcv(df2, "4h")
    common = df1_4h.index.intersection(df2_4h.index)
    if len(common) < LOOKBACK + 5:
        raise ValueError(f"Za mało wspólnych świec dla {asset1}/{asset2}")
    close1 = df1_4h.loc[common, "close"]
    close2 = df2_4h.loc[common, "close"]
    model = LinearRegression().fit(close2.values.reshape(-1,1), close1.values)
    return model.coef_[0]

def get_spread_and_zscore(asset1, asset2, hedge_ratio, lookback=LOOKBACK):
    now = datetime.now(timezone.utc)
    end = now.strftime("%Y-%m-%d")
    start = (now - pd.Timedelta(days=lookback*2)).strftime("%Y-%m-%d")
    df1 = datamod.load_binance_ohlcv(asset1, "1h", start, end, use_cache=False)
    df2 = datamod.load_binance_ohlcv(asset2, "1h", start, end, use_cache=False)
    df1_4h = datamod.resample_ohlcv(df1, "4h")
    df2_4h = datamod.resample_ohlcv(df2, "4h")
    common = df1_4h.index.intersection(df2_4h.index)
    if len(common) < lookback + 5:
        return None, None, None, None
    close1 = df1_4h.loc[common, "close"]
    close2 = df2_4h.loc[common, "close"]
    spread = close1 - hedge_ratio * close2
    rolling_mean = spread.rolling(lookback).mean()
    rolling_std = spread.rolling(lookback).std()
    zscore = (spread - rolling_mean) / rolling_std
    last_spread = spread.iloc[-1]
    last_z = zscore.iloc[-1]
    atr_spread = (spread.rolling(14).max() - spread.rolling(14).min()) / 14
    last_atr = atr_spread.iloc[-1] if not np.isnan(atr_spread.iloc[-1]) else atr_spread.iloc[-2]
    return last_spread, last_z, spread.index[-1], last_atr

def init_log():
    if not BASE_LOG.exists():
        with BASE_LOG.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "pair", "direction", "z_score", "spread",
                             "entry_price", "sl", "tp", "status", "exit_price", "result_R", "notes"])

def main():
    # Stałe hedge ratios z kalibracji 2023-2024
    HEDGE_RATIOS["ltc_ada"]  =  19.0200
    HEDGE_RATIOS["ada_link"] =   0.0199
    HEDGE_RATIOS["bnb_sol"]  =   0.0393
    for name, hedge in HEDGE_RATIOS.items():
        print(f"Hedge ratio {name}: {hedge:.4f}")

    init_log()
    print(f"Paper trading monitor dla par: {[p[2] for p in PAIRS]}")
    print(f"Entry Z: {ENTRY_Z}, Exit Z: {EXIT_Z}, SL_MULT: {SL_MULT}, RR: {RR}")
    print(f"Log: {BASE_LOG}")
    print("Monitoring co 4h...\n")

    positions = {name: 0 for _, _, name in PAIRS}
    entry_data = {name: None for _, _, name in PAIRS}
    # Circuit breaker: ile kolejnych strat na parę
    consecutive_losses = {name: 0 for _, _, name in PAIRS}
    circuit_breaker_until = {name: None for _, _, name in PAIRS}  # datetime do kiedy zablokowane
    CB_MAX_LOSSES = 3      # ile kolejnych strat blokuje
    CB_COOLDOWN_H = 24     # ile godzin blokady

    while True:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
        for a1, a2, name in PAIRS:
            hedge = HEDGE_RATIOS[name]
            spread, z, bar_time, atr = get_spread_and_zscore(a1, a2, hedge)
            if spread is None or np.isnan(z):
                print(f"[{now}] {name}: brak danych")
                continue
            print(f"[{now}] {name}: Z={z:.3f}, spread={spread:.2f}, atr={atr:.2f}")

            pos = positions[name]
            # Sprawdź circuit breaker
            cb_until = circuit_breaker_until[name]
            if cb_until is not None:
                if datetime.now(timezone.utc) < cb_until:
                    print(f"  ⚠️  {name}: CIRCUIT BREAKER aktywny do {cb_until.strftime('%H:%M UTC')}")
                    continue
                else:
                    circuit_breaker_until[name] = None
                    consecutive_losses[name] = 0
                    print(f"  ✅ {name}: circuit breaker wygasł")
            if pos == 0:
                if z > ENTRY_Z:
                    pos = -1
                    entry_price = spread
                    sl = entry_price - SL_MULT * atr
                    tp = entry_price + RR * SL_MULT * atr
                    direction_str = f"SHORT_{a1.split('/')[0]}_LONG_{a2.split('/')[0]}"
                    sizing = position_size_pairs(a1, a2, HEDGE_RATIOS[name], CAPITAL)
                    entry_data[name] = {"entry_time": now, "entry_z": z, "entry_spread": entry_price, "sl": sl, "tp": tp, "sizing": sizing}
                    with BASE_LOG.open("a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([now, name, direction_str, f"{z:.3f}", f"{spread:.2f}",
                                         f"{entry_price:.2f}", f"{sl:.2f}", f"{tp:.2f}", "OPEN", "", "",
                                         f"units={sizing['units_symbol1']:.4f} usd={sizing['usd_per_leg']:.1f}"])
                    print(f"  *** SYGNAŁ: {direction_str} (Z={z:.2f}) | {sizing['units_symbol1']:.4f} {a1[:3]} | ${sizing['usd_per_leg']:.1f}")
                elif z < -ENTRY_Z:
                    pos = 1
                    entry_price = spread
                    sl = entry_price + SL_MULT * atr
                    tp = entry_price - RR * SL_MULT * atr
                    direction_str = f"LONG_{a1.split('/')[0]}_SHORT_{a2.split('/')[0]}"
                    sizing = position_size_pairs(a1, a2, HEDGE_RATIOS[name], CAPITAL)
                    entry_data[name] = {"entry_time": now, "entry_z": z, "entry_spread": entry_price, "sl": sl, "tp": tp, "sizing": sizing}
                    with BASE_LOG.open("a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([now, name, direction_str, f"{z:.3f}", f"{spread:.2f}",
                                         f"{entry_price:.2f}", f"{sl:.2f}", f"{tp:.2f}", "OPEN", "", "",
                                         f"units={sizing['units_symbol1']:.4f} usd={sizing['usd_per_leg']:.1f}"])
                    print(f"  *** SYGNAŁ: {direction_str} (Z={z:.2f}) | {sizing['units_symbol1']:.4f} {a1[:3]} | ${sizing['usd_per_leg']:.1f}")
            else:
                ed = entry_data[name]
                exit_reason = None
                if pos == 1:
                    if spread <= ed["sl"]:
                        exit_reason = "SL"
                    elif spread >= ed["tp"]:
                        exit_reason = "TP"
                    elif z > -EXIT_Z:
                        exit_reason = "EXIT_Z"
                else:  # pos == -1
                    if spread >= ed["sl"]:
                        exit_reason = "SL"
                    elif spread <= ed["tp"]:
                        exit_reason = "TP"
                    elif z < EXIT_Z:
                        exit_reason = "EXIT_Z"
                if exit_reason:
                    direction_str = f"LONG_{a1.split('/')[0]}_SHORT_{a2.split('/')[0]}" if pos == 1 else f"SHORT_{a1.split('/')[0]}_LONG_{a2.split('/')[0]}"
                    # Oblicz R
                    risk = abs(ed["entry_spread"] - ed["sl"])
                    if risk <= 0:
                        risk = 0.02 * abs(ed["entry_spread"])
                    ret = (spread - ed["entry_spread"]) * (1 if pos == 1 else -1)
                    r = ret / risk
                    with BASE_LOG.open("a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([now, name, direction_str, f"{z:.3f}", f"{spread:.2f}",
                                         "", "", "", "CLOSED", f"{spread:.2f}", f"{r:.4f}", exit_reason])
                    print(f"  *** ZAMKNIĘCIE: {direction_str} przyczyna={exit_reason}, R={r:.2f}")
                    # Aktualizuj circuit breaker
                    if r < -0.5:
                        consecutive_losses[name] += 1
                        if consecutive_losses[name] >= CB_MAX_LOSSES:
                            circuit_breaker_until[name] = datetime.now(timezone.utc) + timedelta(hours=CB_COOLDOWN_H)
                            print(f"  🔴 {name}: CIRCUIT BREAKER aktywowany ({CB_MAX_LOSSES} straty z rzędu) – blokada {CB_COOLDOWN_H}H")
                    else:
                        consecutive_losses[name] = 0
                    pos = 0
                    entry_data[name] = None
            positions[name] = pos
        time.sleep(CHECK_EVERY)

if __name__ == "__main__":
    main()
