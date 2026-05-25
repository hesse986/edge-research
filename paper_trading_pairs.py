#!/usr/bin/env python3
"""Paper trading monitor dla pary LTC/XRP – co 4h sprawdza Z-score i generuje sygnały."""

import time
import csv
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import sys
sys.path.insert(0, '.')

import data as datamod
from sklearn.linear_model import LinearRegression

# ==================================================
# KONFIGURACJA – zgodna z backtestem
# ==================================================
ASSETS = ["LTC/USDT", "XRP/USDT"]
TF = "4h"
LOOKBACK = 30           # okno do rolling mean/std
ENTRY_Z = 2.0
EXIT_Z = 0.5
SL_MULT = 1.5           # opcjonalnie, do risk management
RR = 2.0
COST = 0.0002
HISTORY_START = "2023-01-01"   # do kalibracji hedge ratio
LOG_FILE = Path("paper_trades_pairs.csv")
CHECK_EVERY = 60 * 60 * 4       # co 4h

# Kalibracja hedge ratio (stała, z okresu 2023-2024)
def compute_hedge_ratio():
    """Oblicza hedge ratio na podstawie danych z okresu kalibracji."""
    df_ltc = datamod.load_binance_ohlcv("LTC/USDT", "1h", HISTORY_START, "2024-01-01", use_cache=True)
    df_xrp = datamod.load_binance_ohlcv("XRP/USDT", "1h", HISTORY_START, "2024-01-01", use_cache=True)
    df_ltc_4h = datamod.resample_ohlcv(df_ltc, "4h")
    df_xrp_4h = datamod.resample_ohlcv(df_xrp, "4h")
    common = df_ltc_4h.index.intersection(df_xrp_4h.index)
    ltc = df_ltc_4h.loc[common, "close"]
    xrp = df_xrp_4h.loc[common, "close"]
    model = LinearRegression().fit(xrp.values.reshape(-1,1), ltc.values)
    return model.coef_[0]

HEDGE_RATIO = compute_hedge_ratio()
print(f"Hedge ratio (LTC/XRP): {HEDGE_RATIO:.4f}")

def get_current_spread_and_zscore():
    """Pobiera ostatnie dane i oblicza aktualny spread i Z-score."""
    end = datetime.utcnow().strftime("%Y-%m-%d")
    start = (datetime.utcnow() - pd.Timedelta(days=LOOKBACK*2)).strftime("%Y-%m-%d")
    df_ltc = datamod.load_binance_ohlcv("LTC/USDT", "1h", start, end, use_cache=False)
    df_xrp = datamod.load_binance_ohlcv("XRP/USDT", "1h", start, end, use_cache=False)
    df_ltc_4h = datamod.resample_ohlcv(df_ltc, "4h")
    df_xrp_4h = datamod.resample_ohlcv(df_xrp, "4h")
    common = df_ltc_4h.index.intersection(df_xrp_4h.index)
    if len(common) < LOOKBACK + 5:
        return None, None, None
    ltc = df_ltc_4h.loc[common, "close"]
    xrp = df_xrp_4h.loc[common, "close"]
    spread = ltc - HEDGE_RATIO * xrp
    rolling_mean = spread.rolling(LOOKBACK).mean()
    rolling_std = spread.rolling(LOOKBACK).std()
    zscore = (spread - rolling_mean) / rolling_std
    last_spread = spread.iloc[-1]
    last_z = zscore.iloc[-1]
    return last_spread, last_z, spread.index[-1]

def init_log():
    if not LOG_FILE.exists():
        with LOG_FILE.open("w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["timestamp", "asset_pair", "direction", "z_score", "spread",
                             "entry", "sl", "tp", "status", "exit_price", "result_R", "notes"])

def main():
    init_log()
    print(f"Paper trading monitor dla pary LTC/XRP (4h). Hedge ratio: {HEDGE_RATIO:.4f}")
    print(f"Entry Z: {ENTRY_Z}, Exit Z: {EXIT_Z}")
    print(f"Log: {LOG_FILE}")
    print("Monitoring co 4h...\n")
    position = 0   # 1 = long LTC (short XRP), -1 = short LTC (long XRP), 0 = flat
    entry_time = None
    entry_z = None
    entry_spread = None
    while True:
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        spread, z, bar_time = get_current_spread_and_zscore()
        if spread is None or np.isnan(z):
            print(f"[{now}] Brak danych lub za mało świec. Czekam...")
            time.sleep(CHECK_EVERY)
            continue
        print(f"[{now}] Z-score: {z:.3f}, spread: {spread:.2f}")

        if position == 0:
            if z > ENTRY_Z:
                position = -1   # short LTC, long XRP
                entry_time = now
                entry_z = z
                entry_spread = spread
                # Uproszczone SL/TP – docelowo można dodać, ale na razie tylko log
                sl = spread - SL_MULT * (spread * 0.02)   # placeholder
                tp = spread + RR * (spread * 0.02)
                with LOG_FILE.open("a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([now, "LTC/XRP", "SHORT_LTC_LONG_XRP", f"{z:.3f}", f"{spread:.2f}",
                                     f"{spread:.2f}", f"{sl:.2f}", f"{tp:.2f}", "OPEN", "", "", "paper_trade"])
                print(f"  *** SYGNAŁ: SHORT LTC / LONG XRP (Z={z:.2f})")
            elif z < -ENTRY_Z:
                position = 1    # long LTC, short XRP
                entry_time = now
                entry_z = z
                entry_spread = spread
                with LOG_FILE.open("a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([now, "LTC/XRP", "LONG_LTC_SHORT_XRP", f"{z:.3f}", f"{spread:.2f}",
                                     f"{spread:.2f}", "", "", "OPEN", "", "", "paper_trade"])
                print(f"  *** SYGNAŁ: LONG LTC / SHORT XRP (Z={z:.2f})")
        else:
            # Sprawdź wyjście: powrót do EXIT_Z lub przeciwny sygnał
            exit_signal = False
            if position == 1 and z > -EXIT_Z:
                exit_signal = True
            elif position == -1 and z < EXIT_Z:
                exit_signal = True
            if exit_signal:
                # Zamknij pozycję
                direction_str = "LONG_LTC_SHORT_XRP" if position == 1 else "SHORT_LTC_LONG_XRP"
                with LOG_FILE.open("a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([now, "LTC/XRP", direction_str, f"{z:.3f}", f"{spread:.2f}",
                                     "", "", "", "CLOSED", f"{spread:.2f}", "0", "exit_z_threshold"])
                print(f"  *** ZAMKNIĘCIE: {direction_str} przy Z={z:.2f}")
                position = 0
                entry_time = None
            # Opcjonalnie: można dodać SL/TP na podstawie spreadu – na razie pomijamy
        time.sleep(CHECK_EVERY)

if __name__ == "__main__":
    main()
