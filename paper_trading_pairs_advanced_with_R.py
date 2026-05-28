#!/usr/bin/env python3
"""Paper trading monitor dla par z zapisem R i symulacją kapitału."""

import time
import csv
import numpy as np
import pandas as pd
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys
import argparse
sys.path.insert(0, '.')

import data as datamod
from sklearn.linear_model import LinearRegression
import sys
import argparse

def get_zscore_live(sym1, sym2, hedge, lookback=30):
    """Pobiera live Z-score identycznie jak dashboard - ostatnie 40 świec 4H."""
    import requests as _req
    needed = lookback + 10
    url = "https://api.binance.com/api/v3/klines"
    try:
        r1 = _req.get(url, params={"symbol": sym1.replace("/",""), "interval": "4h", "limit": needed}, timeout=10)
        r2 = _req.get(url, params={"symbol": sym2.replace("/",""), "interval": "4h", "limit": needed}, timeout=10)
        p1 = [float(c[4]) for c in r1.json()]
        p2 = [float(c[4]) for c in r2.json()]
        if len(p1) < lookback or len(p2) < lookback:
            return None, None, None
        spread = [p1[i] - hedge * p2[i] for i in range(len(p1))]
        import numpy as _np
        mean = _np.mean(spread[-lookback:])
        std  = _np.std(spread[-lookback:])
        z    = (spread[-1] - mean) / std if std > 0 else 0
        # ATR spreadu
        atr  = _np.mean([abs(spread[i]-spread[i-1]) for i in range(1, len(spread))])
        return float(z), float(spread[-1]), float(atr)
    except Exception as e:
        print(f"  Błąd get_zscore_live: {e}")
        return None, None, None


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
CHECK_EVERY = 3 * 60
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
    """Używa tej samej metody co dashboard - live dane z Binance API."""
    z, spread, atr = get_zscore_live(asset1, asset2, hedge_ratio, lookback)
    if z is None:
        return None, None, None, None
    now = datetime.now(timezone.utc)
    return spread, z, now, atr if atr else 0.0

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
    print("Monitoring co 3 minuty...\n")

    positions = {name: 0 for _, _, name in PAIRS}
    entry_data = {name: None for _, _, name in PAIRS}
    # Circuit breaker: ile kolejnych strat na parę
    consecutive_losses = {name: 0 for _, _, name in PAIRS}
    circuit_breaker_until = {name: None for _, _, name in PAIRS}
    CB_MAX_LOSSES = 3
    CB_COOLDOWN_H = 24

    # Wczytaj stan otwartych pozycji z CSV
    import csv as _csv
    if BASE_LOG.exists():
        with open(BASE_LOG) as _f:
            for _row in _csv.DictReader(_f):
                _name = _row.get("pair","")
                if _row.get("status") == "OPEN" and _name in positions:
                    positions[_name] = 1 if "LONG" in _row.get("direction","") else -1
                    # Przelicz SL/TP na podstawie aktualnego spreadu
                    _entry = float(_row.get("entry_price", 0) or _row.get("spread", 0))
                    _sl    = float(_row.get("sl", 0))
                    _tp    = float(_row.get("tp", 0))
                    entry_data[_name] = {
                        "entry_time":   _row.get("timestamp",""),
                        "entry_z":      float(_row.get("z_score", 0)),
                        "entry_spread": _entry,
                        "sl":           _sl,
                        "tp":           _tp,
                        "sizing":       {},
                    }
                    print(f"  Wczytano pozycję: {_name} entry={_entry:.6f} sl={_sl:.6f} tp={_tp:.6f}")

    while True:  # RUN_ONCE sprawdzane na końcu
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
                    # Minimalny ATR = 1% spreadu żeby uniknąć SL=entry
                    atr_safe = max(atr, abs(entry_price) * 0.01)
                    sl = entry_price - SL_MULT * atr_safe
                    tp = entry_price + RR * SL_MULT * atr_safe
                    direction_str = f"SHORT_{a1.split('/')[0]}_LONG_{a2.split('/')[0]}"
                    sizing = position_size_pairs(a1, a2, HEDGE_RATIOS[name], CAPITAL)
                    entry_data[name] = {"entry_time": now, "entry_z": z, "entry_spread": entry_price, "sl": sl, "tp": tp, "sizing": sizing}
                    with BASE_LOG.open("a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([now, name, direction_str, f"{z:.3f}", f"{spread:.6f}",
                                         f"{entry_price:.6f}", f"{sl:.6f}", f"{tp:.6f}", "OPEN", "", "",
                                         f"units={sizing['units_symbol1']:.4f} usd={sizing['usd_per_leg']:.1f}"])
                    print(f"  *** SYGNAŁ: {direction_str} (Z={z:.2f}) | {sizing['units_symbol1']:.4f} {a1[:3]} | ${sizing['usd_per_leg']:.1f}")
                elif z < -ENTRY_Z:
                    pos = 1
                    entry_price = spread
                    atr_safe = max(atr, abs(entry_price) * 0.01)
                    sl = entry_price + SL_MULT * atr_safe
                    tp = entry_price - RR * SL_MULT * atr_safe
                    direction_str = f"LONG_{a1.split('/')[0]}_SHORT_{a2.split('/')[0]}"
                    sizing = position_size_pairs(a1, a2, HEDGE_RATIOS[name], CAPITAL)
                    entry_data[name] = {"entry_time": now, "entry_z": z, "entry_spread": entry_price, "sl": sl, "tp": tp, "sizing": sizing}
                    with BASE_LOG.open("a", newline="") as f:
                        writer = csv.writer(f)
                        writer.writerow([now, name, direction_str, f"{z:.3f}", f"{spread:.6f}",
                                         f"{entry_price:.6f}", f"{sl:.6f}", f"{tp:.6f}", "OPEN", "", "",
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
                    # Oblicz faktyczne R z kosztami
                    risk = abs(ed["entry_spread"] - ed["sl"])
                    if risk <= 0:
                        risk = 0.02 * abs(ed["entry_spread"])
                    ret  = (spread - ed["entry_spread"]) * (1 if pos == 1 else -1)
                    cost = 0.0002 * abs(ed["entry_spread"]) / risk  # koszty transakcyjne w R
                    r    = ret / risk - cost
                    # Aktualizuj oryginalny wpis OPEN na CLOSED
                    _rows = []
                    _updated = False
                    if BASE_LOG.exists():
                        with open(BASE_LOG, newline="") as _rf:
                            _reader = csv.DictReader(_rf)
                            _fields = _reader.fieldnames
                            for _row in _reader:
                                if (_row["pair"] == name and
                                    _row["status"] == "OPEN" and
                                    not _updated):
                                    _row["status"]     = "CLOSED"
                                    _row["exit_price"] = f"{spread:.6f}"
                                    _row["result_R"]   = f"{r:.4f}"
                                    _row["notes"]      = exit_reason
                                    _updated = True
                                _rows.append(_row)
                    if _updated and _rows:
                        with open(BASE_LOG, "w", newline="") as _wf:
                            _writer = csv.DictWriter(_wf, fieldnames=_fields)
                            _writer.writeheader()
                            _writer.writerows(_rows)
                    else:
                        with BASE_LOG.open("a", newline="") as f:
                            writer = csv.writer(f)
                            writer.writerow([now, name, direction_str, f"{z:.3f}",
                                             f"{spread:.6f}", "", "", "", "CLOSED",
                                             f"{spread:.6f}", f"{r:.4f}", exit_reason])
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
