#!/usr/bin/env python3
"""Paper trading monitor dla par z zapisem R i symulacją kapitału."""

import time
import csv
import numpy as np
import requests
from datetime import datetime, timezone, timedelta
from pathlib import Path
import sys
sys.path.insert(0, '.')

# ==================================================
# KONFIGURACJA
# ==================================================
LOOKBACK   = 30
ENTRY_Z    = 2.0
EXIT_Z     = 0.5
SL_MULT    = 1.5
RR         = 2.0
COST       = 0.0002
CHECK_EVERY = 3 * 60   # 3 minuty

BASE_LOG = Path("paper_trades_pairs_advanced_R.csv")

PAIRS = [
    ("LTC/USDT",  "ADA/USDT",  "ltc_ada",  19.0200),
    ("ADA/USDT",  "LINK/USDT", "ada_link",  0.0199),
    ("BNB/USDT",  "SOL/USDT",  "bnb_sol",   0.0393),
]

CB_MAX_LOSSES = 3
CB_COOLDOWN_H = 24

# ==================================================
# POBIERANIE DANYCH
# ==================================================
def get_zscore_live(sym1, sym2, hedge, lookback=LOOKBACK):
    """Live Z-score z Binance API (identycznie jak dashboard)."""
    needed = lookback + 10
    url = "https://api.binance.com/api/v3/klines"
    try:
        r1 = requests.get(url, params={"symbol": sym1.replace("/",""), "interval": "4h", "limit": needed}, timeout=10)
        r2 = requests.get(url, params={"symbol": sym2.replace("/",""), "interval": "4h", "limit": needed}, timeout=10)
        p1 = [float(c[4]) for c in r1.json()]
        p2 = [float(c[4]) for c in r2.json()]
        if len(p1) < lookback or len(p2) < lookback:
            return None, None, None
        spread = [p1[i] - hedge * p2[i] for i in range(len(p1))]
        mean = np.mean(spread[-lookback:])
        std  = np.std(spread[-lookback:])
        z    = (spread[-1] - mean) / std if std > 0 else 0
        atr  = np.mean([abs(spread[i] - spread[i-1]) for i in range(1, len(spread))])
        return float(z), float(spread[-1]), float(atr)
    except Exception as e:
        print(f"  Błąd get_zscore_live ({sym1}/{sym2}): {e}")
        return None, None, None

# ==================================================
# LOG CSV
# ==================================================
FIELDNAMES = ["timestamp","pair","direction","z_score","spread",
              "entry_price","sl","tp","status","exit_price","result_R","notes"]

def init_log():
    if not BASE_LOG.exists():
        with BASE_LOG.open("w", newline="") as f:
            csv.writer(f).writerow(FIELDNAMES)

def load_open_positions():
    """Wczytaj otwarte pozycje z CSV."""
    if not BASE_LOG.exists():
        return {}
    open_pos = {}
    with open(BASE_LOG, newline="") as f:
        for row in csv.DictReader(f):
            if row["status"] == "OPEN":
                name = row["pair"]
                open_pos[name] = {
                    "direction":    row["direction"],
                    "entry_time":   row["timestamp"],
                    "entry_z":      float(row["z_score"] or 0),
                    "entry_spread": float(row["entry_price"] or row["spread"] or 0),
                    "sl":           float(row["sl"] or 0),
                    "tp":           float(row["tp"] or 0),
                }
    return open_pos

def write_open(now, name, direction, z, spread, entry, sl, tp, units, usd):
    with BASE_LOG.open("a", newline="") as f:
        csv.writer(f).writerow([
            now, name, direction, f"{z:.3f}", f"{spread:.6f}",
            f"{entry:.6f}", f"{sl:.6f}", f"{tp:.6f}",
            "OPEN", "", "", f"units={units:.4f} usd={usd:.1f}"
        ])

def close_position(name, spread, r, exit_reason):
    """Zaktualizuj wpis OPEN na CLOSED w CSV."""
    rows = []
    updated = False
    with open(BASE_LOG, newline="") as f:
        reader = csv.DictReader(f)
        fields = reader.fieldnames
        for row in reader:
            if row["pair"] == name and row["status"] == "OPEN" and not updated:
                row["status"]     = "CLOSED"
                row["exit_price"] = f"{spread:.6f}"
                row["result_R"]   = f"{r:.4f}"
                row["notes"]      = exit_reason
                updated = True
            rows.append(row)
    if updated:
        with open(BASE_LOG, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
    return updated

# ==================================================
# GŁÓWNA PĘTLA
# ==================================================
def main():
    init_log()
    print("=" * 50)
    print("PAIRS TRADING MONITOR")
    print(f"Entry Z: ±{ENTRY_Z} | Exit Z: ±{EXIT_Z}")
    print(f"SL_MULT: {SL_MULT} | RR: {RR} | Interval: {CHECK_EVERY//60}min")
    print("=" * 50)

    # Stan pozycji
    positions = {name: 0        for _,_,name,_ in PAIRS}
    entry_data = {name: None    for _,_,name,_ in PAIRS}
    cons_losses = {name: 0      for _,_,name,_ in PAIRS}
    cb_until    = {name: None   for _,_,name,_ in PAIRS}

    # Wczytaj otwarte pozycje z CSV
    open_pos = load_open_positions()
    for name, data in open_pos.items():
        pos = 1 if "LONG" in data["direction"].split("_")[0] else -1
        positions[name]  = pos
        entry_data[name] = data
        print(f"  Wczytano: {name} {'LONG' if pos==1 else 'SHORT'} "
              f"entry={data['entry_spread']:.6f} sl={data['sl']:.6f} tp={data['tp']:.6f}")

    print()

    while True:
        now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

        for a1, a2, name, hedge in PAIRS:
            # Pobierz dane
            z, spread, atr = get_zscore_live(a1, a2, hedge)
            if z is None or np.isnan(z):
                print(f"[{now}] {name}: brak danych")
                continue

            print(f"[{now}] {name}: Z={z:.3f} spread={spread:.6f} atr={atr:.6f}")

            # Circuit breaker
            if cb_until[name] is not None:
                if datetime.now(timezone.utc) < cb_until[name]:
                    print(f"  ⚠️  {name}: CB aktywny do {cb_until[name].strftime('%H:%M UTC')}")
                    continue
                else:
                    cb_until[name] = None
                    cons_losses[name] = 0
                    print(f"  ✅ {name}: circuit breaker wygasł")

            pos = positions[name]
            atr_safe = max(atr, abs(spread) * 0.005)  # min 0.5% spreadu

            # ── WEJŚCIE ──────────────────────────────
            if pos == 0:
                if z < -ENTRY_Z:
                    # LONG spread: oczekujemy wzrostu spreadu
                    # SL poniżej entry (spread spada dalej = strata)
                    # TP powyżej entry (spread rośnie = zysk)
                    sl = spread - SL_MULT * atr_safe
                    tp = spread + RR * SL_MULT * atr_safe
                    direction = f"LONG_{a1.split('/')[0]}_SHORT_{a2.split('/')[0]}"
                    positions[name]  = 1
                    entry_data[name] = {
                        "entry_spread": spread, "sl": sl, "tp": tp,
                        "entry_z": z, "entry_time": now, "direction": direction
                    }
                    write_open(now, name, direction, z, spread, spread, sl, tp,
                               abs(1.0 / atr_safe) if atr_safe > 0 else 0, spread * 100)
                    print(f"  *** LONG {name} Z={z:.2f} entry={spread:.6f} SL={sl:.6f} TP={tp:.6f}")

                elif z > ENTRY_Z:
                    # SHORT spread: oczekujemy spadku spreadu
                    # SL powyżej entry (spread rośnie dalej = strata)
                    # TP poniżej entry (spread spada = zysk)
                    sl = spread + SL_MULT * atr_safe
                    tp = spread - RR * SL_MULT * atr_safe
                    direction = f"SHORT_{a1.split('/')[0]}_LONG_{a2.split('/')[0]}"
                    positions[name]  = -1
                    entry_data[name] = {
                        "entry_spread": spread, "sl": sl, "tp": tp,
                        "entry_z": z, "entry_time": now, "direction": direction
                    }
                    write_open(now, name, direction, z, spread, spread, sl, tp,
                               abs(1.0 / atr_safe) if atr_safe > 0 else 0, spread * 100)
                    print(f"  *** SHORT {name} Z={z:.2f} entry={spread:.6f} SL={sl:.6f} TP={tp:.6f}")

            # ── WYJŚCIE ──────────────────────────────
            else:
                ed = entry_data[name]
                exit_reason = None

                if pos == 1:  # LONG – oczekujemy wzrostu spreadu
                    if spread <= ed["sl"]:
                        exit_reason = "SL"
                    elif spread >= ed["tp"]:
                        exit_reason = "TP"
                    elif z > -EXIT_Z:
                        exit_reason = "EXIT_Z"

                else:  # SHORT – oczekujemy spadku spreadu
                    if spread >= ed["sl"]:
                        exit_reason = "SL"
                    elif spread <= ed["tp"]:
                        exit_reason = "TP"
                    elif z < EXIT_Z:
                        exit_reason = "EXIT_Z"

                if exit_reason:
                    risk = abs(ed["entry_spread"] - ed["sl"])
                    if risk <= 0:
                        risk = atr_safe
                    ret = (spread - ed["entry_spread"]) * pos
                    cost_R = COST * abs(ed["entry_spread"]) / risk
                    r = ret / risk - cost_R

                    close_position(name, spread, r, exit_reason)
                    print(f"  *** ZAMKNIĘCIE {name}: {exit_reason} R={r:.3f} "
                          f"(entry={ed['entry_spread']:.6f} exit={spread:.6f})")

                    # Circuit breaker
                    if r < -0.5:
                        cons_losses[name] += 1
                        if cons_losses[name] >= CB_MAX_LOSSES:
                            cb_until[name] = datetime.now(timezone.utc) + timedelta(hours=CB_COOLDOWN_H)
                            print(f"  🔴 {name}: CB aktywowany – {CB_COOLDOWN_H}H blokada")
                    else:
                        cons_losses[name] = 0

                    positions[name]  = 0
                    entry_data[name] = None

        time.sleep(CHECK_EVERY)

if __name__ == "__main__":
    main()
