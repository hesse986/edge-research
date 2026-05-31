"""
Grid Search dla Pairs Trading – własna implementacja bez vectorbt.
Testuje setki kombinacji parametrów na danych historycznych.
Używa walk-forward validation żeby uniknąć overfittingu.
"""
import sys, os
sys.path.insert(0, '.')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
from itertools import product
from pathlib import Path
import data as datamod
from hedge import compute_spread, beta_window_for
import csv
from datetime import datetime

# ============================================================
# PARAMETRY DO PRZETESTOWANIA
# ============================================================
PARAM_GRID = {
    "entry_z":  [1.5, 2.0, 2.5, 3.0],
    "exit_z":   [0.3, 0.5, 0.8],
    "lookback": [20, 30, 40, 60],
    "sl_mult":  [1.0, 1.5, 2.0],
}

PAIRS = [
    ("LTC/USDT", "XRP/USDT"),
    ("LTC/USDT", "ADA/USDT"),
    ("XRP/USDT", "ADA/USDT"),
]

CALIB_START   = "2023-01-01"
CALIB_END     = "2024-01-01"
VAL_START     = "2024-01-01"
VAL_END       = "2025-01-01"
FORWARD_START = "2025-01-01"
FORWARD_END   = "2026-05-01"
COST          = 0.0002
RR            = 2.0
MIN_TRADES    = 20

# ============================================================
# SILNIK BACKTESTU
# ============================================================
def run_backtest(spread, entry_z, exit_z, lookback, sl_mult, cost=COST, rr=RR):
    mean   = spread.rolling(lookback).mean()
    std    = spread.rolling(lookback).std()
    zscore = (spread - mean) / std
    atr    = (spread.rolling(14).max() - spread.rolling(14).min()) / 14

    pos = 0; entry_z_val = 0; entry_spread = 0; trades = []
    for i in range(lookback + 5, len(spread)):
        z = zscore.iloc[i]
        if pd.isna(z): continue
        if pos == 0:
            if z > entry_z:
                pos = -1; entry_z_val = z; entry_spread = spread.iloc[i]
            elif z < -entry_z:
                pos = 1;  entry_z_val = z; entry_spread = spread.iloc[i]
        else:
            sl = atr.iloc[i] * sl_mult
            risk = sl if sl > 0 else 0.02 * abs(entry_spread)
            ret  = (spread.iloc[i] - entry_spread) * pos
            exit_cond = (pos == 1 and z > -exit_z) or (pos == -1 and z < exit_z)
            if exit_cond:
                r = ret / risk - cost if risk > 0 else 0
                trades.append(r); pos = 0
    return trades

def compute_metrics(trades):
    if len(trades) < MIN_TRADES:
        return None
    r = np.array(trades)
    exp_R = float(r.mean())
    sr    = float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0
    pf    = float(r[r>0].sum() / abs(r[r<0].sum())) if (r<0).any() else 999.0
    wr    = float((r>0).mean())
    # Bootstrap pctMR
    rng   = np.random.default_rng(42)
    mc    = [np.mean(rng.choice(r, size=len(r))) for _ in range(2000)]
    pctMR = float(100 * np.mean(np.array(mc) < exp_R))
    return {"n": len(r), "exp_R": round(exp_R,4), "sharpe": round(sr,3),
            "pf": round(pf,3), "wr": round(wr,3), "pctMR": round(pctMR,1)}

def get_full_prices(df1, df2):
    """Wyrównane ceny close na wspólnym indeksie (cały ciągły zakres)."""
    common = df1.index.intersection(df2.index)
    return df1.loc[common, "close"], df2.loc[common, "close"]

# ============================================================
# GŁÓWNA PĘTLA GRID SEARCH
# ============================================================
def main():
    print("Ładowanie danych...")
    data = {}
    all_symbols = set(s for p in PAIRS for s in p)
    for sym in all_symbols:
        df1h = datamod.load_binance_ohlcv(sym, "1h", CALIB_START, FORWARD_END, use_cache=True)
        data[sym] = datamod.resample_ohlcv(df1h, "4h")
        print(f"  {sym[:3]}: {len(data[sym])} świec 4H")

    keys   = list(PARAM_GRID.keys())
    values = list(PARAM_GRID.values())
    combos = list(product(*values))
    total  = len(combos) * len(PAIRS)
    print(f"\nGrid search: {len(combos)} kombinacji × {len(PAIRS)} par = {total} testów\n")

    results = []
    done = 0
    for sym1, sym2 in PAIRS:
        # Ciągłe ceny na wspólnym indeksie. Spread liczymy raz na całym szeregu
        # (rolling OLS, beta out-of-sample), a DOPIERO POTEM tniemy na okresy —
        # dzięki temu val/forward dostają warmup bety z danych przeszłych,
        # a beta nigdy nie podgląda przyszłości (audyt 3.1).
        p1_full, p2_full = get_full_prices(data[sym1], data[sym2])

        for combo in combos:
            params = dict(zip(keys, combo))
            ez, xz, lb, sl = params["entry_z"], params["exit_z"], params["lookback"], params["sl_mult"]

            # Okno bety zależy od lookbacku z-score (4×), więc spread per kombinacja.
            spread_full = compute_spread(p1_full, p2_full, beta_window_for(lb))
            idx = spread_full.index
            spread_cal = spread_full[idx < CALIB_END]
            spread_val = spread_full[(idx >= VAL_START) & (idx < VAL_END)]
            spread_fwd = spread_full[idx >= FORWARD_START]

            t_cal = run_backtest(spread_cal, ez, xz, lb, sl)
            t_val = run_backtest(spread_val, ez, xz, lb, sl)
            t_fwd = run_backtest(spread_fwd, ez, xz, lb, sl)

            m_cal = compute_metrics(t_cal)
            m_val = compute_metrics(t_val)
            m_fwd = compute_metrics(t_fwd)

            if m_cal and m_val:
                results.append({
                    "pair":     f"{sym1[:3]}/{sym2[:3]}",
                    "entry_z":  ez, "exit_z": xz,
                    "lookback": lb, "sl_mult": sl,
                    "cal_expR":  m_cal["exp_R"],  "cal_pctMR": m_cal["pctMR"],
                    "val_expR":  m_val["exp_R"],  "val_pctMR": m_val["pctMR"],
                    "val_n":     m_val["n"],
                    "fwd_expR":  m_fwd["exp_R"] if m_fwd else None,
                    "fwd_pctMR": m_fwd["pctMR"] if m_fwd else None,
                })

            done += 1
            if done % 50 == 0:
                print(f"  {done}/{total} testów...")

    df = pd.DataFrame(results)
    print(f"\nZakończono {len(df)} testów z wynikami.")

    # SELEKCJA WYŁĄCZNIE NA WALIDACJI (audyt 3.2).
    # Forward MUSI zostać out-of-sample, więc NIE filtrujemy po fwd_expR — forward
    # jest tylko raportowany obok, raz, i nie wpływa na wybór konfiguracji.
    df_valid = df[df["val_pctMR"] >= 90].copy()
    df_valid = df_valid.sort_values(["val_pctMR", "val_expR"], ascending=False)

    print(f"\n=== TOP 10 konfiguracji (selekcja: val_pctMR>=90%) ===")
    print("(fwd_R pokazany TYLKO informacyjnie — nie był kryterium wyboru)")
    print(f"{'Para':<10} {'entry_z':<8} {'exit_z':<7} {'lb':<4} {'sl':<4} {'cal_R':<7} {'val_pctMR':<10} {'fwd_R':<7}")
    print("-" * 65)
    for _, row in df_valid.head(10).iterrows():
        fwd = f"{row['fwd_expR']:<7.4f}" if pd.notna(row['fwd_expR']) else f"{'n/a':<7}"
        print(f"{row['pair']:<10} {row['entry_z']:<8} {row['exit_z']:<7} {row['lookback']:<4} {row['sl_mult']:<4} "
              f"{row['cal_expR']:<7.4f} {row['val_pctMR']:<10.1f} {fwd}")

    # Zapisz wyniki
    out = Path("trading_system/research/grid_search_results.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nWszystkie wyniki zapisane: {out}")

    # Porównaj z obecnymi parametrami
    current = df[(df["entry_z"]==2.0) & (df["exit_z"]==0.5) & (df["lookback"]==30) & (df["sl_mult"]==1.5)]
    if not current.empty:
        print(f"\n=== Obecne parametry (entry=2.0, exit=0.5, lb=30, sl=1.5) ===")
        for _, row in current.iterrows():
            print(f"{row['pair']}: val_pctMR={row['val_pctMR']}% fwd_R={row['fwd_expR']}")

if __name__ == "__main__":
    main()
