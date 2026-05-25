"""
Edge Research Runner
Uruchomienie w tle: nohup python runner.py > log.txt 2>&1 &
Wyniki: results_TIMESTAMP.csv
"""

import yaml, glob, os, sys, time, csv, traceback
import numpy as np
import pandas as pd
from datetime import datetime

import data as datamod
import edges as edgemod

# ============================================================
# STAŁE
# ============================================================
PCT_THRESHOLD   = 95.0
RANDOM_RUNS     = 3000
SL_MULT         = 1.5
RR              = 2.0
MIN_GAP         = 12

MIN_TRADES_BY_TF = {"5m": 60, "15m": 40, "1h": 30, "4h": 20, "1d": 12}
COST_BY_TF       = {"5m": 0.0008, "15m": 0.0005, "1h": 0.0003, "4h": 0.0002, "1d": 0.00015}
MAX_HOLD_BY_TF   = {"5m": 48, "15m": 32, "1h": 24, "4h": 24, "1d": 10}

VALIDATION_START = "2022-01-01"
VALIDATION_END   = "2025-01-01"

EDGE_MAP = {
    "donchian_acceptance":           edgemod.edge_donchian_acceptance,
    "breakout_retest_v2":            edgemod.edge_breakout_retest_v2,
    "failed_breakout_range_v3":      edgemod.edge_failed_breakout_range_v3,
    "sweep_reclaim_v3_continuation": edgemod.edge_sweep_reclaim_v3_continuation,
    "sweep_reclaim_v3_reversal":     edgemod.edge_sweep_reclaim_v3_reversal,
    "vwap_range_reversion":          edgemod.edge_vwap_range_reversion,
    "keltner_range_reversion":       edgemod.edge_keltner_range_reversion,
    "volume_climax_reversal":        edgemod.edge_volume_climax_reversal,
    "momentum_regime_v2":            edgemod.edge_momentum_regime_v2,
    "compression_breakout_v2":       edgemod.edge_compression_breakout_v2,
    "failed_breakout_range_v2":      edgemod.edge_failed_breakout_range_v2,
    "sweep_reclaim_v2":              edgemod.edge_sweep_reclaim_v2,
    "failed_breakout_range_v3_sweep_high": edgemod.edge_failed_breakout_range_v3_sweep_high,
    "failed_breakout_range_v3_sweep_low":  edgemod.edge_failed_breakout_range_v3_sweep_low,
    "breakout_retest_v2_short":            edgemod.edge_breakout_retest_v2_short,
    "liquidation_proxy":               edgemod.edge_liquidation_proxy,
    "relative_strength_rotation":      edgemod.edge_relative_strength_rotation,
    "funding_momentum_divergence":     edgemod.edge_funding_momentum_divergence,
    "panic_no_followthrough":          edgemod.edge_panic_no_followthrough,
    "sweep_reclaim_v1":                edgemod.edge_sweep_reclaim_v1,
    "failed_breakout_v1":              edgemod.edge_failed_breakout_v1,
    "compression_expansion_v1":        edgemod.edge_compression_expansion_v1,
    "funding_extremes_v1":             edgemod.edge_funding_extremes_v1,
    "breakout_retest_v2_momentum_filter": edgemod.edge_breakout_retest_v2_momentum_filter,
    "breakout_retest_v2_donchian_filter": edgemod.edge_breakout_retest_v2_donchian_filter,
    "breakout_retest_v2_mom_don_filter":  edgemod.edge_breakout_retest_v2_mom_don_filter,
    "ltc_failed_breakout_funding":        edgemod.edge_ltc_failed_breakout_funding,
    "funding_exhaustion_v2":       edgemod.edge_funding_exhaustion_v2,
}

# ============================================================
# WCZYTAJ GRUPY AKTYWÓW
# ============================================================
ASSET_GROUPS = {
    "calibration": ["BTC/USDT", "ETH/USDT"],
    "validation":  ["LTC/USDT", "XRP/USDT"],
    "holdout":     ["ADA/USDT", "DOT/USDT", "AVAX/USDT"],
}

def get_group(asset):
    for group, assets in ASSET_GROUPS.items():
        if asset in assets:
            return group
    return "unknown"

# ============================================================
# FUNKCJE POMOCNICZE
# ============================================================
def atr(df, window=14):
    h, l, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return pd.Series(tr).rolling(window, min_periods=1).mean().to_numpy()

def get_df(asset, tf):
    df1h = datamod.load_binance_ohlcv(asset, "1h",
                                       VALIDATION_START, VALIDATION_END,
                                       use_cache=True)
    tf_map = {"1h": "1h", "4h": "4h", "1d": "1D", "5m": "5min", "15m": "15min"}
    key = tf_map.get(tf, tf)
    if key == "1h":
        return df1h
    return datamod.resample_ohlcv(df1h, key)

def get_signals(df, edge_name, direction, funding=None):
    fn = EDGE_MAP.get(edge_name)
    if fn is None:
        raise ValueError(f"Nieznany edge: {edge_name}")
    if funding is not None and edge_name in ("funding_exhaustion_v2", "funding_extremes_v1", "funding_momentum_divergence", "ltc_failed_breakout_funding"):
        raw = fn(df, funding)
    else:
        raw = fn(df)
    if direction == "both":
        idx  = [s[0] for s in raw]
        dirs = [s[1] for s in raw]
    elif direction == "long":
        idx  = [s[0] for s in raw if s[1] == 1]
        dirs = [1] * len(idx)
    else:
        idx  = [s[0] for s in raw if s[1] == -1]
        dirs = [-1] * len(idx)
    return idx, dirs

def simulate(o, h, l, c, a, entry_idx, direction, cost, max_hold):
    j0 = entry_idx + 1
    if j0 >= len(c):
        return None
    risk = a[entry_idx] * SL_MULT
    if not np.isfinite(risk) or risk <= 0:
        return None
    entry  = o[j0]
    cost_R = cost * entry / risk
    sl     = entry - risk    if direction == 1 else entry + risk
    tp     = entry + RR*risk if direction == 1 else entry - RR*risk
    last   = min(j0 + max_hold, len(c)) - 1
    for j in range(j0, last + 1):
        hit_sl = l[j] <= sl if direction == 1 else h[j] >= sl
        hit_tp = h[j] >= tp if direction == 1 else l[j] <= tp
        if hit_sl:
            return (-1.0 - cost_R, j)
        if hit_tp:
            return (RR - cost_R, j)
    r = ((c[last] - entry) / risk * direction) - cost_R
    return (r, last)

def backtest(df, indices, dirs, cost, max_hold):
    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    l = df["low"].to_numpy()
    c = df["close"].to_numpy()
    a = atr(df)
    results, next_free = [], 0
    for b, d in zip(indices, dirs):
        if b < next_free:
            continue
        sim = simulate(o, h, l, c, a, b, d, cost, max_hold)
        if sim is None:
            continue
        r, exit_bar = sim
        results.append(r)
        next_free = exit_bar + MIN_GAP
    return results

def expectancy(arr):
    return float(np.mean(arr)) if arr else 0.0

def bootstrap_ci(arr):
    if len(arr) < 5:
        return (0.0, 0.0)
    rng = np.random.default_rng(42)
    samples = rng.choice(arr, size=(2000, len(arr)), replace=True)
    means = samples.mean(axis=1)
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))

def matched_random(df, n, cost, max_hold, runs, direction="both"):
    elig = list(range(MIN_GAP, len(df) - max_hold - 1))
    if len(elig) < 5 or n < 1:
        return np.array([])
    rng  = np.random.default_rng(0)
    outs = []
    for _ in range(runs):
        picks = rng.choice(elig, size=min(n, len(elig)), replace=False)
        dirs = rng.choice([-1] if direction=="short" else ([1] if direction=="long" else [-1,1]), size=len(picks))
        outs.append(expectancy(backtest(df, picks, dirs, cost, max_hold)))
    return np.array(outs)

def time_shift(df, indices, dirs, cost, max_hold, runs):
    if not indices:
        return np.array([])
    rng  = np.random.default_rng(1)
    n    = len(df)
    outs = []
    for _ in range(runs):
        shifted, sdirs = [], []
        for s, d in zip(indices, dirs):
            off = int(rng.integers(10, 61)) * (1 if rng.random() < 0.5 else -1)
            j   = s + off
            if 0 <= j < n - 1:
                shifted.append(j)
                sdirs.append(d)
        if shifted:
            outs.append(expectancy(backtest(df, shifted, sdirs, cost, max_hold)))
    return np.array(outs)

def pct_of(value, dist):
    if len(dist) == 0:
        return float("nan")
    return float((dist < value).mean() * 100)

# ============================================================
# TEST JEDNEGO AKTYWA
# ============================================================
def test_asset(asset, hypo, tf, cost, min_trades, max_hold, direction="both"):
    group = get_group(asset)
    try:
        df = get_df(asset, tf)
        funding = None
        if hypo["edge"] in ("funding_exhaustion_v2", "funding_extremes_v1", "funding_momentum_divergence", "ltc_failed_breakout_funding"):
            funding = datamod.load_binance_funding(asset, VALIDATION_START, VALIDATION_END, use_cache=True)
        indices, dirs = get_signals(df, hypo["edge"], hypo["direction"], funding)

        if len(indices) < min_trades:
            return {"group": group, "asset": asset,
                    "trades": len(indices), "verdict": "ZA_MALO_TRADOW",
                    "exp_R": "", "ci_lo": "", "ci_hi": "",
                    "pctMR": "", "pctTS": ""}

        res          = backtest(df, indices, dirs, cost, max_hold)
        exp          = expectancy(res)
        ci_lo, ci_hi = bootstrap_ci(res)

        # holdout: liczymy benchmarki ale nie ogłaszamy werdyktu
        if group == "holdout":
            mr_dist = matched_random(df, len(indices), cost, max_hold, RANDOM_RUNS, direction)
            ts_dist = time_shift(df, indices, dirs, cost, max_hold, RANDOM_RUNS)
            pct_mr  = pct_of(exp, mr_dist)
            pct_ts  = pct_of(exp, ts_dist)
            verdict = "HOLDOUT_NIE_OTWARTY"
        elif group == "calibration":
            mr_dist = matched_random(df, len(indices), cost, max_hold, RANDOM_RUNS, direction)
            ts_dist = time_shift(df, indices, dirs, cost, max_hold, RANDOM_RUNS)
            pct_mr  = pct_of(exp, mr_dist)
            pct_ts  = pct_of(exp, ts_dist)
            verdict = "KALIBRACJA"
        else:
            mr_dist = matched_random(df, len(indices), cost, max_hold, RANDOM_RUNS, direction)
            ts_dist = time_shift(df, indices, dirs, cost, max_hold, RANDOM_RUNS)
            pct_mr  = pct_of(exp, mr_dist)
            pct_ts  = pct_of(exp, ts_dist)
            verdict = ("PRZESZEDL" if pct_mr >= PCT_THRESHOLD
                                   and pct_ts >= PCT_THRESHOLD
                       else "odrzucony")

        return {"group": group, "asset": asset,
                "trades": len(indices),
                "exp_R":  round(exp, 4),
                "ci_lo":  round(ci_lo, 4),
                "ci_hi":  round(ci_hi, 4),
                "pctMR":  round(pct_mr, 1),
                "pctTS":  round(pct_ts, 1),
                "verdict": verdict}

    except Exception as e:
        return {"group": group, "asset": asset, "trades": 0,
                "verdict": f"ERROR: {e}",
                "exp_R": "", "ci_lo": "", "ci_hi": "",
                "pctMR": "", "pctTS": ""}

# ============================================================
# GŁÓWNA PĘTLA
# ============================================================
def run_all():
    yaml_files = sorted(glob.glob("hypotheses/*.yaml"))
    if not yaml_files:
        print("Brak plików YAML w katalogu hypotheses/")
        sys.exit(1)

    timestamp  = datetime.now().strftime("%Y%m%d_%H%M")
    out_csv    = f"results_{timestamp}.csv"
    fieldnames = ["hypothesis", "timeframe", "direction", "edge",
                  "group", "asset", "trades",
                  "exp_R", "ci_lo", "ci_hi",
                  "pctMR", "pctTS", "verdict", "tested_at"]

    with open(out_csv, "w", newline="") as f:
        csv.DictWriter(f, fieldnames=fieldnames).writeheader()

    all_assets = (ASSET_GROUPS["calibration"] +
                  ASSET_GROUPS["validation"] +
                  ASSET_GROUPS["holdout"])

    print(f"\n{'='*55}")
    print(f"Edge Research Runner | {timestamp}")
    print(f"Hipotez: {len(yaml_files)} | Aktywów: {len(all_assets)}")
    print(f"Kalibracja : {ASSET_GROUPS['calibration']}")
    print(f"Walidacja  : {ASSET_GROUPS['validation']}")
    print(f"Holdout    : {ASSET_GROUPS['holdout']} (werdykt ukryty)")
    print(f"Wyniki     : {out_csv}")
    print(f"{'='*55}\n")

    for i, yaml_file in enumerate(yaml_files, 1):
        with open(yaml_file) as f:
            hypo = yaml.safe_load(f)

        name       = hypo["name"]
        direction  = hypo.get("direction", "both")
        timeframes = hypo.get("timeframes", ["4h"])

        print(f"[{i}/{len(yaml_files)}] {name}")

        for tf in timeframes:
            cost       = COST_BY_TF.get(tf, 0.0002)
            min_trades = MIN_TRADES_BY_TF.get(tf, 20)
            max_hold   = MAX_HOLD_BY_TF.get(tf, 24)

            passed_validation = 0
            total_validation  = len(ASSET_GROUPS["validation"])

            print(f"  TF={tf} koszt={cost*100:.3f}%")

            for asset in all_assets:
                group = get_group(asset)
                print(f"    [{group:12s}] {asset}...", end=" ", flush=True)
                t0  = time.time()
                row = test_asset(asset, hypo, tf, cost, min_trades, max_hold, direction)
                row.update({"hypothesis": name, "timeframe": tf,
                            "direction": direction, "edge": hypo["edge"],
                            "tested_at": datetime.now().isoformat()})

                with open(out_csv, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writerow({k: row.get(k, "") for k in fieldnames})

                v = row.get("verdict", "?")
                if v == "PRZESZEDL":
                    passed_validation += 1

                parts = []
                if row["trades"]:
                    parts.append(f"n={row['trades']}")
                if row["exp_R"] != "":
                    parts.append(f"exp={row['exp_R']}")
                if row["pctMR"] != "":
                    parts.append(f"MR={row['pctMR']}%")
                if row["pctTS"] != "":
                    parts.append(f"TS={row['pctTS']}%")
                parts.append(f"-> {v}")
                parts.append(f"({time.time()-t0:.0f}s)")
                print(" ".join(parts))

            print(f"\n  WALIDACJA: {passed_validation}/{total_validation} przeszło")
            if passed_validation >= 1:
                print(f"  *** KANDYDAT – otwórz holdout dla: {name} TF={tf} ***")
            print()

    print(f"{'='*55}")
    print(f"Gotowe. Wyniki: {out_csv}")

if __name__ == "__main__":
    run_all()
