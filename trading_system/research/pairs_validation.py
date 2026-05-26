"""
Właściwa walidacja dla strategii pairs trading.
Zamiast bootstrap na zwrotach, używamy permutation test na spreadzie.

Problem z obecnym pctMR:
- Bootstrap tasuje zwroty z transakcji
- Ale traci strukturę czasową spreadu (stacjonarność)
- Wszystkie konfiguracje dają pctMR~50% bo benchmark jest błędny

Rozwiązanie:
- Permutation test: generuj losowe spreads o tej samej stacjonarności
- Uruchom strategię na losowych spreadach
- Sprawdź czy prawdziwy spread daje lepsze wyniki
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression


def simulate_trades(spread, entry_z, exit_z, lookback, sl_mult, cost=0.0002):
    """Symulacja transakcji na spreadzie."""
    mean   = spread.rolling(lookback).mean()
    std    = spread.rolling(lookback).std()
    zscore = (spread - mean) / std
    atr    = (spread.rolling(14).max() - spread.rolling(14).min()) / 14

    pos = 0; es = 0; trades = []
    for i in range(lookback + 5, len(spread)):
        z = zscore.iloc[i]
        if pd.isna(z): continue
        if pos == 0:
            if z > entry_z:   pos=-1; es=spread.iloc[i]
            elif z < -entry_z: pos=1;  es=spread.iloc[i]
        else:
            sl   = atr.iloc[i] * sl_mult
            risk = sl if sl > 0 else 0.02 * abs(es)
            ret  = (spread.iloc[i] - es) * pos
            if (pos==1 and z>-exit_z) or (pos==-1 and z<exit_z):
                trades.append(ret/risk - cost if risk>0 else 0)
                pos = 0
    return np.array(trades)


def generate_surrogate_spread(spread, method="phase_randomization"):
    """
    Generuje losowy spread zachowujący właściwości statystyczne oryginału.
    
    method="phase_randomization": zachowuje autokorelację i spektrum mocy
    method="block_shuffle": tasuje bloki danych zachowując lokalną strukturę
    """
    s = spread.dropna().values
    
    if method == "phase_randomization":
        # FFT phase randomization – zachowuje widmo amplitudowe
        fft   = np.fft.rfft(s)
        phases = np.random.uniform(0, 2*np.pi, len(fft))
        phases[0] = 0  # zachowaj składową stałą
        fft_rand = np.abs(fft) * np.exp(1j * phases)
        surrogate = np.fft.irfft(fft_rand, n=len(s))
        # Przeskaluj do oryginalnych statystyk
        surrogate = (surrogate - surrogate.mean()) / surrogate.std()
        surrogate = surrogate * s.std() + s.mean()
        
    elif method == "block_shuffle":
        # Block shuffle – tasuje bloki o długości ~sqrt(n)
        block_size = max(10, int(np.sqrt(len(s))))
        blocks = [s[i:i+block_size] for i in range(0, len(s)-block_size, block_size)]
        np.random.shuffle(blocks)
        surrogate = np.concatenate(blocks)[:len(s)]
    
    return pd.Series(surrogate, index=spread.dropna().index)


def pairs_pctMR(spread, entry_z=2.0, exit_z=0.5, lookback=30, sl_mult=1.5,
                n_permutations=500, method="phase_randomization"):
    """
    Permutation test dla pairs trading.
    
    Returns:
        dict: exp_R, pctMR, n_trades, ci_lo, ci_hi
    """
    # Prawdziwe wyniki
    real_trades = simulate_trades(spread, entry_z, exit_z, lookback, sl_mult)
    if len(real_trades) < 10:
        return {"exp_R": 0, "pctMR": 0, "n": 0, "valid": False}
    
    real_exp_R = float(real_trades.mean())
    
    # Permutation test
    surrogate_results = []
    rng = np.random.default_rng(42)
    
    for _ in range(n_permutations):
        # Generuj losowy spread z tymi samymi właściwościami
        np.random.seed(rng.integers(0, 10000))
        surr = generate_surrogate_spread(spread, method=method)
        surr_trades = simulate_trades(surr, entry_z, exit_z, lookback, sl_mult)
        if len(surr_trades) >= 5:
            surrogate_results.append(float(surr_trades.mean()))
    
    if not surrogate_results:
        return {"exp_R": real_exp_R, "pctMR": 0, "n": len(real_trades), "valid": False}
    
    surrogate_results = np.array(surrogate_results)
    pctMR = float(100 * np.mean(surrogate_results < real_exp_R))
    
    # Confidence interval
    ci_lo = float(np.percentile(surrogate_results, 5))
    ci_hi = float(np.percentile(surrogate_results, 95))
    
    return {
        "exp_R":   round(real_exp_R, 4),
        "pctMR":   round(pctMR, 1),
        "n":       len(real_trades),
        "ci_lo":   round(ci_lo, 4),
        "ci_hi":   round(ci_hi, 4),
        "valid":   pctMR >= 95,
        "method":  method,
        "n_perms": n_permutations
    }


def validate_pairs_full(sym1, sym2, periods, params,
                        n_permutations=500):
    """
    Pełna walidacja pary na wielu okresach.
    
    periods: lista (start, end, label)
    params: dict z entry_z, exit_z, lookback, sl_mult
    """
    import sys
    sys.path.insert(0, '.')
    import data as datamod

    results = {}
    for start, end, label in periods:
        df1h_1 = datamod.load_binance_ohlcv(sym1, "1h", start, end, use_cache=True)
        df1h_2 = datamod.load_binance_ohlcv(sym2, "1h", start, end, use_cache=True)

        import data as d
        df1 = d.resample_ohlcv(df1h_1, "4h")
        df2 = d.resample_ohlcv(df1h_2, "4h")

        common = df1.index.intersection(df2.index)
        p1 = df1.loc[common, "close"]
        p2 = df2.loc[common, "close"]

        model = LinearRegression().fit(p2.values.reshape(-1,1), p1.values)
        spread = p1 - model.coef_[0] * p2

        result = pairs_pctMR(
            spread,
            entry_z    = params.get("entry_z",  2.0),
            exit_z     = params.get("exit_z",   0.5),
            lookback   = params.get("lookback", 30),
            sl_mult    = params.get("sl_mult",  1.5),
            n_permutations = n_permutations
        )
        results[label] = result

    return results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, '.')

    PERIODS = [
        ("2023-01-01", "2024-01-01", "Kalibracja"),
        ("2024-01-01", "2025-01-01", "Walidacja"),
        ("2025-01-01", "2026-05-01", "Forward"),
    ]
    PARAMS = {"entry_z": 2.0, "exit_z": 0.5, "lookback": 30, "sl_mult": 1.5}

    PAIRS = [
        ("LTC/USDT", "XRP/USDT"),
        ("LTC/USDT", "ADA/USDT"),
        ("XRP/USDT", "ADA/USDT"),
    ]

    print("=== Pairs Trading – Permutation Test Validation ===")
    print("(500 surrogate spreads per period)\n")

    for sym1, sym2 in PAIRS:
        print(f"\n{sym1[:3]}/{sym2[:3]}:")
        results = validate_pairs_full(sym1, sym2, PERIODS, PARAMS, n_permutations=500)
        for label, r in results.items():
            verdict = "✅ VALID" if r["valid"] else "❌"
            print(f"  {label:<12}: n={r['n']:3d} exp_R={r['exp_R']:>7.4f} "
                  f"pctMR={r['pctMR']:>5.1f}% {verdict}")
