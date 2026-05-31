"""
Validation Agent – ocena strategii pod kątem overfittingu.
Oparte na:
- Bailey et al. "The Probability of Backtest Overfitting" (PBO)
- Bailey & Lopez de Prado "The Deflated Sharpe Ratio" (DSR)
"""

import numpy as np
import pandas as pd
from itertools import combinations
from scipy.stats import norm
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))


# ============================================================
# 1. DEFLATED SHARPE RATIO (Bailey & Lopez de Prado)
# ============================================================

def sharpe_ratio(returns: np.ndarray, periods_per_year: int = 252) -> float:
    """Annualized Sharpe Ratio."""
    if len(returns) == 0 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(periods_per_year))

def deflated_sharpe_ratio(
    returns: np.ndarray,
    n_trials: int,
    periods_per_year: int = 252
) -> dict:
    """
    Deflated Sharpe Ratio – koryguje SR o liczbę testowanych konfiguracji.

    Wzór (Bailey & Lopez de Prado 2014):

        DSR = Φ[ (SR_hat - SR0) · sqrt(T-1) / sqrt(1 - γ3·SR_hat + (γ4-1)/4·SR_hat²) ]

    KLUCZOWE: SR_hat i SR0 są PER-OKRES (NIE zannualizowane). Człon korekcyjny
    skośność/kurtoza oczekuje SR per-okres — wstawienie SR×√252 zawyżałoby SR²
    i psuło statystykę (audyt 3.3). γ4 to kurtoza SUROWA (normalna = 3), a
    pandas.kurt() zwraca kurtozę NADMIAROWĄ, więc dodajemy +3.

    Benchmark SR0 (oczekiwane maksimum SR z n_trials prób) jest w jednostkach
    odchylenia standardowego estymatora SR, więc skalujemy go przez SE(SR_hat),
    by porównanie było spójne jednostkowo z SR per-okres.

    Returns:
        dict: sr (annualized, do raportu), sr_periodic, sr_benchmark, dsr_stat,
              pvalue, is_significant, ...
    """
    T = len(returns)
    if T < 10:
        return {"sr": 0, "sr_periodic": 0, "sr_benchmark": 0,
                "dsr_stat": 0, "pvalue": 1.0, "is_significant": False,
                "n_trials": n_trials, "T": T, "skew": 0, "kurt": 0}

    r = np.asarray(returns, dtype=float)
    std = r.std(ddof=1)
    sr_periodic = float(r.mean() / std) if std > 0 else 0.0   # SR per-okres (NIE annualizowany)
    sr_annual   = float(sr_periodic * np.sqrt(periods_per_year))  # tylko do raportu

    # Skośność i kurtoza zwrotów
    skew = float(pd.Series(r).skew())
    kurt_raw = float(pd.Series(r).kurt()) + 3.0   # pandas.kurt() = nadmiarowa → surowa

    # SE estymatora SR (Mertens/Lo) na SR per-okres
    inner = (1 - skew * sr_periodic + (kurt_raw - 1) / 4 * sr_periodic**2) / (T - 1)
    if inner <= 0:
        inner = 1.0 / (T - 1)
    se_sr = np.sqrt(inner)

    # Oczekiwane maksimum SR z n_trials prób (w jednostkach SD estymatora SR):
    # E[max] ≈ (1-γ)·Z^-1(1-1/N) + γ·Z^-1(1-1/(N·e)), γ = Euler-Mascheroni
    gamma = 0.5772156649
    if n_trials > 1:
        expected_max_z = ((1 - gamma) * norm.ppf(1 - 1/n_trials) +
                          gamma * norm.ppf(1 - 1/(n_trials * np.e)))
    else:
        expected_max_z = 0.0
    sr_benchmark = se_sr * expected_max_z   # benchmark w jednostkach SR per-okres

    # DSR = P(SR_hat > SR0)
    dsr_stat = (sr_periodic - sr_benchmark) / se_sr if se_sr > 0 else 0.0
    pvalue = 1 - norm.cdf(dsr_stat)

    return {
        "sr":              round(sr_annual, 3),      # annualized (czytelność)
        "sr_periodic":     round(sr_periodic, 5),
        "sr_benchmark":    round(sr_benchmark, 5),
        "dsr_stat":        round(dsr_stat, 3),
        "pvalue":          round(pvalue, 4),
        "is_significant":  pvalue < 0.05,
        "n_trials":        n_trials,
        "T":               T,
        "skew":            round(skew, 3),
        "kurt":            round(kurt_raw, 3)
    }


# ============================================================
# 2. PROBABILITY OF BACKTEST OVERFITTING (Bailey et al.)
# ============================================================

def probability_of_backtest_overfitting(
    returns_matrix: np.ndarray,
    n_partitions: int = 10
) -> dict:
    """
    PBO – prawdopodobieństwo że najlepszy backtest jest przypadkowy.
    
    returns_matrix: (T, N) – T okresów, N konfiguracji/strategii
    n_partitions: liczba podziałów IS/OOS
    
    Returns:
        dict: pbo, lambda_bar, is_overfit
    """
    T, N = returns_matrix.shape
    if N < 2 or T < n_partitions * 2:
        return {"pbo": None, "message": "Za mało danych do PBO"}
    
    # Podziel dane na n_partitions równych części
    partition_size = T // n_partitions
    partitions = [
        returns_matrix[i*partition_size:(i+1)*partition_size, :]
        for i in range(n_partitions)
    ]
    
    # Kombinacje IS/OOS: każda połowa jako IS, druga jako OOS
    lambdas = []
    
    for is_mask in combinations(range(n_partitions), n_partitions // 2):
        oos_mask = [i for i in range(n_partitions) if i not in is_mask]
        
        # IS: wybierz najlepszą strategię
        is_data  = np.vstack([partitions[i] for i in is_mask])
        oos_data = np.vstack([partitions[i] for i in oos_mask])
        
        is_sharpes  = is_data.mean(axis=0) / (is_data.std(axis=0) + 1e-10)
        oos_sharpes = oos_data.mean(axis=0) / (oos_data.std(axis=0) + 1e-10)
        
        # Najlepsza strategia IS
        best_is_idx = np.argmax(is_sharpes)
        
        # Ranking OOS
        oos_rank = (oos_sharpes < oos_sharpes[best_is_idx]).sum() / N
        lambdas.append(oos_rank)
    
    lambdas = np.array(lambdas)
    lambda_bar = float(lambdas.mean())
    
    # PBO = P(lambda < 0.5) – prawdopodobieństwo że wybrany jest poniżej mediany OOS
    pbo = float((lambdas < 0.5).mean())
    
    return {
        "pbo":        round(pbo, 3),
        "lambda_bar": round(lambda_bar, 3),
        "is_overfit": pbo > 0.5,
        "n_configs":  N,
        "n_periods":  T,
        "message":    "OVERFIT" if pbo > 0.5 else "OK"
    }


# ============================================================
# 3. GŁÓWNA FUNKCJA WALIDACJI
# ============================================================

def validate_strategy(
    returns: np.ndarray,
    n_trials: int = 1,
    strategy_name: str = "strategy",
    periods_per_year: int = 252
) -> dict:
    """
    Kompletna walidacja strategii.
    
    returns: array zwrotów (R lub % returns)
    n_trials: ile konfiguracji było testowanych (do DSR)
    """
    if len(returns) == 0:
        return {"valid": False, "reason": "Brak danych"}
    
    returns = np.array(returns)
    
    # Podstawowe statystyki
    n = len(returns)
    mean_r = float(returns.mean())
    std_r  = float(returns.std())
    win_rate = float((returns > 0).mean())
    profit_factor = (
        returns[returns > 0].sum() / abs(returns[returns < 0].sum())
        if (returns < 0).any() else float('inf')
    )
    
    # Max drawdown
    cumulative = np.cumprod(1 + returns / 100) if returns.mean() < 1 else np.cumsum(returns)
    rolling_max = np.maximum.accumulate(cumulative)
    drawdown = (cumulative - rolling_max) / (rolling_max + 1e-10)
    max_dd = float(drawdown.min())
    
    # DSR
    dsr = deflated_sharpe_ratio(returns, n_trials, periods_per_year)
    
    # Decyzja
    valid = (
        mean_r > 0 and
        win_rate > 0.45 and
        profit_factor > 1.1 and
        dsr["is_significant"]
    )
    
    result = {
        "strategy":      strategy_name,
        "valid":         valid,
        "n":             n,
        "mean_R":        round(mean_r, 4),
        "win_rate":      round(win_rate, 3),
        "profit_factor": round(profit_factor, 3),
        "max_drawdown":  round(max_dd, 3),
        "sharpe":        dsr["sr"],
        "dsr_stat":      dsr["dsr_stat"],
        "pvalue":        dsr["pvalue"],
        "dsr_ok":        dsr["is_significant"],
        "n_trials":      n_trials,
        "verdict":       "✅ VALID" if valid else "❌ REJECTED"
    }
    
    return result


def print_validation_report(result: dict) -> None:
    print(f"\n{'='*50}")
    print(f"VALIDATION AGENT – {result['strategy']}")
    print(f"{'='*50}")
    print(f"n trades:       {result['n']}")
    print(f"Mean R:         {result['mean_R']:.4f}")
    print(f"Win rate:       {result['win_rate']:.1%}")
    print(f"Profit Factor:  {result['profit_factor']:.2f}")
    print(f"Max Drawdown:   {result['max_drawdown']:.1%}")
    print(f"Sharpe:         {result['sharpe']:.3f}")
    print(f"DSR stat:       {result['dsr_stat']:.3f}")
    print(f"p-value:        {result['pvalue']:.4f}")
    print(f"DSR sig (p<5%): {result['dsr_ok']}")
    print(f"n_trials:       {result['n_trials']}")
    print(f"\n>>> {result['verdict']}")
    print(f"{'='*50}")


if __name__ == "__main__":
    # Test na wynikach pairs trading z backtestu
    import csv
    base = Path(__file__).parent.parent.parent
    
    # Szukaj pliku z wynikami backtestu
    bt_file = base / "backtest_trades_full.csv"
    if bt_file.exists():
        rows = list(csv.DictReader(open(bt_file)))
        returns = np.array([float(r["R"]) for r in rows if r["R"]])
        
        # Walidacja z uwzględnieniem że testowaliśmy wiele konfiguracji
        N_TRIALS = 50  # przybliżona liczba testowanych edge'y w projekcie
        result = validate_strategy(returns, n_trials=N_TRIALS,
                                   strategy_name="pairs_trading_LTC_XRP_ADA")
        print_validation_report(result)
    else:
        # Demo na losowych danych
        print("Brak backtest_trades_full.csv – demo na syntetycznych danych")
        np.random.seed(42)
        good_returns = np.random.normal(0.4, 1.5, 436)
        result = validate_strategy(good_returns, n_trials=50,
                                   strategy_name="pairs_trading_demo")
        print_validation_report(result)
