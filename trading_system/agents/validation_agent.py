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
    DSR = SR* * sqrt(T) / sigma
    gdzie SR* = benchmark SR uwzględniający n_trials
    
    Returns:
        dict: sr, dsr, pvalue, is_significant
    """
    T = len(returns)
    if T < 10:
        return {"sr": 0, "dsr": 0, "pvalue": 1.0, "is_significant": False}
    
    sr = sharpe_ratio(returns, periods_per_year)
    
    # Skewness i kurtosis zwrotów
    skew = float(pd.Series(returns).skew())
    kurt = float(pd.Series(returns).kurt())
    
    # Benchmark SR – oczekiwane maksimum z n_trials losowych testów
    # Przybliżenie: E[max SR] ~ (1 - gamma) * Z^-1(1 - 1/n) + gamma * Z^-1(1 - 1/(n*e))
    # gdzie gamma = 0.5772 (Euler-Mascheroni)
    gamma = 0.5772156649
    if n_trials > 1:
        sr_benchmark = (1 - gamma) * norm.ppf(1 - 1/n_trials) + \
                       gamma * norm.ppf(1 - 1/(n_trials * np.e))
    else:
        sr_benchmark = 0.0
    
    # Korekta o skewness i kurtosis (pełna formuła DSR)
    inner = (1 - skew * sr + (kurt - 1) / 4 * sr**2) / (T - 1)
    if inner <= 0:
        inner = 1.0 / (T - 1)
    correction = np.sqrt(inner)
    
    # DSR = P(SR > SR_benchmark)
    dsr_stat = (sr - sr_benchmark) / correction if correction > 0 else 0
    pvalue = 1 - norm.cdf(dsr_stat)
    
    return {
        "sr":              round(sr, 3),
        "sr_benchmark":    round(sr_benchmark, 3),
        "dsr_stat":        round(dsr_stat, 3),
        "pvalue":          round(pvalue, 4),
        "is_significant":  pvalue < 0.05,
        "n_trials":        n_trials,
        "T":               T,
        "skew":            round(skew, 3),
        "kurt":            round(kurt, 3)
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
