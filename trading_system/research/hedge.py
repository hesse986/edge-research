"""
Współdzielony helper do hedge ratio (bety) i spreadu dla pairs trading.

JEDYNE źródło prawdy dla liczenia spreadu — używane przez:
- trading_system/research/grid_search_pairs.py
- trading_system/research/pairs_validation.py
- runner.py (benchmark pairs inline w test_asset)
- paper_trading_pairs_advanced_with_R.py (live monitor)

Metoda: rolling OLS out-of-sample.
Beta w punkcie i jest estymowana WYŁĄCZNIE z danych sprzed świecy i
(okno kończące się na i-1). To eliminuje lookahead, który był obecny gdy
beta fitowano przez LinearRegression na CAŁYM oknie (również forward).

Spread[i] = p1[i] - beta[i] * p2[i]
gdzie beta[i] = OLS slope(p2 -> p1) na oknie [i-window, i-1].

Okno bety = beta_mult * lookback (domyślnie 4× lookback z-score).
"""

import pandas as pd

# Domyślny mnożnik: okno bety = 4× lookback z-score (przy lookback=30 → 120 świec).
DEFAULT_BETA_MULT = 4


def beta_window_for(lookback, beta_mult=DEFAULT_BETA_MULT):
    """Długość okna estymacji bety dla danego lookbacku z-score."""
    return int(beta_mult * lookback)


def _align(p1, p2):
    """Zwraca (p1, p2) jako wyrównane Series o wspólnym indeksie.

    Akceptuje listy/ndarray (dostają RangeIndex) lub Series (wyrównanie po indeksie).
    """
    if not isinstance(p1, pd.Series):
        p1 = pd.Series(list(p1))
    if not isinstance(p2, pd.Series):
        p2 = pd.Series(list(p2))
    if not p1.index.equals(p2.index):
        common = p1.index.intersection(p2.index)
        p1 = p1.loc[common]
        p2 = p2.loc[common]
    return p1, p2


def rolling_hedge_ratio(p1, p2, window):
    """Rolling OLS slope (p2 -> p1) liczony out-of-sample.

    beta[i] = cov(p2, p1) / var(p2) na oknie [i-window, i-1].
    Shift o 1 świecę gwarantuje, że beta[i] NIE używa danych z indeksu i
    ani późniejszych — czyli brak lookaheadu.

    Pierwsze `window` punktów ma betę NaN (za mało historii).
    """
    p1, p2 = _align(p1, p2)
    cov = p2.rolling(window).cov(p1)   # ddof=1, jak LinearRegression slope
    var = p2.rolling(window).var()
    beta = (cov / var).shift(1)
    return beta


def compute_spread(p1, p2, window):
    """Spread pairs trading z betą liczoną out-of-sample.

    spread[i] = p1[i] - beta[i] * p2[i], beta[i] tylko z danych < i.
    Zwraca pd.Series wyrównany do indeksu wejścia (NaN tam, gdzie beta NaN).
    """
    p1, p2 = _align(p1, p2)
    beta = rolling_hedge_ratio(p1, p2, window)
    return p1 - beta * p2
