"""
Test braku lookaheadu w rolling hedge ratio / spreadzie.

Uruchom: python trading_system/research/test_hedge.py
(Kompatybilny też z pytest, jeśli kiedyś zostanie zainstalowany.)

Teza dowodzona: spread[i] oraz beta[i] zależą WYŁĄCZNIE od danych z indeksów < i
(dla spreadu dodatkowo od p1[i], p2[i] z bieżącej świecy — to obserwacja, nie lookahead).
Podmiana danych po indeksie i NIE może zmienić spreadu/bety w punkcie i.
"""
import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hedge import rolling_hedge_ratio, compute_spread, beta_window_for


def _make_prices(n=500, seed=7):
    rng = np.random.default_rng(seed)
    # Dwa skointegrowane szeregi: p1 ≈ beta*p2 + noise, beta dryfuje lekko.
    p2 = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)))
    beta_true = 1.8 + 0.0005 * np.arange(n)
    p1 = pd.Series(beta_true * p2.values + rng.normal(0, 2, n))
    return p1, p2


def test_no_lookahead_spread():
    """Podmiana danych PO indeksie i nie zmienia spreadu w [0..i]."""
    p1, p2 = _make_prices()
    window = beta_window_for(lookback=30)  # 120
    spread = compute_spread(p1, p2, window)

    i = 300  # punkt kontrolny w środku
    rng = np.random.default_rng(123)
    p1b = p1.copy()
    p2b = p2.copy()
    # Zmień WSZYSTKO po indeksie i (włącznie z i+1)
    p1b.iloc[i + 1:] += rng.normal(0, 50, len(p1) - i - 1)
    p2b.iloc[i + 1:] *= 1.5
    spread_b = compute_spread(p1b, p2b, window)

    a = spread.iloc[:i + 1].to_numpy()
    b = spread_b.iloc[:i + 1].to_numpy()
    # Porównaj tam, gdzie nie ma NaN (warmup okna bety)
    mask = ~np.isnan(a)
    assert mask.any(), "spread powinien mieć niepuste wartości po warmupie"
    np.testing.assert_allclose(a[mask], b[mask], rtol=0, atol=0,
                               err_msg="LOOKAHEAD: spread[<=i] zmienił się po podmianie danych >i")


def test_no_lookahead_beta():
    """beta[i] nie zależy od danych z indeksu >= i (podmiana od i nie zmienia beta[i])."""
    p1, p2 = _make_prices()
    window = beta_window_for(lookback=30)
    beta = rolling_hedge_ratio(p1, p2, window)

    i = 300
    p1b = p1.copy()
    p2b = p2.copy()
    # Zmień dane od indeksu i (włącznie) — beta[i] korzysta z okna [i-window, i-1]
    p1b.iloc[i:] += 999.0
    p2b.iloc[i:] += 999.0
    beta_b = rolling_hedge_ratio(p1b, p2b, window)

    a = beta.iloc[:i + 1].to_numpy()
    b = beta_b.iloc[:i + 1].to_numpy()
    mask = ~np.isnan(a)
    np.testing.assert_allclose(a[mask], b[mask], rtol=0, atol=0,
                               err_msg="LOOKAHEAD: beta[<=i] zależy od danych >=i")


def test_beta_matches_manual_ols():
    """beta[i] == OLS slope(p2->p1) na oknie [i-window, i-1] (zgodność z definicją)."""
    p1, p2 = _make_prices()
    window = beta_window_for(lookback=30)
    beta = rolling_hedge_ratio(p1, p2, window)

    i = 300
    x = p2.iloc[i - window:i].to_numpy()
    y = p1.iloc[i - window:i].to_numpy()
    slope_manual = np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1)
    np.testing.assert_allclose(beta.iloc[i], slope_manual, rtol=1e-9, atol=1e-9,
                               err_msg="beta[i] nie zgadza się z ręcznym OLS slope na oknie [i-window, i-1]")


def test_current_candle_used_in_spread():
    """Sanity: spread[i] UŻYWA p1[i], p2[i] (zmiana w i zmienia spread[i]) — to nie lookahead."""
    p1, p2 = _make_prices()
    window = beta_window_for(lookback=30)
    spread = compute_spread(p1, p2, window)
    i = 300
    p1b = p1.copy()
    p1b.iloc[i] += 10.0
    spread_b = compute_spread(p1b, p2, window)
    assert spread.iloc[i] != spread_b.iloc[i], "spread[i] powinien zależeć od p1[i] (bieżąca obserwacja)"
    # ...ale punkt i-1 (i wcześniejsze) bez zmian
    assert spread.iloc[i - 1] == spread_b.iloc[i - 1]


if __name__ == "__main__":
    tests = [test_no_lookahead_spread, test_no_lookahead_beta,
             test_beta_matches_manual_ols, test_current_candle_used_in_spread]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ✅ {t.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  ❌ {t.__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} testów przeszło")
    sys.exit(1 if failed else 0)
