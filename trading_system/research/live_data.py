"""
Jedno źródło prawdy dla LIVE spreadu / z-score (monitor + dashboard).

Wszystkie miejsca, które liczą spread na żywo (paper_trading_pairs_advanced_with_R.py
oraz trading_system/dashboard/server.py: get_zscore i spread_chart) wołają
`live_spread_zscore` zamiast kopiować logikę z zahardkodowaną betą.

Spread liczony tym samym rolling-OLS helperem co backtest (hedge.compute_spread),
więc live == research. Beta NIE jest hardkodowana — re-estymowana out-of-sample.

Dodatkowo: `_fetch_klines` ma retry z exponential backoff + jitter i obsługę
HTTP 429 (Retry-After), żeby przejściowe błędy/limit Binance nie wywalały pętli.
"""

import os
import sys
import time
import random
import numpy as np
import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from hedge import compute_spread, beta_window_for

KLINES_URL = "https://api.binance.com/api/v3/klines"


def _fetch_klines(symbol, interval="4h", limit=200,
                  max_retries=5, base_delay=0.5, timeout=10):
    """Pobiera świece z Binance z retry/backoff/jitter i obsługą 429.

    Zwraca listę świec (surowy JSON Binance) albo rzuca wyjątek po wyczerpaniu prób.
    """
    params = {"symbol": symbol.replace("/", ""), "interval": interval, "limit": limit}
    last_err = None
    for attempt in range(max_retries):
        try:
            resp = requests.get(KLINES_URL, params=params, timeout=timeout)
            if resp.status_code == 429:
                # Respektuj Retry-After jeśli jest, inaczej exponential backoff.
                retry_after = float(resp.headers.get("Retry-After", 0) or 0)
                delay = retry_after if retry_after > 0 else base_delay * (2 ** attempt)
                time.sleep(delay + random.uniform(0, base_delay))
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as e:
            last_err = e
            if attempt == max_retries - 1:
                break
            time.sleep(base_delay * (2 ** attempt) + random.uniform(0, base_delay))
    raise RuntimeError(f"fetch_klines({symbol}) nieudany po {max_retries} próbach: {last_err}")


def live_spread_zscore(sym1, sym2, lookback=30, beta_mult=4,
                       history=60, closed_only=False, interval="4h"):
    """Live spread + z-score dla pary, jednym helperem (rolling OLS).

    Args:
        lookback:    okno z-score (świece).
        beta_mult:   okno bety = beta_mult × lookback (rolling OLS, out-of-sample).
        history:     ile WAŻNYCH (po warmupie bety) punktów spreadu zwrócić.
        closed_only: jeśli True, pomija formującą się świecę (decyzje tylko po
                     domknięciu) — używane przez monitor. Dashboard używa False.

    Returns:
        dict z polami: timestamps, p1, p2, spread (lista), zscore (lista),
        mean, std, z (ostatni), spread_last, atr, price1, price2, candle_ts.
        Zwraca None, gdy danych jest za mało.
    """
    window = beta_window_for(lookback, beta_mult)
    need = window + lookback + history + 5

    k1 = _fetch_klines(sym1, interval, need)
    k2 = _fetch_klines(sym2, interval, need)

    # Wyrównanie po czasie otwarcia świecy (c[0]); c[4]=close, c[6]=close time.
    o1 = {int(c[0]): (float(c[4]), int(c[6])) for c in k1}
    o2 = {int(c[0]): float(c[4]) for c in k2}
    common_ts = sorted(set(o1) & set(o2))

    if closed_only:
        now_ms = int(time.time() * 1000)
        common_ts = [t for t in common_ts if o1[t][1] <= now_ms]  # świeca domknięta

    if len(common_ts) < window + lookback:
        return None

    p1 = [o1[t][0] for t in common_ts]
    p2 = [o2[t] for t in common_ts]

    spread_series = compute_spread(pd.Series(p1), pd.Series(p2), window).dropna()
    if len(spread_series) < lookback:
        return None

    pos = spread_series.index.to_numpy()      # pozycje ważnych punktów w common_ts
    sp = spread_series.to_numpy()
    ts = [common_ts[i] for i in pos]

    # Przytnij do żądanej historii.
    sp = sp[-history:]
    ts = ts[-history:]

    mean = float(np.mean(sp[-lookback:]))
    std = float(np.std(sp[-lookback:]))
    z = (sp[-1] - mean) / std if std > 0 else 0.0
    zscore = [((s - mean) / std if std > 0 else 0.0) for s in sp]
    atr = float(np.mean(np.abs(np.diff(sp)))) if len(sp) > 1 else 0.0

    return {
        "timestamps":  ts,
        "p1":          p1[-history:],
        "p2":          p2[-history:],
        "spread":      [float(x) for x in sp],
        "zscore":      [round(float(x), 4) for x in zscore],
        "mean":        mean,
        "std":         std,
        "z":           float(z),
        "spread_last": float(sp[-1]),
        "atr":         atr,
        "price1":      float(p1[-1]),
        "price2":      float(p2[-1]),
        "candle_ts":   int(ts[-1]),   # czas otwarcia ostatniej użytej świecy
    }
