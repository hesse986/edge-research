"""Kandydaci na edge do Fazy 0.

Kazda funkcja bierze DataFrame OHLCV i zwraca liste sygnalow:
    [(indeks_swiecy, kierunek), ...]

kierunek: +1 = long, -1 = short.

Zasada anty-lookahead:
- sygnal na swiecy i moze uzywac tylko danych do swiecy i wlacznie,
- silnik wchodzi dopiero na open swiecy i+1.

Plik zawiera 3 grupy:
- core: pierwotne, proste definicje v1,
- v2: ulepszone wersje edge'ow, ktore mialy jakis sygnal,
- new: dodatkowe modele Fazy 0 do odkrywania nowych przewag.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from engine import atr


# ---------------------------------------------------------------------------
# Helpers — zero zewnetrznych bibliotek TA, zeby framework byl lekki.
# ---------------------------------------------------------------------------


def _sma(x, window: int):
    return pd.Series(x).rolling(window, min_periods=max(2, window // 3)).mean().to_numpy()


def _ema(x, span: int):
    return pd.Series(x).ewm(span=span, adjust=False, min_periods=max(2, span // 3)).mean().to_numpy()


def _rolling_std(x, window: int):
    return pd.Series(x).rolling(window, min_periods=max(5, window // 3)).std().to_numpy()


def _rolling_vwap(df, window: int):
    close = df["close"].to_numpy()
    vol = df["volume"].to_numpy()
    pv = pd.Series(close * vol)
    vv = pd.Series(vol)
    denom = vv.rolling(window, min_periods=max(5, window // 3)).sum()
    return (pv.rolling(window, min_periods=max(5, window // 3)).sum() / denom).to_numpy()


def _bar_pos(open_, high, low, close):
    # Pozycja zamkniecia w zakresie swiecy: 0 = close przy low, 1 = close przy high.
    rng = np.maximum(high - low, 1e-12)
    return (close - low) / rng


def _adx(df, window: int = 14):
    """Prosty ADX Wilder-like oparty na rolling mean, wystarczajacy do filtra rezimu."""
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()

    up = np.diff(high, prepend=high[0])
    down = -np.diff(low, prepend=low[0])
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)

    prev_close = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    tr_s = pd.Series(tr).rolling(window, min_periods=max(2, window // 2)).mean().to_numpy()
    plus_s = pd.Series(plus_dm).rolling(window, min_periods=max(2, window // 2)).mean().to_numpy()
    minus_s = pd.Series(minus_dm).rolling(window, min_periods=max(2, window // 2)).mean().to_numpy()

    plus_di = 100 * plus_s / np.maximum(tr_s, 1e-12)
    minus_di = 100 * minus_s / np.maximum(tr_s, 1e-12)
    dx = 100 * np.abs(plus_di - minus_di) / np.maximum(plus_di + minus_di, 1e-12)
    return pd.Series(dx).rolling(window, min_periods=max(2, window // 2)).mean().to_numpy()


def _volume_ma(df, window: int = 20):
    return _sma(df["volume"].to_numpy(), window)


def _range_width_atr(df, lookback: int = 48):
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    a = atr(df, 14)
    width = pd.Series(high).rolling(lookback, min_periods=max(5, lookback // 3)).max().to_numpy() - \
            pd.Series(low).rolling(lookback, min_periods=max(5, lookback // 3)).min().to_numpy()
    return width / np.maximum(a, 1e-12)


# ---------------------------------------------------------------------------
# CORE v1 — oryginalne, proste definicje jako baseline porownawczy.
# ---------------------------------------------------------------------------


def edge_sweep_reclaim_v1(df, lookback=20):
    """V1: prosty liquidity sweep + close back inside."""
    high = df["high"].to_numpy(); low = df["low"].to_numpy(); close = df["close"].to_numpy()
    signals = []
    for i in range(lookback, len(df)):
        prior_high = high[i - lookback:i].max()
        prior_low = low[i - lookback:i].min()
        if high[i] > prior_high and close[i] < prior_high:
            signals.append((i, -1))
        elif low[i] < prior_low and close[i] > prior_low:
            signals.append((i, +1))
    return signals


def edge_failed_breakout_v1(df, range_lookback=30, fail_window=4):
    """V1: breakout w ostatnich swiecach + powrot do zakresu."""
    high = df["high"].to_numpy(); low = df["low"].to_numpy(); close = df["close"].to_numpy()
    signals = []
    start = range_lookback + fail_window
    for i in range(start, len(df)):
        base = slice(i - range_lookback - fail_window, i - fail_window)
        range_high = high[base].max(); range_low = low[base].min()
        recent = close[i - fail_window:i]
        broke_up = bool((recent > range_high).any())
        broke_down = bool((recent < range_low).any())
        if broke_up and close[i] < range_high:
            signals.append((i, -1))
        elif broke_down and close[i] > range_low:
            signals.append((i, +1))
    return signals


def edge_compression_expansion_v1(df, atr_window=14, percentile_window=100, pctl=0.20, expansion_mult=1.5):
    """V1: compression -> kierunek swiecy ekspansji."""
    a = atr(df, atr_window)
    open_ = df["open"].to_numpy(); high = df["high"].to_numpy(); low = df["low"].to_numpy(); close = df["close"].to_numpy()
    rng_bar = high - low
    signals = []
    for i in range(percentile_window, len(df)):
        thresh = np.nanpercentile(a[i - percentile_window:i], pctl * 100)
        compressed = a[i - 1] <= thresh
        expanded = rng_bar[i] > expansion_mult * a[i - 1]
        if compressed and expanded and a[i - 1] > 0:
            signals.append((i, 1 if close[i] > open_[i] else -1))
    return signals


def edge_funding_extremes_v1(df, funding, lookback=180, pctl=0.10):
    """V1: skrajny funding kontrariansko."""
    if funding is None or len(funding) == 0:
        return []
    f = funding.reindex(df.index, method="ffill").to_numpy()
    signals = []
    for i in range(lookback, len(df)):
        window = f[i - lookback:i]
        window = window[~np.isnan(window)]
        if len(window) < lookback // 2 or np.isnan(f[i]):
            continue
        hi = np.nanpercentile(window, (1 - pctl) * 100)
        lo = np.nanpercentile(window, pctl * 100)
        if f[i] >= hi and f[i] > 0:
            signals.append((i, -1))
        elif f[i] <= lo and f[i] < 0:
            signals.append((i, +1))
    return signals


def edge_momentum_v1(df, lookback=72, threshold=0.05):
    """V1: time-series momentum, prosty zwrot z lookback."""
    close = df["close"].to_numpy()
    signals = []
    for i in range(lookback, len(df)):
        ret = close[i] / close[i - lookback] - 1.0
        if ret > threshold:
            signals.append((i, +1))
        elif ret < -threshold:
            signals.append((i, -1))
    return signals


# ---------------------------------------------------------------------------
# V2 — ulepszone modele z kontekstem rynku.
# ---------------------------------------------------------------------------


def edge_sweep_reclaim_v2(df, lookback=30, wick_atr=0.25, close_back_atr=0.05, vol_mult=1.05):
    """V2: liquidity sweep + reclaim z jakoscia knota i wolumenem.

    Logika short:
    - aktualna swieca wybija prior_high z ostatnich N swiec,
    - zamyka sie z powrotem pod tym poziomem z buforem ATR,
    - ma gorny knot >= wick_atr * ATR,
    - close jest w dolnej polowie swiecy,
    - wolumen nie jest martwy.
    """
    open_ = df["open"].to_numpy(); high = df["high"].to_numpy(); low = df["low"].to_numpy(); close = df["close"].to_numpy()
    vol = df["volume"].to_numpy(); vma = _volume_ma(df, 20)
    a = atr(df, 14); pos = _bar_pos(open_, high, low, close)
    signals = []
    for i in range(max(lookback, 25), len(df)):
        if not np.isfinite(a[i - 1]) or a[i - 1] <= 0:
            continue
        prior_high = high[i - lookback:i].max(); prior_low = low[i - lookback:i].min()
        upper_wick = high[i] - max(open_[i], close[i])
        lower_wick = min(open_[i], close[i]) - low[i]
        vol_ok = np.isfinite(vma[i]) and vol[i] >= vol_mult * vma[i]

        if (high[i] > prior_high and
            close[i] < prior_high - close_back_atr * a[i - 1] and
            upper_wick >= wick_atr * a[i - 1] and
            pos[i] <= 0.45 and vol_ok):
            signals.append((i, -1))
        elif (low[i] < prior_low and
              close[i] > prior_low + close_back_atr * a[i - 1] and
              lower_wick >= wick_atr * a[i - 1] and
              pos[i] >= 0.55 and vol_ok):
            signals.append((i, +1))
    return signals


def edge_failed_breakout_range_v2(df, range_lookback=48, fail_window=6, max_adx=24, break_atr=0.10):
    """V2: failed breakout tylko w rynku range / niskiego trendu.

    To eliminuje najwiekszy blad V1: fade'owanie prawdziwego trendu.
    """
    high = df["high"].to_numpy(); low = df["low"].to_numpy(); close = df["close"].to_numpy()
    open_ = df["open"].to_numpy(); a = atr(df, 14); adx = _adx(df, 14)
    pos = _bar_pos(open_, high, low, close)
    signals = []
    start = range_lookback + fail_window + 20
    for i in range(start, len(df)):
        if not np.isfinite(adx[i]) or adx[i] > max_adx or a[i] <= 0:
            continue
        base = slice(i - range_lookback - fail_window, i - fail_window)
        range_high = high[base].max(); range_low = low[base].min()
        recent = close[i - fail_window:i]
        broke_up = bool((recent > range_high + break_atr * a[i]).any())
        broke_down = bool((recent < range_low - break_atr * a[i]).any())

        if broke_up and close[i] < range_high and pos[i] <= 0.55:
            signals.append((i, -1))
        elif broke_down and close[i] > range_low and pos[i] >= 0.45:
            signals.append((i, +1))
    return signals


def edge_compression_breakout_v2(df, range_lookback=36, atr_window=14, percentile_window=120,
                                 pctl=0.20, expansion_mult=1.20, close_buffer_atr=0.10,
                                 vol_mult=1.05):
    """V2: compression jako setup, kierunek dopiero po accepted breakout.

    V1 bralo kierunek koloru swiecy. V2 wymaga zamkniecia poza lokalnym
    zakresem, ekspansji zmiennosci i wolumenu. To lepiej opisuje:
    compression -> expansion -> acceptance.
    """
    open_ = df["open"].to_numpy(); high = df["high"].to_numpy(); low = df["low"].to_numpy(); close = df["close"].to_numpy()
    vol = df["volume"].to_numpy(); vma = _volume_ma(df, 20)
    a = atr(df, atr_window); pos = _bar_pos(open_, high, low, close)
    true_range = high - low
    signals = []
    start = max(percentile_window, range_lookback, 25)
    for i in range(start, len(df)):
        if not np.isfinite(a[i - 1]) or a[i - 1] <= 0:
            continue
        atr_thresh = np.nanpercentile(a[i - percentile_window:i], pctl * 100)
        compressed = a[i - 1] <= atr_thresh
        expanded = true_range[i] >= expansion_mult * a[i - 1]
        vol_ok = np.isfinite(vma[i]) and vol[i] >= vol_mult * vma[i]
        prior_high = high[i - range_lookback:i].max(); prior_low = low[i - range_lookback:i].min()
        buf = close_buffer_atr * a[i - 1]
        if compressed and expanded and vol_ok:
            if close[i] > prior_high + buf and pos[i] >= 0.65:
                signals.append((i, +1))
            elif close[i] < prior_low - buf and pos[i] <= 0.35:
                signals.append((i, -1))
    return signals


def edge_funding_exhaustion_v2(df, funding, lookback=180, pctl=0.10, ema_window=50,
                               dist_atr=1.00):
    """V2: skrajny funding + price exhaustion.

    Funding sam w sobie jest filtrem crowdingu. V2 nie shortuje samego
    dodatniego funding; wymaga jeszcze overextension ceny od EMA i swiecy
    sugerujacej utrate impetu.
    """
    if funding is None or len(funding) == 0:
        return []
    f = funding.reindex(df.index, method="ffill").to_numpy()
    open_ = df["open"].to_numpy(); high = df["high"].to_numpy(); low = df["low"].to_numpy(); close = df["close"].to_numpy()
    a = atr(df, 14); ema = _ema(close, ema_window); pos = _bar_pos(open_, high, low, close)
    signals = []
    start = max(lookback, ema_window + 5)
    for i in range(start, len(df)):
        window = f[i - lookback:i]
        window = window[~np.isnan(window)]
        if len(window) < lookback // 2 or np.isnan(f[i]) or not np.isfinite(ema[i]) or a[i] <= 0:
            continue
        hi = np.nanpercentile(window, (1 - pctl) * 100)
        lo = np.nanpercentile(window, pctl * 100)
        z_atr = (close[i] - ema[i]) / a[i]

        if f[i] >= hi and f[i] > 0 and z_atr >= dist_atr and pos[i] <= 0.55:
            signals.append((i, -1))
        elif f[i] <= lo and f[i] < 0 and z_atr <= -dist_atr and pos[i] >= 0.45:
            signals.append((i, +1))
    return signals


def edge_momentum_regime_v2(df, lookback=72, threshold=0.05, ema_fast=20, ema_slow=80,
                            adx_min=18, trigger_only=True):
    """V2: momentum z filtrem trend regime i tylko na swiezych triggerach.

    V1 generowalo wiele kolejnych sygnalow w tym samym trendzie. V2 pyta:
    czy momentum, EMA i ADX sa zgodne, a sygnal pojawia sie jako nowy stan.
    """
    close = df["close"].to_numpy()
    fast = _ema(close, ema_fast); slow = _ema(close, ema_slow); adx = _adx(df, 14)
    signals = []
    prev_state = 0
    start = max(lookback, ema_slow + 10)
    for i in range(start, len(df)):
        if not np.isfinite(fast[i]) or not np.isfinite(slow[i]) or not np.isfinite(adx[i]):
            continue
        ret = close[i] / close[i - lookback] - 1.0
        state = 0
        if ret > threshold and fast[i] > slow[i] and close[i] > slow[i] and adx[i] >= adx_min:
            state = +1
        elif ret < -threshold and fast[i] < slow[i] and close[i] < slow[i] and adx[i] >= adx_min:
            state = -1
        if state != 0 and (not trigger_only or state != prev_state):
            signals.append((i, state))
        prev_state = state
    return signals


# ---------------------------------------------------------------------------
# NEW — dodatkowe modele Fazy 0.
# ---------------------------------------------------------------------------


def edge_vwap_range_reversion(df, lookback=72, z_entry=1.6, max_adx=22):
    """Mean reversion do rolling VWAP tylko przy slabym trendzie.

    To jest czysty kandydat dla rynku range: cena daleko od value -> powrot.
    """
    close = df["close"].to_numpy()
    vwap = _rolling_vwap(df, lookback)
    spread = close - vwap
    sd = _rolling_std(spread, lookback)
    adx = _adx(df, 14)
    signals = []
    for i in range(max(lookback, 30), len(df)):
        if not np.isfinite(vwap[i]) or not np.isfinite(sd[i]) or sd[i] <= 0 or not np.isfinite(adx[i]):
            continue
        z = spread[i] / sd[i]
        if adx[i] <= max_adx:
            if z >= z_entry:
                signals.append((i, -1))
            elif z <= -z_entry:
                signals.append((i, +1))
    return signals


def edge_keltner_range_reversion(df, ema_window=40, atr_window=20, band_mult=1.8, max_adx=22):
    """Mean reversion z pasm Keltnera w range.

    Inne zrodlo niz VWAP: EMA + ATR, z filtrem ADX.
    """
    close = df["close"].to_numpy(); ema = _ema(close, ema_window); a = atr(df, atr_window); adx = _adx(df, 14)
    signals = []
    for i in range(max(ema_window, atr_window, 30), len(df)):
        if not np.isfinite(ema[i]) or not np.isfinite(adx[i]) or a[i] <= 0:
            continue
        upper = ema[i] + band_mult * a[i]
        lower = ema[i] - band_mult * a[i]
        if adx[i] <= max_adx:
            if close[i] > upper:
                signals.append((i, -1))
            elif close[i] < lower:
                signals.append((i, +1))
    return signals


def edge_donchian_acceptance(df, lookback=55, buffer_atr=0.10, adx_min=18, close_pos_min=0.70, vol_mult=1.00):
    """Trendowy breakout: zamkniecie poza Donchian range + acceptance.

    Ten model nie fade'uje wybicia; gra kontynuacje, gdy rynek akceptuje
    nowy poziom. Dobrze kontrastuje z failed_breakout_range_v2.
    """
    open_ = df["open"].to_numpy(); high = df["high"].to_numpy(); low = df["low"].to_numpy(); close = df["close"].to_numpy()
    vol = df["volume"].to_numpy(); vma = _volume_ma(df, 20)
    a = atr(df, 14); adx = _adx(df, 14); pos = _bar_pos(open_, high, low, close)
    signals = []
    for i in range(max(lookback, 30), len(df)):
        if a[i] <= 0 or not np.isfinite(adx[i]) or adx[i] < adx_min:
            continue
        prior_high = high[i - lookback:i].max(); prior_low = low[i - lookback:i].min()
        vol_ok = np.isfinite(vma[i]) and vol[i] >= vol_mult * vma[i]
        if close[i] > prior_high + buffer_atr * a[i] and pos[i] >= close_pos_min and vol_ok:
            signals.append((i, +1))
        elif close[i] < prior_low - buffer_atr * a[i] and pos[i] <= 1 - close_pos_min and vol_ok:
            signals.append((i, -1))
    return signals


def edge_breakout_retest(df, lookback=48, retest_window=5, buffer_atr=0.10, close_pos_min=0.60):
    """Breakout -> retest -> hold poziomu.

    Sygnał dopiero po obronie poziomu, nie w pierwszej emocji wybicia.
    """
    open_ = df["open"].to_numpy(); high = df["high"].to_numpy(); low = df["low"].to_numpy(); close = df["close"].to_numpy()
    a = atr(df, 14); pos = _bar_pos(open_, high, low, close)
    signals = []
    start = lookback + retest_window + 20
    for i in range(start, len(df)):
        if a[i] <= 0:
            continue
        base = slice(i - lookback - retest_window, i - retest_window)
        range_high = high[base].max(); range_low = low[base].min()
        recent = close[i - retest_window:i]
        broke_up = bool((recent > range_high + buffer_atr * a[i]).any())
        broke_down = bool((recent < range_low - buffer_atr * a[i]).any())
        buf = buffer_atr * a[i]

        if broke_up and low[i] <= range_high + buf and close[i] > range_high and pos[i] >= close_pos_min:
            signals.append((i, +1))
        elif broke_down and high[i] >= range_low - buf and close[i] < range_low and pos[i] <= 1 - close_pos_min:
            signals.append((i, -1))
    return signals


def edge_volume_climax_reversal(df, lookback=50, vol_mult=2.0, wick_atr=0.35, dist_atr=1.0, ema_window=50):
    """Climactic volume + wick + overextension = kontrarianski reversal.

    Kandydat do mean reversion po panice/euforii.
    """
    open_ = df["open"].to_numpy(); high = df["high"].to_numpy(); low = df["low"].to_numpy(); close = df["close"].to_numpy()
    vol = df["volume"].to_numpy(); vma = _volume_ma(df, lookback)
    a = atr(df, 14); ema = _ema(close, ema_window); pos = _bar_pos(open_, high, low, close)
    signals = []
    start = max(lookback, ema_window, 30)
    for i in range(start, len(df)):
        if not np.isfinite(vma[i]) or not np.isfinite(ema[i]) or a[i] <= 0:
            continue
        upper_wick = high[i] - max(open_[i], close[i])
        lower_wick = min(open_[i], close[i]) - low[i]
        z_atr = (close[i] - ema[i]) / a[i]
        climactic = vol[i] >= vol_mult * vma[i]
        if climactic and z_atr >= dist_atr and upper_wick >= wick_atr * a[i] and pos[i] <= 0.55:
            signals.append((i, -1))
        elif climactic and z_atr <= -dist_atr and lower_wick >= wick_atr * a[i] and pos[i] >= 0.45:
            signals.append((i, +1))
    return signals



# ---------------------------------------------------------------------------
# PHASE 0.2 — rozbiorka najlepszych kandydatow.
# Nie dodajemy nowych przypadkowych modeli. Ulepszamy tylko:
# breakout/retest, failed breakout in range, sweep/reclaim oraz filtry
# momentum i Donchian acceptance.
# ---------------------------------------------------------------------------


def _safe_body(open_, close, eps=1e-12):
    return np.maximum(np.abs(close - open_), eps)


def _wick_body_ratios(open_, high, low, close):
    body = _safe_body(open_, close)
    upper = high - np.maximum(open_, close)
    lower = np.minimum(open_, close) - low
    return upper / body, lower / body


def _momentum_bias(df, lookback=72, threshold=0.04, ema_fast=20, ema_slow=80, adx_min=16):
    """Zwraca tablice -1/0/+1 jako filtr kierunku, nie samodzielny trigger."""
    close = df["close"].to_numpy()
    fast = _ema(close, ema_fast)
    slow = _ema(close, ema_slow)
    adx = _adx(df, 14)
    out = np.zeros(len(df), dtype=int)
    start = max(lookback, ema_slow + 10)
    for i in range(start, len(df)):
        if not (np.isfinite(fast[i]) and np.isfinite(slow[i]) and np.isfinite(adx[i])):
            continue
        ret = close[i] / close[i - lookback] - 1.0
        if ret > threshold and fast[i] > slow[i] and close[i] > slow[i] and adx[i] >= adx_min:
            out[i] = +1
        elif ret < -threshold and fast[i] < slow[i] and close[i] < slow[i] and adx[i] >= adx_min:
            out[i] = -1
    return out


def _donchian_acceptance_bias(df, lookback=55, buffer_atr=0.08, close_pos_min=0.62, adx_min=16, vol_mult=0.95):
    """Acceptance filter: czy rynek zaakceptowal wybicie Donchian."""
    open_ = df["open"].to_numpy(); high = df["high"].to_numpy(); low = df["low"].to_numpy(); close = df["close"].to_numpy()
    vol = df["volume"].to_numpy(); vma = _volume_ma(df, 20)
    a = atr(df, 14); adx = _adx(df, 14); pos = _bar_pos(open_, high, low, close)
    out = np.zeros(len(df), dtype=int)
    for i in range(max(lookback, 30), len(df)):
        if a[i] <= 0 or not np.isfinite(adx[i]) or adx[i] < adx_min:
            continue
        prior_high = high[i - lookback:i].max(); prior_low = low[i - lookback:i].min()
        vol_ok = np.isfinite(vma[i]) and vol[i] >= vol_mult * vma[i]
        if vol_ok and close[i] > prior_high + buffer_atr * a[i] and pos[i] >= close_pos_min:
            out[i] = +1
        elif vol_ok and close[i] < prior_low - buffer_atr * a[i] and pos[i] <= 1 - close_pos_min:
            out[i] = -1
    return out


def _filter_direction(signals, direction):
    return [s for s in signals if s[1] == direction]


def edge_breakout_retest_v2(df, lookback=48, breakout_window=8, retest_window=6,
                            breakout_buffer_atr=0.08, retest_tolerance_atr=0.25,
                            max_retest_depth_atr=0.75, min_hold_bars=1,
                            breakout_vol_mult=1.08, adx_min=16, adx_max=45,
                            close_pos_min=0.58, momentum_filter=False,
                            donchian_filter=False):
    """Breakout -> retest -> hold, wersja 0.2.

    Ulepszenia vs poprzedni breakout_retest:
    - volume na swiecy wybicia,
    - jakosc retestu: dotkniecie strefy bez zbyt glebokiego powrotu,
    - minimalne utrzymanie poziomu przez ostatnie swiece,
    - filtr ADX/regime,
    - opcjonalny filtr momentum i/lub Donchian acceptance,
    - meta zapisuje poziom, typ i indeks swiecy wybicia.
    """
    open_ = df["open"].to_numpy(); high = df["high"].to_numpy(); low = df["low"].to_numpy(); close = df["close"].to_numpy()
    vol = df["volume"].to_numpy(); vma = _volume_ma(df, 20)
    a = atr(df, 14); adx = _adx(df, 14); pos = _bar_pos(open_, high, low, close)
    mom = _momentum_bias(df) if momentum_filter else np.zeros(len(df), dtype=int)
    don = _donchian_acceptance_bias(df, lookback=max(lookback, 55)) if donchian_filter else np.zeros(len(df), dtype=int)
    signals = []
    start = lookback + breakout_window + retest_window + 30
    for i in range(start, len(df)):
        if a[i] <= 0 or not np.isfinite(adx[i]) or adx[i] < adx_min or adx[i] > adx_max:
            continue
        base_end = i - breakout_window
        if base_end <= lookback:
            continue
        base = slice(base_end - lookback, base_end)
        level_high = high[base].max(); level_low = low[base].min()
        hold_from = max(base_end, i - min_hold_bars + 1)
        if hold_from < 0:
            continue

        # LONG: breakout powyzej zakresu, pozniej retest poziomu od gory i hold.
        long_breaks = []
        for j in range(base_end, i + 1):
            if a[j] <= 0 or not np.isfinite(vma[j]):
                continue
            vol_ok = vol[j] >= breakout_vol_mult * vma[j]
            if close[j] > level_high + breakout_buffer_atr * a[j] and pos[j] >= 0.62 and vol_ok:
                long_breaks.append(j)
        if long_breaks:
            j = long_breaks[0]
            zone_hi = level_high + retest_tolerance_atr * a[i]
            zone_lo = level_high - max_retest_depth_atr * a[i]
            after = slice(j + 1, i + 1)
            retested = bool((low[after] <= zone_hi).any() and (low[after] >= zone_lo).all())
            held = bool((close[hold_from:i + 1] > level_high).all())
            filters_ok = ((not momentum_filter) or mom[i] >= 1) and ((not donchian_filter) or don[j] >= 1 or don[i] >= 1)
            if retested and held and close[i] > level_high and pos[i] >= close_pos_min and filters_ok:
                signals.append((i, +1, {"pattern": "breakout_retest_v2", "side": "long", "level": float(level_high), "breakout_idx": int(j)}))
                continue

        # SHORT: wybicie dolem, retest od dolu i hold ponizej poziomu.
        short_breaks = []
        for j in range(base_end, i + 1):
            if a[j] <= 0 or not np.isfinite(vma[j]):
                continue
            vol_ok = vol[j] >= breakout_vol_mult * vma[j]
            if close[j] < level_low - breakout_buffer_atr * a[j] and pos[j] <= 0.38 and vol_ok:
                short_breaks.append(j)
        if short_breaks:
            j = short_breaks[0]
            zone_lo = level_low - retest_tolerance_atr * a[i]
            zone_hi = level_low + max_retest_depth_atr * a[i]
            after = slice(j + 1, i + 1)
            retested = bool((high[after] >= zone_lo).any() and (high[after] <= zone_hi).all())
            held = bool((close[hold_from:i + 1] < level_low).all())
            filters_ok = ((not momentum_filter) or mom[i] <= -1) and ((not donchian_filter) or don[j] <= -1 or don[i] <= -1)
            if retested and held and close[i] < level_low and pos[i] <= 1 - close_pos_min and filters_ok:
                signals.append((i, -1, {"pattern": "breakout_retest_v2", "side": "short", "level": float(level_low), "breakout_idx": int(j)}))
    return signals


def edge_breakout_retest_v2_long(df):
    return _filter_direction(edge_breakout_retest_v2(df), +1)


def edge_breakout_retest_v2_short(df):
    return _filter_direction(edge_breakout_retest_v2(df), -1)


def edge_breakout_retest_v2_momentum_filter(df):
    return edge_breakout_retest_v2(df, momentum_filter=True)


def edge_breakout_retest_v2_donchian_filter(df):
    return edge_breakout_retest_v2(df, donchian_filter=True)


def edge_breakout_retest_v2_mom_don_filter(df):
    return edge_breakout_retest_v2(df, momentum_filter=True, donchian_filter=True)


def _range_quality_ok(df, i, base_slice, max_adx=24, min_width_atr=2.0, max_width_atr=9.0, max_slope_atr=0.35):
    high = df["high"].to_numpy(); low = df["low"].to_numpy(); close = df["close"].to_numpy()
    adx = _adx(df, 14); a = atr(df, 14)
    ema_fast = _ema(close, 20); ema_slow = _ema(close, 80)
    if a[i] <= 0 or not np.isfinite(adx[i]) or adx[i] > max_adx:
        return False
    width = high[base_slice].max() - low[base_slice].min()
    width_atr = width / max(a[i], 1e-12)
    slope_atr = abs(ema_fast[i] - ema_slow[i]) / max(a[i], 1e-12) if np.isfinite(ema_fast[i]) and np.isfinite(ema_slow[i]) else 999
    return min_width_atr <= width_atr <= max_width_atr and slope_atr <= max_slope_atr


def edge_failed_breakout_range_v3(df, range_lookback=48, fail_window=8, max_adx=24,
                                  sweep_atr=0.05, close_back_atr=0.02,
                                  min_wick_body=0.60, close_pos_short=0.58,
                                  close_pos_long=0.42, require_range_quality=True,
                                  side="both"):
    """Failed breakout in range, wersja 0.2/v3.

    Ulepszenia:
    - bierze pod uwage sweep wickiem, nie tylko close poza zakresem,
    - wymaga close back inside range,
    - wick/body ratio na swiecy sweepa albo swiecy powrotu,
    - range quality score: niski ADX, sensowna szerokosc range i mala roznica EMA,
    - side pozwala osobno testowac sweep high i sweep low.
    """
    open_ = df["open"].to_numpy(); high = df["high"].to_numpy(); low = df["low"].to_numpy(); close = df["close"].to_numpy()
    a = atr(df, 14); adx = _adx(df, 14); pos = _bar_pos(open_, high, low, close)
    upper_ratio, lower_ratio = _wick_body_ratios(open_, high, low, close)
    signals = []
    start = range_lookback + fail_window + 90
    for i in range(start, len(df)):
        if a[i] <= 0 or not np.isfinite(adx[i]) or adx[i] > max_adx:
            continue
        base = slice(i - range_lookback - fail_window, i - fail_window)
        if require_range_quality and not _range_quality_ok(df, i, base, max_adx=max_adx):
            continue
        range_high = high[base].max(); range_low = low[base].min()
        recent_idx = range(i - fail_window, i + 1)

        if side in ("both", "high"):
            candidates = [j for j in recent_idx if high[j] > range_high + sweep_atr * a[j]]
            if candidates:
                j = candidates[-1]
                wick_ok = max(upper_ratio[j], upper_ratio[i]) >= min_wick_body
                close_back = close[i] < range_high - close_back_atr * a[i]
                if close_back and wick_ok and pos[i] <= close_pos_short:
                    signals.append((i, -1, {"pattern": "failed_breakout_range_v3", "side": "sweep_high", "level": float(range_high), "sweep_idx": int(j)}))
                    continue

        if side in ("both", "low"):
            candidates = [j for j in recent_idx if low[j] < range_low - sweep_atr * a[j]]
            if candidates:
                j = candidates[-1]
                wick_ok = max(lower_ratio[j], lower_ratio[i]) >= min_wick_body
                close_back = close[i] > range_low + close_back_atr * a[i]
                if close_back and wick_ok and pos[i] >= close_pos_long:
                    signals.append((i, +1, {"pattern": "failed_breakout_range_v3", "side": "sweep_low", "level": float(range_low), "sweep_idx": int(j)}))
    return signals


def edge_failed_breakout_range_v3_sweep_high(df):
    return edge_failed_breakout_range_v3(df, side="high")


def edge_failed_breakout_range_v3_sweep_low(df):
    return edge_failed_breakout_range_v3(df, side="low")


def edge_sweep_reclaim_v3_reversal(df, lookback=36, retest_window=5, sweep_atr=0.05,
                                   close_back_atr=0.02, retest_tolerance_atr=0.25,
                                   min_quality=1.45, close_pos_min=0.55):
    """Liquidity sweep -> reclaim -> retest/rejection, wersja reversal.

    Volume nie jest twardym filtrem. Jest czescia quality score razem z knotem,
    dystansem wybicia i jakoscia zamkniecia. Dzieki temu nie odrzucamy dobrych
    setupow tylko dlatego, ze wolumen byl minimalnie ponizej stalego progu.
    """
    open_ = df["open"].to_numpy(); high = df["high"].to_numpy(); low = df["low"].to_numpy(); close = df["close"].to_numpy()
    vol = df["volume"].to_numpy(); vma = _volume_ma(df, 20)
    a = atr(df, 14); pos = _bar_pos(open_, high, low, close)
    upper_ratio, lower_ratio = _wick_body_ratios(open_, high, low, close)
    signals = []
    start = lookback + retest_window + 30
    for i in range(start, len(df)):
        if a[i] <= 0:
            continue
        base_end = i - retest_window
        base = slice(base_end - lookback, base_end)
        level_high = high[base].max(); level_low = low[base].min()
        # szukamy sweepa w ostatnich barach, potem retestu poziomu i odrzucenia.
        for j in range(base_end, i + 1):
            if a[j] <= 0:
                continue
            vol_ratio = vol[j] / vma[j] if np.isfinite(vma[j]) and vma[j] > 0 else 1.0
            # sweep high -> short reversal
            sweep_dist = (high[j] - level_high) / max(a[j], 1e-12)
            close_back = (level_high - close[j]) / max(a[j], 1e-12)
            quality = max(0, sweep_dist) + max(0, close_back) + min(max(vol_ratio, 0), 3.0) / 3.0 + min(upper_ratio[j], 3.0) / 3.0
            retest = high[i] >= level_high - retest_tolerance_atr * a[i]
            reject = close[i] < level_high and pos[i] <= close_pos_min
            if high[j] > level_high + sweep_atr * a[j] and close[j] < level_high - close_back_atr * a[j] and quality >= min_quality and retest and reject:
                signals.append((i, -1, {"pattern": "sweep_reclaim_v3_reversal", "side": "sweep_high", "level": float(level_high), "sweep_idx": int(j), "quality": float(quality)}))
                break
            # sweep low -> long reversal
            sweep_dist = (level_low - low[j]) / max(a[j], 1e-12)
            close_back = (close[j] - level_low) / max(a[j], 1e-12)
            quality = max(0, sweep_dist) + max(0, close_back) + min(max(vol_ratio, 0), 3.0) / 3.0 + min(lower_ratio[j], 3.0) / 3.0
            retest = low[i] <= level_low + retest_tolerance_atr * a[i]
            reject = close[i] > level_low and pos[i] >= 1 - close_pos_min
            if low[j] < level_low - sweep_atr * a[j] and close[j] > level_low + close_back_atr * a[j] and quality >= min_quality and retest and reject:
                signals.append((i, +1, {"pattern": "sweep_reclaim_v3_reversal", "side": "sweep_low", "level": float(level_low), "sweep_idx": int(j), "quality": float(quality)}))
                break
    return signals


def edge_sweep_reclaim_v3_continuation(df, lookback=36, reclaim_window=5, sweep_atr=0.05,
                                       close_back_atr=0.02, acceptance_atr=0.08,
                                       min_quality=1.30, close_pos_min=0.62):
    """Fakeout failed -> continuation.

    Po sweepie i powrocie do zakresu rynek ponownie reclaimuje poziom i utrzymuje
    acceptance. To testuje druga strone zjawiska: nie fade, tylko kontynuacje po
    nieudanym fakeoucie.
    """
    open_ = df["open"].to_numpy(); high = df["high"].to_numpy(); low = df["low"].to_numpy(); close = df["close"].to_numpy()
    vol = df["volume"].to_numpy(); vma = _volume_ma(df, 20)
    a = atr(df, 14); pos = _bar_pos(open_, high, low, close)
    upper_ratio, lower_ratio = _wick_body_ratios(open_, high, low, close)
    signals = []
    start = lookback + reclaim_window + 30
    for i in range(start, len(df)):
        if a[i] <= 0:
            continue
        base_end = i - reclaim_window
        base = slice(base_end - lookback, base_end)
        level_high = high[base].max(); level_low = low[base].min()
        for j in range(base_end, i):
            if a[j] <= 0:
                continue
            vol_ratio = vol[j] / vma[j] if np.isfinite(vma[j]) and vma[j] > 0 else 1.0
            # sweep high, potem reclaim i continuation long
            sweep_dist = (high[j] - level_high) / max(a[j], 1e-12)
            close_back = (level_high - close[j]) / max(a[j], 1e-12)
            quality = max(0, sweep_dist) + max(0, close_back) + min(max(vol_ratio, 0), 3.0) / 3.0 + min(upper_ratio[j], 3.0) / 3.0
            if (high[j] > level_high + sweep_atr * a[j] and close[j] < level_high - close_back_atr * a[j] and
                quality >= min_quality and close[i] > level_high + acceptance_atr * a[i] and pos[i] >= close_pos_min):
                signals.append((i, +1, {"pattern": "sweep_reclaim_v3_continuation", "side": "reclaim_high", "level": float(level_high), "sweep_idx": int(j), "quality": float(quality)}))
                break
            # sweep low, potem reclaim dolem i continuation short
            sweep_dist = (level_low - low[j]) / max(a[j], 1e-12)
            close_back = (close[j] - level_low) / max(a[j], 1e-12)
            quality = max(0, sweep_dist) + max(0, close_back) + min(max(vol_ratio, 0), 3.0) / 3.0 + min(lower_ratio[j], 3.0) / 3.0
            if (low[j] < level_low - sweep_atr * a[j] and close[j] > level_low + close_back_atr * a[j] and
                quality >= min_quality and close[i] < level_low - acceptance_atr * a[i] and pos[i] <= 1 - close_pos_min):
                signals.append((i, -1, {"pattern": "sweep_reclaim_v3_continuation", "side": "reclaim_low", "level": float(level_low), "sweep_idx": int(j), "quality": float(quality)}))
                break
    return signals


EDGE_GROUPS = {
    "core": {
        "sweep_reclaim_v1": edge_sweep_reclaim_v1,
        "failed_breakout_v1": edge_failed_breakout_v1,
        "compression_expansion_v1": edge_compression_expansion_v1,
        "funding_extremes_v1": edge_funding_extremes_v1,
        "momentum_v1": edge_momentum_v1,
    },
    "v2": {
        "sweep_reclaim_v2": edge_sweep_reclaim_v2,
        "failed_breakout_range_v2": edge_failed_breakout_range_v2,
        "compression_breakout_v2": edge_compression_breakout_v2,
        "funding_exhaustion_v2": edge_funding_exhaustion_v2,
        "momentum_regime_v2": edge_momentum_regime_v2,
    },
    "new": {
        "vwap_range_reversion": edge_vwap_range_reversion,
        "keltner_range_reversion": edge_keltner_range_reversion,
        "donchian_acceptance": edge_donchian_acceptance,
        "breakout_retest": edge_breakout_retest,
        "volume_climax_reversal": edge_volume_climax_reversal,
    },
    "phase02": {
        # Glowny kandydat 4H: breakout -> retest -> hold, rozbity na warianty.
        "breakout_retest_v2": edge_breakout_retest_v2,
        "breakout_retest_v2_long": edge_breakout_retest_v2_long,
        "breakout_retest_v2_short": edge_breakout_retest_v2_short,
        "breakout_retest_v2_momentum_filter": edge_breakout_retest_v2_momentum_filter,
        "breakout_retest_v2_donchian_filter": edge_breakout_retest_v2_donchian_filter,
        "breakout_retest_v2_mom_don_filter": edge_breakout_retest_v2_mom_don_filter,

        # Glowny kandydat 1H: failed breakout in range, rozbity na sweep high/low.
        "failed_breakout_range_v3": edge_failed_breakout_range_v3,
        "failed_breakout_range_v3_sweep_high": edge_failed_breakout_range_v3_sweep_high,
        "failed_breakout_range_v3_sweep_low": edge_failed_breakout_range_v3_sweep_low,

        # Liquidity sweep: oddzielamy reversal od continuation.
        "sweep_reclaim_v3_reversal": edge_sweep_reclaim_v3_reversal,
        "sweep_reclaim_v3_continuation": edge_sweep_reclaim_v3_continuation,
    },
}



def edge_ltc_failed_breakout_funding(df, funding=None,
                                      range_lookback=30, fail_window=4,
                                      funding_lookback=180, funding_pctl=0.10,
                                      confirm_window=3):
    """Kombinacja: failed_breakout_v1 AND funding_extremes_v1.
    Sygnał tylko gdy oba warunki zgadzają się co do kierunku
    w oknie confirm_window świec od siebie.
    """
    # --- składnik 1: failed breakout ---
    fb_signals = {}  # {indeks: kierunek}
    high  = df["high"].to_numpy()
    low   = df["low"].to_numpy()
    close = df["close"].to_numpy()
    start = range_lookback + fail_window
    for i in range(start, len(df)):
        base_sl = slice(i - range_lookback - fail_window, i - fail_window)
        r_high  = high[base_sl].max()
        r_low   = low[base_sl].min()
        recent  = close[i - fail_window:i]
        if bool((recent > r_high).any()) and close[i] < r_high:
            fb_signals[i] = -1
        elif bool((recent < r_low).any()) and close[i] > r_low:
            fb_signals[i] = +1

    # --- składnik 2: funding extreme ---
    fe_signals = {}  # {indeks: kierunek}
    if funding is not None and len(funding) > 0:
        f = funding.reindex(df.index, method="ffill").to_numpy()
        for i in range(funding_lookback, len(df)):
            window = f[i - funding_lookback:i]
            window = window[~np.isnan(window)]
            if len(window) < funding_lookback // 2 or np.isnan(f[i]):
                continue
            hi = np.nanpercentile(window, (1 - funding_pctl) * 100)
            lo = np.nanpercentile(window, funding_pctl * 100)
            if f[i] >= hi and f[i] > 0:
                fe_signals[i] = -1
            elif f[i] <= lo and f[i] < 0:
                fe_signals[i] = +1

    # --- kombinacja: oba muszą się zgadzać w oknie ±confirm_window ---
    combined = []
    for fb_idx, fb_dir in fb_signals.items():
        for offset in range(-confirm_window, confirm_window + 1):
            fe_dir = fe_signals.get(fb_idx + offset)
            if fe_dir == fb_dir:
                combined.append((fb_idx, fb_dir))
                break
    return combined

def get_edges(groups=("v2", "new")):
    """Zwraca slownik edge'ow dla wybranych grup."""
    out = {}
    for group in groups:
        if group not in EDGE_GROUPS:
            raise ValueError(f"Nieznana grupa edge'ow: {group}. Dostepne: {list(EDGE_GROUPS)}")
        out.update(EDGE_GROUPS[group])
    return out


# kompatybilnosc wstecz: domyslnie nie odpalamy core, bo juz byl testowany
EDGES = get_edges(("phase02",))


# ---------------------------------------------------------------------------
# NEW PREMIUM EDGES — zakodowane na podstawie literatury akademickiej
# ---------------------------------------------------------------------------

def edge_liquidation_proxy(df, atr_mult=2.0, vol_mult=2.0, reclaim_window=3,
                           min_hold_bars=2):
    """Proxy likwidacji: duży ruch + volume spike + szybki reclaim.
    
    Hipoteza: po gwałtownej wyprzedaży z dużym wolumenem, jeśli rynek
    szybko odzyskuje połowę świecy → wymuszone likwidacje wchłonięte.
    Long po reclaim, short analogicznie.
    """
    open_ = df["open"].to_numpy()
    high  = df["high"].to_numpy()
    low   = df["low"].to_numpy()
    close = df["close"].to_numpy()
    vol   = df["volume"].to_numpy()
    vma   = _volume_ma(df, 20)
    a     = atr(df, 14)
    signals = []
    start = 25
    for i in range(start, len(df) - reclaim_window):
        if a[i] <= 0 or not np.isfinite(vma[i]) or vma[i] <= 0:
            continue
        candle_range = high[i] - low[i]
        vol_spike    = vol[i] >= vol_mult * vma[i]
        big_candle   = candle_range >= atr_mult * a[i]
        if not (vol_spike and big_candle):
            continue
        midpoint = (high[i] + low[i]) / 2
        # Bearish candle → szukamy long reclaim
        if close[i] < open_[i]:
            for j in range(i + 1, min(i + 1 + reclaim_window, len(df))):
                if close[j] > midpoint:
                    signals.append((j, +1, {
                        "pattern": "liquidation_long_reclaim",
                        "candle_idx": i,
                        "midpoint": float(midpoint)
                    }))
                    break
        # Bullish candle → szukamy short reclaim
        else:
            for j in range(i + 1, min(i + 1 + reclaim_window, len(df))):
                if close[j] < midpoint:
                    signals.append((j, -1, {
                        "pattern": "liquidation_short_reclaim",
                        "candle_idx": i,
                        "midpoint": float(midpoint)
                    }))
                    break
    return signals


def edge_relative_strength_rotation(df, lookback=20, top_pct=0.4,
                                    atr_window=14):
    """Relatywna siła: normalizowana zmiana ceny vs własna zmienność.
    
    Hipoteza: aktywa z wysokim z-score zwrotu (relatywna siła)
    kontynuują trend krótkoterminowo (cross-sectional momentum proxy).
    Sygnał long gdy znormalizowany zwrot > 1 std, short gdy < -1 std.
    """
    close = df["close"].to_numpy()
    a     = atr(df, atr_window)
    signals = []
    start = lookback + atr_window + 5
    for i in range(start, len(df)):
        if a[i] <= 0:
            continue
        ret    = (close[i] - close[i - lookback]) / close[i - lookback]
        window = np.array([
            (close[k] - close[k - lookback]) / max(close[k - lookback], 1e-12)
            for k in range(i - lookback, i + 1)
            if k >= lookback
        ])
        if len(window) < 5:
            continue
        mu  = window.mean()
        std = window.std()
        if std <= 0:
            continue
        z = (ret - mu) / std
        if z > 1.0:
            signals.append((i, +1, {"pattern": "rel_strength_long",
                                    "z_score": float(z)}))
        elif z < -1.0:
            signals.append((i, -1, {"pattern": "rel_strength_short",
                                    "z_score": float(z)}))
    return signals


def edge_funding_momentum_divergence(df, funding, lookback=24,
                                     funding_lookback=180,
                                     funding_pctl=0.10,
                                     momentum_threshold=0.03):
    """Funding + momentum divergencja.
    
    Hipoteza: gdy funding ekstremalnie dodatni ALE momentum cenowy
    słabnie (cena robi niższe highs) → divergencja → short.
    Analogicznie long gdy funding ekstremalnie ujemny + cena stabilna.
    """
    if funding is None or len(funding) == 0:
        return []
    close  = df["close"].to_numpy()
    high   = df["high"].to_numpy()
    f      = funding.reindex(df.index, method="ffill").to_numpy()
    a      = atr(df, 14)
    f_ser  = pd.Series(f)
    f_hi   = f_ser.rolling(funding_lookback, min_periods=funding_lookback // 2).quantile(1 - funding_pctl)
    f_lo   = f_ser.rolling(funding_lookback, min_periods=funding_lookback // 2).quantile(funding_pctl)
    f_hi_a = f_hi.to_numpy()
    f_lo_a = f_lo.to_numpy()
    signals = []
    start   = max(funding_lookback, lookback, 20)
    for i in range(start, len(df)):
        if not np.isfinite(f[i]) or a[i] <= 0:
            continue
        # Momentum z lookback świec
        ret = (close[i] - close[i - lookback]) / max(close[i - lookback], 1e-12)
        # Funding ekstremalnie dodatni + momentum słabnie → short
        if (np.isfinite(f_hi_a[i]) and f[i] >= f_hi_a[i] and
                f[i] > 0 and ret < momentum_threshold):
            signals.append((i, -1, {
                "pattern": "funding_momentum_divergence_short",
                "funding": float(f[i]), "ret": float(ret)
            }))
        # Funding ekstremalnie ujemny + cena stabilna → long
        elif (np.isfinite(f_lo_a[i]) and f[i] <= f_lo_a[i] and
              f[i] < 0 and ret > -momentum_threshold):
            signals.append((i, +1, {
                "pattern": "funding_momentum_divergence_long",
                "funding": float(f[i]), "ret": float(ret)
            }))
    return signals


def edge_panic_no_followthrough(df, atr_mult=1.8, vol_mult=1.5,
                                followthrough_window=2, min_hold=1):
    """Panika bez follow-through = mean reversion.
    
    Hipoteza: duża bearish świeca + high volume + kolejne N świec
    nie kontynuuje spadku → wymuszona wyprzedaż wchłonięta.
    """
    open_ = df["open"].to_numpy()
    high  = df["high"].to_numpy()
    low   = df["low"].to_numpy()
    close = df["close"].to_numpy()
    vol   = df["volume"].to_numpy()
    vma   = _volume_ma(df, 20)
    a     = atr(df, 14)
    pos   = _bar_pos(open_, high, low, close)
    signals = []
    start = 25
    for i in range(start, len(df) - followthrough_window - 1):
        if a[i] <= 0 or not np.isfinite(vma[i]) or vma[i] <= 0:
            continue
        bearish     = close[i] < open_[i]
        big_candle  = (high[i] - low[i]) >= atr_mult * a[i]
        vol_spike   = vol[i] >= vol_mult * vma[i]
        low_close   = pos[i] <= 0.35
        if not (bearish and big_candle and vol_spike and low_close):
            continue
        panic_low = low[i]
        # Sprawdź czy follow-through nie nastąpił
        no_follow = all(
            low[j] >= panic_low - 0.1 * a[i]
            for j in range(i + 1, i + 1 + followthrough_window)
        )
        if no_follow:
            entry_bar = i + followthrough_window
            if entry_bar < len(df):
                signals.append((entry_bar, +1, {
                    "pattern": "panic_no_followthrough",
                    "panic_idx": i,
                    "panic_low": float(panic_low)
                }))
    return signals


def edge_feargreed_extreme(df, feargreed=None, extreme_low=25, extreme_high=75,
                            lookback_days=7):
    """Contrarian na ekstremalnym Fear & Greed Index.
    Extreme fear (<25) → long, extreme greed (>75) → short.
    Sygnał gdy FG przekracza próg i utrzymuje się lookback_days dni.
    """
    if feargreed is None or len(feargreed) == 0:
        return []
    fg = feargreed.reindex(df.index, method="ffill")
    signals = []
    for i in range(lookback_days, len(df)):
        window = fg.iloc[i - lookback_days:i]
        if window.isna().all():
            continue
        avg_fg = float(window.mean())
        if avg_fg <= extreme_low:
            signals.append((i, +1))
        elif avg_fg >= extreme_high:
            signals.append((i, -1))
    return signals


def edge_cross_exchange_premium(df, bybit_df=None, premium_pctl=0.90,
                                 lookback=60, min_premium_pct=0.003):
    """Cross-exchange premium: gdy Binance cena odbiega od Bybit o >min_premium_pct.
    Binance droższy → short (mean reversion), tańszy → long.
    """
    if bybit_df is None or len(bybit_df) == 0:
        return []
    bybit_close = bybit_df["close"].reindex(df.index, method="ffill")
    signals = []
    for i in range(lookback, len(df)):
        bn_close  = float(df["close"].iloc[i])
        by_close  = float(bybit_close.iloc[i])
        if by_close == 0 or np.isnan(by_close):
            continue
        premium = (bn_close - by_close) / by_close
        window_premiums = []
        for j in range(i - lookback, i):
            bn = float(df["close"].iloc[j])
            by = float(bybit_close.iloc[j])
            if by > 0 and not np.isnan(by):
                window_premiums.append((bn - by) / by)
        if len(window_premiums) < lookback // 2:
            continue
        hi = np.nanpercentile(window_premiums, premium_pctl * 100)
        lo = np.nanpercentile(window_premiums, (1 - premium_pctl) * 100)
        if premium >= hi and abs(premium) >= min_premium_pct:
            signals.append((i, -1))
        elif premium <= lo and abs(premium) >= min_premium_pct:
            signals.append((i, +1))
    return signals


def edge_wick_pressure(df, lookback=20, wick_ratio_pctl=0.85,
                        min_wick_ratio=1.5):
    """Wick/body pressure jako proxy order book imbalance.
    Długi dolny wick (kupujący absorbują sprzedaż) → long.
    Długi górny wick (sprzedający absorbują zakupy) → short.
    """
    signals = []
    high  = df["high"].to_numpy()
    low   = df["low"].to_numpy()
    open_ = df["open"].to_numpy()
    close = df["close"].to_numpy()
    for i in range(lookback, len(df)):
        body = abs(close[i] - open_[i])
        if body == 0:
            continue
        lower_wick = min(open_[i], close[i]) - low[i]
        upper_wick = high[i] - max(open_[i], close[i])
        lower_ratio = lower_wick / body
        upper_ratio = upper_wick / body
        window_lower = []
        window_upper = []
        for j in range(i - lookback, i):
            b = abs(close[j] - open_[j])
            if b == 0:
                continue
            window_lower.append((min(open_[j], close[j]) - low[j]) / b)
            window_upper.append((high[j] - max(open_[j], close[j])) / b)
        if len(window_lower) < lookback // 2:
            continue
        hi_lower = np.nanpercentile(window_lower, wick_ratio_pctl * 100)
        hi_upper = np.nanpercentile(window_upper, wick_ratio_pctl * 100)
        if lower_ratio >= hi_lower and lower_ratio >= min_wick_ratio:
            signals.append((i, +1))
        elif upper_ratio >= hi_upper and upper_ratio >= min_wick_ratio:
            signals.append((i, -1))
    return signals


def edge_funding_confluence(df, funding=None, lookback=180, pctl=0.10,
                             n_assets_required=3):
    """Multi-asset funding confluence: sygnał gdy wiele assetów
    ma jednocześnie ekstremalny funding w tym samym kierunku.
    Używa tylko funding dla bieżącego assetu jako proxy – 
    wymaga dostosowania w runnerze dla pełnej wersji.
    """
    if funding is None or len(funding) == 0:
        return []
    f = funding.reindex(df.index, method="ffill").to_numpy()
    signals = []
    for i in range(lookback, len(df)):
        window = f[i - lookback:i]
        window = window[~np.isnan(window)]
        if len(window) < lookback // 2 or np.isnan(f[i]):
            continue
        hi = np.nanpercentile(window, (1 - pctl) * 100)
        lo = np.nanpercentile(window, pctl * 100)
        strength = 0
        if f[i] >= hi and f[i] > 0:
            signals.append((i, -1))
        elif f[i] <= lo and f[i] < 0:
            signals.append((i, +1))
    return signals
