"""Silnik backtestu i statystyki dla Fazy 0.

Model transakcji domyslny:
- wejscie po open NASTEPNEJ swiecy po sygnale,
- stop loss = sl_mult * ATR,
- take profit = rr * ryzyko,
- wyjscie czasowe po max_hold swiecach,
- koszt round-trip naliczany do kazdej transakcji.

Wyniki sa w R, czyli wielokrotnosciach ryzyka.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def atr(df, window=14):
    """Average True Range jako tablica numpy wyrownana do df."""
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    prev_close = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    return pd.Series(tr).rolling(window, min_periods=1).mean().to_numpy()


DEFAULTS = dict(
    sl_mult=1.5,
    rr=2.0,
    max_hold=24,
    cost=0.0015,
    min_gap=12,
    exit_mode="fixed",       # fixed albo trailing
    trail_mult=2.0,
    breakeven_after_R=1.2,
)


def _parse_signal(signal):
    """Wspiera tuple (idx, dir) oraz (idx, dir, meta)."""
    if len(signal) == 2:
        return signal[0], signal[1], {}
    if len(signal) == 3:
        return signal[0], signal[1], signal[2] or {}
    raise ValueError(f"Niepoprawny sygnal: {signal}")


def simulate_trade(o, h, l, c, atr_arr, entry_idx, direction, p):
    """Symuluje jedna transakcje. Zwraca wynik w R albo None.

    Od Phase 0.4 wynik liczony jest przez wersje detail, zeby wszystkie
    exit profile byly identyczne w tabeli wynikow i w eksporcie transakcji.
    """
    d = simulate_trade_detail(o, h, l, c, atr_arr, entry_idx, direction, p)
    if d is None:
        return None
    return d.get("result_R")


def simulate_trade_fixed(o, h, l, c, atr_arr, entry_idx, direction, p):
    j0 = entry_idx + 1
    if j0 >= len(c):
        return None
    risk = atr_arr[entry_idx] * p["sl_mult"]
    if not np.isfinite(risk) or risk <= 0:
        return None
    entry = o[j0]
    cost_R = p["cost"] * entry / risk
    if direction == 1:
        sl, tp = entry - risk, entry + risk * p["rr"]
    else:
        sl, tp = entry + risk, entry - risk * p["rr"]
    last = min(j0 + p["max_hold"], len(c)) - 1
    for j in range(j0, last + 1):
        if direction == 1:
            hit_sl, hit_tp = l[j] <= sl, h[j] >= tp
        else:
            hit_sl, hit_tp = h[j] >= sl, l[j] <= tp
        if hit_sl:             # pesymistycznie: SL przed TP w tej samej swiecy
            return -1.0 - cost_R
        if hit_tp:
            return p["rr"] - cost_R
    exit_price = c[last]
    return ((exit_price - entry) / risk * direction) - cost_R


def simulate_trade_trailing(o, h, l, c, atr_arr, entry_idx, direction, p):
    """Alternatywny exit do testow trend/momentum: trailing ATR bez stalego TP.

    Nie jest domyslny. Uzywaj jako osobny test wrazliwosci, bo zwieksza liczbe
    porownan i ryzyko falszywych pozytywow.
    """
    j0 = entry_idx + 1
    if j0 >= len(c):
        return None
    initial_risk = atr_arr[entry_idx] * p["sl_mult"]
    if not np.isfinite(initial_risk) or initial_risk <= 0:
        return None
    entry = o[j0]
    cost_R = p["cost"] * entry / initial_risk
    if direction == 1:
        stop = entry - initial_risk
        best = entry
    else:
        stop = entry + initial_risk
        best = entry
    last = min(j0 + p["max_hold"], len(c)) - 1
    for j in range(j0, last + 1):
        a = atr_arr[j] if np.isfinite(atr_arr[j]) and atr_arr[j] > 0 else atr_arr[entry_idx]
        if direction == 1:
            best = max(best, h[j])
            if (best - entry) / initial_risk >= p.get("breakeven_after_R", 1.2):
                stop = max(stop, entry)
            stop = max(stop, best - p.get("trail_mult", 2.0) * a)
            if l[j] <= stop:
                return ((stop - entry) / initial_risk) - cost_R
        else:
            best = min(best, l[j])
            if (entry - best) / initial_risk >= p.get("breakeven_after_R", 1.2):
                stop = min(stop, entry)
            stop = min(stop, best + p.get("trail_mult", 2.0) * a)
            if h[j] >= stop:
                return ((entry - stop) / initial_risk) - cost_R
    exit_price = c[last]
    return ((exit_price - entry) / initial_risk * direction) - cost_R


def backtest_edge(df, signals, params=None):
    """Przepuszcza sygnaly przez model transakcji. Zwraca liste wynikow R."""
    p = {**DEFAULTS, **(params or {})}
    o = df["open"].to_numpy(); h = df["high"].to_numpy(); l = df["low"].to_numpy(); c = df["close"].to_numpy()
    a = atr(df, 14)
    results, last_entry = [], -10 ** 9
    for sig in signals:
        idx, direction, _meta = _parse_signal(sig)
        if idx - last_entry < p["min_gap"]:
            continue
        r = simulate_trade(o, h, l, c, a, idx, direction, p)
        if r is not None and np.isfinite(r):
            results.append(float(r))
            last_entry = idx
    return results


def expectancy(results):
    return float(np.mean(results)) if results else 0.0


def win_rate(results):
    return float(np.mean(np.array(results) > 0)) if results else 0.0


def profit_factor(results):
    if not results:
        return 0.0
    arr = np.array(results, dtype=float)
    gains = arr[arr > 0].sum()
    losses = -arr[arr < 0].sum()
    if losses <= 0:
        return float("inf") if gains > 0 else 0.0
    return float(gains / losses)


def max_drawdown_R(results):
    if not results:
        return 0.0
    eq = np.cumsum(np.array(results, dtype=float))
    peak = np.maximum.accumulate(eq)
    dd = eq - peak
    return float(dd.min())


def bootstrap_ci(results, n=5000, lo=2.5, hi=97.5, seed=0):
    if len(results) < 2:
        return (0.0, 0.0)
    rng = np.random.default_rng(seed)
    arr = np.array(results, dtype=float)
    sample = arr[rng.integers(0, len(arr), size=(n, len(arr)))]
    means = sample.mean(axis=1)
    return (float(np.percentile(means, lo)), float(np.percentile(means, hi)))


def random_benchmark_distribution(df, n_trades, params=None, runs=200, seed=0, valid_indices=None):
    """Rozklad expectancy losowych wejsc.

    Zwraca tablice srednich wynikow losowych strategii. To lepsze niz jedna
    liczba, bo mozemy sprawdzic czy edge bije typowa losowosc, czy tylko srednia.
    """
    p = {**DEFAULTS, **(params or {})}
    o = df["open"].to_numpy(); h = df["high"].to_numpy(); l = df["low"].to_numpy(); c = df["close"].to_numpy()
    a = atr(df, 14)
    rng = np.random.default_rng(seed)
    if valid_indices is None:
        valid = np.arange(20, max(21, len(df) - p["max_hold"] - 2))
    else:
        valid = np.array([int(x) for x in valid_indices if 20 <= int(x) < len(df) - p["max_hold"] - 2], dtype=int)
    if len(valid) == 0 or n_trades <= 0:
        return np.array([0.0])
    run_means = []
    for _ in range(runs):
        k = min(n_trades, len(valid))
        idxs = rng.choice(valid, size=k, replace=False)
        dirs = rng.choice([-1, 1], size=k)
        res = [simulate_trade(o, h, l, c, a, int(i), int(d), p) for i, d in zip(idxs, dirs)]
        res = [r for r in res if r is not None and np.isfinite(r)]
        if res:
            run_means.append(float(np.mean(res)))
    return np.array(run_means, dtype=float) if run_means else np.array([0.0])


def random_benchmark(df, n_trades, params=None, runs=200, seed=0, valid_indices=None):
    return float(np.mean(random_benchmark_distribution(df, n_trades, params, runs, seed, valid_indices=valid_indices)))


def period_stability(df, signals, params=None, n_periods=4):
    bounds = np.linspace(0, len(df), n_periods + 1, dtype=int)
    out = []
    for k in range(n_periods):
        lo, hi = bounds[k], bounds[k + 1]
        seg = []
        for sig in signals:
            idx, direction, meta = _parse_signal(sig)
            if lo <= idx < hi:
                seg.append((idx - lo, direction, meta))
        sub = df.iloc[lo:hi]
        res = backtest_edge(sub, seg, params)
        out.append((expectancy(res), len(res)))
    return out

# ---------------------------------------------------------------------------
# Phase 0.3 — trade-level diagnostics
# ---------------------------------------------------------------------------

def _volume_ma(df, window=20):
    return pd.Series(df["volume"].to_numpy()).rolling(window, min_periods=max(5, window // 3)).mean().to_numpy()


def _bar_pos_arr(open_, high, low, close):
    rng = np.maximum(high - low, 1e-12)
    return (close - low) / rng


def adx_array(df, window: int = 14):
    """Lekki ADX do diagnostyki, niezalezny od edges.py."""
    high = df["high"].to_numpy()
    low = df["low"].to_numpy()
    close = df["close"].to_numpy()
    up = np.diff(high, prepend=high[0])
    down = -np.diff(low, prepend=low[0])
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    prev_close = np.concatenate([[close[0]], close[:-1]])
    tr = np.maximum(high - low, np.maximum(np.abs(high - prev_close), np.abs(low - prev_close)))
    atr_like = pd.Series(tr).rolling(window, min_periods=window).mean().to_numpy()
    plus_di = 100 * pd.Series(plus_dm).rolling(window, min_periods=window).mean().to_numpy() / np.maximum(atr_like, 1e-12)
    minus_di = 100 * pd.Series(minus_dm).rolling(window, min_periods=window).mean().to_numpy() / np.maximum(atr_like, 1e-12)
    dx = 100 * np.abs(plus_di - minus_di) / np.maximum(plus_di + minus_di, 1e-12)
    return pd.Series(dx).rolling(window, min_periods=window).mean().to_numpy()


def simulate_trade_fixed_detail(o, h, l, c, atr_arr, entry_idx, direction, p):
    """Fixed SL/TP z pelna diagnostyka trade'u."""
    j0 = entry_idx + 1
    if j0 >= len(c):
        return None
    risk = atr_arr[entry_idx] * p["sl_mult"]
    if not np.isfinite(risk) or risk <= 0:
        return None
    entry = o[j0]
    cost_R = p["cost"] * entry / risk
    if direction == 1:
        sl, tp = entry - risk, entry + risk * p["rr"]
    else:
        sl, tp = entry + risk, entry - risk * p["rr"]
    last = min(j0 + p["max_hold"], len(c)) - 1
    best_fav = 0.0
    worst_adv = 0.0
    exit_idx = last
    exit_price = c[last]
    exit_reason = "TIME"
    result_R = ((exit_price - entry) / risk * direction) - cost_R

    for j in range(j0, last + 1):
        if direction == 1:
            fav = (h[j] - entry) / risk
            adv = (l[j] - entry) / risk
            hit_sl, hit_tp = l[j] <= sl, h[j] >= tp
        else:
            fav = (entry - l[j]) / risk
            adv = (entry - h[j]) / risk
            hit_sl, hit_tp = h[j] >= sl, l[j] <= tp
        best_fav = max(best_fav, fav)
        worst_adv = min(worst_adv, adv)
        # Pesymistycznie: jesli w tej samej swiecy SL i TP, zakladamy SL pierwszy.
        if hit_sl:
            exit_idx = j
            exit_price = sl
            exit_reason = "SL"
            result_R = -1.0 - cost_R
            break
        if hit_tp:
            exit_idx = j
            exit_price = tp
            exit_reason = "TP"
            result_R = p["rr"] - cost_R
            break

    return dict(
        result_R=float(result_R),
        entry_idx=int(j0),
        exit_idx=int(exit_idx),
        entry=float(entry),
        exit_price=float(exit_price),
        risk=float(risk),
        cost_R=float(cost_R),
        sl=float(sl),
        tp=float(tp),
        bars_held=int(exit_idx - j0 + 1),
        exit_reason=exit_reason,
        mfe_R=float(best_fav),
        mae_R=float(worst_adv),
    )


def simulate_trade_trailing_detail(o, h, l, c, atr_arr, entry_idx, direction, p):
    """Trailing exit z pelna diagnostyka. Uzywany tylko w testach wrazliwosci."""
    j0 = entry_idx + 1
    if j0 >= len(c):
        return None
    initial_risk = atr_arr[entry_idx] * p["sl_mult"]
    if not np.isfinite(initial_risk) or initial_risk <= 0:
        return None
    entry = o[j0]
    cost_R = p["cost"] * entry / initial_risk
    if direction == 1:
        stop = entry - initial_risk
        best_price = entry
    else:
        stop = entry + initial_risk
        best_price = entry
    last = min(j0 + p["max_hold"], len(c)) - 1
    best_fav = 0.0
    worst_adv = 0.0
    exit_idx = last
    exit_price = c[last]
    exit_reason = "TIME"
    for j in range(j0, last + 1):
        a = atr_arr[j] if np.isfinite(atr_arr[j]) and atr_arr[j] > 0 else atr_arr[entry_idx]
        if direction == 1:
            best_price = max(best_price, h[j])
            fav = (h[j] - entry) / initial_risk
            adv = (l[j] - entry) / initial_risk
            if (best_price - entry) / initial_risk >= p.get("breakeven_after_R", 1.2):
                stop = max(stop, entry)
            stop = max(stop, best_price - p.get("trail_mult", 2.0) * a)
            if l[j] <= stop:
                exit_idx = j
                exit_price = stop
                exit_reason = "TRAIL"
                break
        else:
            best_price = min(best_price, l[j])
            fav = (entry - l[j]) / initial_risk
            adv = (entry - h[j]) / initial_risk
            if (entry - best_price) / initial_risk >= p.get("breakeven_after_R", 1.2):
                stop = min(stop, entry)
            stop = min(stop, best_price + p.get("trail_mult", 2.0) * a)
            if h[j] >= stop:
                exit_idx = j
                exit_price = stop
                exit_reason = "TRAIL"
                break
        best_fav = max(best_fav, fav)
        worst_adv = min(worst_adv, adv)
    result_R = ((exit_price - entry) / initial_risk * direction) - cost_R
    return dict(
        result_R=float(result_R),
        entry_idx=int(j0),
        exit_idx=int(exit_idx),
        entry=float(entry),
        exit_price=float(exit_price),
        risk=float(initial_risk),
        cost_R=float(cost_R),
        sl=np.nan,
        tp=np.nan,
        bars_held=int(exit_idx - j0 + 1),
        exit_reason=exit_reason,
        mfe_R=float(best_fav),
        mae_R=float(worst_adv),
    )




def simulate_trade_breakeven_detail(o, h, l, c, atr_arr, entry_idx, direction, p):
    """Fixed TP, ale po osiagnieciu progu MFE stop przesuwa sie na breakeven.

    Parametry:
    - rr: finalny TP w R,
    - breakeven_after_R: prog aktywacji BE, domyslnie 1R,
    - max_hold, sl_mult, cost jak w baseline.
    """
    j0 = entry_idx + 1
    if j0 >= len(c):
        return None
    risk = atr_arr[entry_idx] * p["sl_mult"]
    if not np.isfinite(risk) or risk <= 0:
        return None
    entry = o[j0]
    cost_R = p["cost"] * entry / risk
    rr = p.get("rr", 2.0)
    be_after = p.get("breakeven_after_R", 1.0)
    if direction == 1:
        initial_sl = entry - risk
        stop = initial_sl
        tp = entry + risk * rr
    else:
        initial_sl = entry + risk
        stop = initial_sl
        tp = entry - risk * rr
    last = min(j0 + p["max_hold"], len(c)) - 1
    best_fav = 0.0
    worst_adv = 0.0
    exit_idx = last
    exit_price = c[last]
    exit_reason = "TIME"
    result_R = ((exit_price - entry) / risk * direction) - cost_R
    be_armed = False

    for j in range(j0, last + 1):
        if direction == 1:
            fav = (h[j] - entry) / risk
            adv = (l[j] - entry) / risk
            hit_sl = l[j] <= stop
            hit_tp = h[j] >= tp
            hit_be = h[j] >= entry + be_after * risk
        else:
            fav = (entry - l[j]) / risk
            adv = (entry - h[j]) / risk
            hit_sl = h[j] >= stop
            hit_tp = l[j] <= tp
            hit_be = l[j] <= entry - be_after * risk
        best_fav = max(best_fav, fav)
        worst_adv = min(worst_adv, adv)
        # Pesymistycznie: aktualny stop przed targetem.
        if hit_sl:
            exit_idx = j
            exit_price = stop
            exit_reason = "BE" if be_armed else "SL"
            result_R = ((exit_price - entry) / risk * direction) - cost_R
            break
        if hit_tp:
            exit_idx = j
            exit_price = tp
            exit_reason = "TP"
            result_R = rr - cost_R
            break
        if hit_be and not be_armed:
            be_armed = True
            stop = entry

    return dict(
        result_R=float(result_R), entry_idx=int(j0), exit_idx=int(exit_idx),
        entry=float(entry), exit_price=float(exit_price), risk=float(risk), cost_R=float(cost_R),
        sl=float(initial_sl), tp=float(tp), bars_held=int(exit_idx - j0 + 1),
        exit_reason=exit_reason, mfe_R=float(best_fav), mae_R=float(worst_adv),
    )


def simulate_trade_partial_detail(o, h, l, c, atr_arr, entry_idx, direction, p):
    """Partial TP: czesc pozycji wychodzi na tp1, reszta na tp2/SL/TIME.

    Parametry:
    - partial_tp1_R: pierwszy target, domyslnie 1R,
    - partial_tp2_R albo rr: drugi target, domyslnie 2R,
    - partial_fraction: jaka czesc zamykamy na TP1, domyslnie 0.5,
    - move_stop_to_be: jesli True, po TP1 stop reszty idzie na entry.
    """
    j0 = entry_idx + 1
    if j0 >= len(c):
        return None
    risk = atr_arr[entry_idx] * p["sl_mult"]
    if not np.isfinite(risk) or risk <= 0:
        return None
    entry = o[j0]
    cost_R = p["cost"] * entry / risk
    tp1_R = p.get("partial_tp1_R", 1.0)
    tp2_R = p.get("partial_tp2_R", p.get("rr", 2.0))
    frac = p.get("partial_fraction", 0.5)
    move_be = bool(p.get("move_stop_to_be", False))
    if direction == 1:
        initial_sl = entry - risk
        stop = initial_sl
        tp1 = entry + risk * tp1_R
        tp2 = entry + risk * tp2_R
    else:
        initial_sl = entry + risk
        stop = initial_sl
        tp1 = entry - risk * tp1_R
        tp2 = entry - risk * tp2_R
    last = min(j0 + p["max_hold"], len(c)) - 1
    best_fav = 0.0
    worst_adv = 0.0
    leg1_done = False
    result_before_cost = 0.0
    exit_idx = last
    exit_price = c[last]
    exit_reason = "TIME"

    for j in range(j0, last + 1):
        if direction == 1:
            fav = (h[j] - entry) / risk
            adv = (l[j] - entry) / risk
            hit_stop = l[j] <= stop
            hit_tp1 = h[j] >= tp1
            hit_tp2 = h[j] >= tp2
        else:
            fav = (entry - l[j]) / risk
            adv = (entry - h[j]) / risk
            hit_stop = h[j] >= stop
            hit_tp1 = l[j] <= tp1
            hit_tp2 = l[j] <= tp2
        best_fav = max(best_fav, fav)
        worst_adv = min(worst_adv, adv)

        # Pesymistycznie: stop przed targetami. Jesli leg1 zamkniety, dotyczy tylko reszty.
        if hit_stop:
            exit_idx = j
            exit_price = stop
            stop_R = ((stop - entry) / risk * direction)
            if leg1_done:
                result_before_cost += (1.0 - frac) * stop_R
                exit_reason = "TP1_BE" if move_be and abs(stop_R) < 1e-9 else "TP1_SL"
            else:
                result_before_cost = -1.0
                exit_reason = "SL"
            break

        if not leg1_done and hit_tp1:
            leg1_done = True
            result_before_cost += frac * tp1_R
            if move_be:
                stop = entry
            # Jesli ta sama swieca dobila TP2, uznajemy TP2 dla reszty po TP1.
            if hit_tp2:
                exit_idx = j
                exit_price = tp2
                result_before_cost += (1.0 - frac) * tp2_R
                exit_reason = "TP1_TP2"
                break

        if leg1_done and hit_tp2:
            exit_idx = j
            exit_price = tp2
            result_before_cost += (1.0 - frac) * tp2_R
            exit_reason = "TP1_TP2"
            break

    else:
        # TIME exit dla niezrealizowanych czesci.
        exit_price = c[last]
        exit_idx = last
        time_R = ((exit_price - entry) / risk * direction)
        if leg1_done:
            result_before_cost += (1.0 - frac) * time_R
            exit_reason = "TP1_TIME"
        else:
            result_before_cost = time_R
            exit_reason = "TIME"

    result_R = result_before_cost - cost_R
    return dict(
        result_R=float(result_R), entry_idx=int(j0), exit_idx=int(exit_idx),
        entry=float(entry), exit_price=float(exit_price), risk=float(risk), cost_R=float(cost_R),
        sl=float(initial_sl), tp=float(tp2), bars_held=int(exit_idx - j0 + 1),
        exit_reason=exit_reason, mfe_R=float(best_fav), mae_R=float(worst_adv),
    )


def simulate_trade_trail_after_detail(o, h, l, c, atr_arr, entry_idx, direction, p):
    """Hybrid: klasyczny SL, po osiagnieciu triggera stop idzie do BE i trailing ATR.

    Nie ma stalego TP, chyba ze ustawisz hybrid_tp_R. To testujemy jako wrazliwosc,
    bo wczesniejszy pure trailing byl slaby.
    """
    j0 = entry_idx + 1
    if j0 >= len(c):
        return None
    risk = atr_arr[entry_idx] * p["sl_mult"]
    if not np.isfinite(risk) or risk <= 0:
        return None
    entry = o[j0]
    cost_R = p["cost"] * entry / risk
    trigger_R = p.get("trail_after_R", 1.0)
    trail_mult = p.get("trail_mult", 1.5)
    hybrid_tp_R = p.get("hybrid_tp_R", None)
    if direction == 1:
        initial_sl = entry - risk
        stop = initial_sl
        best_price = entry
        tp = entry + risk * hybrid_tp_R if hybrid_tp_R is not None else np.nan
    else:
        initial_sl = entry + risk
        stop = initial_sl
        best_price = entry
        tp = entry - risk * hybrid_tp_R if hybrid_tp_R is not None else np.nan
    last = min(j0 + p["max_hold"], len(c)) - 1
    best_fav = 0.0
    worst_adv = 0.0
    exit_idx = last
    exit_price = c[last]
    exit_reason = "TIME"
    active = False

    for j in range(j0, last + 1):
        a = atr_arr[j] if np.isfinite(atr_arr[j]) and atr_arr[j] > 0 else atr_arr[entry_idx]
        if direction == 1:
            best_price = max(best_price, h[j])
            fav = (h[j] - entry) / risk
            adv = (l[j] - entry) / risk
            if fav >= trigger_R:
                active = True
            if active:
                stop = max(stop, entry, best_price - trail_mult * a)
            hit_stop = l[j] <= stop
            hit_tp = hybrid_tp_R is not None and h[j] >= tp
        else:
            best_price = min(best_price, l[j])
            fav = (entry - l[j]) / risk
            adv = (entry - h[j]) / risk
            if fav >= trigger_R:
                active = True
            if active:
                stop = min(stop, entry, best_price + trail_mult * a)
            hit_stop = h[j] >= stop
            hit_tp = hybrid_tp_R is not None and l[j] <= tp
        best_fav = max(best_fav, fav)
        worst_adv = min(worst_adv, adv)
        if hit_stop:
            exit_idx = j
            exit_price = stop
            exit_reason = "TRAIL" if active else "SL"
            break
        if hit_tp:
            exit_idx = j
            exit_price = tp
            exit_reason = "TP"
            break
    result_R = ((exit_price - entry) / risk * direction) - cost_R
    return dict(
        result_R=float(result_R), entry_idx=int(j0), exit_idx=int(exit_idx),
        entry=float(entry), exit_price=float(exit_price), risk=float(risk), cost_R=float(cost_R),
        sl=float(initial_sl), tp=float(tp) if np.isfinite(tp) else np.nan,
        bars_held=int(exit_idx - j0 + 1), exit_reason=exit_reason,
        mfe_R=float(best_fav), mae_R=float(worst_adv),
    )

def simulate_trade_detail(o, h, l, c, atr_arr, entry_idx, direction, p):
    mode = p.get("exit_mode", "fixed")
    if mode == "trailing":
        return simulate_trade_trailing_detail(o, h, l, c, atr_arr, entry_idx, direction, p)
    if mode == "breakeven":
        return simulate_trade_breakeven_detail(o, h, l, c, atr_arr, entry_idx, direction, p)
    if mode == "partial":
        return simulate_trade_partial_detail(o, h, l, c, atr_arr, entry_idx, direction, p)
    if mode == "trail_after":
        return simulate_trade_trail_after_detail(o, h, l, c, atr_arr, entry_idx, direction, p)
    return simulate_trade_fixed_detail(o, h, l, c, atr_arr, entry_idx, direction, p)


def backtest_edge_trades(df, signals, params=None, edge_name="", timeframe="", extra=None):
    """Zwraca DataFrame z kazda transakcja i diagnostyka wejscia/wyjscia."""
    p = {**DEFAULTS, **(params or {})}
    o = df["open"].to_numpy(); h = df["high"].to_numpy(); l = df["low"].to_numpy(); c = df["close"].to_numpy()
    v = df["volume"].to_numpy()
    a = atr(df, 14)
    adx = adx_array(df, 14)
    vma = _volume_ma(df, 20)
    bar_pos = _bar_pos_arr(o, h, l, c)
    rows, last_entry = [], -10 ** 9
    for sig in signals:
        idx, direction, meta = _parse_signal(sig)
        if idx - last_entry < p["min_gap"]:
            continue
        d = simulate_trade_detail(o, h, l, c, a, idx, direction, p)
        if d is None or not np.isfinite(d.get("result_R", np.nan)):
            continue
        sig_time = df.index[idx]
        entry_i = d["entry_idx"]
        exit_i = d["exit_idx"]
        vol_ratio = v[idx] / vma[idx] if np.isfinite(vma[idx]) and vma[idx] > 0 else np.nan
        bar_range_atr = (h[idx] - l[idx]) / a[idx] if np.isfinite(a[idx]) and a[idx] > 0 else np.nan
        level = meta.get("level", np.nan) if isinstance(meta, dict) else np.nan
        dist_level_atr = ((c[idx] - level) / a[idx] * direction) if np.isfinite(level) and np.isfinite(a[idx]) and a[idx] > 0 else np.nan
        row = {
            "edge": edge_name,
            "timeframe": timeframe,
            "exit_mode": p.get("exit_mode", "fixed"),
            "signal_idx": int(idx),
            "signal_time": sig_time,
            "entry_time": df.index[entry_i],
            "exit_time": df.index[exit_i],
            "year": int(pd.Timestamp(sig_time).year),
            "month": int(pd.Timestamp(sig_time).month),
            "weekday": int(pd.Timestamp(sig_time).weekday()),
            "hour": int(pd.Timestamp(sig_time).hour) if hasattr(pd.Timestamp(sig_time), "hour") else np.nan,
            "direction": "long" if direction == 1 else "short",
            "direction_num": int(direction),
            "R": d["result_R"],
            "entry": d["entry"],
            "exit_price": d["exit_price"],
            "risk": d["risk"],
            "cost_R": d["cost_R"],
            "sl": d["sl"],
            "tp": d["tp"],
            "bars_held": d["bars_held"],
            "exit_reason": d["exit_reason"],
            "mfe_R": d["mfe_R"],
            "mae_R": d["mae_R"],
            "atr": a[idx],
            "atr_pct": a[idx] / c[idx] if c[idx] != 0 else np.nan,
            "adx": adx[idx],
            "volume_ratio": vol_ratio,
            "bar_pos": bar_pos[idx],
            "bar_range_atr": bar_range_atr,
            "level": level,
            "distance_to_level_atr": dist_level_atr,
        }
        if isinstance(meta, dict):
            for k, val in meta.items():
                if k not in row:
                    row[k] = val
            # przydatne pochodne, jesli edge zapisuje indeks swiecy bazowej
            for idx_key in ("breakout_idx", "sweep_idx"):
                if idx_key in meta and meta[idx_key] is not None:
                    j = int(meta[idx_key])
                    if 0 <= j < len(df):
                        row[f"{idx_key}_time"] = df.index[j]
                        row[f"bars_since_{idx_key.replace('_idx','')}"] = int(idx - j)
        if extra:
            row.update(extra)
        rows.append(row)
        last_entry = idx
    return pd.DataFrame(rows)


def diagnostics_summary(trades: pd.DataFrame):
    """Zbiorczy raport diagnostyczny po edge/timeframe/year/direction/exit_reason."""
    if trades is None or trades.empty:
        return pd.DataFrame()

    def _agg(g):
        arr = g["R"].astype(float)
        wins = arr[arr > 0]
        losses = arr[arr < 0]
        losses_abs = -losses.sum()
        return pd.Series({
            "trades": len(g),
            "exp_R": arr.mean(),
            "median_R": arr.median(),
            "winR": (arr > 0).mean(),
            "PF": (wins.sum() / losses_abs) if losses_abs > 0 else (np.inf if wins.sum() > 0 else 0.0),
            "sum_R": arr.sum(),
            "maxDD_R": max_drawdown_R(arr.tolist()),
            "avg_MFE_R": g["mfe_R"].mean(),
            "avg_MAE_R": g["mae_R"].mean(),
            "avg_bars_held": g["bars_held"].mean(),
            "avg_adx": g["adx"].mean(),
            "avg_vol_ratio": g["volume_ratio"].mean(),
            "avg_atr_pct": g["atr_pct"].mean(),
        })

    frames = []
    base_keys = ["edge", "timeframe"]
    if "exit_profile" in trades.columns:
        base_keys = ["edge", "timeframe", "exit_profile"]
    grouping_plan = [
        (base_keys, "all"),
        (base_keys + ["year"], "year"),
        (base_keys + ["direction"], "direction"),
        (base_keys + ["exit_reason"], "exit_reason"),
    ]
    for keys, bucket in grouping_plan:
        part = trades.groupby(keys, dropna=False).apply(_agg).reset_index()
        part.insert(0, "bucket", bucket)
        frames.append(part)
    return pd.concat(frames, ignore_index=True, sort=False)
