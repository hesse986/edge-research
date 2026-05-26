from pathlib import Path
import numpy as np
import pandas as pd
from pathlib import Path

def atr(df, window=14):
    h, l, c = df["high"].to_numpy(), df["low"].to_numpy(), df["close"].to_numpy()
    pc = np.concatenate([[c[0]], c[:-1]])
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return pd.Series(tr).rolling(window, min_periods=1).mean().to_numpy()

def _ema(close, span):
    return pd.Series(close).ewm(span=span, adjust=False).mean().to_numpy()

def short_term_reversal_liquidity(df, lookback=1, threshold=-0.005, vol_percentile=60):
    close = df["close"].to_numpy()
    volume = df["volume"].to_numpy()
    returns = np.diff(close, prepend=close[0]) / close
    vol_ma = pd.Series(volume).rolling(20).mean().to_numpy()
    vol_thresh = np.nanpercentile(vol_ma, vol_percentile)
    signals = []
    for i in range(lookback+10, len(df)):
        if returns[i] < threshold and vol_ma[i] < vol_thresh:
            signals.append((i, 1))
        elif returns[i] > -threshold and vol_ma[i] < vol_thresh:
            signals.append((i, -1))
    return signals

def trend_following(df, fast=20, slow=50, adx_min=25):
    close = df["close"].to_numpy()
    fast_ema = _ema(close, fast)
    slow_ema = _ema(close, slow)
    # użyj atr jako proxy ADX – dla uproszczenia
    a = atr(df, 14)
    signals = []
    for i in range(slow+10, len(df)):
        if fast_ema[i] > slow_ema[i] and a[i] > adx_min:
            signals.append((i, 1))
        elif fast_ema[i] < slow_ema[i] and a[i] > adx_min:
            signals.append((i, -1))
    return signals

def oi_price_divergence(df, oi=None, lookback=20, threshold=0.02):
    if oi is None or len(oi) == 0:
        return []
    oi_series = oi.reindex(df.index, method='ffill').to_numpy()
    close = df["close"].to_numpy()
    signals = []
    for i in range(lookback+10, len(df)):
        oi_change = (oi_series[i] - oi_series[i-lookback]) / oi_series[i-lookback]
        price_change = (close[i] - close[i-lookback]) / close[i-lookback]
        if oi_change > threshold and abs(price_change) < threshold/2:
            signals.append((i, -1))
        elif oi_change < -threshold and abs(price_change) < threshold/2:
            signals.append((i, 1))
    return signals

def cross_exchange_premium(df, bybit_df=None, premium_pctl=0.90, lookback=60, min_premium_pct=0.003):
    if bybit_df is None or len(bybit_df) == 0:
        return []
    bybit_close = bybit_df["close"].reindex(df.index, method="ffill").to_numpy()
    close = df["close"].to_numpy()
    signals = []
    for i in range(lookback, len(df)):
        if bybit_close[i] == 0 or np.isnan(bybit_close[i]):
            continue
        premium = (close[i] - bybit_close[i]) / bybit_close[i]
        window_premiums = []
        for j in range(i-lookback, i):
            if bybit_close[j] > 0 and not np.isnan(bybit_close[j]):
                window_premiums.append((close[j] - bybit_close[j]) / bybit_close[j])
        if len(window_premiums) < lookback // 2:
            continue
        hi = np.percentile(window_premiums, premium_pctl * 100)
        lo = np.percentile(window_premiums, (1 - premium_pctl) * 100)
        if premium >= hi and abs(premium) >= min_premium_pct:
            signals.append((i, -1))
        elif premium <= lo and abs(premium) >= min_premium_pct:
            signals.append((i, 1))
    return signals

# Puste funkcje dla pozostałych, aby uniknąć błędów importu
def funding_crowding_unwind(df, funding): return []
def volatility_compression_breakout(df): return []
def liquidity_cascade_reclaim(df): return []
def msb_vwap_confluence(df): return []
def msb_ema_cross_confluence(df): return []


def hash_ribbon(df, hashrate=None, fast=30, slow=60):
    """
    Hash Ribbon dla BTC.
    Sygnał kupna: fast SMA hashrate przecina slow SMA od dołu (po capitulation).
    Sygnał sprzedaży: fast SMA spada poniżej slow SMA.
    hashrate: pd.Series z dziennymi danymi hashrate.
    df: DataFrame OHLCV BTC (dzienny lub 4H).
    """
    import pandas as pd
    import numpy as np

    if hashrate is None or len(hashrate) == 0:
        return []

    # Resample hashrate do dziennego
    hr_daily = hashrate.resample("1D").last().ffill()
    hr_fast = hr_daily.rolling(fast).mean()
    hr_slow = hr_daily.rolling(slow).mean()

    # Połącz z indeksem df
    df_dates = df.index.normalize() if hasattr(df.index, "normalize") else pd.DatetimeIndex(df.index).normalize()
    signals = []
    prev_above = None

    for i in range(slow + 5, len(df)):
        date = df_dates[i]
        # Znajdź najbliższy dzień w hashrate
        try:
            f = hr_fast.asof(date)
            s = hr_slow.asof(date)
        except Exception:
            continue
        if pd.isna(f) or pd.isna(s) or s == 0:
            continue

        above = f > s
        if prev_above is not None:
            if above and not prev_above:
                signals.append((i, 1))   # crossover up = buy
            elif not above and prev_above:
                signals.append((i, -1))  # crossover down = sell
        prev_above = above

    return signals


def peer_reversal(df, peers_df=None, signal_threshold=2.0):
    """
    Peer co-movement reversal.
    Gdy peer asset spada >2sigma, wejdź long na tym assecie następnego dnia.
    peers_df: dict {symbol: DataFrame} z danymi innych assetów.
    """
    import numpy as np
    import pandas as pd

    if peers_df is None:
        return []

    signals = []
    close = df["close"]
    returns = close.pct_change()

    for peer_symbol, peer_df in peers_df.items():
        peer_returns = peer_df["close"].pct_change()
        peer_std = peer_returns.rolling(60).std()
        # Zsynchronizuj indeksy
        common = df.index.intersection(peer_df.index)
        if len(common) < 60:
            continue
        peer_ret = peer_returns.reindex(common)
        peer_s = peer_std.reindex(common)

        for i in range(60, len(common) - 1):
            date = common[i]
            if pd.isna(peer_ret.iloc[i]) or pd.isna(peer_s.iloc[i]):
                continue
            if peer_s.iloc[i] == 0:
                continue
            z = peer_ret.iloc[i] / peer_s.iloc[i]
            # Gdy peer spada mocno – kup ten asset następnego dnia
            if z < -signal_threshold:
                # Znajdź indeks w df
                try:
                    idx = df.index.get_loc(common[i + 1])
                    signals.append((idx, 1))
                except KeyError:
                    continue

    return signals


def day_of_week_effect(df, long_days=None, short_days=None):
    """
    Weekend/day-of-week effect.
    long_days: lista dni (0=Pon, 6=Nd) do kupna na zamknięciu
    short_days: lista dni do shortu na zamknięciu
    """
    import pandas as pd
    if long_days is None:
        long_days  = [0, 4]   # Poniedziałek, Piątek
    if short_days is None:
        short_days = [6]       # Niedziela

    signals = []
    index = df.index
    if not hasattr(index, "dayofweek"):
        index = pd.DatetimeIndex(index)

    for i in range(1, len(df)):
        dow = index[i].dayofweek
        if dow in long_days:
            signals.append((i, 1))
        elif dow in short_days:
            signals.append((i, -1))
    return signals


def volume_spike_momentum(df, vol_z_threshold=2.0, lookback=20):
    """
    Volume spike momentum.
    Gdy wolumen > mean+2*std i cena rośnie -> long następna świeca.
    Gdy wolumen > mean+2*std i cena spada -> short następna świeca.
    """
    import pandas as pd
    import numpy as np

    close  = df["close"].to_numpy()
    volume = df["volume"].to_numpy()

    vol_s  = pd.Series(volume)
    vol_ma = vol_s.rolling(lookback).mean().to_numpy()
    vol_std= vol_s.rolling(lookback).std().to_numpy()

    signals = []
    for i in range(lookback + 1, len(df) - 1):
        if vol_std[i] == 0 or np.isnan(vol_std[i]):
            continue
        vol_z = (volume[i] - vol_ma[i]) / vol_std[i]
        if vol_z < vol_z_threshold:
            continue
        ret_curr = (close[i] - close[i-1]) / close[i-1]
        if ret_curr > 0.005:
            signals.append((i+1, 1))
        elif ret_curr < -0.005:
            signals.append((i+1, -1))
    return signals


def btc_leadlag_reversal(df, btc_df=None, threshold_pct=80, lookback=1):
    """
    BTC lead-lag reversal.
    Gdy BTC mocno spada (bottom threshold%), kup alt następnej świecy.
    Gdy BTC mocno rośnie (top threshold%), short alt następnej świecy.
    btc_df: DataFrame z danymi BTC
    """
    import pandas as pd
    import numpy as np

    if btc_df is None:
        return []

    btc_ret = btc_df["close"].pct_change()
    alt_ret = df["close"].pct_change()
    common  = df.index.intersection(btc_df.index)

    if len(common) < 50:
        return []

    btc_r   = btc_ret.reindex(common)
    low_thr = btc_r.quantile((100 - threshold_pct) / 100)
    hi_thr  = btc_r.quantile(threshold_pct / 100)

    signals = []
    for i in range(1, len(common) - 1):
        date = common[i]
        try:
            z = btc_r.loc[date]
        except:
            continue
        if pd.isna(z):
            continue
        idx = df.index.get_loc(common[i + 1]) if common[i + 1] in df.index else None
        if idx is None:
            continue
        if z < low_thr:
            signals.append((idx, 1))   # BTC spada mocno -> long alt
        elif z > hi_thr:
            signals.append((idx, -1))  # BTC rośnie mocno -> short alt
    return signals


def funding_extreme_1h(df, funding=None, extreme_pct=5, direction_filter="low"):
    """
    Funding rate extremes na 1H.
    Ekstremalnie niski funding -> long (rynek oversold w perpetuals).
    Ekstremalnie wysoki funding -> short (rynek overbought).
    extreme_pct: procent ekstremalnych wartości (5 = top/bottom 5%)
    """
    import pandas as pd
    import numpy as np

    if funding is None or len(funding) == 0:
        return []

    f = funding.reindex(df.index, method="nearest", tolerance=pd.Timedelta('4h'))
    if f.isna().all():
        return []

    lo = f.quantile(extreme_pct / 100)
    hi = f.quantile(1 - extreme_pct / 100)

    signals = []
    for i in range(1, len(df) - 1):
        fval = f.iloc[i]
        if pd.isna(fval):
            continue
        if fval <= lo:
            signals.append((i + 1, 1))   # long
        elif fval >= hi:
            signals.append((i + 1, -1))  # short
    return signals
