"""Wczytywanie danych dla testow Fazy 0 — z cache i resamplingiem.

Strategia: pobieramy RAZ dane godzinowe (1h), zapisujemy do cache na dysku,
a interwaly 4h i Daily wyliczamy lokalnie przez resampling. Dzieki temu
kolejne uruchomienia sa natychmiastowe i wszystkie interwaly sa spojne.
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd

CACHE_DIR = "cache"


def _cache_path(name):
    os.makedirs(CACHE_DIR, exist_ok=True)
    safe = name.replace("/", "_").replace(":", "_").replace(" ", "_")
    return os.path.join(CACHE_DIR, safe + ".pkl")


def load_binance_ohlcv(symbol="BTC/USDT", timeframe="1h",
                       start="2020-01-01", end="2025-01-01", use_cache=True):
    """Pobiera swiece OHLCV z Binance przez ccxt. Cache'uje na dysku."""
    key = f"ohlcv_{symbol}_{timeframe}_{start}_{end}"
    path = _cache_path(key)
    if use_cache and os.path.exists(path):
        print(f"  cache: wczytuje {timeframe} z dysku")
        return pd.read_pickle(path)
    print(f"  pobieram {timeframe} z Binance (to potrwa kilka minut) ...")
    import ccxt
    exchange = ccxt.binance({"enableRateLimit": True})
    since = exchange.parse8601(f"{start}T00:00:00Z")
    end_ms = exchange.parse8601(f"{end}T00:00:00Z")
    tf_ms = exchange.parse_timeframe(timeframe) * 1000
    rows = []
    while since < end_ms:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not batch:
            break
        rows += batch
        since = batch[-1][0] + tf_ms
        if len(batch) < 1000:
            break
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low",
                                     "close", "volume"])
    df = df.drop_duplicates("ts")
    df["ts"] = pd.to_datetime(df["ts"], unit="ms", utc=True)
    df = df.set_index("ts").sort_index()
    if use_cache:
        df.to_pickle(path)
    return df


def load_binance_funding(symbol="BTC/USDT", start="2020-01-01",
                         end="2025-01-01", use_cache=True):
    """Pobiera historie funding rate. Cache'uje na dysku."""
    key = f"funding_{symbol}_{start}_{end}"
    path = _cache_path(key)
    if use_cache and os.path.exists(path):
        print("  cache: wczytuje funding z dysku")
        return pd.read_pickle(path)
    print("  pobieram funding z Binance ...")
    import ccxt
    exchange = ccxt.binance({"enableRateLimit": True,
                             "options": {"defaultType": "future"}})
    since = exchange.parse8601(f"{start}T00:00:00Z")
    end_ms = exchange.parse8601(f"{end}T00:00:00Z")
    rows = []
    while since < end_ms:
        batch = exchange.fetch_funding_rate_history(symbol, since=since,
                                                    limit=1000)
        if not batch:
            break
        rows += batch
        since = batch[-1]["timestamp"] + 1
        if len(batch) < 1000:
            break
    if not rows:
        s = pd.Series(dtype=float)
    else:
        s = pd.Series({pd.to_datetime(r["timestamp"], unit="ms", utc=True):
                       float(r["fundingRate"]) for r in rows}).sort_index()
    if use_cache:
        s.to_pickle(path)
    return s


def resample_ohlcv(df, rule):
    """Resampluje swiece 1h na wyzszy interwal (np. '4h', '1D')."""
    agg = {"open": "first", "high": "max", "low": "min",
           "close": "last", "volume": "sum"}
    out = df.resample(rule, label="left", closed="left").agg(agg)
    return out.dropna()


def generate_synthetic_ohlcv(n_bars=40000, seed=7):
    """Syntetyczny szereg OHLCV z klastrowaniem zmiennosci (tylko smoke test)."""
    rng = np.random.default_rng(seed)
    vol = np.zeros(n_bars)
    vol[0] = 0.01
    for i in range(1, n_bars):
        vol[i] = abs(0.92 * vol[i - 1] + 0.08 * 0.01
                     + rng.normal(0, 0.0025))
    rets = rng.normal(0, 1, n_bars) * vol
    close = 20000 * np.exp(np.cumsum(rets))
    open_ = np.empty(n_bars)
    open_[0] = close[0]
    open_[1:] = close[:-1]
    high = np.maximum(open_, close) * (1 + abs(rng.normal(0, 0.5, n_bars))
                                       * vol)
    low = np.minimum(open_, close) * (1 - abs(rng.normal(0, 0.5, n_bars))
                                      * vol)
    volume = rng.lognormal(10, 1, n_bars)
    idx = pd.date_range("2020-01-01", periods=n_bars, freq="1h", tz="UTC")
    return pd.DataFrame({"open": open_, "high": high, "low": low,
                         "close": close, "volume": volume}, index=idx)


def generate_synthetic_funding(index, seed=11):
    """Syntetyczny funding wyrownany do indeksu swiec (co 8h)."""
    rng = np.random.default_rng(seed)
    fund_idx = index[::8]
    vals = rng.normal(0.0001, 0.0005, len(fund_idx))
    return pd.Series(vals, index=fund_idx)


def load_feargreed(start: str, end: str, use_cache=True) -> "pd.Series":
    """Fear & Greed Index (0-100) z alternative.me. Dzienny."""
    import pandas as pd, json, ssl, urllib.request
    cache = Path(_cache_path(f"feargreed_{start}_{end}.pkl"))
    if use_cache and cache.exists():
        return pd.read_pickle(cache)
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    url = "https://api.alternative.me/fng/?limit=2000&format=json&date_format=us"
    with urllib.request.urlopen(url, timeout=15, context=ctx) as r:
        raw = json.loads(r.read())["data"]
    records = {pd.to_datetime(d["date"]): int(d["value"]) for d in raw}
    s = pd.Series(records).sort_index()
    s = s[s.index >= pd.to_datetime(start)]
    s = s[s.index <= pd.to_datetime(end)]
    if use_cache:
        s.to_pickle(cache)
    return s


def load_bybit_ohlcv(symbol: str, timeframe: str,
                     start: str, end: str, use_cache=True) -> "pd.DataFrame":
    """OHLCV z Bybit przez ccxt."""
    import pandas as pd, ccxt, time as _time
    tf_map = {"1h": "1h", "4h": "4h", "1d": "1D", "15m": "15min", "5m": "5min"}
    cache = Path(_cache_path(f"bybit_{symbol.replace('/','_')}_{timeframe}_{start}_{end}.pkl"))
    if use_cache and cache.exists():
        return pd.read_pickle(cache)
    exchange = ccxt.bybit({"enableRateLimit": True})
    since = exchange.parse8601(start + "T00:00:00Z")
    until = exchange.parse8601(end   + "T00:00:00Z")
    all_ohlcv = []
    while since < until:
        batch = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not batch:
            break
        all_ohlcv.extend(batch)
        since = batch[-1][0] + 1
        _time.sleep(0.2)
    df = pd.DataFrame(all_ohlcv, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.set_index("timestamp")
    df = df[df.index <= pd.to_datetime(end)]
    resample_key = tf_map.get(timeframe, timeframe)
    if use_cache:
        df.to_pickle(cache)
    return df


def load_binance_open_interest(symbol, timeframe="4h", start=None, end=None, use_cache=True):
    """Pobiera historyczny Open Interest z Binance Futures (REST). Timestamps w UTC.

    Jedyny loader OI w projekcie (wcześniej istniało 7 wariantów-iteracji).
    Paginacja przez startTime/endTime, max 500 rekordów na request,
    następnie trim do [start, end] i resample do docelowego interwału.
    """
    import requests
    import pandas as pd
    from pathlib import Path
    import time

    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)
    safe = symbol.replace("/", "_")
    cache_file = cache_dir / f"oi_{safe}_{timeframe}_{start}_{end}.pkl"
    if use_cache and cache_file.exists():
        return pd.read_pickle(cache_file)

    sym = symbol.replace("/", "")
    url = "https://fapi.binance.com/futures/data/openInterestHist"
    all_data = []

    # UTC timestamps
    current_start = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000) if start else None
    end_ts = int(pd.Timestamp(end, tz="UTC").timestamp() * 1000) if end else None

    while True:
        params = {"symbol": sym, "period": timeframe, "limit": 500}
        if current_start:
            params["startTime"] = current_start
        if end_ts:
            params["endTime"] = end_ts

        resp = requests.get(url, params=params)
        if resp.status_code != 200:
            print(f"Błąd API: {resp.status_code} {resp.text}")
            break

        data = resp.json()
        if not data:
            break

        all_data.extend(data)
        if len(data) < 500:
            break

        current_start = data[-1]["timestamp"] + 1
        if end_ts and current_start >= end_ts:
            break
        time.sleep(0.1)

    if not all_data:
        return pd.Series(dtype=float)

    df = pd.DataFrame(all_data)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp").sort_index()
    oi = df["sumOpenInterest"].astype(float)

    # Trim do żądanego zakresu i resample do docelowego interwału
    if start:
        oi = oi[oi.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        oi = oi[oi.index <= pd.Timestamp(end, tz="UTC")]
    oi = oi.resample(timeframe).last()

    if use_cache:
        oi.to_pickle(cache_file)
    return oi


def load_hashrate(start=None, end=None, use_cache=True):
    """Pobiera historyczny hashrate BTC z blockchain.info."""
    import requests
    import pandas as pd
    from pathlib import Path

    cache_dir = Path("cache")
    cache_dir.mkdir(exist_ok=True)
    cache_file = cache_dir / f"hashrate_{start}_{end}.pkl"
    if use_cache and cache_file.exists():
        return pd.read_pickle(cache_file)

    resp = requests.get(
        "https://api.blockchain.info/charts/hash-rate",
        params={"timespan": "all", "format": "json", "sampled": "false"}
    )
    if resp.status_code != 200:
        print(f"Błąd API hashrate: {resp.status_code}")
        return pd.Series(dtype=float)

    values = resp.json()["values"]
    df = pd.DataFrame(values)
    df["timestamp"] = pd.to_datetime(df["x"], unit="s", utc=True)
    df = df.set_index("timestamp")["y"].rename("hashrate")
    df = df.sort_index()

    if start:
        df = df[df.index >= pd.Timestamp(start, tz="UTC")]
    if end:
        df = df[df.index <= pd.Timestamp(end, tz="UTC")]

    if use_cache:
        df.to_pickle(cache_file)
    return df
