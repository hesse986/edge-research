"""
Risk Agent – position sizing oparty na frameworku Carvera.
Carver "Systematic Trading" – Chapter 9: Position Sizing

Zasada: pozycja w jednostkach = (cel_ryzyka * kapitał) / (cena * ann_vol)
gdzie ann_vol = dzienna_vol * sqrt(252)
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
import data as datamod

# ============================================================
# KONFIGURACJA
# ============================================================
CAPITAL         = 1000.0    # wirtualny kapitał USD
RISK_TARGET     = 0.12      # 12% rocznego ryzyka (Carver: 10-25%)
MAX_RISK_TRADE  = 0.02      # max 2% kapitału na jeden trade
MIN_RISK_TRADE  = 0.005     # min 0.5% kapitału na jeden trade
VOL_LOOKBACK    = 30        # dni do obliczenia zmienności
VOL_FLOOR       = 0.05      # minimalna annualized vol (5%)

def get_daily_vol(symbol: str, lookback: int = VOL_LOOKBACK) -> float:
    """Oblicza dzienną zmienność jako std zwrotów dziennych."""
    from datetime import datetime, timedelta
    end   = datetime.now().strftime("%Y-%m-%d")
    start = (datetime.now() - timedelta(days=lookback * 2)).strftime("%Y-%m-%d")
    try:
        df1h = datamod.load_binance_ohlcv(symbol, "1h", start, end, use_cache=False)
        df   = datamod.resample_ohlcv(df1h, "1D")
        returns = df["close"].pct_change().dropna()
        if len(returns) < 10:
            return 0.03  # fallback
        return float(returns.tail(lookback).std())
    except Exception as e:
        print(f"Błąd vol dla {symbol}: {e}")
        return 0.03

def annualized_vol(daily_vol: float) -> float:
    """Przelicza dzienną vol na roczną."""
    ann = daily_vol * np.sqrt(252)
    return max(ann, VOL_FLOOR)

def position_size_carver(
    symbol: str,
    current_price: float,
    capital: float = CAPITAL,
    risk_target: float = RISK_TARGET,
    daily_vol: float = None
) -> dict:
    """
    Oblicza wielkość pozycji według Carvera.
    
    Returns:
        dict z: units, usd_value, risk_pct, ann_vol
    """
    if daily_vol is None:
        daily_vol = get_daily_vol(symbol)
    
    ann_v = annualized_vol(daily_vol)
    
    # Carver formula: N = (risk_target * capital) / (price * ann_vol)
    units = (risk_target * capital) / (current_price * ann_v)
    usd_value = units * current_price
    risk_pct  = (usd_value * ann_v) / capital
    
    # Clamp do min/max
    risk_pct_clamped = np.clip(risk_pct, MIN_RISK_TRADE, MAX_RISK_TRADE)
    if risk_pct_clamped != risk_pct:
        units     = (risk_pct_clamped * capital) / (current_price * ann_v)
        usd_value = units * current_price
        risk_pct  = risk_pct_clamped
    
    return {
        "symbol":    symbol,
        "units":     round(units, 6),
        "usd_value": round(usd_value, 2),
        "risk_pct":  round(risk_pct * 100, 2),
        "ann_vol":   round(ann_v * 100, 1),
        "daily_vol": round(daily_vol * 100, 2),
        "capital":   capital
    }

def position_size_pairs(
    symbol1: str,
    symbol2: str,
    hedge_ratio: float,
    capital: float = CAPITAL,
    risk_target: float = RISK_TARGET
) -> dict:
    """
    Position sizing dla pairs trading.
    Ryzyko liczymy na spreadzie, nie na pojedynczym assecie.
    """
    # Oblicz vol bezposrednio ze spreadu
    from datetime import datetime, timedelta
    end_   = datetime.now().strftime("%Y-%m-%d")
    start_ = (datetime.now() - timedelta(days=90)).strftime("%Y-%m-%d")
    try:
        df1h_1 = datamod.load_binance_ohlcv(symbol1, "1h", start_, end_, use_cache=False)
        df1h_2 = datamod.load_binance_ohlcv(symbol2, "1h", start_, end_, use_cache=False)
        d1 = datamod.resample_ohlcv(df1h_1, "1D")
        d2 = datamod.resample_ohlcv(df1h_2, "1D")
        common_ = d1.index.intersection(d2.index)
        spread_series = d1.loc[common_, "close"] - hedge_ratio * d2.loc[common_, "close"]
        spread_vol = float(spread_series.pct_change().dropna().std())
    except:
        spread_vol = 0.02
    ann_spread_vol = annualized_vol(spread_vol)
    
    # Liczba jednostek spreadu
    # Zakładamy 1 jednostkę = 1 LTC i hedge_ratio XRP
    # Wartość spreadu w USD ≈ cena LTC (uproszczenie)
    try:
        from datetime import datetime, timedelta
        end   = datetime.now().strftime("%Y-%m-%d")
        start = (datetime.now() - timedelta(days=5)).strftime("%Y-%m-%d")
        df1   = datamod.load_binance_ohlcv(symbol1, "1h", start, end, use_cache=False)
        price1 = float(df1["close"].iloc[-1])
    except:
        price1 = 100.0  # fallback
    
    units = (risk_target * capital) / (price1 * ann_spread_vol)
    usd_value = units * price1
    risk_pct  = np.clip((usd_value * ann_spread_vol) / capital, MIN_RISK_TRADE, MAX_RISK_TRADE)
    
    return {
        "symbol1":        symbol1,
        "symbol2":        symbol2,
        "hedge_ratio":    round(hedge_ratio, 4),
        "units_symbol1":  round(units, 6),
        "units_symbol2":  round(units * hedge_ratio, 6),
        "usd_per_leg":    round(usd_value, 2),
        "spread_ann_vol": round(ann_spread_vol * 100, 1),
        "risk_pct":       round(risk_pct * 100, 2),
        "capital":        capital
    }

def risk_report(capital: float = CAPITAL) -> None:
    """Wyświetla raport ryzyka dla aktywnych par."""
    pairs = [
        ("LTC/USDT", "XRP/USDT", -26.72),
        ("LTC/USDT", "ADA/USDT",  19.02),
        ("XRP/USDT", "ADA/USDT",   0.24),
    ]
    print(f"\n{'='*55}")
    print(f"RISK AGENT – Position Sizing Report")
    print(f"Kapitał: ${capital:.0f} | Risk target: {RISK_TARGET*100:.0f}%/rok")
    print(f"{'='*55}")
    for s1, s2, hedge in pairs:
        r = position_size_pairs(s1, s2, hedge, capital)
        print(f"\n{s1[:3]}/{s2[:3]}:")
        print(f"  Spread vol (ann): {r['spread_ann_vol']}%")
        print(f"  Jednostki {s1[:3]}: {r['units_symbol1']}")
        print(f"  Jednostki {s2[:3]}: {r['units_symbol2']}")
        print(f"  USD per leg:      ${r['usd_per_leg']}")
        print(f"  Ryzyko na trade:  {r['risk_pct']}%")
    print(f"\n{'='*55}")

if __name__ == "__main__":
    risk_report()
