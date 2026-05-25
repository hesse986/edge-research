#!/usr/bin/env python3
"""
Analiza paper tradingu na podstawie logów.
- Odczytuje pliki CSV z monitorów.
- Dla każdego trade oblicza zwrot w R (ryzyko = odległość entry do SL lub 2% ruchu spreadu).
- Symuluje kapitał początkowy 1000 USD z ryzykiem 1% na trade.
- Generuje raport i wykres equity.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

CAPITAL_START = 1000.0
RISK_PERCENT = 0.01   # 1% kapitału na trade (ryzyko)

def load_logs():
    """Wczytuje logi z obu monitorów (jeśli istnieją)."""
    logs = {}
    pairs_log = Path("paper_trades_pairs_advanced.csv")
    breakout_log = Path("paper_trades.csv")
    if pairs_log.exists():
        logs['pairs'] = pd.read_csv(pairs_log)
        print(f"Wczytano {len(logs['pairs'])} wierszy z {pairs_log}")
    else:
        print("Brak pliku pairs")
        logs['pairs'] = None
    if breakout_log.exists():
        logs['breakout'] = pd.read_csv(breakout_log)
        print(f"Wczytano {len(logs['breakout'])} wierszy z {breakout_log}")
    else:
        print("Brak pliku breakout")
        logs['breakout'] = None
    return logs

def compute_r(row):
    """
    Oblicza zwrot w R dla pojedynczego trade.
    Ryzyko = |entry - SL| (jeśli SL istnieje), inaczej 0.02 * entry spread.
    Zwrot = (exit - entry) / risk * sign
    """
    entry = row.get('entry_price')
    if entry is None or pd.isna(entry):
        # stary format: entry_price to spread
        entry = row.get('entry_price')
    sl = row.get('sl')
    exit_price = row.get('exit_price')
    direction = row.get('direction', '')
    # Określenie znaku (+1 long, -1 short)
    if 'LONG' in str(direction):
        sign = 1
    elif 'SHORT' in str(direction):
        sign = -1
    else:
        # Domyślnie z kierunku (może być 'LONG' w kolumnie)
        sign = 1
    if pd.isna(entry) or pd.isna(exit_price):
        return np.nan
    # Ryzyko
    if not pd.isna(sl):
        risk = abs(entry - sl)
    else:
        # domyślne ryzyko 2% entry price
        risk = 0.02 * abs(entry)
    if risk <= 0:
        return 0.0
    ret = (exit_price - entry) * sign
    r = ret / risk
    return r

def simulate_capital(trades):
    """Symuluje kapitał na podstawie trades (DataFrame z R)."""
    if trades is None or trades.empty:
        return [], []
    equity = [CAPITAL_START]
    capital = CAPITAL_START
    timestamps = []
    for idx, row in trades.iterrows():
        if row['R'] is None or pd.isna(row['R']):
            continue
        risk_capital = capital * RISK_PERCENT
        profit = risk_capital * row['R']
        capital += profit
        equity.append(capital)
        # timestamp – używamy timestampu zamknięcia lub otwarcia
        ts = row.get('timestamp', '')
        if pd.isna(ts):
            ts = idx
        timestamps.append(ts)
    return timestamps, equity

def main():
    logs = load_logs()
    all_trades = []
    for name, df in logs.items():
        if df is None or df.empty:
            continue
        # Filtruj tylko zamknięte trade (status == CLOSED)
        if 'status' in df.columns:
            df = df[df['status'] == 'CLOSED']
        print(f"Liczba zamkniętych trade dla {name}: {len(df)}")
        # Oblicz R dla każdego trade
        df['R'] = df.apply(compute_r, axis=1)
        all_trades.append(df)
    if not all_trades:
        print("Brak trade do analizy.")
        return
    trades_combined = pd.concat(all_trades, ignore_index=True)
    # Usuń wiersze z brakującym R
    trades_combined = trades_combined.dropna(subset=['R'])
    print(f"Łączna liczba trade z R: {len(trades_combined)}")
    if len(trades_combined) == 0:
        return
    # Statystyki
    exp_R = trades_combined['R'].mean()
    win_rate = (trades_combined['R'] > 0).mean()
    profit_factor = trades_combined[trades_combined['R']>0]['R'].sum() / abs(trades_combined[trades_combined['R']<0]['R'].sum())
    total_R = trades_combined['R'].sum()
    # Symulacja kapitału
    timestamps, equity = simulate_capital(trades_combined)
    final_capital = equity[-1] if equity else CAPITAL_START
    total_return_pct = (final_capital - CAPITAL_START) / CAPITAL_START * 100
    print("\n=== RAPORT PAPER TRADING (kapitał wirtualny 1000 USD) ===")
    print(f"Liczba trade:          {len(trades_combined)}")
    print(f"Expectancy (R):        {exp_R:.2f}")
    print(f"Win rate:              {win_rate:.1%}")
    print(f"Profit Factor:         {profit_factor:.2f}")
    print(f"Suma R:                {total_R:.2f}")
    print(f"Końcowy kapitał:       {final_capital:.2f} USD")
    print(f"Całkowity zwrot:       {total_return_pct:.1f}%")
    # Generuj wykres equity
    if len(equity) > 1:
        plt.figure(figsize=(10,6))
        plt.plot(equity, marker='o', linestyle='-', linewidth=1, markersize=3)
        plt.axhline(y=CAPITAL_START, color='gray', linestyle='--', label='Start 1000 USD')
        plt.title('Equity Curve – Virtual Paper Trading')
        plt.xlabel('Trade number')
        plt.ylabel('Capital (USD)')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig('equity_curve.png', dpi=150)
        print("Wykres equity zapisany jako equity_curve.png")
    else:
        print("Za mało danych do wykresu.")
    # Opcjonalnie zapisz szczegóły trade do CSV
    trades_combined.to_csv('paper_trades_analysis.csv', index=False)
    print("Szczegółowa lista trade z R zapisana w paper_trades_analysis.csv")

if __name__ == "__main__":
    main()
