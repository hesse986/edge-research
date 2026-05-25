#!/usr/bin/env python3
"""Uruchamia backtest pairs trading i zapisuje transakcje do CSV, a następnie analizuje kapitał."""

import sys
sys.path.insert(0, '.')
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from backtest_pairs_full import run_backtest, CALIB_START, CALIB_END, VAL_START, VAL_END, FORWARD_START, FORWARD_END

CAPITAL_START = 1000.0
RISK_PERCENT = 0.01

def collect_trades():
    """Uruchamia backtest dla trzech okresów i zbiera transakcje z R."""
    trades = []
    # Kalibracja
    t_cal = run_backtest(CALIB_START, CALIB_END, "Kalibracja")
    if t_cal:
        for t in t_cal:
            trades.append(("kalibracja", t[0], t[1], t[2], t[3], t[4], t[5]))
    # Walidacja
    t_val = run_backtest(VAL_START, VAL_END, "Walidacja")
    if t_val:
        for t in t_val:
            trades.append(("walidacja", t[0], t[1], t[2], t[3], t[4], t[5]))
    # Forward
    t_for = run_backtest(FORWARD_START, FORWARD_END, "Forward")
    if t_for:
        for t in t_for:
            trades.append(("forward", t[0], t[1], t[2], t[3], t[4], t[5]))
    return trades

def simulate_capital(trades_df):
    """Symuluje kapitał na podstawie DataFrame z kolumną R."""
    capital = CAPITAL_START
    equity = [capital]
    for _, row in trades_df.iterrows():
        r = row['R']
        if pd.isna(r):
            continue
        risk_cap = capital * RISK_PERCENT
        profit = risk_cap * r
        capital += profit
        equity.append(capital)
    return equity

def main():
    print("Zbieranie transakcji z backtestu...")
    trades = collect_trades()
    if not trades:
        print("Brak transakcji.")
        return
    df = pd.DataFrame(trades, columns=["period","entry_bar","exit_bar","position","entry_z","exit_z","R"])
    df.to_csv("backtest_trades_full.csv", index=False)
    print(f"Zapisano {len(df)} transakcji do backtest_trades_full.csv")
    # Statystyki
    exp_R = df['R'].mean()
    win_rate = (df['R'] > 0).mean()
    profit_factor = df[df['R']>0]['R'].sum() / abs(df[df['R']<0]['R'].sum())
    total_R = df['R'].sum()
    print("\n=== Statystyki backtestu ===")
    print(f"Liczba trade: {len(df)}")
    print(f"Expectancy (R): {exp_R:.4f}")
    print(f"Win rate: {win_rate:.2%}")
    print(f"Profit Factor: {profit_factor:.2f}")
    print(f"Suma R: {total_R:.4f}")
    # Symulacja kapitału
    equity = simulate_capital(df)
    final_cap = equity[-1]
    total_return = (final_cap - CAPITAL_START) / CAPITAL_START * 100
    print(f"\n=== Wirtualny kapitał (start {CAPITAL_START} USD, ryzyko {RISK_PERCENT*100}% na trade) ===")
    print(f"Końcowy kapitał: {final_cap:.2f} USD")
    print(f"Całkowity zwrot: {total_return:.1f}%")
    # Wykres
    plt.figure(figsize=(10,6))
    plt.plot(equity, marker='o', markersize=3, linestyle='-')
    plt.axhline(y=CAPITAL_START, color='gray', linestyle='--', label='Start 1000 USD')
    plt.title('Equity Curve – Pairs Trading LTC/XRP (backtest)')
    plt.xlabel('Trade number')
    plt.ylabel('Capital (USD)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig('equity_curve_backtest.png', dpi=150)
    print("Wykres zapisany jako equity_curve_backtest.png")
    # Dodatkowo zapisz szczegóły trade z okresami
    print("\nPodsumowanie według okresów:")
    for period, group in df.groupby('period'):
        print(f"{period:10s} trades={len(group)} exp_R={group['R'].mean():.4f}")

if __name__ == "__main__":
    main()
