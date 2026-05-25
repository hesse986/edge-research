# Trading System Context

## Potwierdzony edge
- Pairs trading spread mean reversion: LTC/XRP, LTC/ADA, XRP/ADA
- Timeframe: 4H
- Parametry: LOOKBACK=30, ENTRY_Z=2.0, EXIT_Z=0.5, SL_MULT=1.5, RR=2.0
- Backtest: exp_R=0.40, PF=1.76, forward pctMR=98.7%
- Monitor działa: paper_trading_pairs_advanced_with_R.py

## Framework testowania
- Kalibracja → Walidacja → Holdout → Forward test
- Próg: pctMR >= 95% na walidacji
- Benchmark: matched random (bootstrap 5000 iteracji)

## Odrzucone edge'e
- failed_breakout_range_v2 (degradacja w forward teście)
- short_term_reversal_liquidity, hash_ribbon, peer_reversal
- momentum, funding_exhaustion, compression_breakout

## Stack
- Python 3.12, ccxt, pandas, numpy, sklearn
- runner.py, edges.py, premium_edges.py, data.py
- Binance OHLCV 1H jako baza danych

## Cel: multi-agent trading system
- Research Agent: hipotezy z literatury
- Backtest Agent: runner.py z PBO
- Validation Agent: Deflated Sharpe, pctMR
- Risk Agent: position sizing, drawdown control
- Execution Agent: papier → live
