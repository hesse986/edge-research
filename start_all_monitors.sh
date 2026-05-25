#!/bin/bash
cd /Users/oletomasen/Downloads/edge_research
source venv/bin/activate
nohup python3 paper_trading_pairs_advanced.py > log_paper_pairs_advanced.txt 2>&1 &
echo "Advanced pairs monitor PID: $!"
# Tu możesz dodać inne monitory (np. dla failed breakout)
