#!/bin/bash
cd /Users/oletomasen/Downloads/edge_research
source venv/bin/activate
REPORT_FILE="weekly_report_$(date +%Y%m%d).txt"
python analyze_paper_trades.py > "$REPORT_FILE"

# Wyślij e-mail
mail -s "Weekly Paper Trading Report $(date +%Y-%m-%d)" tomaszoleszkowicz@gmail.com < "$REPORT_FILE"

# Opcjonalnie: usuń stary plik (zostaw na wypadek)
# rm "$REPORT_FILE"
