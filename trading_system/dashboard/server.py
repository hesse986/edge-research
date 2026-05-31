"""
Dashboard Server – FastAPI backend dla lokalnego dashboardu.
Uruchomienie: python3 trading_system/dashboard/server.py
Dostęp: http://localhost:8000
"""
import csv
import json
import sys
import numpy as np
from pathlib import Path
from datetime import datetime, timezone
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

BASE = Path(__file__).parent.parent.parent
sys.path.insert(0, str(BASE))
from trading_system.research.live_data import live_spread_zscore

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"])

# ─── helpers ────────────────────────────────────────────────

def load_csv(name):
    p = BASE / name
    if not p.exists():
        return []
    return list(csv.DictReader(open(p)))

def compute_stats(returns):
    if not returns:
        return {}
    r = np.array(returns, dtype=float)
    sr = float(r.mean() / r.std() * np.sqrt(252)) if r.std() > 0 else 0
    wins = r[r > 0].sum()
    loss = abs(r[r < 0].sum())
    pf   = float(wins / loss) if loss > 0 else 999.0
    cum  = np.cumprod(1 + r * 0.01)
    roll = np.maximum.accumulate(cum)
    dd   = float(((cum - roll) / roll).min())
    capital = round(1000.0 * float(cum[-1]), 2)
    return {
        "n":              len(r),
        "mean_R":         round(float(r.mean()), 4),
        "win_rate":       round(float((r > 0).mean()) * 100, 1),
        "profit_factor":  round(pf, 2),
        "sharpe":         round(sr, 2),
        "max_drawdown":   round(dd * 100, 1),
        "total_R":        round(float(r.sum()), 2),
        "capital":        capital,
    }

def compute_equity(returns):
    eq = [1000.0]
    for r in returns:
        eq.append(round(eq[-1] * (1 + float(r) * 0.01), 2))
    return eq

def get_zscore(sym1, sym2, lookback=30):
    """Live Z-score przez wspólny helper (rolling OLS, jedno źródło prawdy).

    Dashboard = podgląd, więc closed_only=False (pokazuje formującą się świecę).
    """
    try:
        data = live_spread_zscore(sym1, sym2, lookback=lookback, closed_only=False)
        if data is None:
            return None
        return {
            "z":       round(data["z"], 3),
            "spread":  round(data["spread_last"], 3),
            "price1":  round(data["price1"], 4),
            "price2":  round(data["price2"], 4),
        }
    except Exception as e:
        return {"error": str(e)}

# ─── API endpoints ───────────────────────────────────────────

@app.get("/api/stats")
def stats():
    bt = load_csv("backtest_trades_full.csv")
    bt_r = [row["R"] for row in bt if row.get("R")]
    pt = load_csv("paper_trades_pairs_advanced_R.csv")
    pt_closed = [r for r in pt if r.get("status") == "CLOSED"]
    pt_r = [r["result_R"] for r in pt_closed if r.get("result_R")]
    return {
        "backtest":     compute_stats(bt_r),
        "paper":        compute_stats(pt_r),
        "updated_at":   datetime.now(timezone.utc).isoformat(),
    }

@app.get("/api/equity")
def equity():
    bt = load_csv("backtest_trades_full.csv")
    bt_r = [row["R"] for row in bt if row.get("R")]
    return {"equity": compute_equity(bt_r), "n": len(bt_r)}

@app.get("/api/positions")
def positions():
    pt = load_csv("paper_trades_pairs_advanced_R.csv")
    open_pos  = [r for r in pt if r.get("status") == "OPEN"]
    closed    = [r for r in pt if r.get("status") == "CLOSED"]
    return {
        "open":   open_pos,
        "closed": closed[-20:][::-1],
    }

@app.get("/api/zscore")
def zscore():
    pairs = [
        ("LTC/USDT",  "ADA/USDT",  "ltc_ada"),
        ("ADA/USDT",  "LINK/USDT", "ada_link"),
        ("BNB/USDT",  "SOL/USDT",  "bnb_sol"),
    ]
    result = {}
    for s1, s2, name in pairs:
        result[name] = get_zscore(s1, s2)
        result[name]["pair"] = f"{s1[:3]}/{s2[:3]}"
        result[name]["signal"] = (
            "SHORT" if result[name].get("z", 0) >  2.0 else
            "LONG"  if result[name].get("z", 0) < -2.0 else
            "FLAT"
        )
    return result

@app.get("/api/hypotheses")
def hypotheses():
    """Triage kolejki hipotez (GREEN/ORANGE/RED + powód). Tylko klasyfikacja."""
    try:
        from trading_system.research.triage_hypotheses import triage_hypotheses
        return {"hypotheses": triage_hypotheses()}
    except Exception as e:
        return {"hypotheses": [], "error": str(e)}

@app.get("/", response_class=HTMLResponse)
def dashboard():
    return (Path(__file__).parent / "index.html").read_text()


@app.get("/api/processes")
def processes():
    import subprocess
    result = {}
    
    # Sprawdz monitor pairs trading
    r1 = subprocess.run(["pgrep", "-f", "paper_trading_pairs_advanced_with_R"], 
                       capture_output=True, text=True)
    result["monitor"] = {
        "name": "Pairs Trading Monitor",
        "pid": r1.stdout.strip() or None,
        "status": "running" if r1.stdout.strip() else "stopped",
        "file": "paper_trading_pairs_advanced_with_R.py"
    }
    
    # Sprawdz dashboard
    r2 = subprocess.run(["lsof", "-ti", "tcp:8000"],
                       capture_output=True, text=True)
    pid = r2.stdout.strip().split("\n")[0]
    result["dashboard"] = {
        "name": "Dashboard Server",
        "pid": pid or None,
        "status": "running" if pid else "stopped",
        "file": "trading_system/dashboard/server.py"
    }
    
    # Sprawdz cron
    r3 = subprocess.run(["crontab", "-l"], capture_output=True, text=True)
    has_cron = "send_weekly_report" in r3.stdout
    result["cron"] = {
        "name": "Weekly Report (cron)",
        "pid": None,
        "status": "configured" if has_cron else "not configured",
        "file": "send_weekly_report.sh"
    }
    
    # Sprawdz auto_search
    r4 = subprocess.run(["pgrep", "-f", "auto_search"],
                       capture_output=True, text=True)
    result["auto_search"] = {
        "name": "Auto Search",
        "pid": r4.stdout.strip() or None,
        "status": "running" if r4.stdout.strip() else "stopped",
        "file": "auto_search.py"
    }
    
    return result


@app.get("/api/spread_chart/{pair_name}")
def spread_chart(pair_name: str):
    """Zwraca historię spreadu dla pary z liniami entry/TP/SL.

    Spread liczony wspólnym helperem (rolling OLS) — bez stałej bety.
    """
    pair_config = {
        "ltc_ada":  ("LTC/USDT", "ADA/USDT"),
        "ada_link": ("ADA/USDT", "LINK/USDT"),
        "bnb_sol":  ("BNB/USDT", "SOL/USDT"),
    }

    if pair_name not in pair_config:
        return {"error": "Unknown pair"}

    sym1, sym2 = pair_config[pair_name]

    try:
        data = live_spread_zscore(sym1, sym2, lookback=30, history=60, closed_only=False)
        if data is None:
            return {"error": "Za mało danych"}

        # Pobierz otwarte pozycje dla tej pary
        open_positions = []
        pt = load_csv("paper_trades_pairs_advanced_R.csv")
        for row in pt:
            if row.get("pair") == pair_name and row.get("status") == "OPEN":
                open_positions.append({
                    "entry":     float(row.get("entry_price", 0) or 0),
                    "sl":        float(row.get("sl", 0) or 0),
                    "tp":        float(row.get("tp", 0) or 0),
                    "direction": row.get("direction", ""),
                    "timestamp": row.get("timestamp", ""),
                })

        return {
            "timestamps": data["timestamps"],
            "spread":     data["spread"],
            "zscore":     data["zscore"],
            "mean":       data["mean"],
            "std":        data["std"],
            "open_positions": open_positions,
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
