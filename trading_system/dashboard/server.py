"""
Dashboard Server – FastAPI backend dla lokalnego dashboardu.
Uruchomienie: python3 trading_system/dashboard/server.py
Dostęp: http://localhost:8000
"""
import csv
import json
import numpy as np
import requests
from pathlib import Path
from datetime import datetime, timezone
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware

BASE = Path(__file__).parent.parent.parent
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

def get_zscore(sym1, sym2, hedge, lookback=30):
    """Pobiera live Z-score z Binance."""
    try:
        needed = lookback + 10
        url = "https://api.binance.com/api/v3/klines"
        p1 = [float(c[4]) for c in requests.get(url, params={"symbol": sym1.replace("/",""), "interval": "4h", "limit": needed}, timeout=5).json()]
        p2 = [float(c[4]) for c in requests.get(url, params={"symbol": sym2.replace("/",""), "interval": "4h", "limit": needed}, timeout=5).json()]
        if len(p1) < lookback or len(p2) < lookback:
            return None
        spread = [p1[i] - hedge * p2[i] for i in range(len(p1))]
        mean   = np.mean(spread[-lookback:])
        std    = np.std(spread[-lookback:])
        z      = (spread[-1] - mean) / std if std > 0 else 0
        return {
            "z":       round(float(z), 3),
            "spread":  round(float(spread[-1]), 3),
            "price1":  round(p1[-1], 4),
            "price2":  round(p2[-1], 4),
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
        ("LTC/USDT", "XRP/USDT", -26.72, "ltc_xrp"),
        ("LTC/USDT", "ADA/USDT",  19.02, "ltc_ada"),
        ("XRP/USDT", "ADA/USDT",   0.24, "xrp_ada"),
    ]
    result = {}
    for s1, s2, h, name in pairs:
        result[name] = get_zscore(s1, s2, h)
        result[name]["pair"] = f"{s1[:3]}/{s2[:3]}"
        result[name]["signal"] = (
            "SHORT" if result[name].get("z", 0) >  2.0 else
            "LONG"  if result[name].get("z", 0) < -2.0 else
            "FLAT"
        )
    return result

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
    r2 = subprocess.run(["pgrep", "-f", "uvicorn"], 
                       capture_output=True, text=True)
    pids = [p for p in r2.stdout.strip().split() if p]
    result["dashboard"] = {
        "name": "Dashboard Server",
        "pid": pids[0] if pids else None,
        "status": "running" if pids else "stopped",
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

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
