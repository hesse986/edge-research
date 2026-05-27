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
        ("LTC/USDT",  "ADA/USDT",   19.02, "ltc_ada"),
        ("ADA/USDT",  "LINK/USDT",   0.0199, "ada_link"),
        ("BNB/USDT",  "SOL/USDT",    0.0393, "bnb_sol"),
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
    """Zwraca historię spreadu dla pary z liniami entry/TP/SL."""
    import requests as _req
    import numpy as _np

    pair_config = {
        "ltc_ada":  ("LTC/USDT", "ADA/USDT",  19.02),
        "ada_link": ("ADA/USDT", "LINK/USDT",  0.0199),
        "bnb_sol":  ("BNB/USDT", "SOL/USDT",   0.0393),
    }

    if pair_name not in pair_config:
        return {"error": "Unknown pair"}

    sym1, sym2, hedge = pair_config[pair_name]
    url = "https://api.binance.com/api/v3/klines"

    try:
        r1 = _req.get(url, params={"symbol": sym1.replace("/",""), "interval": "4h", "limit": 60}, timeout=5)
        r2 = _req.get(url, params={"symbol": sym2.replace("/",""), "interval": "4h", "limit": 60}, timeout=5)
        candles1 = r1.json()
        candles2 = r2.json()

        timestamps = [int(c[0]) for c in candles1]
        p1 = [float(c[4]) for c in candles1]
        p2 = [float(c[4]) for c in candles2]
        spread = [p1[i] - hedge * p2[i] for i in range(len(p1))]

        lookback = 30
        mean = _np.mean(spread[-lookback:])
        std  = _np.std(spread[-lookback:])
        zscore = [(s - mean) / std if std > 0 else 0 for s in spread]

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
            "timestamps": timestamps,
            "spread":     spread,
            "zscore":     zscore,
            "mean":       float(mean),
            "std":        float(std),
            "open_positions": open_positions,
        }
    except Exception as e:
        return {"error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=False)
