#!/usr/bin/env python3
"""Paper Trading Monitor – failed_breakout_range_v2 4H
Assets: LTC/USDT, ADA/USDT
"""
import time, csv, sys
from datetime import datetime
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import data as datamod
import edges as edgemod

ASSETS      = ["LTC/USDT", "ADA/USDT"]
EDGE_FN     = edgemod.edge_failed_breakout_range_v2
TF          = "4h"
LOOKBACK    = "2024-01-01"
LOG_FILE    = Path(__file__).parent / "paper_trades.csv"
CHECK_EVERY = 60 * 60 * 4

fieldnames = ["timestamp","asset","direction","signal_bar","close_at_signal",
              "status","entry","sl","tp","exit_price","result_R","notes"]

def init_log():
    if not LOG_FILE.exists():
        with LOG_FILE.open("w", newline="") as f:
            csv.DictWriter(f, fieldnames=fieldnames).writeheader()
        print("Stworzono: " + str(LOG_FILE))

def get_signals_now(asset):
    try:
        end  = datetime.utcnow().strftime("%Y-%m-%d")
        df1h = datamod.load_binance_ohlcv(asset, "1h", LOOKBACK, end, use_cache=False)
        df   = datamod.resample_ohlcv(df1h, "4h")
        sigs = EDGE_FN(df)
        if not sigs:
            return None
        last_bar, direction = sigs[-1]
        if last_bar < len(df) - 2:
            return None
        close   = float(df["close"].iloc[last_bar])
        atr_val = float(df["high"].rolling(14).max().iloc[last_bar] -
                        df["low"].rolling(14).min().iloc[last_bar]) / 14
        sl_dist = atr_val * 1.5
        dir_str = "LONG" if direction == 1 else "SHORT"
        entry   = float(df["open"].iloc[last_bar + 1]) if last_bar + 1 < len(df) else close
        sl      = entry - sl_dist if direction == 1 else entry + sl_dist
        tp      = entry + sl_dist * 2 if direction == 1 else entry - sl_dist * 2
        return {"timestamp": datetime.utcnow().isoformat(),
                "asset": asset, "direction": dir_str,
                "signal_bar": str(df.index[last_bar]),
                "close_at_signal": f"{close:.4f}", "status": "OPEN",
                "entry": f"{entry:.4f}", "sl": f"{sl:.4f}", "tp": f"{tp:.4f}",
                "exit_price": "", "result_R": "", "notes": "paper_trade"}
    except Exception as e:
        print("  ERROR " + asset + ": " + str(e))
        return None

def run():
    init_log()
    print("Paper Trading Monitor – " + str(ASSETS))
    print("Edge: failed_breakout_range_v2 @ " + TF)
    print("Log: " + str(LOG_FILE))
    print("Sprawdzanie co 4h")
    while True:
        now = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")
        print("[" + now + "] Sprawdzam sygnaly...")
        for asset in ASSETS:
            sig = get_signals_now(asset)
            if sig:
                msg = "  *** SYGNAL: " + asset + " " + sig["direction"]
                msg += " entry=" + sig["entry"] + " sl=" + sig["sl"] + " tp=" + sig["tp"] + " ***"
                print(msg)
                with LOG_FILE.open("a", newline="") as f:
                    csv.DictWriter(f, fieldnames=fieldnames).writerow(sig)
            else:
                print("  " + asset + ": brak sygnalu")
        print("  Nastepne sprawdzenie za 4h")
        time.sleep(CHECK_EVERY)

if __name__ == "__main__":
    run()
