"""
Triage kolejki hipotez: klasyfikuje każdy hypotheses/*.yaml jako GREEN/ORANGE/RED.
NIE testuje edge'ów — ocenia tylko, czy warto je w ogóle odpalać.

Reguły (pierwsza dopasowana wygrywa; RED > ORANGE > GREEN):

RED:
  - `edge` nie istnieje w EDGE_MAP (import z runner.py) — silnik nieuruchamialny
    (np. brak runnera dla 'combination'),
  - mechanizm wymaga źródła danych, którego data.py nie ma loadera (dane aspiracyjne),
  - ten sam `edge` jest już w hypotheses/done/ z takim samym setupem (duplikat).

ORANGE:
  - edge uruchamialny, ale wymaga danych ISTNIEJĄCYCH lecz NIEpodpiętych do runnera,
  - edge silnie pokrywa się z już testowaną hipotezą (ten sam edge, inny setup).

GREEN:
  - uruchamialny, dane dostępne i podpięte, brak duplikatu w done/.
"""
import os
import sys
import glob
import re
import yaml

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.abspath(os.path.join(_HERE, "..", ".."))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

# Źródło prawdy dla uruchamialności: EDGE_MAP z runnera.
try:
    from runner import EDGE_MAP
    EDGE_KEYS = set(EDGE_MAP.keys())
    EDGE_MAP_OK = True
except Exception:
    EDGE_KEYS = set()
    EDGE_MAP_OK = False

QUEUE_DIR = os.path.join(_ROOT, "hypotheses")
DONE_DIR = os.path.join(_ROOT, "hypotheses", "done")

# Loadery faktycznie obecne w data.py: OHLCV (cena/wolumen), funding, fear&greed,
# open interest, hashrate, bybit (cross-exchange). Poniżej to, czego NIE MA wcale.
# Mechanizm wspominający którekolwiek z tych źródeł → RED (dane aspiracyjne).
UNAVAILABLE_DATA = [
    (("mvrv",), "MVRV (on-chain)"),
    (("nupl",), "NUPL (on-chain)"),
    (("etf",), "przepływy ETF"),
    (("stablecoin",), "przepływy stablecoinów"),
    (("social", "sentiment", "twitter", "reddit"), "dane social/sentiment"),
    (("chainlet", "transaction network", "transaction patterns",
      "network clustering", "spectral analysis", "on-chain", "onchain"),
     "dane on-chain / graf transakcji"),
    (("realized price", "realized cap", "realized value"),
     "realized price/cap (on-chain)"),
    (("dex",), "dane DEX (np. spread CEX/DEX)"),
    (("order book", "orderbook", "order flow", "level 2", "bid/ask", "bid-ask"),
     "dane order book / L2"),
    (("expiry", "expiration", "options expiry"), "kalendarz wygasania / opcje"),
]

STATUS_RANK = {"GREEN": 0, "ORANGE": 1, "RED": 2}
PRIO_RANK = {"high": 0, "medium": 1, "low": 2}


def _text(h):
    """Złączony, zlowercase'owany tekst do skanowania słów-kluczy o danych."""
    return " ".join(str(h.get(k, "")) for k in ("name", "mechanism", "notes")).lower()


def _match_unavailable(text):
    for kws, reason in UNAVAILABLE_DATA:
        for kw in kws:
            if kw in text:
                return reason
    return None


def _needs_open_interest(text):
    """OI ma loader w data.py, ale runner go NIE podpina do edge'ów → ORANGE."""
    return ("open interest" in text or "open-interest" in text
            or bool(re.search(r"\boi\b", text)))


def _load_done_index(done_dir):
    """edge -> lista (name, (direction, frozenset(timeframes))) z hypotheses/done/."""
    idx = {}
    for f in glob.glob(os.path.join(done_dir, "*.yaml")):
        try:
            h = yaml.safe_load(open(f)) or {}
        except Exception:
            continue
        edge = h.get("edge")
        if not edge:
            continue
        setup = (h.get("direction"), frozenset(h.get("timeframes") or []))
        idx.setdefault(edge, []).append((h.get("name", os.path.basename(f)), setup))
    return idx


def _classify(h, done_idx):
    """Zwraca (status, jednozdaniowy_powód) dla pojedynczej hipotezy."""
    edge = h.get("edge")
    direction = h.get("direction")
    tfs = frozenset(h.get("timeframes") or [])
    text = _text(h)

    # 1. Edge nieuruchamialny (brak w EDGE_MAP).
    if not edge or edge not in EDGE_KEYS:
        return "RED", (f"edge '{edge}' nie istnieje w EDGE_MAP — silnik nieuruchamialny "
                       f"(np. brak runnera 'combination')")

    # 2. Mechanizm wymaga danych, których nie ma loadera.
    reason = _match_unavailable(text)
    if reason:
        return "RED", f"mechanizm wymaga: {reason} — brak loadera w data.py (dane aspiracyjne)"

    # 3. Dokładny duplikat w done/ (ten sam edge + setup).
    done_matches = done_idx.get(edge, [])
    exact = [n for (n, s) in done_matches if s == (direction, tfs)]
    if exact:
        return "RED", (f"ten sam edge '{edge}' i setup ({direction}, {sorted(tfs)}) "
                       f"już przetestowany w done/ ({exact[0]})")

    # 4. Dane istnieją, ale niepodpięte do runnera.
    if _needs_open_interest(text):
        return "ORANGE", ("wymaga Open Interest — loader jest w data.py, ale runner "
                          "nie podpina OI do edge'ów")

    # 5. Silne pokrycie z już testowaną hipotezą (ten sam edge, inny setup).
    if done_matches:
        return "ORANGE", (f"edge '{edge}' już testowany w done/ ({done_matches[0][0]}) "
                          f"przy innym setupie — silne pokrycie")

    # 6. Czysty kandydat.
    return "GREEN", "uruchamialny, dane dostępne i podpięte, brak duplikatu w done/"


def triage_hypotheses(queue_dir=QUEUE_DIR, done_dir=DONE_DIR):
    """Klasyfikuje całą kolejkę. Zwraca listę dictów posortowaną GREEN→ORANGE→RED."""
    done_idx = _load_done_index(done_dir)
    out = []
    for f in sorted(glob.glob(os.path.join(queue_dir, "*.yaml"))):
        base = os.path.basename(f).replace(".yaml", "")
        try:
            h = yaml.safe_load(open(f)) or {}
        except Exception as e:
            out.append({"name": base, "edge": None, "status": "RED",
                        "reason": f"nie udało się sparsować YAML: {e}",
                        "priority": "?", "direction": "?", "timeframes": [], "source": ""})
            continue
        status, reason = _classify(h, done_idx)
        if not EDGE_MAP_OK:
            reason += " [UWAGA: nie udało się zaimportować EDGE_MAP z runner.py]"
        out.append({
            "name":       h.get("name", base),
            "edge":       h.get("edge"),
            "status":     status,
            "reason":     reason,
            "priority":   h.get("priority", "?"),
            "direction":  h.get("direction", "?"),
            "timeframes": h.get("timeframes") or [],
            "source":     h.get("source", ""),
        })
    out.sort(key=lambda r: (STATUS_RANK.get(r["status"], 3),
                            PRIO_RANK.get(r["priority"], 3), r["name"]))
    return out


if __name__ == "__main__":
    rows = triage_hypotheses()
    counts = {"GREEN": 0, "ORANGE": 0, "RED": 0}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    print(f"Triage kolejki hipotez: {len(rows)} plików "
          f"(🟢 {counts['GREEN']}  🟡 {counts['ORANGE']}  🔴 {counts['RED']})\n")
    icon = {"GREEN": "🟢", "ORANGE": "🟡", "RED": "🔴"}
    for r in rows:
        print(f"{icon.get(r['status'],'?')} {r['status']:<6} {r['name']:<42} "
              f"[{r['edge']}] {r['reason']}")
