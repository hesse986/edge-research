"""
Auto Search - automatyczne wyszukiwanie hipotez co 4 godziny.
Uruchomienie: nohup python auto_search.py > auto_search.log 2>&1 &
"""

import anthropic
import json
import yaml
import os
import time
import hashlib
from datetime import datetime
from pathlib import Path

client = anthropic.Anthropic()

TESTED_DB      = "tested_hypotheses.json"
HYPOTHESES_DIR = "hypotheses"
SEARCH_LOG     = "search_history.json"
INTERVAL_HOURS = 4

SOURCES = [
    {
        "name": "quantpedia",
        "query": "site:quantpedia.com crypto trading strategy 2024 2025",
        "focus": "Szukaj strategii z konkretnym mechanizmem i parametrami"
    },
    {
        "name": "arxiv_momentum",
        "query": "arxiv cryptocurrency cross-sectional momentum anomaly 2024",
        "focus": "Szukaj akademickich dowodów na momentum w krypto"
    },
    {
        "name": "ssrn_funding",
        "query": "SSRN perpetual futures funding rate predictability crypto",
        "focus": "Szukaj strategii opartych na funding rate"
    },
    {
        "name": "arxiv_liquidation",
        "query": "arxiv crypto liquidation cascade order flow 2024 2025",
        "focus": "Szukaj strategii opartych na likwidacjach i przepływach"
    },
    {
        "name": "quantpedia_volatility",
        "query": "site:quantpedia.com crypto volatility clustering mean reversion",
        "focus": "Szukaj strategii opartych na zmienności"
    },
    {
        "name": "ssrn_basis",
        "query": "SSRN crypto basis trading perpetual spot arbitrage 2024",
        "focus": "Szukaj strategii basis i carry"
    },
    {
        "name": "arxiv_onchain",
        "query": "arxiv on-chain metrics bitcoin price prediction 2024",
        "focus": "Szukaj on-chain indykatorów jako sygnałów"
    },
    {
        "name": "combinations",
        "query": "crypto multi-factor strategy funding momentum volatility combined",
        "focus": "Szukaj kombinacji wielu sygnałów"
    },
]

AVAILABLE_EDGES = """
Dostępne funkcje edge:
- vwap_range_reversion, keltner_range_reversion
- compression_breakout_v2, failed_breakout_range_v2
- funding_exhaustion_v2, funding_momentum_divergence
- momentum_regime_v2, relative_strength_rotation
- liquidation_proxy, panic_no_followthrough
- sweep_reclaim_v2, breakout_retest_v2
- volume_climax_reversal, donchian_acceptance
"""

def load_tested():
    if os.path.exists(TESTED_DB):
        with open(TESTED_DB) as f:
            return json.load(f)
    return {"tested": [], "rejected": [], "candidates": []}

def save_tested(db):
    with open(TESTED_DB, "w") as f:
        json.dump(db, f, indent=2)

def load_search_history():
    if os.path.exists(SEARCH_LOG):
        with open(SEARCH_LOG) as f:
            return json.load(f)
    return {"searches": []}

def save_search_history(history):
    with open(SEARCH_LOG, "w") as f:
        json.dump(history, f, indent=2)

def hypothesis_hash(h):
    key = f"{h.get('edge','')}-{h.get('direction','')}-{'-'.join(h.get('timeframes',[]))}"
    return hashlib.md5(key.encode()).hexdigest()[:8]

def search_source(source, tested_names, previous_results):
    print(f"\n  Przeszukuję: {source['name']}...")

    prev_summary = ""
    if previous_results:
        top = sorted(previous_results, key=lambda x: x.get("pctTS", 0), reverse=True)[:5]
        prev_summary = "\n".join([
            f"- {r['name']}: edge={r['edge']}, pctTS={r.get('pctTS','?')}, exp_R={r.get('exp_R','?')}"
            for r in top
        ])

    prompt = f"""Przeszukaj internet używając zapytania: "{source['query']}"
Fokus: {source['focus']}

{AVAILABLE_EDGES}

Już przetestowane (NIE powtarzaj):
{chr(10).join(tested_names[-20:]) if tested_names else 'brak'}

Najlepsze dotychczasowe wyniki (użyj jako inspirację do kombinacji):
{prev_summary if prev_summary else 'brak danych'}

Na podstawie wyników wyszukiwania zaproponuj 3-5 NOWYCH hipotez które:
1. Mają jasny mechanizm ekonomiczny
2. Mapują na dostępne funkcje edge LUB opisują nową funkcję do zakodowania
3. Nie są powtórzeniem już przetestowanych
4. Jeśli widzisz możliwość kombinacji dwóch silnych sygnałów - zaproponuj ją

Zwróć TYLKO JSON array:
[
  {{
    "name": "unique_name",
    "description": "krótki opis",
    "mechanism": "dlaczego powinno działać",
    "edge_type": "nazwa_funkcji lub 'new_function_needed'",
    "direction": "long/short/both",
    "timeframes": ["4h"],
    "priority": "high/medium/low",
    "source_url": "url źródła",
    "combination_of": ["edge1", "edge2"],
    "notes": "dodatkowe uwagi"
  }}
]"""

    try:
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=3000,
            tools=[{"type": "web_search_20250305", "name": "web_search"}],
            messages=[{"role": "user", "content": prompt}]
        )

        full_text = ""
        for block in response.content:
            if hasattr(block, 'text'):
                full_text += block.text

        start = full_text.find('[')
        end   = full_text.rfind(']') + 1
        if start == -1 or end == 0:
            return []

        hypotheses = json.loads(full_text[start:end])
        return hypotheses

    except Exception as e:
        print(f"    Błąd: {e}")
        return []

def save_hypothesis_yaml(h, today):
    name = h.get("name", "unknown").replace(" ", "_").lower()
    filepath = f"{HYPOTHESES_DIR}/{name}.yaml"

    if os.path.exists(filepath):
        return False, "exists"

    content = {
        "name":        name,
        "edge":        h.get("edge_type", "unknown"),
        "direction":   h.get("direction", "both"),
        "timeframes":  h.get("timeframes", ["4h"]),
        "prereg_date": today,
        "source":      h.get("source_url", "auto_search"),
        "mechanism":   h.get("mechanism", ""),
        "notes":       h.get("notes", ""),
        "priority":    h.get("priority", "medium"),
        "combination_of": h.get("combination_of", [])
    }

    with open(filepath, "w") as f:
        yaml.dump(content, f, default_flow_style=False, allow_unicode=True)

    return True, filepath

def update_tested_from_results():
    db = load_tested()
    import glob
    for csv_file in sorted(glob.glob("results_*.csv")):
        with open(csv_file) as f:
            for line in f:
                parts = line.strip().split(",")
                if len(parts) < 13 or parts[0] == "hypothesis":
                    continue
                name    = parts[0]
                pct_mr  = parts[10]
                pct_ts  = parts[11]
                verdict = parts[12]
                if name not in [t["name"] for t in db["tested"]]:
                    db["tested"].append({
                        "name":    name,
                        "pctMR":   pct_mr,
                        "pctTS":   pct_ts,
                        "verdict": verdict
                    })
    save_tested(db)
    return db

def run_search_cycle(cycle_num):
    print(f"\n{'='*55}")
    print(f"Cykl #{cycle_num} | {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"{'='*55}")

    db      = update_tested_from_results()
    history = load_search_history()
    today   = datetime.now().strftime("%Y-%m-%d")
    os.makedirs(HYPOTHESES_DIR, exist_ok=True)

    tested_names = [t["name"] for t in db["tested"]]
    prev_results = db.get("candidates", [])

    source_idx    = cycle_num % len(SOURCES)
    source        = SOURCES[source_idx]
    new_count     = 0
    skipped_count = 0

    hypotheses = search_source(source, tested_names, prev_results)

    print(f"  Znaleziono {len(hypotheses)} propozycji")

    for h in hypotheses:
        name = h.get("name", "").replace(" ", "_").lower()
        h_hash = hypothesis_hash(h)

        already_tested = any(
            t["name"] == name or t["name"].startswith(name[:15])
            for t in db["tested"]
        )

        if already_tested:
            skipped_count += 1
            print(f"  [skip] {name} – już testowane")
            continue

        if h.get("edge_type") == "new_function_needed":
            print(f"  [new_fn] {name} – wymaga nowej funkcji: {h.get('notes','')}")
            db.setdefault("new_functions_needed", []).append({
                "name":      name,
                "mechanism": h.get("mechanism", ""),
                "notes":     h.get("notes", ""),
                "date":      today
            })
            save_tested(db)
            continue

        saved, result = save_hypothesis_yaml(h, today)
        if saved:
            new_count += 1
            print(f"  [new]  {name}")
            print(f"         edge: {h.get('edge_type')} | dir: {h.get('direction')} | priority: {h.get('priority')}")
            if h.get("combination_of"):
                print(f"         kombinacja: {h.get('combination_of')}")
        else:
            skipped_count += 1

    history["searches"].append({
        "cycle":     cycle_num,
        "source":    source["name"],
        "date":      today,
        "new":       new_count,
        "skipped":   skipped_count,
        "total":     len(hypotheses)
    })
    save_search_history(history)

    print(f"\n  Nowych hipotez: {new_count} | Pominięto: {skipped_count}")

    if new_count > 0:
        print(f"\n  Uruchamianie testów...")
        os.system("nohup python runner.py >> runner_auto.log 2>&1 &")

    return new_count

def checklist_report():
    history = load_search_history()
    sources_searched = set(s["source"] for s in history.get("searches", []))
    all_sources      = set(s["name"] for s in SOURCES)
    missing          = all_sources - sources_searched

    print(f"\n{'='*55}")
    print("CHECKLIST ŹRÓDEŁ")
    print(f"{'='*55}")
    for s in SOURCES:
        status = "✓" if s["name"] in sources_searched else "✗ NIEODWIEDZONE"
        print(f"  {status} {s['name']}")
    if missing:
        print(f"\n  UWAGA: {len(missing)} źródeł nieodwiedzonych: {missing}")
    else:
        print("\n  Wszystkie źródła odwiedzone przynajmniej raz.")

def main():
    print("\nAuto Search System v1.0")
    print(f"Interval: co {INTERVAL_HOURS} godziny")
    print(f"Źródła: {len(SOURCES)}")
    print("Ctrl+C aby zatrzymać\n")

    cycle = 0
    while True:
        try:
            run_search_cycle(cycle)
            checklist_report()
            cycle += 1
            print(f"\nSleep {INTERVAL_HOURS}h do następnego cyklu...")
            time.sleep(INTERVAL_HOURS * 3600)
        except KeyboardInterrupt:
            print("\nZatrzymano przez użytkownika.")
            break
        except Exception as e:
            print(f"Błąd w cyklu: {e}")
            time.sleep(300)

if __name__ == "__main__":
    main()
