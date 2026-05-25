"""
Hypothesis Generator - przeszukuje źródła i generuje hipotezy do testowania.
Użycie: python hypothesis_generator.py
"""

import anthropic
import json
import yaml
import os
from datetime import datetime

client = anthropic.Anthropic()

SYSTEM_PROMPT = """Jesteś ekspertem od ilościowego tradingu kryptowalut.
Twoim zadaniem jest przeszukanie internetu i znalezienie konkretnych, 
testowalnych strategii tradingowych dla kryptowalut.

Dla każdej strategii musisz określić:
1. Jasny mechanizm (dlaczego powinno działać)
2. Konkretny sygnał wejścia (mierzalny)
3. Kierunek (long/short/both)
4. Timeframe (1h/4h/1d)
5. Które aktywa (BTC, ETH, alty)

Skup się na strategiach opartych na:
- Funding rate i open interest (perpetual futures)
- Cross-sectional momentum (relatywna siła między aktywami)
- Volatility clustering i mean reversion
- On-chain data anomalies
- Liquidation cascades
- Order flow imbalance

Zwróć TYLKO JSON array z hipotezami. Każda hipoteza ma pola:
name, description, mechanism, edge_type, direction, timeframes, 
assets, entry_signal, exit_signal, expected_source
"""

AVAILABLE_EDGES = """
Dostępne funkcje edge w systemie:
- vwap_range_reversion: mean reversion do VWAP w range
- keltner_range_reversion: mean reversion z pasm Keltnera
- compression_breakout_v2: kompresja ATR -> wybicie
- funding_exhaustion_v2: ekstremalny funding + price overextension
- momentum_regime_v2: trend momentum z filtrem EMA/ADX
- failed_breakout_range_v2: failed breakout w range
- sweep_reclaim_v2: liquidity sweep + reclaim
- breakout_retest_v2: breakout + retest + hold
- volume_climax_reversal: reversal po climactic volume
- donchian_acceptance: breakout acceptance z Donchiana
"""

def search_and_generate():
    print("Przeszukuję źródła akademickie i Quantpedia...")
    print("=" * 55)
    
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        tools=[{"type": "web_search_20250305", "name": "web_search"}],
        system=SYSTEM_PROMPT,
        messages=[{
            "role": "user",
            "content": f"""Przeszukaj następujące źródła i znajdź 8 konkretnych strategii tradingowych dla kryptowalut:

1. https://quantpedia.com/strategies/ (szukaj crypto strategies)
2. Przeszukaj arXiv dla: "cryptocurrency momentum anomaly"
3. Przeszukaj SSRN dla: "crypto funding rate predictability"
4. Przeszukaj dla: "perpetual futures basis trading strategy"
5. Przeszukaj dla: "crypto cross-sectional momentum strategy"

{AVAILABLE_EDGES}

Dla każdej znalezionej strategii oceń czy można ją zmapować 
na jedną z dostępnych funkcji edge lub czy potrzeba nowej.

Zwróć wyniki jako JSON array. Przykład formatu:
[
  {{
    "name": "funding_momentum_divergence",
    "description": "Wejście short gdy funding ekstremalnie dodatni ale momentum słabnie",
    "mechanism": "Zatłoczenie longów + brak kontynuacji = wymuszone odwijanie",
    "edge_type": "funding_exhaustion_v2",
    "direction": "short",
    "timeframes": ["4h"],
    "priority": "high",
    "source": "url lub opis źródła",
    "notes": "dodatkowe uwagi"
  }}
]

Odpowiedz TYLKO JSON array, nic więcej."""
        }]
    )
    
    # Wyciągnij tekst z odpowiedzi
    full_text = ""
    for block in response.content:
        if hasattr(block, 'text'):
            full_text += block.text
    
    print("Odpowiedź otrzymana. Parsowanie hipotez...")
    
    # Znajdź JSON w odpowiedzi
    start = full_text.find('[')
    end = full_text.rfind(']') + 1
    if start == -1 or end == 0:
        print("Brak JSON w odpowiedzi. Treść:")
        print(full_text[:500])
        return []
    
    json_str = full_text[start:end]
    
    try:
        hypotheses = json.loads(json_str)
        return hypotheses
    except json.JSONDecodeError as e:
        print(f"Błąd parsowania JSON: {e}")
        print("Raw JSON:")
        print(json_str[:500])
        return []

def save_hypotheses(hypotheses):
    os.makedirs("hypotheses", exist_ok=True)
    saved = 0
    
    today = datetime.now().strftime("%Y-%m-%d")
    
    for h in hypotheses:
        name = h.get("name", "unknown").replace(" ", "_").lower()
        edge = h.get("edge_type", "unknown")
        direction = h.get("direction", "both")
        timeframes = h.get("timeframes", ["4h"])
        priority = h.get("priority", "medium")
        source = h.get("source", "AI generated")
        notes = h.get("notes", "")
        
        yaml_content = {
            "name": name,
            "edge": edge,
            "direction": direction,
            "timeframes": timeframes,
            "prereg_date": today,
            "source": source,
            "mechanism": h.get("mechanism", ""),
            "notes": notes,
            "priority": priority
        }
        
        filepath = f"hypotheses/{name}.yaml"
        with open(filepath, "w") as f:
            yaml.dump(yaml_content, f, default_flow_style=False, allow_unicode=True)
        
        print(f"  [{priority:6s}] {name}")
        print(f"           edge: {edge} | dir: {direction} | tf: {timeframes}")
        print(f"           {h.get('description', '')[:80]}")
        print()
        saved += 1
    
    return saved

def main():
    print("\nHypothesis Generator v1.0")
    print("=" * 55)
    
    hypotheses = search_and_generate()
    
    if not hypotheses:
        print("Nie znaleziono hipotez.")
        return
    
    print(f"\nZnaleziono {len(hypotheses)} hipotez:\n")
    saved = save_hypotheses(hypotheses)
    
    print(f"{'=' * 55}")
    print(f"Zapisano {saved} hipotez do katalogu hypotheses/")
    print(f"Uruchom: python runner.py aby przetestować")

if __name__ == "__main__":
    main()
