#!/usr/bin/env python3
"""Monitor Quantocracy – pobiera RSS i zapisuje nowe wpisy z kodem."""
import feedparser
import re
from pathlib import Path
from datetime import datetime

LOG_FILE = Path("/Users/oletomasen/Downloads/edge_research/quantocracy_entries.txt")
RSS_URL = "https://quantocracy.com/feed/"

def fetch_entries():
    feed = feedparser.parse(RSS_URL)
    new_entries = []
    for entry in feed.entries[:20]:
        title = entry.title
        link = entry.link
        summary = entry.summary
        has_code = bool(re.search(r'```python|```r|library\(|import ', summary, re.IGNORECASE))
        if has_code:
            new_entries.append((title, link, summary[:500]))
    return new_entries

def main():
    if not LOG_FILE.exists():
        LOG_FILE.touch()
    old = LOG_FILE.read_text()
    new = fetch_entries()
    if new:
        with LOG_FILE.open("a") as f:
            f.write(f"\n--- {datetime.now().isoformat()} ---\n")
            for title, link, summary in new:
                if title not in old:
                    f.write(f"TITLE: {title}\nLINK: {link}\nSUMMARY: {summary}\n\n")
        print(f"Zapisano {len(new)} nowych wpisów")
    else:
        print("Brak nowych wpisów z kodem")

if __name__ == "__main__":
    main()
