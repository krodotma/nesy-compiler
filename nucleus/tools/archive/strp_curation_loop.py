#!/usr/bin/env python3
"""
strp_curation_loop.py - Automated SOTA Curation Daemon

Fetches RSS/Atom feeds, filters for new items, and ingests them into the Rhizome.
Emits 'curation.cycle.*' events to the bus.

Usage:
  python3 strp_curation_loop.py --seeds ../../sota_international_seeds.md
"""

import sys
import os
import argparse
import json
import time
import subprocess
import urllib.request
import xml.etree.ElementTree as ET
from datetime import datetime

# --- Configuration ---
DEFAULT_SEEDS = [
    {"url": "https://deepmind.com/blog/rss.xml", "name": "DeepMind"},
    {"url": "https://openai.com/news/rss.xml", "name": "OpenAI"},
    {"url": "https://raw.githubusercontent.com/Olshansk/rss-feeds/main/feeds/feed_anthropic_news.xml", "name": "Anthropic"},
    {"url": "http://export.arxiv.org/rss/cs.AI", "name": "arXiv cs.AI"},
    {"url": "http://export.arxiv.org/rss/cs.LG", "name": "arXiv cs.LG"}
]

BUS_TOOL = os.path.join(os.path.dirname(__file__), "agent_bus.py")
RHIZOME_TOOL = os.path.join(os.path.dirname(__file__), "rhizome.py")

def log(topic, kind, data):
    """Emit an event to the bus."""
    level = "info"
    if kind == "error":
        kind = "log"
        level = "error"
    
    print(f"[LOG] {topic}: {json.dumps(data)}")
    subprocess.run([sys.executable, BUS_TOOL, "pub", "--topic", topic, "--kind", kind, "--level", level, "--data", json.dumps(data)], check=False)

def fetch_rss(url):
    """Fetch and parse RSS feed (naive implementation)."""
    try:
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        with urllib.request.urlopen(req, timeout=10) as response:
            xml_data = response.read()
            root = ET.fromstring(xml_data)
            # Handle RSS 2.0 and Atom
            items = []
            # RSS 2.0
            for item in root.findall(".//item"):
                title = item.find("title").text if item.find("title") is not None else "No Title"
                link = item.find("link").text if item.find("link") is not None else ""
                desc = item.find("description").text if item.find("description") is not None else ""
                items.append({"title": title, "link": link, "summary": desc})
            
            # Atom (namespaces make this tricky in naive ET, skipping for MVP/assuming RSS)
            # If empty, try Atom namespace-agnostic search (hacky)
            if not items:
                for entry in root.findall(".//{http://www.w3.org/2005/Atom}entry"):
                    title = entry.find("{http://www.w3.org/2005/Atom}title").text
                    link_elem = entry.find("{http://www.w3.org/2005/Atom}link")
                    link = link_elem.attrib.get("href") if link_elem is not None else ""
                    summary = entry.find("{http://www.w3.org/2005/Atom}summary").text if entry.find("{http://www.w3.org/2005/Atom}summary") is not None else ""
                    items.append({"title": title, "link": link, "summary": summary})
            
            return items
    except Exception as e:
        log("curation.fetch.error", "error", {"url": url, "error": str(e)})
        return []

def ingest_item(item, source_name):
    """Ingest a single item into the Rhizome."""
    import tempfile
    
    # Enrich item with metadata before saving
    item["_meta"] = {
        "source": source_name,
        "ingested_at": datetime.utcnow().isoformat() + "Z",
        "type": "curation_item"
    }
    
    content = json.dumps(item, indent=2)
    
    with tempfile.NamedTemporaryFile(mode='w', suffix=".json", delete=False) as tmp:
        tmp.write(content)
        tmp_path = tmp.name
    
    # Use rhizome CLI to ingest
    cmd = [
        sys.executable, RHIZOME_TOOL, "ingest",
        tmp_path,
        "--store",
        "--emit-bus",
        "--tag", "sota_curation"
    ]
    
    try:
        subprocess.run(cmd, capture_output=True, check=True)
        os.remove(tmp_path)
        return True
    except subprocess.CalledProcessError as e:
        log("curation.ingest.error", "error", {"item": item["title"], "error": e.stderr.decode('utf-8') if e.stderr else str(e)})
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
        return False

def main():
    parser = argparse.ArgumentParser(description="STRp Curation Loop")
    parser.add_argument("--dry-run", action="store_true", help="Do not ingest, just print")
    args = parser.parse_args()

    log("curation.cycle.start", "metric", {"sources": len(DEFAULT_SEEDS)})

    total_ingested = 0
    
    for seed in DEFAULT_SEEDS:
        items = fetch_rss(seed["url"])
        log("curation.feed.fetched", "log", {"source": seed["name"], "items": len(items)})
        
        # Limit to 5 items per feed for MVP/Safety
        for item in items[:5]:
            if args.dry_run:
                print(f"[DRY RUN] Would ingest: {item['title']} ({item['link']})")
            else:
                if ingest_item(item, seed["name"]):
                    total_ingested += 1

    log("curation.cycle.complete", "metric", {"ingested_count": total_ingested})

if __name__ == "__main__":
    main()
