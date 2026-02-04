#!/usr/bin/env python3
"""
SOTA Manager
============
Manages the lifecycle of SOTA tools: Discovery -> Tracking -> Assimilation -> Integration.
Maintains `.pluribus/index/sota.ndjson` as the Source of Truth.
"""
import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

sys.dont_write_bytecode = True

# Core Integrations (The Neon List)
INTEGRATED_TOOLS = {
    "sqlite-vec": {"status": "integrated", "tags": ["memory", "vector", "backbone"], "desc": "Local vector search extension for SQLite"},
    "vllm": {"status": "integrated", "tags": ["backbone", "llm", "serving"], "desc": "High-throughput LLM serving engine"},
    "unstructured": {"status": "integrated", "tags": ["memory", "etl"], "desc": "Universal document parsing (PDF/HTML/Img)"},
    "webllm": {"status": "integrated", "tags": ["backbone", "browser", "edge"], "desc": "In-browser LLM inference (WASM/WebGPU)"},
    "microsandbox": {"status": "partial", "tags": ["agent", "safety"], "desc": "Firecracker-based code isolation (simulated via code_executor)"},
    "wasmedge": {"status": "integrated", "tags": ["backbone", "wasm", "runtime"], "desc": "Lightweight WASM runtime for agent skills"},
    "tensorzero": {"status": "planned", "tags": ["mlops", "gateway"], "desc": "Unified LLM gateway and evaluation"},
    "mastra": {"status": "integrated", "tags": ["agent", "orchestration"], "desc": "TypeScript agent workflows"},
    "maestro": {"status": "integrated", "tags": ["testing", "mobile", "e2e"], "desc": "Mobile UI testing with Maestro CLI"},
    "agent-s": {"status": "integrated", "tags": ["agent", "gui", "cua"], "desc": "GUI automation (OSWorld-grade Agent-S)"},
    "agent0": {"status": "integrated", "tags": ["agent", "evolution"], "desc": "Self-evolving curriculum/executor loop (Agent0)"},
    "mem0": {"status": "integrated", "tags": ["memory", "agent"], "desc": "Long-term memory store with adapters"},
    "graphiti": {"status": "integrated", "tags": ["memory", "kg"], "desc": "Temporal knowledge graph"},
    "lit": {"status": "integrated", "tags": ["ui", "web-components"], "desc": "Lightweight web components"},,
}

def get_root():
    # Find .pluribus root
    cwd = Path.cwd().resolve()
    for p in [cwd, *cwd.parents]:
        if (p / ".pluribus").exists():
            return p
    return cwd

def load_index(root):
    path = root / ".pluribus" / "index" / "sota.ndjson"
    items = {}
    if path.exists():
        with open(path, 'r') as f:
            for line in f:
                if line.strip():
                    try:
                        obj = json.loads(line)
                        items[obj['id']] = obj
                    except: pass
    return items

def save_index(root, items):
    path = root / ".pluribus" / "index" / "sota.ndjson"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        for k in sorted(items.keys()):
            f.write(json.dumps(items[k]) + "\n")

def cmd_rebuild(args):
    root = get_root()
    items = load_index(root)
    
    # 1. Update/Inject Integrated Tools
    for key, info in INTEGRATED_TOOLS.items():
        if key not in items:
            items[key] = {
                "id": key,
                "title": key, # Normalized name
                "url": f"https://github.com/search?q={key}", # Placeholder if unknown
                "org": "sota",
                "region": "global",
                "type": "tool",
                "priority": 1,
                "cadence_days": 30,
                "tags": info['tags'],
                "notes": info['desc'],
                "distill_status": info['status'] == 'integrated' and 'completed' or 'queued'
            }
        
        # Force status update
        items[key]['integration_status'] = info['status']
        items[key]['tags'] = list(set(items[key].get('tags', []) + info['tags']))
        items[key]['notes'] = info['desc']

    # 2. Ingest from Markdown Catalogs (Simple Parse)
    # TODO: Parse sota_tools_catalog.md and sota_long_tail_distilled.md
    # For now, we rely on the INTEGRATED list as the "Neon" set.
    
    save_index(root, items)
    print(f"Rebuilt SOTA index with {len(items)} items. {len(INTEGRATED_TOOLS)} highlighted.")

def cmd_list(args):
    root = get_root()
    items = load_index(root)
    print(f"{'ID':<20} | {'STATUS':<12} | {'DESC'}")
    print("-" * 60)
    for k, v in items.items():
        status = v.get('integration_status', 'candidate')
        if args.filter and args.filter not in status: continue
        print(f"{k:<20} | {status:<12} | {v.get('notes', '')[:40]}")

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd")
    
    rebuild = sub.add_parser("rebuild")
    ls = sub.add_parser("list")
    ls.add_argument("--filter", help="Filter by status (integrated, planned, candidate)")
    
    args = p.parse_args()
    if args.cmd == "rebuild":
        cmd_rebuild(args)
    elif args.cmd == "list":
        cmd_list(args)
    else:
        p.print_help()

if __name__ == "__main__":
    main()
