#!/usr/bin/env python3
import json
import time
import uuid
import sys
import os
from pathlib import Path

# Ensure we can import from nucleus/tools
current_dir = Path(__file__).parent
sys.path.append(str(current_dir.parent.parent))

from nucleus.tools.event_semantics import create_semantic_event, EntelexisState, SemioticalysisLayer

# Configuration
DOMAINS_PATH = Path("nucleus/config/domains.ndjson")
KG_PATH = Path(".pluribus/index/kg_nodes.ndjson")
BUS_PATH = Path(".pluribus/bus/events.ndjson")

# Data from the ingest pattern
VIDEOS = [
    {
        "id": "9Gv7eZemHrE",
        "title": "Anthropic's Ralph Loop + Claude Code",
        "claims": ["Anthropic framework runs Claude Code 24/7", "Ralph Loop enables continuous agent operation"],
        "workflow": "Claude Code Continuous Integration",
        "tools": ["Claude Code", "Ralph Loop"],
        "date": "2025-12-28"
    },
    {
        "id": "uuV1DcvObsg",
        "title": "Oh My OpenCode (5 SUPER Agent/MCP/Prompt Config)",
        "claims": ["OpenCode configuration enhances agent capabilities", "MCP servers integrate with OpenCode"],
        "workflow": "OpenCode Agent Configuration",
        "tools": ["OpenCode", "MCP"],
        "date": "2025-12-27"
    },
    {
        "id": "xGIxusdJr0w",
        "title": "GLM-4.7 + KingMode + Frontend Skill",
        "claims": ["KingMode prompt enhances GLM-4.7 performance", "Frontend skill allows direct UI generation"],
        "workflow": "GLM-4.7 Frontend Generation",
        "tools": ["GLM-4.7", "KingMode", "Frontend Skill"],
        "date": "2025-12-26"
    },
    {
        "id": "_rKybKSmiLs",
        "title": "Gemini 3.0 Flash + Opus 4.5 + Antigravity",
        "claims": ["Antigravity workflow combines Gemini and Opus", "Free tier usage is maximized"],
        "workflow": "Antigravity Multi-Model Workflow",
        "tools": ["Gemini 3.0 Flash", "Opus 4.5", "Antigravity"],
        "date": "2025-12-25"
    },
    {
        "id": "wxhSrzjZOm4",
        "title": "Gemini 3.0 Flash Designer",
        "claims": ["Gemini 3.0 Flash excels at frontend design", "Cost effective for UI prototyping"],
        "workflow": "Gemini Flash UI Design",
        "tools": ["Gemini 3.0 Flash"],
        "date": "2025-12-24"
    }
]

def ensure_dir(p: Path) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)

def append_ndjson(path: Path, obj: dict) -> None:
    ensure_dir(path)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def run():
    print(f"Starting Sagent Iteration 1...")
    
    # 1. Load Domains
    if not DOMAINS_PATH.exists():
        print(f"Error: {DOMAINS_PATH} not found.")
        return
    
    print(f"Loaded domains from {DOMAINS_PATH}")
    
    # 2. Iterate Videos
    for video in VIDEOS:
        print(f"Processing video: {video['title']}")
        
        # A. Create Semantic Event
        semioticalysis = SemioticalysisLayer(
            syntactic="Video Ingest",
            semantic=f"Distillation of '{video['title']}'",
            pragmatic="Extracting SOTA patterns for replication",
            metalinguistic="AICodeKing Pattern"
        )
        
        entelexis = EntelexisState(
            phase="actualizing",
            form_signature="Knowledge Graph Node",
            material_context="Video Transcript/Distillation",
            actualization_progress=1.0
        )
        
        event = create_semantic_event(
            topic="sota.ingest.distill",
            kind="artifact",
            level="info",
            actor="sagent",
            data={
                "video_id": video["id"],
                "title": video["title"],
                "action": "distill_complete"
            },
            semantic=f"Completed distillation of {video['title']} into KG nodes",
            reasoning="Pattern ingestion required for evolutionary updates",
            actionable=["Review KG nodes", "Test new workflows"],
            impact="medium",
            semioticalysis=semioticalysis,
            entelexis=entelexis
        )
        
        # B. Emit Event
        append_ndjson(BUS_PATH, event.to_dict())
        print(f"  Emitted event {event.id}")
        
        # C. Create KG Nodes
        # 1. SOTA Item Node
        sota_node = {
            "id": f"sota:video:{video['id']}",
            "type": "sota_item",
            "label": video["title"],
            "props": {
                "url": f"https://youtu.be/{video['id']}",
                "date": video["date"],
                "channel": "AICodeKing"
            },
            "tags": ["sota", "video", "aicodeking"]
        }
        append_ndjson(KG_PATH, sota_node)
        
        # 2. Claims Nodes
        for i, claim in enumerate(video["claims"]):
            claim_node = {
                "id": f"claim:{video['id']}:{i}",
                "type": "claim",
                "label": claim,
                "props": {
                    "source": f"sota:video:{video['id']}"
                },
                "tags": ["claim", "insight"]
            }
            append_ndjson(KG_PATH, claim_node)
            
            # Edge: Video -> Claim
            edge = {
                "from": sota_node["id"],
                "to": claim_node["id"],
                "rel": "supports_claim",
                "props": {}
            }
            append_ndjson(KG_PATH, edge)
            
        # 3. Workflow Node
        wf_node = {
            "id": f"workflow:{video['id']}",
            "type": "workflow",
            "label": video["workflow"],
            "props": {
                "source": f"sota:video:{video['id']}"
            },
            "tags": ["workflow", "process"]
        }
        append_ndjson(KG_PATH, wf_node)
        
        # Edge: Video -> Workflow
        edge = {
            "from": sota_node["id"],
            "to": wf_node["id"],
            "rel": "demonstrates_workflow",
            "props": {}
        }
        append_ndjson(KG_PATH, edge)
        
        # 4. Tool Nodes
        for tool in video["tools"]:
            tool_id = tool.lower().replace(" ", "_").replace(".", "")
            tool_node = {
                "id": f"tool:{tool_id}",
                "type": "tool",
                "label": tool,
                "props": {},
                "tags": ["tool", "sota_tool"]
            }
            append_ndjson(KG_PATH, tool_node)
            
            # Edge: Workflow -> Tool
            edge = {
                "from": wf_node["id"],
                "to": tool_node["id"],
                "rel": "requires_tool",
                "props": {}
            }
            append_ndjson(KG_PATH, edge)
            
        print(f"  Created KG nodes for {video['title']}")

if __name__ == "__main__":
    run()
