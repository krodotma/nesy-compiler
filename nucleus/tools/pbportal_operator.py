#!/usr/bin/env python3
"""
pbportal_operator.py - Portal Inception Operator

PBPORTAL handles the inception of new portals from etymon seeds.
Integrates rhizome DAG, HEXIS buffers, CMP scoring, and iso_git crystallization.

Ring: 1 (Infrastructure)
Protocol: DKIN v29 | PAIP v15 | Citizen v1

Usage:
    python3 pbportal_operator.py incept <etymon> <content_path> [--tags TAG]...
    python3 pbportal_operator.py status <portal_id>
    python3 pbportal_operator.py list [--etymon ETYMON]

Bus Topics:
    portal.incepted
    portal.crystallized
    cmp.portal.scored
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Any, Optional, List


# Configuration
BUS_DIR = Path(os.environ.get("PLURIBUS_BUS_DIR", ".pluribus/bus"))
RHIZOME_DIR = Path(os.environ.get("PLURIBUS_RHIZOME_DIR", ".pluribus/rhizome"))
PORTAL_DIR = Path(os.environ.get("PLURIBUS_PORTAL_DIR", ".pluribus/portals"))
TOOLS_DIR = Path(__file__).parent


@dataclass
class InceptionResult:
    """Result of a portal inception."""
    portal_id: str
    etymon: str
    content_hash: str
    rhizome_node_id: Optional[str] = None
    embedding: Optional[List[float]] = None
    cmp_score: Optional[float] = None
    tags: List[str] = field(default_factory=list)
    created_iso: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    status: str = "incepted"  # incepted, crystallized, failed
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "InceptionResult":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class PortalOperator:
    """
    PBPORTAL - Portal Inception Operator
    
    Flow:
    1. Receive etymon seed + content
    2. Generate multi-modal embedding
    3. Insert into rhizome DAG
    4. Notify agents via HEXIS
    5. Trigger CMP scoring
    6. Crystallize to iso_git when ready
    """

    def __init__(self, bus_dir: Path = None, portal_dir: Path = None):
        self.bus_dir = bus_dir or BUS_DIR
        self.portal_dir = portal_dir or PORTAL_DIR
        self.bus_path = self.bus_dir / "events.ndjson"
        self.portal_ledger = self.portal_dir / "portals.ndjson"
        self.portal_dir.mkdir(parents=True, exist_ok=True)

    def _emit_bus_event(self, topic: str, data: Dict[str, Any], level: str = "info"):
        """Emit event to the Pluribus bus."""
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)
        
        event = {
            "id": uuid.uuid4().hex,
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat(),
            "topic": topic,
            "kind": "event",
            "level": level,
            "actor": "pbportal",
            "protocol_version": "v29",
            "data": data,
        }
        
        with open(self.bus_path, "a") as f:
            f.write(json.dumps(event) + "\n")
        
        return event["id"]

    def _call_rhizome(self, file_path: Path, tags: List[str]) -> Optional[str]:
        """Call rhizome.py to ingest content."""
        rhizome_py = TOOLS_DIR / "rhizome.py"
        if not rhizome_py.exists():
            return None
        
        cmd = [
            sys.executable,
            str(rhizome_py),
            "ingest",
            str(file_path),
            "--emit-bus",
        ]
        for tag in tags:
            cmd.extend(["--tag", tag])
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                # Parse node ID from output
                for line in result.stdout.split("\n"):
                    if line.startswith("Node:"):
                        return line.split(":")[1].strip()
        except Exception:
            pass
        return None

    def _call_hexis_notify(self, etymon: str, portal_id: str, agents: List[str] = None):
        """Notify agents via HEXIS buffer."""
        hexis_py = TOOLS_DIR / "hexis_buffer.py"
        if not hexis_py.exists():
            return
        
        agents = agents or ["claude", "codex", "omega"]
        message = f"[PORTAL INCEPTED] {etymon} → {portal_id[:12]}"
        
        for agent in agents:
            try:
                subprocess.run([
                    sys.executable,
                    str(hexis_py),
                    "publish",
                    agent,
                    message,
                    "--data", json.dumps({"etymon": etymon, "portal_id": portal_id}),
                ], capture_output=True, timeout=10)
            except Exception:
                pass

    def _generate_embedding(self, content: bytes) -> Optional[List[float]]:
        """Generate semantic embedding for content.
        
        TODO: Integrate with sentence-transformers or local embedding model.
        For now, returns a placeholder based on content hash.
        """
        # Placeholder: return hash-derived pseudo-embedding
        h = hashlib.sha256(content).hexdigest()
        # Convert hex to float values (not real embeddings, just for structure)
        embedding = [int(h[i:i+2], 16) / 255.0 for i in range(0, 64, 2)]
        return embedding

    def _calculate_cmp_score(self, etymon: str, embedding: List[float]) -> float:
        """Calculate CMP score for portal inception.
        
        TODO: Call cmp_engine_v2.py for real scoring.
        Placeholder returns hash-based score.
        """
        # Placeholder score based on etymon
        h = hashlib.md5(etymon.encode()).hexdigest()
        base_score = int(h[:4], 16) / 65535.0
        return 0.5 + (base_score * 0.5)  # Score between 0.5 and 1.0

    def incept(
        self,
        etymon: str,
        content_path: Path,
        tags: List[str] = None,
        notify_agents: bool = True,
    ) -> InceptionResult:
        """
        Incept a new portal from etymon seed.
        
        Args:
            etymon: The etymon domain (e.g., 'aiwa.re', 'kro.ma')
            content_path: Path to content file to incept
            tags: Optional semantic tags
            notify_agents: Whether to send HEXIS notifications
            
        Returns:
            InceptionResult with portal details
        """
        if not content_path.exists():
            raise FileNotFoundError(f"Content not found: {content_path}")
        
        content = content_path.read_bytes()
        content_hash = hashlib.sha256(content).hexdigest()
        portal_id = f"portal-{uuid.uuid4().hex[:12]}"
        
        # Default tags
        tags = tags or []
        if etymon not in tags:
            tags.append(etymon)
        tags.extend(["portal", "inception"])
        
        # Generate embedding
        embedding = self._generate_embedding(content)
        
        # Calculate CMP score
        cmp_score = self._calculate_cmp_score(etymon, embedding)
        
        # Insert into rhizome DAG
        rhizome_node_id = self._call_rhizome(content_path, tags)
        
        # Create inception result
        result = InceptionResult(
            portal_id=portal_id,
            etymon=etymon,
            content_hash=content_hash,
            rhizome_node_id=rhizome_node_id,
            embedding=embedding[:8] if embedding else None,  # Store truncated
            cmp_score=cmp_score,
            tags=tags,
            metadata={
                "content_path": str(content_path),
                "content_size": len(content),
                "inception_host": os.environ.get("HOSTNAME", "unknown"),
            },
        )
        
        # Save to ledger
        with open(self.portal_ledger, "a") as f:
            f.write(json.dumps(result.to_dict()) + "\n")
        
        # Emit bus event
        self._emit_bus_event("portal.incepted", {
            "portal_id": portal_id,
            "etymon": etymon,
            "content_hash": content_hash[:16],
            "rhizome_node_id": rhizome_node_id,
            "cmp_score": cmp_score,
            "tags": tags,
        })
        
        # Emit CMP scoring event
        self._emit_bus_event("cmp.portal.scored", {
            "portal_id": portal_id,
            "etymon": etymon,
            "score": cmp_score,
            "components": {
                "semantic_coherence": embedding[0] if embedding else 0.0,
                "etymon_alignment": 0.8,
                "inception_quality": 0.9,
            },
        })
        
        # Notify agents via HEXIS
        if notify_agents:
            self._call_hexis_notify(etymon, portal_id)
        
        return result

    def get_status(self, portal_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a portal by ID."""
        if not self.portal_ledger.exists():
            return None
        
        for line in self.portal_ledger.read_text().strip().split("\n"):
            if line:
                try:
                    data = json.loads(line)
                    if data.get("portal_id") == portal_id:
                        return data
                except json.JSONDecodeError:
                    continue
        return None

    def list_portals(self, etymon: str = None, limit: int = 20) -> List[Dict[str, Any]]:
        """List portals, optionally filtered by etymon."""
        if not self.portal_ledger.exists():
            return []
        
        portals = []
        for line in self.portal_ledger.read_text().strip().split("\n"):
            if line:
                try:
                    data = json.loads(line)
                    if etymon is None or data.get("etymon") == etymon:
                        portals.append(data)
                except json.JSONDecodeError:
                    continue
        
        return portals[-limit:]


def cmd_incept(args):
    """Incept a new portal."""
    operator = PortalOperator()
    
    try:
        result = operator.incept(
            etymon=args.etymon,
            content_path=Path(args.content),
            tags=args.tags or [],
            notify_agents=not args.no_notify,
        )
        
        print(f"Portal ID: {result.portal_id}")
        print(f"Etymon: {result.etymon}")
        print(f"Hash: {result.content_hash[:16]}...")
        print(f"Rhizome: {result.rhizome_node_id or '(not indexed)'}")
        print(f"CMP Score: {result.cmp_score:.3f}")
        print(f"Tags: {', '.join(result.tags)}")
        print(f"Status: {result.status}")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_status(args):
    """Get portal status."""
    operator = PortalOperator()
    status = operator.get_status(args.portal_id)
    
    if status:
        print(json.dumps(status, indent=2))
        return 0
    else:
        print(f"Portal not found: {args.portal_id}")
        return 1


def cmd_list(args):
    """List portals."""
    operator = PortalOperator()
    portals = operator.list_portals(etymon=args.etymon, limit=args.limit or 20)
    
    if not portals:
        print("No portals found")
        return 0
    
    for p in portals:
        cmp = p.get("cmp_score", 0)
        print(f"  {p['portal_id'][:16]} | {p['etymon']:20} | CMP: {cmp:.2f} | {p['status']}")
    
    print(f"\n{len(portals)} portal(s)")
    return 0


def main():
    parser = argparse.ArgumentParser(description="PBPORTAL - Portal Inception Operator")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # incept
    p_incept = subparsers.add_parser("incept", help="Incept a new portal")
    p_incept.add_argument("etymon", help="Etymon domain (e.g., aiwa.re)")
    p_incept.add_argument("content", help="Content file path")
    p_incept.add_argument("--tags", action="append", help="Add tag (repeatable)")
    p_incept.add_argument("--no-notify", action="store_true", help="Skip HEXIS notifications")
    
    # status
    p_status = subparsers.add_parser("status", help="Get portal status")
    p_status.add_argument("portal_id", help="Portal ID")
    
    # list
    p_list = subparsers.add_parser("list", help="List portals")
    p_list.add_argument("--etymon", help="Filter by etymon")
    p_list.add_argument("--limit", type=int, help="Max portals to show")
    
    args = parser.parse_args()
    
    if args.command == "incept":
        return cmd_incept(args)
    elif args.command == "status":
        return cmd_status(args)
    elif args.command == "list":
        return cmd_list(args)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
