#!/usr/bin/env python3
"""
motif_extractor.py - ω-Motif Extraction & Recurrence Detection

DUALITY-BIND E3: Extract recurring subgraphs from successful traces.
Recurrence = Büchi acceptance for "what good behavior looks like."

Ring: 1 (Operator)
Protocol: DKIN v29
"""

import argparse
import json
import hashlib
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Set, Any, Optional
import os

BUS_DIR = Path(os.environ.get("PLURIBUS_BUS_DIR", ".pluribus/bus"))
MOTIF_DIR = Path(os.environ.get("PLURIBUS_MOTIF_DIR", ".pluribus/motifs"))


@dataclass
class OmegaMotif:
    """A recurring pattern that indicates good behavior."""
    motif_id: str
    pattern_hash: str
    sequence: List[str]  # Topic sequence
    occurrence_count: int = 0
    success_rate: float = 0.0
    avg_reward: float = 0.0
    first_seen: float = 0.0
    last_seen: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MotifExtractor:
    """Extract ω-motifs from bus event traces."""
    
    def __init__(self, bus_dir: Path = None, motif_dir: Path = None):
        self.bus_dir = bus_dir or BUS_DIR
        self.motif_dir = motif_dir or MOTIF_DIR
        self.bus_path = self.bus_dir / "events.ndjson"
        self.motif_path = self.motif_dir / "omega_motifs.ndjson"
        self.motif_dir.mkdir(parents=True, exist_ok=True)
        self.motifs: Dict[str, OmegaMotif] = {}
    
    def _hash_sequence(self, seq: List[str]) -> str:
        return hashlib.md5("|".join(seq).encode()).hexdigest()[:12]
    
    def extract_ngrams(self, topics: List[str], n: int = 3) -> List[List[str]]:
        """Extract n-grams from topic sequence."""
        return [topics[i:i+n] for i in range(len(topics) - n + 1)]
    
    def extract_from_bus(self, window_size: int = 3, min_occurrences: int = 2) -> int:
        """Extract motifs from bus events."""
        if not self.bus_path.exists():
            return 0
        
        # Load events
        events = []
        for line in self.bus_path.read_text().strip().split("\n"):
            if line:
                try:
                    events.append(json.loads(line))
                except:
                    continue
        
        # Group by lineage
        by_lineage = defaultdict(list)
        for e in events:
            lid = e.get("lineage_id", "default")
            by_lineage[lid].append(e)
        
        # Extract n-grams per lineage
        pattern_counts = Counter()
        pattern_successes = defaultdict(list)
        pattern_rewards = defaultdict(list)
        pattern_times = defaultdict(list)
        
        for lid, trace in by_lineage.items():
            topics = [e.get("topic", "") for e in trace]
            ngrams = self.extract_ngrams(topics, window_size)
            
            for i, ngram in enumerate(ngrams):
                h = self._hash_sequence(ngram)
                pattern_counts[h] += 1
                
                # Check success of this pattern
                if i + window_size < len(trace):
                    next_event = trace[i + window_size]
                    data = next_event.get("data", {})
                    success = data.get("success", data.get("status") == "success")
                    reward = data.get("reward", 0.0)
                    pattern_successes[h].append(1 if success else 0)
                    pattern_rewards[h].append(reward)
                    pattern_times[h].append(next_event.get("ts", 0))
        
        # Create motifs for recurring patterns
        new_count = 0
        for h, count in pattern_counts.items():
            if count >= min_occurrences and h not in self.motifs:
                successes = pattern_successes[h]
                rewards = pattern_rewards[h]
                times = pattern_times[h]
                
                self.motifs[h] = OmegaMotif(
                    motif_id=f"omega-{h}",
                    pattern_hash=h,
                    sequence=[],  # Would need to store actual sequence
                    occurrence_count=count,
                    success_rate=sum(successes) / len(successes) if successes else 0.0,
                    avg_reward=sum(rewards) / len(rewards) if rewards else 0.0,
                    first_seen=min(times) if times else 0.0,
                    last_seen=max(times) if times else 0.0,
                )
                new_count += 1
        
        self._save()
        return new_count
    
    def _save(self):
        with open(self.motif_path, "w") as f:
            for m in self.motifs.values():
                f.write(json.dumps(m.to_dict()) + "\n")
    
    def get_stats(self) -> Dict[str, Any]:
        self._load()
        if not self.motifs:
            return {"total": 0, "avg_success_rate": 0.0, "top_motifs": []}
        
        return {
            "total": len(self.motifs),
            "avg_success_rate": sum(m.success_rate for m in self.motifs.values()) / len(self.motifs),
            "top_motifs": sorted(
                [(m.motif_id, m.occurrence_count, m.success_rate) for m in self.motifs.values()],
                key=lambda x: x[1], reverse=True
            )[:5],
        }
    
    def _load(self):
        if self.motif_path.exists():
            for line in self.motif_path.read_text().strip().split("\n"):
                if line:
                    try:
                        data = json.loads(line)
                        m = OmegaMotif(**data)
                        self.motifs[m.pattern_hash] = m
                    except:
                        continue


def main():
    parser = argparse.ArgumentParser(description="ω-Motif Extractor")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    p_extract = subparsers.add_parser("extract", help="Extract motifs from bus")
    p_extract.add_argument("--window", type=int, default=3, help="N-gram window size")
    p_extract.add_argument("--min", type=int, default=2, help="Min occurrences")
    
    subparsers.add_parser("stats", help="Show motif statistics")
    
    args = parser.parse_args()
    
    extractor = MotifExtractor()
    
    if args.command == "extract":
        count = extractor.extract_from_bus(args.window, args.min)
        print(f"Extracted {count} new ω-motifs")
        return 0
    elif args.command == "stats":
        stats = extractor.get_stats()
        print(f"ω-Motif Statistics")
        print(f"  Total: {stats['total']}")
        print(f"  Avg Success Rate: {stats['avg_success_rate']:.2%}")
        print(f"  Top Motifs: {stats.get('top_motifs', [])}")
        return 0


if __name__ == "__main__":
    sys.exit(main())
