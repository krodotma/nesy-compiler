#!/usr/bin/env python3
"""
pattern_memory.py - Pattern and Anti-Pattern Memory for ARK CL

P2-023: Create pattern memory (successful commits)
P2-024: Create anti-pattern memory (rejected commits)

Implements episodic memory for:
- Successful commit patterns (high CMP, passed gates)
- Anti-patterns (rejected commits, low CMP, failed gates)
- Pattern matching for new commits
"""

import json
import time
import sqlite3
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path
import logging

logger = logging.getLogger("ARK.CL.PatternMemory")


@dataclass
class Pattern:
    """A successful commit pattern to remember."""
    pattern_id: str
    etymology_keywords: List[str]
    file_extensions: List[str]
    entropy_profile: Dict[str, float]  # Typical entropy values
    cmp_range: Tuple[float, float]  # (min, max) CMP values
    gate_profile: Dict[str, float]  # gate -> pass rate
    frequency: int = 1  # How often this pattern appears
    last_seen: float = field(default_factory=time.time)
    reward_sum: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            **asdict(self),
            "cmp_range": list(self.cmp_range)
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "Pattern":
        data["cmp_range"] = tuple(data.get("cmp_range", [0.5, 0.5]))
        return cls(**data)
    
    def match_score(self, keywords: List[str], extensions: List[str]) -> float:
        """Compute match score against this pattern."""
        keyword_overlap = len(set(keywords) & set(self.etymology_keywords))
        ext_overlap = len(set(extensions) & set(self.file_extensions))
        
        keyword_score = keyword_overlap / max(len(self.etymology_keywords), 1)
        ext_score = ext_overlap / max(len(self.file_extensions), 1)
        
        # Weighted combination
        return 0.6 * keyword_score + 0.4 * ext_score


@dataclass
class AntiPattern:
    """A pattern to avoid (rejected/failed commits)."""
    pattern_id: str
    etymology_keywords: List[str]
    file_extensions: List[str]
    failure_mode: str  # Which gate failed or why rejected
    entropy_profile: Dict[str, float]
    typical_cmp: float
    frequency: int = 1
    last_seen: float = field(default_factory=time.time)
    severity: float = 1.0  # How bad is violating this pattern
    
    def to_dict(self) -> Dict:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict) -> "AntiPattern":
        return cls(**data)
    
    def match_score(self, keywords: List[str], extensions: List[str]) -> float:
        """Compute match score - high score means this anti-pattern applies."""
        keyword_overlap = len(set(keywords) & set(self.etymology_keywords))
        ext_overlap = len(set(extensions) & set(self.file_extensions))
        
        keyword_score = keyword_overlap / max(len(self.etymology_keywords), 1)
        ext_score = ext_overlap / max(len(self.file_extensions), 1)
        
        return 0.6 * keyword_score + 0.4 * ext_score


class PatternMemory:
    """
    Episodic memory for commit patterns.
    
    Stores:
    - Patterns: Successful commit patterns to emulate
    - AntiPatterns: Failed patterns to avoid
    """
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = Path(db_path) if db_path else Path("~/.ark/cl/pattern_memory.db").expanduser()
        self.patterns: Dict[str, Pattern] = {}
        self.anti_patterns: Dict[str, AntiPattern] = {}
        self._ensure_db()
        self._load()
    
    def _ensure_db(self) -> None:
        """Create SQLite database."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("""
            CREATE TABLE IF NOT EXISTS patterns (
                id TEXT PRIMARY KEY,
                is_anti INTEGER DEFAULT 0,
                data TEXT NOT NULL,
                frequency INTEGER DEFAULT 1,
                ts REAL NOT NULL
            )
        """)
        conn.commit()
        conn.close()
    
    def _load(self) -> None:
        """Load patterns from database."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            cur = conn.execute("SELECT id, is_anti, data FROM patterns")
            for row in cur:
                try:
                    pid, is_anti, data_str = row
                    data = json.loads(data_str)
                    if is_anti:
                        self.anti_patterns[pid] = AntiPattern.from_dict(data)
                    else:
                        self.patterns[pid] = Pattern.from_dict(data)
                except Exception:
                    pass
            conn.close()
            logger.debug("Loaded %d patterns, %d anti-patterns", 
                        len(self.patterns), len(self.anti_patterns))
        except Exception as e:
            logger.warning("Failed to load pattern memory: %s", e)
    
    def _save_pattern(self, pattern: Pattern) -> None:
        """Persist pattern to database."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.execute(
                "INSERT OR REPLACE INTO patterns (id, is_anti, data, frequency, ts) VALUES (?, 0, ?, ?, ?)",
                (pattern.pattern_id, json.dumps(pattern.to_dict()), 
                 pattern.frequency, pattern.last_seen)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning("Failed to save pattern: %s", e)
    
    def _save_anti_pattern(self, anti: AntiPattern) -> None:
        """Persist anti-pattern to database."""
        try:
            conn = sqlite3.connect(str(self.db_path))
            conn.execute(
                "INSERT OR REPLACE INTO patterns (id, is_anti, data, frequency, ts) VALUES (?, 1, ?, ?, ?)",
                (anti.pattern_id, json.dumps(anti.to_dict()),
                 anti.frequency, anti.last_seen)
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning("Failed to save anti-pattern: %s", e)
    
    def record_success(
        self,
        etymology: str,
        files: List[str],
        entropy: Dict[str, float],
        cmp: float,
        gate_results: Dict[str, bool],
        reward: float = 0.0
    ) -> Pattern:
        """
        Record a successful commit pattern.
        
        P2-023: Create pattern memory (successful commits)
        """
        # Extract keywords and extensions
        keywords = [w.lower() for w in etymology.split() if len(w) > 3]
        extensions = list(set(f.split(".")[-1] for f in files if "." in f))
        
        # Generate pattern ID from content hash
        content_str = f"{sorted(keywords)}{sorted(extensions)}"
        pattern_id = f"pat_{hash(content_str) % 10000:04d}"
        
        if pattern_id in self.patterns:
            # Update existing pattern
            pattern = self.patterns[pattern_id]
            pattern.frequency += 1
            pattern.last_seen = time.time()
            pattern.reward_sum += reward
            # Update CMP range
            pattern.cmp_range = (
                min(pattern.cmp_range[0], cmp),
                max(pattern.cmp_range[1], cmp)
            )
            # Update gate profile
            for gate, passed in gate_results.items():
                old_rate = pattern.gate_profile.get(gate, 0.5)
                pattern.gate_profile[gate] = old_rate * 0.9 + (1.0 if passed else 0.0) * 0.1
        else:
            # Create new pattern
            pattern = Pattern(
                pattern_id=pattern_id,
                etymology_keywords=keywords,
                file_extensions=extensions,
                entropy_profile=entropy,
                cmp_range=(cmp, cmp),
                gate_profile={g: 1.0 if p else 0.0 for g, p in gate_results.items()},
                reward_sum=reward
            )
            self.patterns[pattern_id] = pattern
        
        self._save_pattern(pattern)
        logger.debug("Recorded success pattern %s (freq=%d)", pattern_id, pattern.frequency)
        return pattern
    
    def record_failure(
        self,
        etymology: str,
        files: List[str],
        entropy: Dict[str, float],
        cmp: float,
        failed_gate: str,
        reason: str = ""
    ) -> AntiPattern:
        """
        Record a failed commit anti-pattern.
        
        P2-024: Create anti-pattern memory (rejected commits)
        """
        keywords = [w.lower() for w in etymology.split() if len(w) > 3]
        extensions = list(set(f.split(".")[-1] for f in files if "." in f))
        
        # Include failure mode in ID
        content_str = f"{sorted(keywords)}{sorted(extensions)}{failed_gate}"
        pattern_id = f"anti_{hash(content_str) % 10000:04d}"
        
        if pattern_id in self.anti_patterns:
            anti = self.anti_patterns[pattern_id]
            anti.frequency += 1
            anti.last_seen = time.time()
            # Increase severity with frequency
            anti.severity = min(2.0, anti.severity + 0.1)
        else:
            anti = AntiPattern(
                pattern_id=pattern_id,
                etymology_keywords=keywords,
                file_extensions=extensions,
                failure_mode=failed_gate,
                entropy_profile=entropy,
                typical_cmp=cmp,
                severity=1.0
            )
            self.anti_patterns[pattern_id] = anti
        
        self._save_anti_pattern(anti)
        logger.debug("Recorded anti-pattern %s (freq=%d)", pattern_id, anti.frequency)
        return anti
    
    def find_similar_patterns(
        self,
        etymology: str,
        files: List[str],
        top_k: int = 5
    ) -> List[Tuple[Pattern, float]]:
        """Find patterns similar to proposed commit."""
        keywords = [w.lower() for w in etymology.split() if len(w) > 3]
        extensions = list(set(f.split(".")[-1] for f in files if "." in f))
        
        scored = []
        for pattern in self.patterns.values():
            score = pattern.match_score(keywords, extensions)
            if score > 0.1:  # Minimum threshold
                scored.append((pattern, score))
        
        # Sort by score and frequency
        scored.sort(key=lambda x: x[1] * (1 + x[0].frequency * 0.1), reverse=True)
        return scored[:top_k]
    
    def find_matching_anti_patterns(
        self,
        etymology: str,
        files: List[str],
        threshold: float = 0.3
    ) -> List[Tuple[AntiPattern, float]]:
        """Find anti-patterns that match proposed commit (warnings)."""
        keywords = [w.lower() for w in etymology.split() if len(w) > 3]
        extensions = list(set(f.split(".")[-1] for f in files if "." in f))
        
        matches = []
        for anti in self.anti_patterns.values():
            score = anti.match_score(keywords, extensions)
            if score >= threshold:
                matches.append((anti, score * anti.severity))
        
        matches.sort(key=lambda x: x[1], reverse=True)
        return matches
    
    def predict_cmp(self, etymology: str, files: List[str]) -> Tuple[float, float]:
        """
        Predict CMP range based on similar patterns.
        
        Returns (predicted_cmp, confidence)
        """
        similar = self.find_similar_patterns(etymology, files, top_k=3)
        
        if not similar:
            return 0.5, 0.0  # No prediction, zero confidence
        
        # Weight by match score and frequency
        total_weight = 0.0
        cmp_sum = 0.0
        
        for pattern, score in similar:
            weight = score * (1 + pattern.frequency * 0.05)
            avg_cmp = (pattern.cmp_range[0] + pattern.cmp_range[1]) / 2
            cmp_sum += avg_cmp * weight
            total_weight += weight
        
        predicted = cmp_sum / total_weight if total_weight > 0 else 0.5
        confidence = min(1.0, total_weight / 3.0)  # Normalize to [0, 1]
        
        return predicted, confidence
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get memory statistics."""
        return {
            "pattern_count": len(self.patterns),
            "anti_pattern_count": len(self.anti_patterns),
            "total_pattern_frequency": sum(p.frequency for p in self.patterns.values()),
            "total_anti_frequency": sum(a.frequency for a in self.anti_patterns.values()),
            "top_patterns": [
                {"id": p.pattern_id, "freq": p.frequency}
                for p in sorted(self.patterns.values(), 
                               key=lambda x: x.frequency, reverse=True)[:5]
            ],
            "top_anti_patterns": [
                {"id": a.pattern_id, "failure": a.failure_mode, "freq": a.frequency}
                for a in sorted(self.anti_patterns.values(),
                               key=lambda x: x.frequency, reverse=True)[:5]
            ]
        }
