#!/usr/bin/env python3
"""
MetaLearner - Pluribus Learning System Backend

Central engine for specific learning and pattern recognition.
Consumes:
- Optimization telemetry
- Bus events (meta.ingest)
- User feedback

Produces:
- Learned Patterns (Graphiti)
- Design Suggestions
- Vector Embeddings (Phase 4)

Usage:
    python meta_learner.py run              # Daemon mode
    python meta_learner.py record <source> <ctx> <json_signal>
    python meta_learner.py pattern <pattern> <confidence>
    python meta_learner.py suggest <context>
    python meta_learner.py nearest <query>  # Semantic search
"""

import os
import sys
import sqlite3
import json
import time
import struct
import argparse
import hashlib
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass


# Phase 4: Universal Encoder
try:
    from universal_encoder import UniversalEncoder, EncoderConfig
    _ENCODER_AVAILABLE = True
except ImportError:
    _ENCODER_AVAILABLE = False

# Phase 5: Circuit Breaker Integration (Self-Repair Loop)
try:
    from circuit_breaker import CircuitBreaker, CircuitState
    _CIRCUIT_BREAKER_AVAILABLE = True
except ImportError:
    _CIRCUIT_BREAKER_AVAILABLE = False


# ============================================================================
# Graphiti Bridge
# ============================================================================

class GraphitiService:
    """Interface to Graphiti Knowledge Graph."""
    
    def __init__(self, root_dir: str = None):
        self.root_dir = root_dir or os.environ.get("PLURIBUS_ROOT", os.getcwd())
        self.graphiti_dir = os.path.join(self.root_dir, ".pluribus", "graphiti")
        os.makedirs(self.graphiti_dir, exist_ok=True)
    
    def add_entity(self, entity: Dict):
        """Add entity to Graphiti."""
        path = os.path.join(self.graphiti_dir, "entities.ndjson")
        with open(path, "a") as f:
            f.write(json.dumps(entity) + "\n")
            
    def add_fact(self, fact: Dict):
        """Add fact to Graphiti."""
        path = os.path.join(self.graphiti_dir, "facts.ndjson")
        with open(path, "a") as f:
            f.write(json.dumps(fact) + "\n")
            
    def query_facts(self, predicate: str = None) -> List[Dict]:
        """Query facts from Graphiti."""
        facts = []
        path = os.path.join(self.graphiti_dir, "facts.ndjson")
        if not os.path.exists(path):
            return []
        
        with open(path, "r") as f:
            for line in f:
                try:
                    fact = json.loads(line)
                    if predicate is None or fact.get("predicate") == predicate:
                        facts.append(fact)
                except:
                    pass
        return facts


# ============================================================================
# Semantic FCOS & Relational Expander
# ============================================================================

class SemanticFCOS:
    """
    Feature Centrality & Object Salience (FCOS) Detector.
    
    Identifies if a signal is 'central' or 'salient' based on simple
    statistical anomaly detection on the signal data size or complexity.
    """
    def __init__(self):
        self.baseline_complexity = 0.0
        self.alpha = 0.1  # Moving average factor

    def analyze(self, signal_data: Dict) -> Dict:
        """
        Analyze signal for salience.
        Returns metadata with score.
        """
        # Simple heuristic: complexity ~ length of stringified keys/values
        raw_str = json.dumps(signal_data)
        complexity = len(raw_str)
        
        # Update baseline
        if self.baseline_complexity == 0:
            self.baseline_complexity = complexity
        else:
            self.baseline_complexity = (1 - self.alpha) * self.baseline_complexity + self.alpha * complexity
            
        # Deviation
        ratio = complexity / (self.baseline_complexity + 1e-9)
        
        is_salient = ratio > 1.5 or ratio < 0.5
        
        return {
            "salience_score": ratio,
            "is_anamolous": is_salient,
            "complexity": complexity
        }

class RelationalExpander:
    """
    Expands an experience context by querying Graphiti for related nodes.
    MAXIMIZES context via lateral thinking (graph traversal).
    """
    def __init__(self, graphiti_service: GraphitiService):
        self.graphiti = graphiti_service

    def expand(self, context_str: str) -> List[str]:
        """
        Find related concepts in graphiti based on keywords in context.
        """
        # Naive keyword extraction check
        related = []
        # In a real impl, we'd extract entities from context_str and query them
        # For now, let's just query broad facts if context implies "pattern"
        if "pattern" in context_str.lower():
            facts = self.graphiti.query_facts(predicate="RECOMMENDS")
            for f in facts:
                if f.get("source") == "meta_learner":
                    related.append(f"Related Pattern: {f.get('object_id')} (conf={f.get('confidence')})")
        
        return related

# ============================================================================
# MetaLearner Core
# ============================================================================

class MetaLearner:
    """
    Core learning engine with Circuit Breaker self-repair integration.
    
    Persists experiences to SQLite.
    Syncs high-confidence patterns to Graphiti.
    Pauses learning during upstream outages (Phase 5).
    """
    
    def __init__(self, db_path: str = None, enable_vectors: bool = True, enforce_encoder: bool = True):
        self.root_dir = os.environ.get("PLURIBUS_ROOT", os.getcwd())
        default_db = os.path.join(self.root_dir, ".pluribus", "metalearner", "experience.db")
        self.db_path = db_path or default_db
        self.enforce_encoder = enforce_encoder
        self._learning_paused = False
        self._pause_reason = None
        
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        # Initialize Encoder (Phase 4)
        self.encoder = None
        self.vector_dim = 0
        if enable_vectors and _ENCODER_AVAILABLE:
            try:
                self.encoder = UniversalEncoder()
                self.vector_dim = self.encoder.config.vector_dim
            except Exception as e:
                print(f"Warning: Failed to init encoder: {e}")
        
        # Initialize Circuit Breaker (Phase 5)
        self.circuit_breaker = None
        if _CIRCUIT_BREAKER_AVAILABLE:
            try:
                bus_path = os.path.join(self.root_dir, ".pluribus", "bus")
                self.circuit_breaker = CircuitBreaker(bus_path=bus_path)
                print(f"[MetaLearner] Circuit Breaker linked (state: {self.circuit_breaker.state.value})")
            except Exception as e:
                print(f"Warning: Failed to init circuit breaker: {e}")
        
        self._init_db()
        self.graphiti = GraphitiService(self.root_dir)
        
        # Initialize specialized sub-minds
        self.fcos = SemanticFCOS()
        self.expander = RelationalExpander(self.graphiti)

    def _is_circuit_open(self) -> bool:
        """Check if circuit breaker is OPEN (blocking learning)."""
        if self.circuit_breaker is None:
            return False
        return self.circuit_breaker.state == CircuitState.OPEN
    
    def _emit_learning_event(self, topic: str, data: Dict):
        """Emit event to bus for observability."""
        bus_path = os.path.join(self.root_dir, ".pluribus", "bus", "events.ndjson")
        try:
            event = {
                "ts": time.time(),
                "topic": topic,
                "data": data,
                "actor": "meta_learner"
            }
            os.makedirs(os.path.dirname(bus_path), exist_ok=True)
            with open(bus_path, "a") as f:
                f.write(json.dumps(event) + "\n")
        except Exception:
            pass  # Non-critical

    def pause_learning(self, reason: str = "circuit_breaker_open"):
        """Pause learning operations (called by self-repair loop)."""
        if not self._learning_paused:
            self._learning_paused = True
            self._pause_reason = reason
            self._emit_learning_event("learning.paused", {"reason": reason})
            print(f"[MetaLearner] Learning PAUSED: {reason}")
    
    def resume_learning(self, reason: str = "circuit_breaker_closed"):
        """Resume learning operations."""
        if self._learning_paused:
            self._learning_paused = False
            self._emit_learning_event("learning.resumed", {"reason": reason})
            print(f"[MetaLearner] Learning RESUMED: {reason}")
            self._pause_reason = None

    def _semantic_hash(self, text: str) -> List[float]:
        """Fallback: Create pseudo-vector from MD5 hash."""
        h = hashlib.md5(text.encode("utf-8")).digest()
        # Unpack 16 bytes into 4 floats
        return list(struct.unpack("4f", h))

    def _init_db(self):
        """Initialize SQLite schema."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Experience Buffer
        # Added vector_embedding BLOB in Phase 4
        c.execute('''
            CREATE TABLE IF NOT EXISTS experience_buffer (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                source TEXT,
                context TEXT,
                signal_data TEXT,
                outcome TEXT,
                vector_embedding BLOB
            )
        ''')
        
        # Learned Patterns
        c.execute('''
            CREATE TABLE IF NOT EXISTS learned_patterns (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pattern_key TEXT UNIQUE,
                pattern_desc TEXT,
                confidence REAL,
                occurrences INTEGER,
                last_seen REAL
            )
        ''')
        
        # Check for vector_embedding column migration
        c.execute("PRAGMA table_info(experience_buffer)")
        columns = [info[1] for info in c.fetchall()]
        if "vector_embedding" not in columns:
            print("Migrating schema: Adding vector_embedding column...")
            c.execute("ALTER TABLE experience_buffer ADD COLUMN vector_embedding BLOB")
            
        conn.commit()
        conn.close()

    def record_experience(self, source: str, context: str, signal_data: Dict, outcome: str = None) -> bool:
        """
        Record a raw experience signal.
        Auto-encodes context to vector if encoder available.
        
        Returns:
            bool: True if recorded, False if learning is paused
        """
        # Phase 5: Circuit Breaker Self-Repair Check
        if self._is_circuit_open():
            if not self._learning_paused:
                self.pause_learning("circuit_breaker_open")
            return False
        
        # Resume if previously paused but circuit now closed
        if self._learning_paused and not self._is_circuit_open():
            self.resume_learning("circuit_breaker_recovered")
            
        # Phase 5: FCOS Salience Check
        fcos_meta = self.fcos.analyze(signal_data)
        signal_data = {**signal_data, "_fcos": fcos_meta}
        
        # Phase 5: Relational Expansion (augment context)
        # Only expand if significant
        if fcos_meta.get("is_anamolous", False):
            expanded_concepts = self.expander.expand(context)
            if expanded_concepts:
                context += f" [EXPANDED: {'; '.join(expanded_concepts)}]"
        
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        vector_blob = None
        if self.encoder:
            # Encode context + signal summary
            text_to_encode = f"{context} {json.dumps(signal_data)}"
            vector = self.encoder.encode(text_to_encode)
            # Pack float list to bytes
            if hasattr(vector, 'tolist'): vector = vector.tolist()
            vector_blob = struct.pack(f'{len(vector)}f', *vector)
        elif self.enforce_encoder:
            # Fallback to semantic hash
            text_to_encode = f"{context} {json.dumps(signal_data)}"
            vector = self._semantic_hash(text_to_encode)
            vector_blob = struct.pack(f'{len(vector)}f', *vector)

            
        c.execute('''
            INSERT INTO experience_buffer 
            (timestamp, source, context, signal_data, outcome, vector_embedding)
            VALUES (?, ?, ?, ?, ?, ?)
        ''', (time.time(), source, context, json.dumps(signal_data), outcome, vector_blob))
        
        conn.commit()
        conn.close()
        
        # Emit learning event for observability
        self._emit_learning_event("learning.experience_recorded", {
            "source": source,
            "context_preview": context[:100] if context else "",
            "salience": fcos_meta.get("salience_score", 0)
        })
        
        return True
        
    def update_pattern(self, pattern_key: str, description: str, confidence_delta: float = 0.1):
        """Update confidence of a learned pattern."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute('SELECT confidence, occurrences FROM learned_patterns WHERE pattern_key = ?', (pattern_key,))
        row = c.fetchone()
        
        if row:
            new_conf = min(1.0, row[0] + confidence_delta)
            new_occ = row[1] + 1
            c.execute('''
                UPDATE learned_patterns 
                SET confidence = ?, occurrences = ?, last_seen = ?
                WHERE pattern_key = ?
            ''', (new_conf, new_occ, time.time(), pattern_key))
            conf = new_conf
        else:
            conf = 0.5  # Initial confidence
            c.execute('''
                INSERT INTO learned_patterns 
                (pattern_key, pattern_desc, confidence, occurrences, last_seen)
                VALUES (?, ?, ?, 1, ?)
            ''', (pattern_key, description, conf, time.time()))
            
        conn.commit()
        conn.close()
        
        # Sync to Graphiti if high confidence
        if conf > 0.8:
            self.sync_to_graphiti(pattern_key, description, conf)

    def sync_to_graphiti(self, pattern_key: str, description: str, confidence: float):
        """Promote learned pattern to Knowledge Graph."""
        import uuid
        entity_id = str(uuid.uuid4())
        
        # Create Pattern Entity
        self.graphiti.add_entity({
            "id": entity_id,
            "name": pattern_key,
            "type": "learned_pattern",
            "description": description,
            "confidence": confidence
        })
        
        # Create Fact (Process -> RECOMMENDS -> Pattern)
        self.graphiti.add_fact({
            "subject_id": "system",  # simplified
            "predicate": "RECOMMENDS",
            "object_id": entity_id,
            "confidence": confidence,
            "source": "meta_learner"
        })

    def suggest(self, context: str) -> List[Dict]:
        """Get suggestions based on context."""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Semantic search (Phase 4)? Or just keyword match for now
        c.execute('SELECT pattern_key, pattern_desc, confidence FROM learned_patterns WHERE confidence > 0.6 ORDER BY confidence DESC LIMIT 5')
        results = [{"pattern": r[0], "desc": r[1], "conf": r[2]} for r in c.fetchall()]
        
        conn.close()
        return results

    def nearest(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Semantic search for experiences (Phase 4).
        Requires UniversalEncoder.
        """
        if not self.encoder:
            return [{"error": "Encoder not available"}]
            
        query_vec = self.encoder.encode(query)
        if hasattr(query_vec, 'tolist'): query_vec = query_vec.tolist()
        
        results = []
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # Naive linear scan (OK for small DB, use vector DB for prod)
        c.execute('SELECT id, context, signal_data, vector_embedding FROM experience_buffer WHERE vector_embedding IS NOT NULL')
        
        for row in c.fetchall():
            rid, ctx, signal, blob = row
            if not blob: continue
            
            # Unpack vector
            count = len(blob) // 4
            vec = struct.unpack(f'{count}f', blob)
            
            # Cosine similarity
            score = self.encoder.similarity(query_vec, vec)
            results.append({
                "id": rid,
                "context": ctx,
                "signal": signal,
                "score": score
            })
            
        conn.close()
        
        # Sort and return top_k
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def run(self):
        """Daemon loop (placeholder)."""
        print(f"MetaLearner Daemon active. DB: {self.db_path}")
        print("Tailing bus (mock)... Use 'record' command to simulate events.")
        # In real impl, would tail .pluribus/bus/events.ndjson
        import time
        while True:
            time.sleep(1)


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Pluribus MetaLearner")
    subparsers = parser.add_subparsers(dest="command")
    
    # Run
    subparsers.add_parser("run")
    
    # Record
    rec_parser = subparsers.add_parser("record")
    rec_parser.add_argument("source")
    rec_parser.add_argument("context")
    rec_parser.add_argument("signal_data")
    
    # Pattern
    pat_parser = subparsers.add_parser("pattern")
    pat_parser.add_argument("pattern_key")
    pat_parser.add_argument("confidence", type=float)
    
    # Suggest
    sug_parser = subparsers.add_parser("suggest")
    sug_parser.add_argument("context")
    
    # Nearest (Phase 4)
    near_parser = subparsers.add_parser("nearest")
    near_parser.add_argument("query")
    near_parser.add_argument("--top_k", type=int, default=5)
    
    args = parser.parse_args()
    
    learner = MetaLearner()
    
    if args.command == "run":
        learner.run()
    elif args.command == "record":
        try:
            data = json.loads(args.signal_data)
        except:
            data = {"raw": args.signal_data}
        learner.record_experience(args.source, args.context, data)
        print("Experience recorded.")
    elif args.command == "pattern":
        learner.update_pattern(args.pattern_key, "Manual update", args.confidence)
        print("Pattern updated.")
    elif args.command == "suggest":
        print(json.dumps(learner.suggest(args.context), indent=2))
    elif args.command == "nearest":
        results = learner.nearest(args.query, args.top_k)
        print(json.dumps(results, indent=2))
    else:
        parser.print_help()

if __name__ == "__main__":
    main()
