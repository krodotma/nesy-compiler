#!/usr/bin/env python3
"""
client.py - MetaLearnerClient for ARK Integration

Wraps the Pluribus MetaLearner server (/suggest, /health endpoints)
with ARK-specific functionality:

- Etymology suggestion from commit context
- CMP prediction (commit quality forecast)  
- Entropy forecasting
- Experience feedback for learning
- Prescient analysis (predict outcome before commit)

Phase 2.1 Implementation (P2-001 through P2-020)
"""

import os
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

logger = logging.getLogger("ARK.MetaLearner")


@dataclass
class MetaLearnerConfig:
    """Configuration for MetaLearner client."""
    host: str = "localhost"
    port: int = 8001
    timeout_seconds: float = 5.0
    retry_count: int = 2
    fallback_enabled: bool = True
    
    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


@dataclass
class PredictionResult:
    """Result of a MetaLearner prediction."""
    success: bool
    prediction_type: str  # cmp, entropy, etymology, quality
    value: Any
    confidence: float = 0.5
    latency_ms: float = 0.0
    source: str = "metalearner"  # metalearner, fallback, cache
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperienceRecord:
    """Record for MetaLearner feedback loop."""
    commit_sha: str
    etymology: str
    cmp_before: float
    cmp_after: float
    entropy_before: Dict[str, float]
    entropy_after: Dict[str, float]
    gate_results: Dict[str, bool]
    success: bool  # Did commit succeed?
    timestamp: str = ""


class MetaLearnerClient:
    """
    ARK client for MetaLearner integration.
    
    Enables:
    - P2-002: /suggest API access
    - P2-003: Etymology extraction
    - P2-006: CMP prediction
    - P2-007: Entropy prediction
    - P2-008: Prescient S-phase analysis
    - P2-010: Feedback loop (commit → outcome → learn)
    - P2-011: Experience buffer for ICL
    """
    
    def __init__(self, config: Optional[MetaLearnerConfig] = None):
        self.config = config or MetaLearnerConfig()
        self._available: Optional[bool] = None
        self._experience_buffer: List[ExperienceRecord] = []
        self._icl_context: List[Dict] = []  # In-context learning examples
        self._cache: Dict[str, PredictionResult] = {}
        self._cache_ttl_seconds = 60
    
    @property
    def available(self) -> bool:
        """Check if MetaLearner server is available."""
        if self._available is None:
            self._available = self._check_health()
        return self._available
    
    def _check_health(self) -> bool:
        """Ping MetaLearner health endpoint."""
        if not HAS_REQUESTS:
            logger.warning("requests library not available - MetaLearner disabled")
            return False
        
        try:
            resp = requests.get(
                f"{self.config.base_url}/health",
                timeout=self.config.timeout_seconds
            )
            return resp.status_code == 200
        except Exception as e:
            logger.debug("MetaLearner health check failed: %s", e)
            return False
    
    def suggest(
        self, 
        context: Dict[str, Any],
        suggestion_type: str = "etymology"
    ) -> PredictionResult:
        """
        Request suggestion from MetaLearner.
        
        Args:
            context: Commit context (files, message, entropy, etc.)
            suggestion_type: Type of suggestion (etymology, design, structure)
            
        Returns:
            PredictionResult with suggestion
        """
        start_time = time.time()
        
        if not self.available:
            return self._fallback_suggestion(context, suggestion_type)
        
        try:
            payload = {
                "type": suggestion_type,
                "context": context,
                "icl_examples": self._icl_context[-5:]  # Last 5 examples
            }
            
            resp = requests.post(
                f"{self.config.base_url}/suggest",
                json=payload,
                timeout=self.config.timeout_seconds
            )
            
            latency = (time.time() - start_time) * 1000
            
            if resp.status_code == 200:
                data = resp.json()
                return PredictionResult(
                    success=True,
                    prediction_type=suggestion_type,
                    value=data,
                    confidence=data.get("confidence", 0.7),
                    latency_ms=latency,
                    source="metalearner"
                )
            else:
                logger.warning("MetaLearner /suggest failed: %s", resp.status_code)
                return self._fallback_suggestion(context, suggestion_type)
                
        except Exception as e:
            logger.warning("MetaLearner request failed: %s", e)
            return self._fallback_suggestion(context, suggestion_type)
    
    def predict_cmp(
        self,
        files: List[str],
        message: str,
        current_entropy: Dict[str, float]
    ) -> PredictionResult:
        """
        Predict CMP (commit quality) before commit.
        
        P2-006: Create CMP prediction from MetaLearner
        
        Args:
            files: Files to be committed
            message: Commit message
            current_entropy: Current H* vector
            
        Returns:
            Predicted CMP value (0-1)
        """
        context = {
            "files": files,
            "message": message,
            "entropy": current_entropy,
            "file_count": len(files),
            "message_length": len(message)
        }
        
        # Check cache
        cache_key = f"cmp:{hash(tuple(sorted(files)))}"
        if cache_key in self._cache:
            cached = self._cache[cache_key]
            if (time.time() - cached.metadata.get("cache_time", 0)) < self._cache_ttl_seconds:
                cached.source = "cache"
                return cached
        
        result = self.suggest(context, "cmp_prediction")
        
        if result.success and isinstance(result.value, dict):
            cmp_value = result.value.get("predicted_cmp", 0.5)
            result.value = cmp_value
        else:
            # Fallback: estimate from entropy
            h_total = sum(current_entropy.values()) / max(len(current_entropy), 1)
            result.value = 1.0 - h_total  # Low entropy = high CMP
        
        # Cache result
        result.metadata["cache_time"] = time.time()
        self._cache[cache_key] = result
        
        return result
    
    def predict_entropy(
        self,
        files: List[str],
        code_snippets: List[str]
    ) -> PredictionResult:
        """
        Predict entropy vector for staged changes.
        
        P2-007: Add entropy prediction (pre-commit)
        """
        context = {
            "files": files,
            "snippets": code_snippets[:5],  # Limit for context window
            "total_lines": sum(len(s.split('\n')) for s in code_snippets)
        }
        
        result = self.suggest(context, "entropy_prediction")
        
        if result.success and isinstance(result.value, dict):
            return result
        
        # Fallback: neutral entropy
        result.value = {
            "h_info": 0.5, "h_miss": 0.5, "h_conj": 0.3,
            "h_alea": 0.3, "h_epis": 0.3, "h_struct": 0.5,
            "c_load": 0.3, "h_goal_drift": 0.3
        }
        result.source = "fallback"
        return result
    
    def suggest_etymology(
        self,
        files: List[str],
        message: str,
        diff_summary: str = ""
    ) -> PredictionResult:
        """
        Suggest etymology (semantic origin) for commit.
        
        P2-003: Wire etymology extraction to MetaLearner
        """
        context = {
            "files": files,
            "message": message,
            "diff_summary": diff_summary[:500]  # Limit
        }
        
        result = self.suggest(context, "etymology")
        
        if result.success and isinstance(result.value, dict):
            result.value = result.value.get("etymology", message)
        else:
            # Fallback: extract from message
            result.value = self._extract_etymology_simple(message, files)
            result.source = "fallback"
        
        return result
    
    def prescient_analysis(
        self,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Prescient S-phase analysis: predict outcome before commit.
        
        P2-008: Implement prescient analysis in S-phase
        
        Returns dict with:
        - predicted_cmp: Expected CMP
        - risk_level: low/medium/high
        - suggestions: List of improvement suggestions
        - gate_predictions: Expected gate results
        """
        files = context.get("files", [])
        message = context.get("message", "")
        entropy = context.get("entropy", {})
        
        # Get predictions
        cmp_result = self.predict_cmp(files, message, entropy)
        entropy_result = self.predict_entropy(files, context.get("snippets", []))
        
        predicted_cmp = cmp_result.value if isinstance(cmp_result.value, (int, float)) else 0.5
        
        # Determine risk level
        if predicted_cmp < 0.3:
            risk_level = "high"
        elif predicted_cmp < 0.6:
            risk_level = "medium"
        else:
            risk_level = "low"
        
        # Generate suggestions if risky
        suggestions = []
        if risk_level == "high":
            suggestions.append("Consider breaking this into smaller commits")
            suggestions.append("Add documentation or tests to reduce entropy")
        elif risk_level == "medium":
            suggestions.append("Review entropy dimensions for improvement")
        
        # Predict gate outcomes
        predicted_entropy = entropy_result.value if isinstance(entropy_result.value, dict) else entropy
        h_total = sum(predicted_entropy.values()) / max(len(predicted_entropy), 1)
        
        gate_predictions = {
            "inertia": predicted_cmp > 0.4,
            "entelecheia": bool(message.strip()),
            "homeostasis": h_total < 0.8
        }
        
        return {
            "predicted_cmp": predicted_cmp,
            "predicted_entropy": predicted_entropy,
            "risk_level": risk_level,
            "suggestions": suggestions,
            "gate_predictions": gate_predictions,
            "confidence": min(cmp_result.confidence, entropy_result.confidence),
            "source": cmp_result.source
        }
    
    def record_experience(self, record: ExperienceRecord) -> None:
        """
        Record commit outcome for feedback learning.
        
        P2-010: Create feedback loop: commit → outcome → learn
        """
        record.timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ")
        self._experience_buffer.append(record)
        
        # Add to ICL context
        icl_example = {
            "etymology": record.etymology,
            "cmp_delta": record.cmp_after - record.cmp_before,
            "success": record.success,
            "files_changed": len(record.gate_results)
        }
        self._icl_context.append(icl_example)
        
        # Trim context window
        if len(self._icl_context) > 20:
            self._icl_context = self._icl_context[-20:]
        
        # Forward to MetaLearner if available
        if self.available:
            try:
                requests.post(
                    f"{self.config.base_url}/feedback",
                    json={
                        "sha": record.commit_sha,
                        "success": record.success,
                        "cmp_delta": record.cmp_after - record.cmp_before,
                        "etymology": record.etymology
                    },
                    timeout=self.config.timeout_seconds
                )
            except Exception as e:
                logger.debug("Failed to send feedback to MetaLearner: %s", e)
    
    def get_icl_context(self) -> List[Dict]:
        """
        Get current ICL context for prompting.
        
        P2-011: Implement experience buffer for ICL
        """
        return self._icl_context.copy()
    
    def flush_experience_buffer(self) -> int:
        """Flush experience buffer to persistent storage."""
        count = len(self._experience_buffer)
        
        # Save to file
        buffer_path = Path("~/.ark/experience_buffer.json").expanduser()
        buffer_path.parent.mkdir(parents=True, exist_ok=True)
        
        existing = []
        if buffer_path.exists():
            try:
                existing = json.loads(buffer_path.read_text())
            except Exception:
                pass
        
        for record in self._experience_buffer:
            existing.append({
                "sha": record.commit_sha,
                "etymology": record.etymology,
                "cmp_before": record.cmp_before,
                "cmp_after": record.cmp_after,
                "success": record.success,
                "timestamp": record.timestamp
            })
        
        buffer_path.write_text(json.dumps(existing, indent=2))
        self._experience_buffer.clear()
        
        logger.info("Flushed %d experiences to buffer", count)
        return count
    
    def _fallback_suggestion(
        self, 
        context: Dict[str, Any], 
        suggestion_type: str
    ) -> PredictionResult:
        """Fallback suggestions when MetaLearner unavailable."""
        if not self.config.fallback_enabled:
            return PredictionResult(
                success=False,
                prediction_type=suggestion_type,
                value=None,
                confidence=0.0,
                source="none"
            )
        
        # Generate fallback based on type
        if suggestion_type == "etymology":
            message = context.get("message", "")
            files = context.get("files", [])
            value = self._extract_etymology_simple(message, files)
        elif suggestion_type == "cmp_prediction":
            # Use entropy heuristic
            entropy = context.get("entropy", {})
            h_total = sum(entropy.values()) / max(len(entropy), 1) if entropy else 0.5
            value = {"predicted_cmp": 1.0 - h_total}
        else:
            value = {"suggestion": "No MetaLearner available"}
        
        return PredictionResult(
            success=True,
            prediction_type=suggestion_type,
            value=value,
            confidence=0.3,  # Low confidence for fallback
            source="fallback"
        )
    
    def _extract_etymology_simple(self, message: str, files: List[str]) -> str:
        """Simple etymology extraction from message and files."""
        # Extract key terms from message
        words = message.lower().split()
        keywords = [w for w in words if len(w) > 4 and w.isalpha()]
        
        # Add file-based context
        file_types = set()
        for f in files:
            if '.' in f:
                ext = f.split('.')[-1]
                file_types.add(ext)
        
        if file_types:
            context = f"[{','.join(sorted(file_types))}]"
        else:
            context = ""
        
        if keywords:
            return f"{context} {' '.join(keywords[:5])}"
        return f"{context} {message[:50]}"


# Convenience function
def get_metalearner_client(**kwargs) -> MetaLearnerClient:
    """Get a configured MetaLearner client."""
    config = MetaLearnerConfig(**kwargs)
    return MetaLearnerClient(config)
