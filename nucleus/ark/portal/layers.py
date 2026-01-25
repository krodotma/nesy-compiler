#!/usr/bin/env python3
"""
layers.py - 3-Layer Source Architecture Management

Manages the promotion pipeline:
L1 (Raw) → L2 (Curated) → L3 (Core)

Each layer has different quality thresholds and gates.
"""

import os
import shutil
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from pathlib import Path
from datetime import datetime
from enum import Enum

from nucleus.ark.ribosome.gene import Gene
from nucleus.ark.portal.distill import CrystallizationGate

logger = logging.getLogger("ARK.Portal.Layers")


class Layer(Enum):
    """Source code layers."""
    RAW = 1        # L1: Ephemeral/Exploratory
    CURATED = 2    # L2: Spec-Aligned/Validated
    CORE = 3       # L3: Ratified/Production


@dataclass
class LayerConfig:
    """Configuration for a source layer."""
    name: str
    path: Path
    fitness_threshold: float
    requires_tests: bool = False
    requires_docs: bool = False
    requires_witness: bool = False


@dataclass
class PromotionResult:
    """Result of a promotion attempt."""
    gene_path: str
    source_layer: Layer
    target_layer: Layer
    success: bool
    reason: str = ""
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())


class LayerManager:
    """
    Manages the 3-Layer source architecture.
    
    Layers:
    - L1 (Raw): /tmp/swarm-*, agent_reports/ - No strict linting
    - L2 (Curated): nucleus/tools/v2/, nucleus/specs/v2/ - Aligned with specs
    - L3 (Core): nucleus/tools/*.py, nucleus/specs/*.md - Production quality
    
    Promotion requires passing layer-specific gates.
    """
    
    def __init__(self, repo_root: Path):
        self.repo_root = repo_root
        
        # Define layer configurations
        self.layers = {
            Layer.RAW: LayerConfig(
                name="raw",
                path=repo_root / "tmp" / "raw",
                fitness_threshold=0.0,
            ),
            Layer.CURATED: LayerConfig(
                name="curated",
                path=repo_root / "nucleus" / "tools" / "v2",
                fitness_threshold=0.4,
                requires_docs=True,
            ),
            Layer.CORE: LayerConfig(
                name="core",
                path=repo_root / "nucleus" / "tools",
                fitness_threshold=0.7,
                requires_tests=True,
                requires_docs=True,
                requires_witness=True,
            ),
        }
        
        # Crystallization gate
        self.crystallize_gate = CrystallizationGate()
    
    def identify_layer(self, file_path: str) -> Optional[Layer]:
        """Identify which layer a file belongs to."""
        path = Path(file_path)
        
        # Check each layer's path
        for layer, config in self.layers.items():
            try:
                path.relative_to(config.path)
                return layer
            except ValueError:
                continue
        
        # Special cases
        if "/tmp/" in str(path) or "agent_reports" in str(path):
            return Layer.RAW
        if "/v2/" in str(path):
            return Layer.CURATED
        if "/nucleus/tools/" in str(path) or "/nucleus/specs/" in str(path):
            return Layer.CORE
        
        return None
    
    def can_promote(self, gene: Gene, target_layer: Layer) -> tuple[bool, str]:
        """
        Check if a gene can be promoted to target layer.
        
        Returns:
            (can_promote, reason)
        """
        config = self.layers[target_layer]
        
        # Fitness check
        if gene.fitness < config.fitness_threshold:
            return False, f"Fitness {gene.fitness:.2f} < required {config.fitness_threshold}"
        
        # Check file content requirements
        try:
            content = Path(gene.path).read_text()
        except Exception as e:
            return False, f"Cannot read file: {e}"
        
        # Documentation check
        if config.requires_docs:
            if gene.path.endswith('.py'):
                if '"""' not in content and "'''" not in content:
                    return False, "Missing docstrings"
        
        # For crystallization (L3), use the full gate
        if target_layer == Layer.CORE:
            passed, reason = self.crystallize_gate.check(gene, content)
            if not passed:
                return False, reason
        
        return True, "Ready for promotion"
    
    def promote(self, gene: Gene, target_layer: Layer, witness: Optional[str] = None) -> PromotionResult:
        """
        Promote a gene to a higher layer.
        
        Args:
            gene: The gene to promote
            target_layer: Target layer
            witness: Optional witness for Core promotion
        
        Returns:
            PromotionResult
        """
        source_layer = self.identify_layer(gene.path)
        
        if source_layer is None:
            return PromotionResult(
                gene_path=gene.path,
                source_layer=Layer.RAW,
                target_layer=target_layer,
                success=False,
                reason="Cannot identify source layer"
            )
        
        # Check if promotion is valid (must be upward)
        if source_layer.value >= target_layer.value:
            return PromotionResult(
                gene_path=gene.path,
                source_layer=source_layer,
                target_layer=target_layer,
                success=False,
                reason=f"Cannot promote from {source_layer.name} to {target_layer.name}"
            )
        
        # Check witness requirement
        config = self.layers[target_layer]
        if config.requires_witness and not witness:
            return PromotionResult(
                gene_path=gene.path,
                source_layer=source_layer,
                target_layer=target_layer,
                success=False,
                reason="Witness required for Core promotion"
            )
        
        # Check promotion eligibility
        can_promote, reason = self.can_promote(gene, target_layer)
        if not can_promote:
            return PromotionResult(
                gene_path=gene.path,
                source_layer=source_layer,
                target_layer=target_layer,
                success=False,
                reason=reason
            )
        
        # Perform the promotion (copy file)
        try:
            source_path = Path(gene.path)
            target_config = self.layers[target_layer]
            target_path = target_config.path / source_path.name
            
            target_config.path.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, target_path)
            
            logger.info(f"✅ Promoted {source_path.name}: {source_layer.name} → {target_layer.name}")
            
            return PromotionResult(
                gene_path=str(target_path),
                source_layer=source_layer,
                target_layer=target_layer,
                success=True,
                reason=f"Promoted to {target_config.path}"
            )
            
        except Exception as e:
            return PromotionResult(
                gene_path=gene.path,
                source_layer=source_layer,
                target_layer=target_layer,
                success=False,
                reason=f"Copy failed: {e}"
            )
    
    def layer_status(self) -> Dict[str, Dict]:
        """Get status of all layers."""
        status = {}
        
        for layer, config in self.layers.items():
            file_count = 0
            if config.path.exists():
                file_count = sum(1 for _ in config.path.glob("**/*.py"))
            
            status[layer.name] = {
                "path": str(config.path),
                "fitness_threshold": config.fitness_threshold,
                "file_count": file_count,
                "requires_tests": config.requires_tests,
                "requires_docs": config.requires_docs,
                "requires_witness": config.requires_witness,
            }
        
        return status
