"""
Theia Metagrammar — Multi-language AST transformation rules.

Extends EGGP with type-coherent transformations across languages.

Future roadmap:
    - Python ↔ JavaScript AST mappings
    - Type inference for polymorphic transforms
    - Semantic-preserving refactoring
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Set


@dataclass
class TypeMapping:
    """
    Type mapping between languages.
    
    Example: Python `int` → JavaScript `number`
    """
    source_lang: str
    target_lang: str
    source_type: str
    target_type: str
    is_exact: bool = True  # False if lossy conversion


@dataclass
class ASTPattern:
    """
    Pattern for matching AST structures.
    """
    node_type: str
    children_patterns: List["ASTPattern"] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    captures: Dict[str, str] = field(default_factory=dict)  # name -> node path


@dataclass
class TransformRule:
    """
    Multi-language transformation rule.
    """
    name: str
    source_lang: str
    target_lang: str
    pattern: ASTPattern
    replacement: ASTPattern
    type_constraints: List[TypeMapping] = field(default_factory=list)


class MetagrammarRegistry:
    """
    Registry of multi-language transformation rules.
    """
    
    def __init__(self):
        self._rules: Dict[str, TransformRule] = {}
        self._type_mappings: List[TypeMapping] = []
        self._supported_languages: Set[str] = {"python"}
    
    def register_rule(self, rule: TransformRule) -> None:
        """Register a transformation rule."""
        self._rules[rule.name] = rule
        self._supported_languages.add(rule.source_lang)
        self._supported_languages.add(rule.target_lang)
    
    def register_type_mapping(self, mapping: TypeMapping) -> None:
        """Register a type mapping."""
        self._type_mappings.append(mapping)
    
    def find_rules(
        self,
        source_lang: str,
        target_lang: Optional[str] = None,
    ) -> List[TransformRule]:
        """Find applicable rules for language pair."""
        results = []
        for rule in self._rules.values():
            if rule.source_lang == source_lang:
                if target_lang is None or rule.target_lang == target_lang:
                    results.append(rule)
        return results
    
    def can_transform(self, source_lang: str, target_lang: str) -> bool:
        """Check if transformation path exists."""
        return bool(self.find_rules(source_lang, target_lang))
    
    def get_type_mapping(
        self,
        source_lang: str,
        target_lang: str,
        source_type: str,
    ) -> Optional[str]:
        """Get target type for source type."""
        for mapping in self._type_mappings:
            if (mapping.source_lang == source_lang and
                mapping.target_lang == target_lang and
                mapping.source_type == source_type):
                return mapping.target_type
        return None
    
    def summary(self) -> Dict[str, Any]:
        """Get registry summary."""
        return {
            "n_rules": len(self._rules),
            "n_type_mappings": len(self._type_mappings),
            "languages": list(self._supported_languages),
        }


# =============================================================================
# DEFAULT REGISTRY WITH COMMON RULES
# =============================================================================

def create_default_registry() -> MetagrammarRegistry:
    """Create registry with common Python transformations."""
    registry = MetagrammarRegistry()
    
    # Add common type mappings for Python → JS
    registry.register_type_mapping(TypeMapping(
        source_lang="python",
        target_lang="javascript",
        source_type="int",
        target_type="number",
    ))
    
    registry.register_type_mapping(TypeMapping(
        source_lang="python",
        target_lang="javascript",
        source_type="str",
        target_type="string",
    ))
    
    registry.register_type_mapping(TypeMapping(
        source_lang="python",
        target_lang="javascript",
        source_type="list",
        target_type="Array",
    ))
    
    registry.register_type_mapping(TypeMapping(
        source_lang="python",
        target_lang="javascript",
        source_type="dict",
        target_type="Object",
    ))
    
    return registry


__all__ = [
    "TypeMapping",
    "ASTPattern",
    "TransformRule",
    "MetagrammarRegistry",
    "create_default_registry",
]
