#!/usr/bin/env python3
"""
etymology.py - EtymologyExtractor: Semantic origin discovery

Extracts the semantic origin/purpose from code and commits
using AST analysis, docstrings, and pattern matching.
"""

import ast
import re
from dataclasses import dataclass
from typing import Optional, List, Dict
from pathlib import Path


@dataclass
class Etymology:
    """Semantic origin of a code unit."""
    primary: str           # Main purpose
    keywords: List[str]    # Semantic keywords
    domain: str            # Domain area (e.g., "evolution", "bus", "neural")
    confidence: float      # 0-1 confidence in extraction


class EtymologyExtractor:
    """
    Extracts semantic etymology from code and commits.
    
    Uses multiple strategies:
    1. Docstring analysis
    2. Function/class name decomposition
    3. Import pattern matching
    4. Comment mining
    """
    
    # Domain keyword mappings
    DOMAIN_PATTERNS = {
        "evolution": ["evolve", "mutate", "clade", "gene", "cmp", "fitness", "dna"],
        "bus": ["bus", "event", "topic", "emit", "subscribe", "message"],
        "neural": ["neural", "model", "predict", "feature", "train", "inference"],
        "synthesis": ["synthesize", "ltl", "spec", "grammar", "ast", "parse"],
        "rhizom": ["rhizom", "dag", "lineage", "ancestry", "etymology"],
        "portal": ["ingest", "portal", "distill", "crystallize", "entropy"],
        "core": ["router", "dispatcher", "orchestrator", "scheduler"],
    }
    
    def extract_from_code(self, code: str, filepath: str = "") -> Etymology:
        """Extract etymology from Python source code."""
        keywords = []
        purposes = []
        
        try:
            tree = ast.parse(code)
            
            # 1. Module docstring
            module_doc = ast.get_docstring(tree)
            if module_doc:
                purposes.append(self._extract_purpose(module_doc))
                keywords.extend(self._extract_keywords(module_doc))
            
            # 2. Class/function names
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    keywords.extend(self._decompose_name(node.name))
                    doc = ast.get_docstring(node)
                    if doc:
                        purposes.append(self._extract_purpose(doc))
                elif isinstance(node, ast.FunctionDef):
                    keywords.extend(self._decompose_name(node.name))
            
            # 3. Import patterns
            imports = self._extract_imports(tree)
            keywords.extend(imports)
            
        except SyntaxError:
            # Fallback to filename
            if filepath:
                keywords.extend(self._decompose_name(Path(filepath).stem))
        
        # Determine domain
        domain = self._classify_domain(keywords)
        
        # Build primary purpose
        primary = purposes[0] if purposes else self._build_purpose(keywords, filepath)
        
        return Etymology(
            primary=primary,
            keywords=list(set(keywords)),
            domain=domain,
            confidence=min(len(keywords) / 10, 1.0)
        )
    
    def extract_from_commit_message(self, message: str) -> Etymology:
        """Extract etymology from a commit message."""
        # Parse conventional commit
        match = re.match(r"^(\w+)(?:\(([^)]+)\))?:\s*(.+)$", message.split("\n")[0])
        
        keywords = []
        if match:
            commit_type, scope, subject = match.groups()
            keywords.append(commit_type)
            if scope:
                keywords.append(scope)
            keywords.extend(self._extract_keywords(subject))
        else:
            keywords.extend(self._extract_keywords(message))
        
        domain = self._classify_domain(keywords)
        
        return Etymology(
            primary=message.split("\n")[0][:100],
            keywords=keywords,
            domain=domain,
            confidence=0.8 if match else 0.5
        )
    
    def _extract_purpose(self, text: str) -> str:
        """Extract first sentence as purpose."""
        sentences = text.split(".")
        if sentences:
            return sentences[0].strip()[:200]
        return text[:200]
    
    def _extract_keywords(self, text: str) -> List[str]:
        """Extract semantic keywords from text."""
        # Remove punctuation and split
        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', text.lower())
        
        # Filter stopwords
        stopwords = {"the", "a", "an", "is", "are", "was", "were", "be", "been",
                    "this", "that", "it", "to", "for", "of", "in", "on", "with"}
        
        return [w for w in words if w not in stopwords and len(w) > 2]
    
    def _decompose_name(self, name: str) -> List[str]:
        """Decompose CamelCase or snake_case into keywords."""
        # Split CamelCase
        parts = re.sub(r'([a-z])([A-Z])', r'\1_\2', name)
        # Split snake_case
        return [p.lower() for p in parts.split('_') if p]
    
    def _extract_imports(self, tree: ast.AST) -> List[str]:
        """Extract imported module names."""
        imports = []
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name.split('.')[0])
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module.split('.')[0])
        return imports
    
    def _classify_domain(self, keywords: List[str]) -> str:
        """Classify into a domain based on keywords."""
        scores = {domain: 0 for domain in self.DOMAIN_PATTERNS}
        
        for keyword in keywords:
            for domain, patterns in self.DOMAIN_PATTERNS.items():
                if keyword in patterns:
                    scores[domain] += 1
        
        best_domain = max(scores, key=scores.get)
        return best_domain if scores[best_domain] > 0 else "general"
    
    def _build_purpose(self, keywords: List[str], filepath: str) -> str:
        """Build a purpose string from keywords."""
        if not keywords:
            return f"Code from {Path(filepath).name}" if filepath else "Unknown purpose"
        return f"Module for {' '.join(keywords[:5])}"
