#!/usr/bin/env python3
"""
features.py - Feature Extraction Pipeline for Neural Gates

P2-043: Implement feature extraction pipeline
P2-044: Add AST embedding for code understanding
P2-045: Create commit embedding model

Extracts features from:
- Code content (tokenization, AST)
- Commit context (files, message, diff)
- Entropy vector (H* dimensions)
"""

import re
import ast
import time
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

logger = logging.getLogger("ARK.Neural.Features")


@dataclass
class CodeFeatures:
    """Features extracted from code content."""
    token_count: int
    line_count: int
    function_count: int
    class_count: int
    import_count: int
    complexity_score: float  # Cyclomatic-like complexity
    docstring_ratio: float   # Documented functions/classes
    type_hint_ratio: float   # Type-annotated items
    ast_depth: int           # Max AST depth
    token_ids: List[int] = field(default_factory=list)  # For neural model
    
    def to_dict(self) -> Dict:
        return {
            "token_count": self.token_count,
            "line_count": self.line_count,
            "function_count": self.function_count,
            "class_count": self.class_count,
            "import_count": self.import_count,
            "complexity_score": self.complexity_score,
            "docstring_ratio": self.docstring_ratio,
            "type_hint_ratio": self.type_hint_ratio,
            "ast_depth": self.ast_depth
        }
    
    def to_vector(self) -> List[float]:
        """Convert to feature vector for ML."""
        return [
            self.token_count / 1000,  # Normalize
            self.line_count / 500,
            self.function_count / 50,
            self.class_count / 20,
            self.import_count / 30,
            self.complexity_score,
            self.docstring_ratio,
            self.type_hint_ratio,
            self.ast_depth / 20
        ]


@dataclass 
class CommitFeatures:
    """Features extracted from commit context."""
    file_count: int
    total_additions: int
    total_deletions: int
    file_types: List[str]
    message_length: int
    etymology_keywords: List[str]
    entropy_vector: Dict[str, float]
    code_features: List[CodeFeatures] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "file_count": self.file_count,
            "total_additions": self.total_additions,
            "total_deletions": self.total_deletions,
            "file_types": self.file_types,
            "message_length": self.message_length,
            "etymology_keywords": self.etymology_keywords,
            "entropy_vector": self.entropy_vector
        }
    
    def to_vector(self) -> List[float]:
        """Convert to feature vector."""
        # Entropy values
        entropy_keys = ["h_info", "h_miss", "h_conj", "h_alea",
                       "h_epis", "h_struct", "c_load", "h_goal_drift"]
        entropy_values = [self.entropy_vector.get(k, 0.5) for k in entropy_keys]
        
        # Commit metadata
        commit_values = [
            self.file_count / 20,
            self.total_additions / 500,
            self.total_deletions / 500,
            len(self.file_types) / 10,
            self.message_length / 200,
            len(self.etymology_keywords) / 10
        ]
        
        # Aggregate code features
        if self.code_features:
            code_avg = [0.0] * 9
            for cf in self.code_features:
                for i, v in enumerate(cf.to_vector()):
                    code_avg[i] += v
            code_avg = [v / len(self.code_features) for v in code_avg]
        else:
            code_avg = [0.5] * 9
        
        return entropy_values + commit_values + code_avg


class FeatureExtractor:
    """
    Extracts features from code and commits for neural gates.
    
    P2-043: Feature extraction pipeline
    P2-044: AST embedding for code
    P2-045: Commit embedding
    """
    
    def __init__(self, vocab_path: Optional[str] = None):
        self.vocab: Dict[str, int] = {}
        self._special_tokens = {
            "<PAD>": 0,
            "<UNK>": 1,
            "<CLS>": 2,
            "<SEP>": 3
        }
        self.vocab.update(self._special_tokens)
        
        if vocab_path and Path(vocab_path).exists():
            self._load_vocab(vocab_path)
        else:
            self._init_basic_vocab()
    
    def _init_basic_vocab(self) -> None:
        """Initialize with Python keywords and common tokens."""
        python_keywords = [
            "def", "class", "return", "if", "else", "elif", "for", "while",
            "try", "except", "finally", "with", "as", "import", "from",
            "True", "False", "None", "and", "or", "not", "in", "is",
            "break", "continue", "pass", "raise", "yield", "lambda",
            "async", "await", "self", "cls"
        ]
        
        for i, token in enumerate(python_keywords, start=len(self._special_tokens)):
            self.vocab[token] = i
    
    def _load_vocab(self, path: str) -> None:
        """Load vocabulary from file."""
        try:
            with open(path) as f:
                for i, line in enumerate(f, start=len(self._special_tokens)):
                    token = line.strip()
                    if token:
                        self.vocab[token] = i
        except Exception as e:
            logger.warning("Failed to load vocab: %s", e)
    
    def extract_code_features(self, code: str) -> CodeFeatures:
        """
        Extract features from Python code.
        
        P2-044: AST embedding for code understanding
        """
        lines = code.split('\n') if code else []
        
        # Basic counts
        token_count = len(code.split()) if code else 0
        line_count = len(lines)
        
        # Parse AST
        function_count = 0
        class_count = 0
        import_count = 0
        docstring_count = 0
        type_hint_count = 0
        total_items = 0
        ast_depth = 0
        
        try:
            tree = ast.parse(code)
            ast_depth = self._get_ast_depth(tree)
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef) or isinstance(node, ast.AsyncFunctionDef):
                    function_count += 1
                    total_items += 1
                    # Check for docstring
                    if (node.body and isinstance(node.body[0], ast.Expr) 
                        and isinstance(node.body[0].value, ast.Constant)
                        and isinstance(node.body[0].value.value, str)):
                        docstring_count += 1
                    # Check for return type hint
                    if node.returns:
                        type_hint_count += 1
                        
                elif isinstance(node, ast.ClassDef):
                    class_count += 1
                    total_items += 1
                    # Check for docstring
                    if (node.body and isinstance(node.body[0], ast.Expr)
                        and isinstance(node.body[0].value, ast.Constant)
                        and isinstance(node.body[0].value.value, str)):
                        docstring_count += 1
                        
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    import_count += 1
                    
        except SyntaxError:
            # Can't parse - estimate from patterns
            function_count = len(re.findall(r'\bdef\s+\w+', code))
            class_count = len(re.findall(r'\bclass\s+\w+', code))
            import_count = len(re.findall(r'\bimport\s+', code))
        
        # Calculate ratios
        docstring_ratio = docstring_count / max(total_items, 1)
        type_hint_ratio = type_hint_count / max(function_count, 1)
        
        # Complexity score (simplified cyclomatic)
        complexity_score = self._estimate_complexity(code)
        
        # Tokenize for neural model
        token_ids = self._tokenize(code)
        
        return CodeFeatures(
            token_count=token_count,
            line_count=line_count,
            function_count=function_count,
            class_count=class_count,
            import_count=import_count,
            complexity_score=complexity_score,
            docstring_ratio=docstring_ratio,
            type_hint_ratio=type_hint_ratio,
            ast_depth=ast_depth,
            token_ids=token_ids
        )
    
    def _get_ast_depth(self, node: ast.AST, depth: int = 0) -> int:
        """Get maximum depth of AST."""
        max_depth = depth
        for child in ast.iter_child_nodes(node):
            child_depth = self._get_ast_depth(child, depth + 1)
            max_depth = max(max_depth, child_depth)
        return max_depth
    
    def _estimate_complexity(self, code: str) -> float:
        """
        Estimate cyclomatic complexity from code patterns.
        
        Returns normalized complexity score [0, 1].
        """
        # Count decision points
        decision_patterns = [
            r'\bif\b', r'\bfor\b', r'\bwhile\b', r'\band\b', r'\bor\b',
            r'\btry\b', r'\bexcept\b', r'\bwith\b', r'\belif\b'
        ]
        
        decision_count = 0
        for pattern in decision_patterns:
            decision_count += len(re.findall(pattern, code))
        
        # Normalize by lines
        lines = len(code.split('\n'))
        if lines == 0:
            return 0.0
        
        # Complexity per line, capped at 1.0
        return min(1.0, decision_count / (lines * 0.1 + 1))
    
    def _tokenize(self, code: str, max_tokens: int = 512) -> List[int]:
        """Tokenize code for neural model."""
        # Simple whitespace tokenization
        tokens = re.findall(r'\w+|[^\w\s]', code)
        
        token_ids = [self._special_tokens["<CLS>"]]
        for token in tokens[:max_tokens - 2]:
            if token in self.vocab:
                token_ids.append(self.vocab[token])
            else:
                token_ids.append(self._special_tokens["<UNK>"])
        token_ids.append(self._special_tokens["<SEP>"])
        
        # Pad to max length
        while len(token_ids) < max_tokens:
            token_ids.append(self._special_tokens["<PAD>"])
        
        return token_ids[:max_tokens]
    
    def extract_commit_features(
        self,
        files: List[str],
        file_contents: Dict[str, str],
        message: str,
        etymology: str,
        entropy: Dict[str, float],
        additions: int = 0,
        deletions: int = 0
    ) -> CommitFeatures:
        """
        Extract features from commit context.
        
        P2-045: Create commit embedding model
        """
        # Extract file types
        file_types = list(set(
            f.split('.')[-1] for f in files if '.' in f
        ))
        
        # Extract code features for each file
        code_features = []
        for path, content in file_contents.items():
            if path.endswith('.py'):
                try:
                    cf = self.extract_code_features(content)
                    code_features.append(cf)
                except Exception:
                    pass
        
        # Extract keywords from etymology
        etymology_keywords = [
            w.lower() for w in etymology.split() 
            if len(w) > 3 and w.isalpha()
        ]
        
        return CommitFeatures(
            file_count=len(files),
            total_additions=additions,
            total_deletions=deletions,
            file_types=file_types,
            message_length=len(message),
            etymology_keywords=etymology_keywords,
            entropy_vector=entropy,
            code_features=code_features
        )
    
    def get_embedding_dim(self) -> int:
        """Get total embedding dimension for commit features."""
        # 8 (entropy) + 6 (commit meta) + 9 (code features) = 23
        return 23
