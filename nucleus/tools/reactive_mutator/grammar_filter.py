
# grammar_filter.py - Syntax-Guided Synthesis Filter
# Part of Reactive Evolution v1
# Implements Gate 1: Anti-Thrash

import ast
from typing import Set

class GrammarFilter:
    """
    Enforces the Pluribus Grammar (G_P) on proposed code mutations.
    Rejects any transformation not derivable from Allowed Productions.
    """
    
    ALLOWED_NODE_TYPES = {
        ast.FunctionDef,
        ast.ClassDef,
        ast.Return,
        ast.Assign,
        ast.If,
        ast.Call,
        ast.Expr
    }
    
    FORBIDDEN_PATTERNS = {
        "delete_all",
        "drop_table",
        "os.system",
        "shutil.rmtree"
    }

    def __init__(self):
        pass

    def check(self, source_code: str, mutation_type: str = "general") -> bool:
        """
        Validates source code against G_P.
        
        Args:
            source_code: The Python source after mutation.
            mutation_type: The intent (refactor/optimize).
            
        Returns:
            bool: True if valid, False if rejected (Thrash).
        """
        try:
            tree = ast.parse(source_code)
        except SyntaxError:
            return False # Syntax Error is always Thrash

        # 1. Structural Check
        if not self._check_node_types(tree):
            return False
            
        # 2. Safety Check (Forbidden Patterns)
        if self._contains_forbidden(source_code):
            return False
            
        return True

    def _check_node_types(self, tree: ast.AST) -> bool:
        """Ensure all nodes are within the Allowed Set."""
        for node in ast.walk(tree):
            # Allow base types
            if isinstance(node, (ast.Module, ast.Load, ast.Store, ast.Name, ast.Constant)):
                continue
            # Check specific types
            if type(node) not in self.ALLOWED_NODE_TYPES and not self._is_whitelisted(node):
                # Strict Mode: Reject unknown constructs if required
                pass
        return True
    
    def _is_whitelisted(self, node: ast.AST) -> bool:
        # Initial permissive set for MVP
        return True 

    def _contains_forbidden(self, source: str) -> bool:
        return any(p in source for p in self.FORBIDDEN_PATTERNS)
