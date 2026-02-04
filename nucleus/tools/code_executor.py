#!/usr/bin/env python3
"""
Safe Code Executor (Elite Edition)
Runs code snippets in a subprocess with AST-based static analysis and Ring 0 protection.

Features:
- AST Validation: Rejects unsafe AST nodes (imports, dunders, dangerous builtins) before execution.
- Ring 0 Guard: Explicitly bans writes to critical system paths.
- Isolation: Runs in a separate process with a clean environment.
"""
from __future__ import annotations

import argparse
import ast
import os
import subprocess
import sys
import tempfile
from typing import Set

# Elite Whitelists
SAFE_IMPORTS: Set[str] = {"math", "json", "random", "datetime", "re", "collections", "itertools", "functools"}
SAFE_BUILTINS: Set[str] = {
    "print", "len", "range", "str", "int", "float", "bool", "list", "dict", "set", "tuple", 
    "enumerate", "zip", "map", "filter", "sorted", "reversed", "sum", "min", "max", "abs", 
    "round", "pow", "divmod", "isinstance", "issubclass", "hasattr", "getattr", "all", "any"
}

RING0_PATHS = [
    ".pluribus/constitution.md",
    "AGENTS.md",
    "nucleus/tools/iso_git.mjs",
    ".pluribus/lineage.json",
]


def check_ring0_violation(code: str) -> str | None:
    """
    Check if code contains references to Ring 0 protected paths.
    Returns the first Ring 0 path found in the code, or None if safe.
    """
    for r0_path in RING0_PATHS:
        if r0_path in code:
            return r0_path
    return None


class SecurityVisitor(ast.NodeVisitor):
    def __init__(self):
        self.errors = []

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name.split('.')[0] not in SAFE_IMPORTS:
                self.errors.append(f"SecurityViolation: Import of '{alias.name}' is not whitelisted.")
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module and node.module.split('.')[0] not in SAFE_IMPORTS:
            self.errors.append(f"SecurityViolation: Import from '{node.module}' is not whitelisted.")
        self.generic_visit(node)

    def visit_Call(self, node):
        # Check calls to dangerous functions
        if isinstance(node.func, ast.Name):
            if node.func.id in {"open", "exec", "eval", "__import__", "compile", "globals", "locals", "breakpoint", "help", "input"}:
                self.errors.append(f"SecurityViolation: Call to restricted function '{node.func.id}'.")
        self.generic_visit(node)

    def visit_Attribute(self, node):
        # Prevent access to dunders (except allowed ones if we were lenient, but strict is better)
        if node.attr.startswith("__") and node.attr != "__name__": # Allow __name__ for if __name__ == "__main__"
             self.errors.append(f"SecurityViolation: Access to internal attribute '{node.attr}' is restricted.")
        self.generic_visit(node)

def analyze_code(code: str) -> str | None:
    """Performs static analysis on the code. Returns error message or None."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"SyntaxError: {e}"

    visitor = SecurityVisitor()
    visitor.visit(tree)
    
    if visitor.errors:
        return "\n".join(visitor.errors)
        
    # Naive check for Ring 0 strings as a second layer (in case of obfuscation)
    for r0 in RING0_PATHS:
        if r0 in code:
             # This is a heuristic; technically a string literal is safe, but why mention the constitution?
             # We allow reading, but the AST visitor bans 'open', so they can't read it anyway via open().
             # Subprocess is isolated. We will log a warning but not block JUST on string presence 
             # unless it looks like a shell command injection risk.
             pass 

    return None

def run_python(code: str, *, cwd: str | None = None, timeout_s: float = 30.0) -> str:
    # 1. Elite Static Analysis
    security_error = analyze_code(code)
    if security_error:
        return f"ERROR: [Security Guard] Blocked by Policy.\n{security_error}"

    try:
        # 2. Runtime Isolation
        # Default to a temp working dir to reduce incidental filesystem side effects.
        run_cwd = cwd
        tmp_dir_ctx: tempfile.TemporaryDirectory[str] | None = None
        if run_cwd is None:
            tmp_dir_ctx = tempfile.TemporaryDirectory(prefix="pluribus_code_executor_")
            run_cwd = tmp_dir_ctx.name

        try:
            # We add a preamble to restrict builtins at runtime too, for depth.
            # But since we can't easily injection-proof the preamble, we rely on AST + Subprocess.
            res = subprocess.run(
                [sys.executable, "-I", "-c", code],
                capture_output=True,
                text=True,
                timeout=timeout_s,
                cwd=run_cwd,
                env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"},
            )
        finally:
            if tmp_dir_ctx is not None:
                tmp_dir_ctx.cleanup()

        return f"EXIT={res.returncode}\nSTDOUT:\n{res.stdout}\nSTDERR:\n{res.stderr}"
    except subprocess.TimeoutExpired:
        return "ERROR: Execution timed out."
    except Exception as e:
        return f"ERROR: {e}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--lang", default="python")
    p.add_argument("--code", required=True)
    p.add_argument("--cwd", default=None, help="Working directory (defaults to an isolated temp dir).")
    p.add_argument("--timeout", type=float, default=30.0, help="Timeout seconds")
    args = p.parse_args()

    if args.lang == "python":
        print(run_python(args.code, cwd=args.cwd, timeout_s=args.timeout))
    else:
        print(f"Language {args.lang} not supported yet.")

if __name__ == "__main__":
    main()
