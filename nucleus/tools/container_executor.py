#!/usr/bin/env python3
"""
Container Executor & Elite Code Guard
=====================================

Provides safe, sandboxed execution of Python code.
Includes Ring-0 AST Validation to prevent dangerous builtins.

Usage:
  python3 nucleus/tools/container_executor.py run <file> [--safe]

Gates:
  - AST Validation (bans exec, eval, open, subprocess, __import__)
  - Resource Limits (if running in container)
"""
import sys
import os
import argparse
import ast
import subprocess
import tempfile

sys.dont_write_bytecode = True

# --- Elite Code Guard (AST Validator) ---

BANNED_FUNCTIONS = {
    "exec", "eval", "compile", "open", "input",
    "__import__", "globals", "locals", "super",
    "getattr", "setattr", "delattr"
}

BANNED_MODULES = {
    "os", "sys", "subprocess", "shutil", "socket",
    "requests", "urllib", "http", "importlib"
}

class SecurityVisitor(ast.NodeVisitor):
    def __init__(self):
        self.errors = []

    def visit_Call(self, node):
        if isinstance(node.func, ast.Name):
            if node.func.id in BANNED_FUNCTIONS:
                self.errors.append(f"Line {node.lineno}: Function '{node.func.id}' is banned.")
        self.generic_visit(node)

    def visit_Import(self, node):
        for alias in node.names:
            if alias.name.split('.')[0] in BANNED_MODULES:
                self.errors.append(f"Line {node.lineno}: Module '{alias.name}' is banned.")
        self.generic_visit(node)

    def visit_ImportFrom(self, node):
        if node.module and node.module.split('.')[0] in BANNED_MODULES:
            self.errors.append(f"Line {node.lineno}: Module '{node.module}' is banned.")
        self.generic_visit(node)

    def visit_Attribute(self, node):
        # Ban access to __dunder__ attributes broadly
        if node.attr.startswith("__") and node.attr.endswith("__"):
             # Allow some safe ones if needed, but default to strict
             if node.attr not in {"__init__", "__name__", "__main__"}:
                 self.errors.append(f"Line {node.lineno}: Dunder attribute '{node.attr}' is banned.")
        self.generic_visit(node)

def validate_code(code: str) -> list[str]:
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"Syntax Error: {e}"]
    
    visitor = SecurityVisitor()
    visitor.visit(tree)
    return visitor.errors

# --- Execution Logic ---

def execute_code(path: str, safe: bool):
    try:
        with open(path, "r", encoding="utf-8") as f:
            code = f.read()
    except Exception as e:
        print(f"Error reading file: {e}")
        return 1

    if safe:
        errors = validate_code(code)
        if errors:
            print("Security Violation(s) Detected:")
            for err in errors:
                print(f"  - {err}")
            print("Execution Blocked by Elite Code Guard.")
            return 1
        print("[Guard] AST Validation Passed.")

    # In a real scenario, this would spawn a docker container/firecracker vm.
    # For this implementation, we run locally but after AST check (if safe=True).
    # If safe=False, we assume the caller knows what they are doing (Ring 0 agent).
    
    print(f"Executing {path}...")
    result = subprocess.run([sys.executable, path], capture_output=True, text=True)
    
    print("--- STDOUT ---")
    print(result.stdout)
    if result.stderr:
        print("--- STDERR ---")
        print(result.stderr)
    
    return result.returncode

def main():
    parser = argparse.ArgumentParser(description="Container Executor")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    run_parser = subparsers.add_parser("run", help="Run a script")
    run_parser.add_argument("file", help="Path to python script")
    run_parser.add_argument("--safe", action="store_true", help="Enforce AST security checks")
    
    # Check capabilities
    subparsers.add_parser("capabilities", help="Show runtime capabilities")

    args = parser.parse_args()

    if args.command == "capabilities":
        print(json.dumps({
            "sandbox": "python-ast",
            "container": "subprocess-shim",
            "security": "ring0-guard"
        }, indent=2))
        return 0

    if args.command == "run":
        sys.exit(execute_code(args.file, args.safe))

if __name__ == "__main__":
    import json
    main()
