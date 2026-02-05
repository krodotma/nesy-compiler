#!/usr/bin/env python3
"""
Code Agent - Neural-Guided Code Generation and AST Manipulation

Part of OAGENT 300-Step Plan: Steps 51-80

This module provides:
- Bootstrap and lifecycle management (Step 51)
- Neural code proposal generation (Step 52)
- AST transformations for Python/TypeScript (Steps 53-55)
- Multi-file edit coordination (Step 56)
- PAIP isolation clone management (Step 57)
- Import dependency resolution (Step 58)
- Code style enforcement (Step 59)
- Incremental compilation (Step 60)
- Diff optimization (Step 61)
- Conflict resolution (Step 62)
- Semantic merging (Step 63)
- Rollback management (Step 64)
- Checkpoint system (Step 65)
- Transaction logging (Step 66)
- Undo/redo stack (Step 67)
- Workspace sync (Step 68)
- File watching (Step 69)
- Code orchestration (Step 70)
- Code generation (Step 71)
- Template engine (Step 72)
- Snippet manager (Step 73)
- Refactoring engine (Step 74)
- Code formatter (Step 75)
- Unified linter (Step 76)
- Type checker (Step 77)
- Documentation generator (Step 78)
- Code API (Step 79)
- Code CLI (Step 80)

Protocol: DKIN v30, CITIZEN v2, PAIP v16
Bus Topics: a2a.code.*, code.*, paip.*
A2A Heartbeat: 300s interval, 900s timeout
FalkorDB: Port 6380
"""

from .bootstrap import CodeAgentBootstrap, CodeAgentConfig

# Steps 71-80 imports
from .generator import CodeGenerator, GeneratorConfig, GenerationRequest, GenerationResult
from .template import TemplateEngine, TemplateConfig, CodeTemplate, TemplateResult
from .snippet import SnippetManager, SnippetConfig, CodeSnippet, SnippetSearchResult
from .refactor import RefactoringEngine, RefactoringConfig, RefactoringOperation, RefactoringResult
from .formatter import CodeFormatter, FormatterConfig, FormatResult
from .linter import UnifiedLinter, LinterConfig, LintIssue, LintResult
from .typechecker import TypeChecker, TypeCheckerConfig, TypeIssue, TypeCheckResult
from .docs import DocumentationGenerator, DocConfig, DocResult
from .api import CodeAPI, APIConfig
from .cli import CodeCLI, CLIConfig

__all__ = [
    # Bootstrap (Step 51)
    "CodeAgentBootstrap",
    "CodeAgentConfig",
    # Generator (Step 71)
    "CodeGenerator",
    "GeneratorConfig",
    "GenerationRequest",
    "GenerationResult",
    # Template (Step 72)
    "TemplateEngine",
    "TemplateConfig",
    "CodeTemplate",
    "TemplateResult",
    # Snippet (Step 73)
    "SnippetManager",
    "SnippetConfig",
    "CodeSnippet",
    "SnippetSearchResult",
    # Refactor (Step 74)
    "RefactoringEngine",
    "RefactoringConfig",
    "RefactoringOperation",
    "RefactoringResult",
    # Formatter (Step 75)
    "CodeFormatter",
    "FormatterConfig",
    "FormatResult",
    # Linter (Step 76)
    "UnifiedLinter",
    "LinterConfig",
    "LintIssue",
    "LintResult",
    # TypeChecker (Step 77)
    "TypeChecker",
    "TypeCheckerConfig",
    "TypeIssue",
    "TypeCheckResult",
    # Documentation (Step 78)
    "DocumentationGenerator",
    "DocConfig",
    "DocResult",
    # API (Step 79)
    "CodeAPI",
    "APIConfig",
    # CLI (Step 80)
    "CodeCLI",
    "CLIConfig",
]

__version__ = "0.2.0"
