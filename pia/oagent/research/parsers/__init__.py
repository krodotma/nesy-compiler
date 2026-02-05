#!/usr/bin/env python3
"""
parsers - AST Parser Registry (Step 3)

Provides a registry of AST parsers for different programming languages.

PBTSO Phase: RESEARCH

Bus Topics:
- research.parser.register
- research.capability.register

Protocol: DKIN v30, PAIP v16
"""
from __future__ import annotations

from .base import ASTParser, ParserRegistry, ParseResult
from .python_parser import PythonASTParser

__all__ = [
    "ASTParser",
    "ParserRegistry",
    "ParseResult",
    "PythonASTParser",
]

# Auto-register available parsers
try:
    from .typescript_parser import TypeScriptASTParser
    __all__.append("TypeScriptASTParser")
except ImportError:
    pass
