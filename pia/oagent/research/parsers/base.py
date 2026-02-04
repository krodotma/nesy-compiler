#!/usr/bin/env python3
"""
base.py - AST Parser Registry (Step 3)

Provides base classes and registry for AST parsers.

PBTSO Phase: RESEARCH

Bus Topics:
- research.parser.register
- research.capability.register

Protocol: DKIN v30, PAIP v16
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Type

from ..bootstrap import AgentBus


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class SymbolInfo:
    """Information about a code symbol."""

    name: str
    kind: str  # class, function, method, variable, import, constant
    line: int
    end_line: Optional[int] = None
    column: int = 0
    signature: Optional[str] = None
    docstring: Optional[str] = None
    decorators: List[str] = field(default_factory=list)
    parent: Optional[str] = None  # Parent class/function name
    visibility: str = "public"  # public, private, protected
    is_async: bool = False
    return_type: Optional[str] = None
    parameters: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ImportInfo:
    """Information about an import statement."""

    module: str
    names: List[str]  # Imported names (empty for module import)
    alias: Optional[str] = None
    line: int = 0
    is_relative: bool = False
    level: int = 0  # Number of dots for relative imports


@dataclass
class ParseResult:
    """Result of parsing a source file."""

    language: str
    path: str
    success: bool = True
    error: Optional[str] = None

    classes: List[SymbolInfo] = field(default_factory=list)
    functions: List[SymbolInfo] = field(default_factory=list)
    methods: List[SymbolInfo] = field(default_factory=list)
    variables: List[SymbolInfo] = field(default_factory=list)
    imports: List[ImportInfo] = field(default_factory=list)
    exports: List[str] = field(default_factory=list)  # For JS/TS

    # Metadata
    line_count: int = 0
    has_docstring: bool = False
    module_docstring: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "language": self.language,
            "path": self.path,
            "success": self.success,
            "error": self.error,
            "classes": [
                {
                    "name": c.name,
                    "line": c.line,
                    "end_line": c.end_line,
                    "docstring": c.docstring,
                    "decorators": c.decorators,
                }
                for c in self.classes
            ],
            "functions": [
                {
                    "name": f.name,
                    "line": f.line,
                    "signature": f.signature,
                    "is_async": f.is_async,
                    "return_type": f.return_type,
                    "parameters": f.parameters,
                    "docstring": f.docstring,
                }
                for f in self.functions
            ],
            "methods": [
                {
                    "name": m.name,
                    "line": m.line,
                    "parent": m.parent,
                    "visibility": m.visibility,
                    "is_async": m.is_async,
                }
                for m in self.methods
            ],
            "imports": [
                {
                    "module": i.module,
                    "names": i.names,
                    "alias": i.alias,
                    "is_relative": i.is_relative,
                }
                for i in self.imports
            ],
            "exports": self.exports,
            "line_count": self.line_count,
            "has_docstring": self.has_docstring,
        }


# ============================================================================
# Base Parser
# ============================================================================


class ASTParser(ABC):
    """
    Abstract base class for AST parsers.

    Subclasses implement language-specific parsing logic.
    """

    # File extensions this parser handles
    extensions: List[str] = []

    # Language name
    language: str = "unknown"

    def __init__(self, bus: Optional[AgentBus] = None):
        """
        Initialize the parser.

        Args:
            bus: AgentBus for event emission
        """
        self.bus = bus

    @abstractmethod
    def parse(self, content: str, path: str) -> ParseResult:
        """
        Parse source code and return structured AST data.

        Args:
            content: Source code content
            path: File path (for error reporting)

        Returns:
            ParseResult with extracted symbols
        """
        pass

    def parse_file(self, file_path: str) -> ParseResult:
        """
        Parse a file from disk.

        Args:
            file_path: Path to the file

        Returns:
            ParseResult with extracted symbols
        """
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                content = f.read()
            return self.parse(content, file_path)
        except Exception as e:
            return ParseResult(
                language=self.language,
                path=file_path,
                success=False,
                error=str(e),
            )

    def _emit_event(self, topic: str, data: Dict[str, Any]) -> None:
        """Emit an event if bus is available."""
        if self.bus:
            self.bus.emit({
                "topic": topic,
                "kind": "parse",
                "data": data,
            })


# ============================================================================
# Parser Registry
# ============================================================================


class ParserRegistry:
    """
    Registry of AST parsers by file extension.

    Provides a central lookup for finding the right parser
    for a given file type.

    Example:
        ParserRegistry.register(PythonASTParser)
        parser = ParserRegistry.get_parser(".py")
        result = parser.parse(content, path)
    """

    _parsers: Dict[str, Type[ASTParser]] = {}
    _instances: Dict[str, ASTParser] = {}
    _bus: Optional[AgentBus] = None

    @classmethod
    def set_bus(cls, bus: AgentBus) -> None:
        """Set the bus for all parsers."""
        cls._bus = bus

    @classmethod
    def register(cls, parser_class: Type[ASTParser]) -> None:
        """
        Register a parser class for its extensions.

        Args:
            parser_class: Parser class to register
        """
        for ext in parser_class.extensions:
            cls._parsers[ext] = parser_class

        # Emit registration event
        if cls._bus:
            cls._bus.emit({
                "topic": "research.parser.register",
                "kind": "capability",
                "data": {
                    "language": parser_class.language,
                    "extensions": parser_class.extensions,
                }
            })

    @classmethod
    def get_parser(cls, extension: str) -> Optional[ASTParser]:
        """
        Get a parser instance for a file extension.

        Args:
            extension: File extension (e.g., ".py")

        Returns:
            Parser instance or None if no parser registered
        """
        parser_class = cls._parsers.get(extension)
        if parser_class is None:
            return None

        # Return cached instance or create new one
        if extension not in cls._instances:
            cls._instances[extension] = parser_class(bus=cls._bus)

        return cls._instances[extension]

    @classmethod
    def get_supported_extensions(cls) -> List[str]:
        """Get list of supported file extensions."""
        return list(cls._parsers.keys())

    @classmethod
    def get_parser_for_language(cls, language: str) -> Optional[ASTParser]:
        """
        Get a parser instance for a language name.

        Args:
            language: Language name (e.g., "python")

        Returns:
            Parser instance or None if no parser registered
        """
        for ext, parser_class in cls._parsers.items():
            if parser_class.language == language:
                if ext not in cls._instances:
                    cls._instances[ext] = parser_class(bus=cls._bus)
                return cls._instances[ext]
        return None

    @classmethod
    def parse_file(cls, file_path: str) -> Optional[ParseResult]:
        """
        Parse a file using the appropriate parser.

        Args:
            file_path: Path to the file

        Returns:
            ParseResult or None if no parser available
        """
        from pathlib import Path
        ext = Path(file_path).suffix.lower()
        parser = cls.get_parser(ext)
        if parser is None:
            return None
        return parser.parse_file(file_path)

    @classmethod
    def clear(cls) -> None:
        """Clear all registered parsers (for testing)."""
        cls._parsers.clear()
        cls._instances.clear()
