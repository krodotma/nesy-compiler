#!/usr/bin/env python3
"""
extractor.py - Documentation Extractor (Step 7)

Extracts docstrings, comments, and inline documentation from source code.

PBTSO Phase: RESEARCH

Bus Topics:
- research.docs.extracted
- research.docs.indexed

Protocol: DKIN v30, PAIP v16
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..bootstrap import AgentBus


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class DocBlock:
    """Represents an extracted documentation block."""

    text: str
    kind: str  # docstring, comment, block_comment, jsdoc, rst
    start_line: int
    end_line: int
    language: str
    associated_symbol: Optional[str] = None  # Name of associated function/class
    parsed: Dict[str, Any] = field(default_factory=dict)  # Parsed sections

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "text": self.text,
            "kind": self.kind,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "language": self.language,
            "associated_symbol": self.associated_symbol,
            "parsed": self.parsed,
        }


# ============================================================================
# Documentation Extractor
# ============================================================================


class DocumentationExtractor:
    """
    Extract docstrings, comments, and inline documentation.

    Supports multiple languages and documentation styles:
    - Python: triple-quoted docstrings, # comments
    - JavaScript/TypeScript: JSDoc, // comments, /* */ blocks
    - Go: // comments, godoc conventions
    - Rust: /// and //! doc comments

    Example:
        extractor = DocumentationExtractor()
        docs = extractor.extract(content, "python")
        for doc in docs:
            print(f"{doc.kind}: {doc.text[:50]}...")
    """

    # Language-specific patterns
    PATTERNS = {
        "python": {
            "docstring_triple_double": re.compile(
                r'"""(.*?)"""',
                re.DOTALL
            ),
            "docstring_triple_single": re.compile(
                r"'''(.*?)'''",
                re.DOTALL
            ),
            "comment": re.compile(
                r'^\s*#\s*(.*)$',
                re.MULTILINE
            ),
        },
        "typescript": {
            "jsdoc": re.compile(
                r'/\*\*(.*?)\*/',
                re.DOTALL
            ),
            "block_comment": re.compile(
                r'/\*(.*?)\*/',
                re.DOTALL
            ),
            "line_comment": re.compile(
                r'//\s*(.*)$',
                re.MULTILINE
            ),
        },
        "javascript": {
            "jsdoc": re.compile(
                r'/\*\*(.*?)\*/',
                re.DOTALL
            ),
            "block_comment": re.compile(
                r'/\*(.*?)\*/',
                re.DOTALL
            ),
            "line_comment": re.compile(
                r'//\s*(.*)$',
                re.MULTILINE
            ),
        },
        "go": {
            "godoc": re.compile(
                r'^//\s*(.*)$',
                re.MULTILINE
            ),
            "block_comment": re.compile(
                r'/\*(.*?)\*/',
                re.DOTALL
            ),
        },
        "rust": {
            "doc_outer": re.compile(
                r'///\s*(.*)$',
                re.MULTILINE
            ),
            "doc_inner": re.compile(
                r'//!\s*(.*)$',
                re.MULTILINE
            ),
            "block_comment": re.compile(
                r'/\*(.*?)\*/',
                re.DOTALL
            ),
        },
    }

    # Docstring section patterns (Google/NumPy style)
    SECTION_PATTERNS = {
        "args": re.compile(r'(?:Args?|Parameters?|Params?):\s*\n((?:\s+\w+.*\n?)+)', re.IGNORECASE),
        "returns": re.compile(r'(?:Returns?|Return):\s*\n?(.*?)(?=\n\s*\w+:|$)', re.IGNORECASE | re.DOTALL),
        "raises": re.compile(r'(?:Raises?|Exceptions?):\s*\n((?:\s+\w+.*\n?)+)', re.IGNORECASE),
        "yields": re.compile(r'(?:Yields?):\s*\n?(.*?)(?=\n\s*\w+:|$)', re.IGNORECASE | re.DOTALL),
        "examples": re.compile(r'(?:Examples?|Usage):\s*\n((?:.*\n?)+?)(?=\n\s*\w+:|$)', re.IGNORECASE),
        "notes": re.compile(r'(?:Notes?):\s*\n((?:.*\n?)+?)(?=\n\s*\w+:|$)', re.IGNORECASE),
        "todo": re.compile(r'(?:TODO|FIXME|XXX):\s*(.*)', re.IGNORECASE),
    }

    # JSDoc tag patterns
    JSDOC_PATTERNS = {
        "param": re.compile(r'@param\s+(?:\{([^}]+)\})?\s*(\w+)\s*(?:-?\s*(.*))?'),
        "returns": re.compile(r'@returns?\s+(?:\{([^}]+)\})?\s*(.*)'),
        "type": re.compile(r'@type\s+\{([^}]+)\}'),
        "typedef": re.compile(r'@typedef\s+\{([^}]+)\}\s*(\w+)'),
        "example": re.compile(r'@example\s*\n?(.*?)(?=@|$)', re.DOTALL),
        "description": re.compile(r'^([^@\n].*?)(?=@|\n\s*\n|$)', re.DOTALL),
        "deprecated": re.compile(r'@deprecated\s*(.*)'),
        "since": re.compile(r'@since\s+(.*)'),
        "see": re.compile(r'@see\s+(.*)'),
        "throws": re.compile(r'@throws?\s+(?:\{([^}]+)\})?\s*(.*)'),
    }

    def __init__(self, bus: Optional[AgentBus] = None):
        """
        Initialize the documentation extractor.

        Args:
            bus: AgentBus for event emission
        """
        self.bus = bus

    def extract(
        self,
        content: str,
        language: str,
        path: Optional[str] = None,
    ) -> List[DocBlock]:
        """
        Extract documentation from source code.

        Args:
            content: Source code content
            language: Programming language
            path: File path (for event emission)

        Returns:
            List of DocBlock objects
        """
        docs = []
        patterns = self.PATTERNS.get(language, self.PATTERNS.get("python"))

        if patterns is None:
            return docs

        for pattern_name, pattern in patterns.items():
            for match in pattern.finditer(content):
                text = match.group(1) if match.lastindex else match.group(0)
                text = self._clean_doc_text(text)

                if not text.strip():
                    continue

                start_line = content[:match.start()].count('\n') + 1
                end_line = content[:match.end()].count('\n') + 1

                # Determine kind
                kind = "docstring" if "docstring" in pattern_name else pattern_name

                doc = DocBlock(
                    text=text.strip(),
                    kind=kind,
                    start_line=start_line,
                    end_line=end_line,
                    language=language,
                )

                # Parse sections for docstrings
                if kind in ("docstring", "jsdoc", "doc_outer"):
                    doc.parsed = self._parse_sections(text, language)

                docs.append(doc)

        # Emit event
        if self.bus and path:
            self.bus.emit({
                "topic": "research.docs.extracted",
                "kind": "parse",
                "data": {
                    "path": path,
                    "language": language,
                    "count": len(docs),
                    "kinds": list(set(d.kind for d in docs)),
                }
            })

        return docs

    def extract_file(self, file_path: str, language: Optional[str] = None) -> List[DocBlock]:
        """
        Extract documentation from a file.

        Args:
            file_path: Path to the file
            language: Programming language (auto-detected if not provided)

        Returns:
            List of DocBlock objects
        """
        from pathlib import Path

        path = Path(file_path)

        if language is None:
            ext_map = {
                ".py": "python",
                ".pyi": "python",
                ".js": "javascript",
                ".jsx": "javascript",
                ".ts": "typescript",
                ".tsx": "typescript",
                ".go": "go",
                ".rs": "rust",
            }
            language = ext_map.get(path.suffix.lower(), "python")

        try:
            content = path.read_text(encoding="utf-8", errors="ignore")
            return self.extract(content, language, str(path))
        except Exception:
            return []

    def _clean_doc_text(self, text: str) -> str:
        """Clean extracted documentation text."""
        # Remove leading asterisks from block comments
        lines = text.split('\n')
        cleaned = []
        for line in lines:
            # Remove leading * from JSDoc style
            line = re.sub(r'^\s*\*\s?', '', line)
            cleaned.append(line)
        return '\n'.join(cleaned)

    def _parse_sections(self, text: str, language: str) -> Dict[str, Any]:
        """Parse documentation into structured sections."""
        parsed = {}

        if language in ("javascript", "typescript"):
            # Parse JSDoc style
            parsed = self._parse_jsdoc(text)
        else:
            # Parse Google/NumPy style
            parsed = self._parse_google_style(text)

        return parsed

    def _parse_google_style(self, text: str) -> Dict[str, Any]:
        """Parse Google/NumPy style docstrings."""
        parsed = {}

        # Extract description (everything before first section header)
        first_section = text.find('\n\n')
        if first_section > 0:
            parsed["description"] = text[:first_section].strip()
        else:
            # Check for section headers
            for section_name in self.SECTION_PATTERNS:
                match = re.search(rf'\b{section_name}s?:', text, re.IGNORECASE)
                if match:
                    parsed["description"] = text[:match.start()].strip()
                    break
            else:
                parsed["description"] = text.strip()

        # Extract sections
        for section_name, pattern in self.SECTION_PATTERNS.items():
            match = pattern.search(text)
            if match:
                content = match.group(1).strip()
                if section_name == "args":
                    parsed["args"] = self._parse_arg_list(content)
                elif section_name in ("returns", "yields"):
                    parsed[section_name] = content
                elif section_name == "raises":
                    parsed["raises"] = self._parse_raises_list(content)
                else:
                    parsed[section_name] = content

        return parsed

    def _parse_jsdoc(self, text: str) -> Dict[str, Any]:
        """Parse JSDoc style documentation."""
        parsed = {}

        # Description
        desc_match = self.JSDOC_PATTERNS["description"].search(text)
        if desc_match:
            parsed["description"] = desc_match.group(1).strip()

        # Parameters
        params = []
        for match in self.JSDOC_PATTERNS["param"].finditer(text):
            param = {
                "name": match.group(2),
            }
            if match.group(1):
                param["type"] = match.group(1)
            if match.group(3):
                param["description"] = match.group(3).strip()
            params.append(param)
        if params:
            parsed["params"] = params

        # Returns
        ret_match = self.JSDOC_PATTERNS["returns"].search(text)
        if ret_match:
            ret = {}
            if ret_match.group(1):
                ret["type"] = ret_match.group(1)
            if ret_match.group(2):
                ret["description"] = ret_match.group(2).strip()
            parsed["returns"] = ret

        # Example
        example_match = self.JSDOC_PATTERNS["example"].search(text)
        if example_match:
            parsed["example"] = example_match.group(1).strip()

        # Throws
        throws = []
        for match in self.JSDOC_PATTERNS["throws"].finditer(text):
            throw = {}
            if match.group(1):
                throw["type"] = match.group(1)
            if match.group(2):
                throw["description"] = match.group(2).strip()
            throws.append(throw)
        if throws:
            parsed["throws"] = throws

        # Deprecated
        dep_match = self.JSDOC_PATTERNS["deprecated"].search(text)
        if dep_match:
            parsed["deprecated"] = dep_match.group(1).strip() or True

        return parsed

    def _parse_arg_list(self, content: str) -> List[Dict[str, str]]:
        """Parse argument list from docstring."""
        args = []
        # Pattern: name (type): description
        pattern = re.compile(r'(\w+)\s*(?:\(([^)]+)\))?\s*:\s*(.*)', re.MULTILINE)

        for match in pattern.finditer(content):
            arg = {
                "name": match.group(1),
            }
            if match.group(2):
                arg["type"] = match.group(2)
            if match.group(3):
                arg["description"] = match.group(3).strip()
            args.append(arg)

        return args

    def _parse_raises_list(self, content: str) -> List[Dict[str, str]]:
        """Parse raises list from docstring."""
        raises = []
        pattern = re.compile(r'(\w+)\s*:\s*(.*)', re.MULTILINE)

        for match in pattern.finditer(content):
            raises.append({
                "type": match.group(1),
                "description": match.group(2).strip(),
            })

        return raises


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> int:
    """CLI entry point for Documentation Extractor."""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="Documentation Extractor (Step 7)"
    )
    parser.add_argument(
        "file",
        help="Source file to extract documentation from"
    )
    parser.add_argument(
        "--language", "-l",
        help="Programming language (auto-detected if not provided)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )

    args = parser.parse_args()

    extractor = DocumentationExtractor()
    docs = extractor.extract_file(args.file, args.language)

    if args.json:
        print(json.dumps([d.to_dict() for d in docs], indent=2))
    else:
        print(f"Extracted {len(docs)} documentation blocks from {args.file}:")
        for doc in docs:
            print(f"\n  [{doc.kind}] lines {doc.start_line}-{doc.end_line}:")
            preview = doc.text[:100].replace('\n', ' ')
            if len(doc.text) > 100:
                preview += "..."
            print(f"    {preview}")
            if doc.parsed:
                print(f"    Parsed sections: {list(doc.parsed.keys())}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
