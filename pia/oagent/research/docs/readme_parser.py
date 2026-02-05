#!/usr/bin/env python3
"""
readme_parser.py - README Parser (Step 8)

Parses README.md files for project structure understanding.

PBTSO Phase: RESEARCH

Bus Topics:
- research.readme.parsed
- research.project.structure

Protocol: DKIN v30, PAIP v16
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..bootstrap import AgentBus


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class ReadmeSection:
    """Represents a section in a README file."""

    title: str
    level: int  # Heading level (1-6)
    content: str
    line: int
    subsections: List["ReadmeSection"] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "title": self.title,
            "level": self.level,
            "content": self.content[:500],  # Truncate for serialization
            "line": self.line,
            "subsections": [s.to_dict() for s in self.subsections],
        }


@dataclass
class CodeBlock:
    """Represents a code block in a README file."""

    language: str
    code: str
    line: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "language": self.language,
            "code": self.code[:500],  # Truncate for serialization
            "line": self.line,
        }


@dataclass
class Link:
    """Represents a link in a README file."""

    text: str
    url: str
    line: int
    is_image: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "text": self.text,
            "url": self.url,
            "line": self.line,
            "is_image": self.is_image,
        }


@dataclass
class ParsedReadme:
    """Result of parsing a README file."""

    path: str
    title: Optional[str]
    description: Optional[str]
    sections: List[ReadmeSection]
    code_blocks: List[CodeBlock]
    links: List[Link]
    badges: List[Link]
    install_commands: List[str]
    dependencies: List[str]
    toc: List[Dict[str, Any]]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "path": self.path,
            "title": self.title,
            "description": self.description,
            "sections": [s.to_dict() for s in self.sections],
            "code_blocks": [c.to_dict() for c in self.code_blocks],
            "links": [l.to_dict() for l in self.links],
            "badges": [b.to_dict() for b in self.badges],
            "install_commands": self.install_commands,
            "dependencies": self.dependencies,
            "toc": self.toc,
        }


# ============================================================================
# README Parser
# ============================================================================


class ReadmeParser:
    """
    Parse README.md files for project structure understanding.

    Extracts:
    - Title and description
    - Section hierarchy
    - Code blocks with language detection
    - Links and images
    - Badges
    - Installation commands
    - Dependencies mentioned

    Example:
        parser = ReadmeParser()
        result = parser.parse_file("/path/to/README.md")
        print(f"Title: {result.title}")
        for section in result.sections:
            print(f"  {section.title}")
    """

    # Patterns for extraction
    PATTERNS = {
        "heading": re.compile(r'^(#{1,6})\s+(.+)$', re.MULTILINE),
        "code_block": re.compile(r'```(\w*)\n(.*?)```', re.DOTALL),
        "link": re.compile(r'\[([^\]]+)\]\(([^)]+)\)'),
        "image": re.compile(r'!\[([^\]]*)\]\(([^)]+)\)'),
        "badge": re.compile(r'!\[([^\]]*)\]\((https?://[^)]*(?:badge|shield|img\.shields)[^)]*)\)'),
        "install_npm": re.compile(r'npm\s+install\s+(\S+)'),
        "install_pip": re.compile(r'pip\s+install\s+(\S+)'),
        "install_yarn": re.compile(r'yarn\s+add\s+(\S+)'),
        "install_cargo": re.compile(r'cargo\s+install\s+(\S+)'),
        "install_go": re.compile(r'go\s+get\s+(\S+)'),
        "install_brew": re.compile(r'brew\s+install\s+(\S+)'),
    }

    # Common README file names
    README_NAMES = [
        "README.md",
        "readme.md",
        "README.MD",
        "Readme.md",
        "README.rst",
        "readme.rst",
        "README.txt",
        "readme.txt",
        "README",
    ]

    def __init__(self, bus: Optional[AgentBus] = None):
        """
        Initialize the README parser.

        Args:
            bus: AgentBus for event emission
        """
        self.bus = bus

    def parse(self, content: str, path: str) -> ParsedReadme:
        """
        Parse README content.

        Args:
            content: README file content
            path: File path

        Returns:
            ParsedReadme with extracted information
        """
        # Extract sections
        sections = self._extract_sections(content)

        # Extract title and description
        title = self._extract_title(content)
        description = self._extract_description(content, title)

        # Extract code blocks
        code_blocks = self._extract_code_blocks(content)

        # Extract links
        links, images = self._extract_links(content)

        # Extract badges
        badges = self._extract_badges(content)

        # Extract install commands
        install_commands = self._extract_install_commands(content)

        # Extract dependencies
        dependencies = self._extract_dependencies(content, code_blocks)

        # Build TOC
        toc = self._build_toc(sections)

        result = ParsedReadme(
            path=path,
            title=title,
            description=description,
            sections=sections,
            code_blocks=code_blocks,
            links=links + images,
            badges=badges,
            install_commands=install_commands,
            dependencies=dependencies,
            toc=toc,
        )

        # Emit event
        if self.bus:
            self.bus.emit({
                "topic": "research.readme.parsed",
                "kind": "parse",
                "data": {
                    "path": path,
                    "title": title,
                    "sections": len(sections),
                    "code_blocks": len(code_blocks),
                    "links": len(links),
                }
            })

        return result

    def parse_file(self, file_path: str) -> ParsedReadme:
        """
        Parse a README file from disk.

        Args:
            file_path: Path to the README file

        Returns:
            ParsedReadme with extracted information
        """
        path = Path(file_path)
        content = path.read_text(encoding="utf-8", errors="ignore")
        return self.parse(content, str(path))

    def find_readme(self, directory: str) -> Optional[str]:
        """
        Find README file in a directory.

        Args:
            directory: Directory to search

        Returns:
            Path to README file or None
        """
        dir_path = Path(directory)
        for name in self.README_NAMES:
            readme_path = dir_path / name
            if readme_path.exists():
                return str(readme_path)
        return None

    def _extract_title(self, content: str) -> Optional[str]:
        """Extract the main title (first H1)."""
        match = re.search(r'^#\s+(.+)$', content, re.MULTILINE)
        if match:
            return match.group(1).strip()

        # Try first line if it's not a heading
        first_line = content.strip().split('\n')[0].strip()
        if first_line and not first_line.startswith('#'):
            return first_line

        return None

    def _extract_description(self, content: str, title: Optional[str]) -> Optional[str]:
        """Extract project description (text after title before next heading)."""
        lines = content.split('\n')
        in_description = False
        description_lines = []

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Skip title
            if title and stripped == f"# {title}":
                in_description = True
                continue

            # Skip badges at the top
            if in_description and (
                stripped.startswith('![') or
                stripped.startswith('[![') or
                stripped.startswith('<img') or
                not stripped
            ):
                continue

            # Stop at next heading
            if in_description and stripped.startswith('#'):
                break

            # Collect description
            if in_description and stripped:
                description_lines.append(stripped)

                # Stop after a reasonable description length
                if len(' '.join(description_lines)) > 500:
                    break

        return ' '.join(description_lines).strip() or None

    def _extract_sections(self, content: str) -> List[ReadmeSection]:
        """Extract all sections from the README."""
        sections = []
        lines = content.split('\n')
        current_section = None
        content_lines = []

        for i, line in enumerate(lines):
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)

            if heading_match:
                # Save previous section
                if current_section is not None:
                    current_section.content = '\n'.join(content_lines).strip()
                    sections.append(current_section)

                # Start new section
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()

                current_section = ReadmeSection(
                    title=title,
                    level=level,
                    content="",
                    line=i + 1,
                )
                content_lines = []
            elif current_section is not None:
                content_lines.append(line)

        # Save last section
        if current_section is not None:
            current_section.content = '\n'.join(content_lines).strip()
            sections.append(current_section)

        return sections

    def _extract_code_blocks(self, content: str) -> List[CodeBlock]:
        """Extract code blocks from the README."""
        code_blocks = []

        for match in self.PATTERNS["code_block"].finditer(content):
            language = match.group(1) or "text"
            code = match.group(2)
            line = content[:match.start()].count('\n') + 1

            code_blocks.append(CodeBlock(
                language=language,
                code=code.strip(),
                line=line,
            ))

        return code_blocks

    def _extract_links(self, content: str) -> tuple[List[Link], List[Link]]:
        """Extract links and images from the README."""
        links = []
        images = []

        for match in self.PATTERNS["link"].finditer(content):
            text = match.group(1)
            url = match.group(2)
            line = content[:match.start()].count('\n') + 1

            links.append(Link(
                text=text,
                url=url,
                line=line,
            ))

        for match in self.PATTERNS["image"].finditer(content):
            alt = match.group(1)
            url = match.group(2)
            line = content[:match.start()].count('\n') + 1

            images.append(Link(
                text=alt,
                url=url,
                line=line,
                is_image=True,
            ))

        return links, images

    def _extract_badges(self, content: str) -> List[Link]:
        """Extract badges from the README."""
        badges = []

        for match in self.PATTERNS["badge"].finditer(content):
            alt = match.group(1)
            url = match.group(2)
            line = content[:match.start()].count('\n') + 1

            badges.append(Link(
                text=alt,
                url=url,
                line=line,
                is_image=True,
            ))

        return badges

    def _extract_install_commands(self, content: str) -> List[str]:
        """Extract installation commands from the README."""
        commands = []

        # Look for install patterns in code blocks
        for match in self.PATTERNS["code_block"].finditer(content):
            code = match.group(2)

            for pattern_name, pattern in self.PATTERNS.items():
                if pattern_name.startswith("install_"):
                    # Find the full command line
                    for line in code.split('\n'):
                        if re.search(pattern, line):
                            cmd = line.strip()
                            if cmd and cmd not in commands:
                                commands.append(cmd)

        return commands

    def _extract_dependencies(
        self,
        content: str,
        code_blocks: List[CodeBlock],
    ) -> List[str]:
        """Extract mentioned dependencies from the README."""
        dependencies = []

        # Extract from install commands
        for pattern_name, pattern in self.PATTERNS.items():
            if pattern_name.startswith("install_"):
                for match in pattern.finditer(content):
                    dep = match.group(1)
                    if dep and dep not in dependencies:
                        # Clean up dependency name
                        dep = dep.split('@')[0].split(':')[0]
                        dependencies.append(dep)

        return dependencies

    def _build_toc(self, sections: List[ReadmeSection]) -> List[Dict[str, Any]]:
        """Build table of contents from sections."""
        toc = []
        for section in sections:
            toc.append({
                "title": section.title,
                "level": section.level,
                "line": section.line,
            })
        return toc


# ============================================================================
# CLI Entry Point
# ============================================================================


def main() -> int:
    """CLI entry point for README Parser."""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description="README Parser (Step 8)"
    )
    parser.add_argument(
        "file",
        nargs="?",
        help="README file to parse (searches in current directory if not provided)"
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON"
    )
    parser.add_argument(
        "--toc",
        action="store_true",
        help="Show table of contents only"
    )

    args = parser.parse_args()

    readme_parser = ReadmeParser()

    if args.file:
        file_path = args.file
    else:
        file_path = readme_parser.find_readme(".")
        if not file_path:
            print("No README file found in current directory")
            return 1

    result = readme_parser.parse_file(file_path)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    elif args.toc:
        print(f"Table of Contents for {result.path}:")
        for item in result.toc:
            indent = "  " * (item["level"] - 1)
            print(f"{indent}- {item['title']} (line {item['line']})")
    else:
        print(f"README Analysis: {result.path}")
        print(f"\nTitle: {result.title}")
        if result.description:
            print(f"Description: {result.description[:200]}...")
        print(f"\nSections: {len(result.sections)}")
        for section in result.sections[:10]:
            indent = "  " * (section.level - 1)
            print(f"  {indent}{section.title}")
        if len(result.sections) > 10:
            print(f"  ... and {len(result.sections) - 10} more")
        print(f"\nCode Blocks: {len(result.code_blocks)}")
        for cb in result.code_blocks[:5]:
            print(f"  - {cb.language} (line {cb.line})")
        print(f"\nLinks: {len(result.links)}")
        print(f"Badges: {len(result.badges)}")
        if result.install_commands:
            print(f"\nInstall Commands:")
            for cmd in result.install_commands[:5]:
                print(f"  $ {cmd}")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
