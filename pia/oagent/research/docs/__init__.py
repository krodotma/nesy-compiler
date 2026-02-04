#!/usr/bin/env python3
"""
docs - Documentation Extractor (Steps 7-8)

Extracts documentation from source code and README files.

PBTSO Phase: RESEARCH

Bus Topics:
- research.docs.extracted
- research.docs.indexed
- research.readme.parsed
- research.project.structure

Protocol: DKIN v30, PAIP v16
"""
from __future__ import annotations

from .extractor import DocumentationExtractor, DocBlock
from .readme_parser import ReadmeParser, ReadmeSection

__all__ = [
    "DocumentationExtractor",
    "DocBlock",
    "ReadmeParser",
    "ReadmeSection",
]
