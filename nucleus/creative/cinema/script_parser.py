"""
Script Parser Module
====================

Film script parsing with Fountain format support.
Parses screenplays into structured scene elements.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Optional, Iterator
from pathlib import Path


class ElementType(Enum):
    """Types of script elements in a screenplay."""
    SCENE_HEADING = auto()
    ACTION = auto()
    CHARACTER = auto()
    DIALOGUE = auto()
    PARENTHETICAL = auto()
    TRANSITION = auto()
    CENTERED = auto()
    NOTE = auto()
    BONEYARD = auto()
    SECTION = auto()
    SYNOPSIS = auto()
    PAGE_BREAK = auto()
    TITLE_PAGE = auto()
    DUAL_DIALOGUE = auto()
    LYRIC = auto()


@dataclass
class SceneElement:
    """
    A single element in a screenplay script.

    Attributes:
        element_type: The type of script element (heading, action, dialogue, etc.)
        content: The text content of the element
        character_name: For dialogue, the character speaking
        scene_number: The scene number if present
        is_dual_dialogue: Whether this is dual dialogue
        metadata: Additional metadata for the element
    """
    element_type: ElementType
    content: str
    character_name: Optional[str] = None
    scene_number: Optional[str] = None
    is_dual_dialogue: bool = False
    metadata: dict = field(default_factory=dict)

    def is_scene_heading(self) -> bool:
        """Check if this element is a scene heading."""
        return self.element_type == ElementType.SCENE_HEADING

    def is_dialogue(self) -> bool:
        """Check if this element is dialogue."""
        return self.element_type == ElementType.DIALOGUE

    def is_action(self) -> bool:
        """Check if this element is action."""
        return self.element_type == ElementType.ACTION

    def get_location(self) -> Optional[str]:
        """Extract location from scene heading."""
        if not self.is_scene_heading():
            return None
        # Parse INT./EXT. LOCATION - TIME format
        match = re.match(r'^(?:INT\.|EXT\.|INT\./EXT\.|I/E\.?)\s*(.+?)(?:\s*-\s*.+)?$',
                         self.content, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return self.content

    def get_time_of_day(self) -> Optional[str]:
        """Extract time of day from scene heading."""
        if not self.is_scene_heading():
            return None
        match = re.search(r'-\s*(.+)$', self.content)
        if match:
            return match.group(1).strip()
        return None

    def __repr__(self) -> str:
        if self.character_name:
            return f"SceneElement({self.element_type.name}, {self.character_name}: {self.content[:50]}...)"
        return f"SceneElement({self.element_type.name}, {self.content[:50]}...)"


@dataclass
class Script:
    """
    A complete screenplay script.

    Attributes:
        title: The title of the screenplay
        author: The author(s) of the screenplay
        elements: List of scene elements in order
        metadata: Additional script metadata (draft date, contact, etc.)
    """
    title: str = ""
    author: str = ""
    elements: List[SceneElement] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    def add_element(self, element: SceneElement) -> None:
        """Add an element to the script."""
        self.elements.append(element)

    def get_scenes(self) -> List[List[SceneElement]]:
        """
        Split script into scenes.

        Returns:
            List of scenes, where each scene is a list of elements
            starting with a scene heading.
        """
        scenes = []
        current_scene = []

        for element in self.elements:
            if element.is_scene_heading():
                if current_scene:
                    scenes.append(current_scene)
                current_scene = [element]
            else:
                current_scene.append(element)

        if current_scene:
            scenes.append(current_scene)

        return scenes

    def get_characters(self) -> List[str]:
        """
        Extract all character names from the script.

        Returns:
            Sorted list of unique character names.
        """
        characters = set()
        for element in self.elements:
            if element.element_type == ElementType.CHARACTER:
                # Remove parentheticals and extensions from character names
                name = re.sub(r'\s*\(.*?\)\s*', '', element.content)
                name = name.strip()
                if name:
                    characters.add(name)
        return sorted(characters)

    def get_dialogue_for_character(self, character: str) -> List[SceneElement]:
        """
        Get all dialogue for a specific character.

        Args:
            character: The character name to filter by

        Returns:
            List of dialogue elements for that character.
        """
        dialogue = []
        for element in self.elements:
            if element.character_name and element.character_name.upper() == character.upper():
                if element.is_dialogue():
                    dialogue.append(element)
        return dialogue

    def scene_count(self) -> int:
        """Return the number of scenes in the script."""
        return sum(1 for e in self.elements if e.is_scene_heading())

    def __iter__(self) -> Iterator[SceneElement]:
        return iter(self.elements)

    def __len__(self) -> int:
        return len(self.elements)


class FountainParser:
    """
    Parser for Fountain screenplay format.

    Fountain is a plain text markup language for screenplays.
    See: https://fountain.io/syntax
    """

    # Regex patterns for Fountain elements
    SCENE_HEADING_PATTERN = re.compile(
        r'^(?:\.)?\s*(INT\.|EXT\.|INT\./EXT\.|I/E\.?|EST\.?)\s*(.+?)(?:\s*#(.+)#)?$',
        re.IGNORECASE | re.MULTILINE
    )
    FORCED_SCENE_HEADING_PATTERN = re.compile(r'^\.\s*(.+?)(?:\s*#(.+)#)?$')
    CHARACTER_PATTERN = re.compile(r'^([A-Z][A-Z0-9 .\'-]*?)(\s*\(.*?\))?\s*$')
    FORCED_CHARACTER_PATTERN = re.compile(r'^@(.+)$')
    TRANSITION_PATTERN = re.compile(r'^(.*TO:)\s*$', re.IGNORECASE)
    FORCED_TRANSITION_PATTERN = re.compile(r'^>\s*(.+)$')
    CENTERED_PATTERN = re.compile(r'^>\s*(.+)\s*<$')
    PARENTHETICAL_PATTERN = re.compile(r'^\((.+)\)$')
    NOTE_PATTERN = re.compile(r'\[\[(.+?)\]\]', re.DOTALL)
    BONEYARD_PATTERN = re.compile(r'/\*(.+?)\*/', re.DOTALL)
    SECTION_PATTERN = re.compile(r'^(#{1,6})\s*(.+)$')
    SYNOPSIS_PATTERN = re.compile(r'^=\s*(.+)$')
    PAGE_BREAK_PATTERN = re.compile(r'^===+$')
    LYRIC_PATTERN = re.compile(r'^~\s*(.+)$')

    def __init__(self):
        """Initialize the Fountain parser."""
        self._current_character: Optional[str] = None

    def parse(self, text: str) -> Script:
        """
        Parse a Fountain-formatted screenplay.

        Args:
            text: The Fountain-formatted text to parse

        Returns:
            A Script object containing all parsed elements.
        """
        script = Script()

        # First, extract and remove boneyards and notes
        text = self._extract_boneyards(text)

        # Parse title page if present
        text, title_page = self._parse_title_page(text)
        script.metadata.update(title_page)
        if 'Title' in title_page:
            script.title = title_page['Title']
        if 'Author' in title_page:
            script.author = title_page['Author']
        elif 'Authors' in title_page:
            script.author = title_page['Authors']

        # Split into lines and parse
        lines = text.split('\n')
        i = 0

        while i < len(lines):
            line = lines[i]
            stripped = line.strip()

            # Skip empty lines (but track for paragraph detection)
            if not stripped:
                i += 1
                self._current_character = None
                continue

            # Parse different element types
            element = self._parse_line(stripped, lines, i)
            if element:
                script.add_element(element)

            i += 1

        return script

    def parse_file(self, path: Path | str) -> Script:
        """
        Parse a Fountain file.

        Args:
            path: Path to the Fountain file

        Returns:
            A Script object containing all parsed elements.
        """
        path = Path(path)
        text = path.read_text(encoding='utf-8')
        return self.parse(text)

    def _parse_title_page(self, text: str) -> tuple[str, dict]:
        """
        Parse title page at the beginning of the script.

        Returns:
            Tuple of (remaining text, title page metadata dict)
        """
        metadata = {}
        lines = text.split('\n')

        # Title page ends at first blank line after content
        in_title_page = False
        title_page_end = 0
        current_key = None

        for i, line in enumerate(lines):
            stripped = line.strip()

            # Check for key: value format
            if ':' in stripped and not stripped.startswith(':'):
                parts = stripped.split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    if key and (key[0].isupper() or key.lower() in ['title', 'author', 'authors', 'draft date', 'contact', 'copyright', 'credit', 'source']):
                        in_title_page = True
                        current_key = key
                        metadata[key] = value
                        title_page_end = i + 1
                        continue

            # Continuation of previous key (indented)
            if in_title_page and line.startswith((' ', '\t')) and current_key:
                metadata[current_key] += '\n' + stripped
                title_page_end = i + 1
                continue

            # Empty line after title page ends it
            if in_title_page and not stripped:
                break

            # Non-title-page content
            if not in_title_page and stripped:
                break

        remaining = '\n'.join(lines[title_page_end:])
        return remaining, metadata

    def _extract_boneyards(self, text: str) -> str:
        """Remove boneyard (commented out) sections."""
        return self.BONEYARD_PATTERN.sub('', text)

    def _parse_line(self, stripped: str, lines: list, index: int) -> Optional[SceneElement]:
        """Parse a single line and return appropriate element."""

        # Page break
        if self.PAGE_BREAK_PATTERN.match(stripped):
            return SceneElement(ElementType.PAGE_BREAK, stripped)

        # Section
        section_match = self.SECTION_PATTERN.match(stripped)
        if section_match:
            level = len(section_match.group(1))
            return SceneElement(
                ElementType.SECTION,
                section_match.group(2),
                metadata={'level': level}
            )

        # Synopsis
        synopsis_match = self.SYNOPSIS_PATTERN.match(stripped)
        if synopsis_match:
            return SceneElement(ElementType.SYNOPSIS, synopsis_match.group(1))

        # Centered text
        centered_match = self.CENTERED_PATTERN.match(stripped)
        if centered_match:
            return SceneElement(ElementType.CENTERED, centered_match.group(1))

        # Lyric
        lyric_match = self.LYRIC_PATTERN.match(stripped)
        if lyric_match:
            return SceneElement(ElementType.LYRIC, lyric_match.group(1))

        # Scene heading (forced or standard)
        if stripped.startswith('.') and not stripped.startswith('..'):
            forced_match = self.FORCED_SCENE_HEADING_PATTERN.match(stripped)
            if forced_match:
                return SceneElement(
                    ElementType.SCENE_HEADING,
                    forced_match.group(1),
                    scene_number=forced_match.group(2)
                )

        scene_match = self.SCENE_HEADING_PATTERN.match(stripped)
        if scene_match:
            content = f"{scene_match.group(1)} {scene_match.group(2)}".strip()
            return SceneElement(
                ElementType.SCENE_HEADING,
                content,
                scene_number=scene_match.group(3)
            )

        # Transition (forced or standard)
        forced_trans_match = self.FORCED_TRANSITION_PATTERN.match(stripped)
        if forced_trans_match and not self.CENTERED_PATTERN.match(stripped):
            return SceneElement(ElementType.TRANSITION, forced_trans_match.group(1))

        trans_match = self.TRANSITION_PATTERN.match(stripped)
        if trans_match:
            return SceneElement(ElementType.TRANSITION, trans_match.group(1))

        # Character (forced or standard)
        if stripped.startswith('@'):
            forced_char_match = self.FORCED_CHARACTER_PATTERN.match(stripped)
            if forced_char_match:
                char_name = forced_char_match.group(1).strip()
                self._current_character = char_name
                return SceneElement(
                    ElementType.CHARACTER,
                    char_name,
                    character_name=char_name
                )

        char_match = self.CHARACTER_PATTERN.match(stripped)
        if char_match and self._is_valid_character_context(lines, index):
            char_name = char_match.group(1).strip()
            extension = char_match.group(2) if char_match.group(2) else ""
            self._current_character = char_name
            return SceneElement(
                ElementType.CHARACTER,
                stripped,
                character_name=char_name,
                metadata={'extension': extension.strip('() ')} if extension else {}
            )

        # Parenthetical (after character)
        paren_match = self.PARENTHETICAL_PATTERN.match(stripped)
        if paren_match and self._current_character:
            return SceneElement(
                ElementType.PARENTHETICAL,
                paren_match.group(1),
                character_name=self._current_character
            )

        # Dialogue (after character or parenthetical)
        if self._current_character:
            return SceneElement(
                ElementType.DIALOGUE,
                stripped,
                character_name=self._current_character
            )

        # Default: Action
        return SceneElement(ElementType.ACTION, stripped)

    def _is_valid_character_context(self, lines: list, index: int) -> bool:
        """
        Check if a potential character cue is in valid context.
        Character cues must be preceded by an empty line.
        """
        if index == 0:
            return True

        prev_line = lines[index - 1].strip()
        return not prev_line


__all__ = [
    'ElementType',
    'SceneElement',
    'Script',
    'FountainParser',
]
