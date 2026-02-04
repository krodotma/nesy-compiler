#!/usr/bin/env python3
"""
Skills Scanner - Agent Skills Protocol Implementation for Pluribus
==================================================================

Implements the Agent Skills Protocol (agentskills.io) for cross-model skill discovery.
Scans .pluribus/skills/ directories for SKILL.md files and exposes them to all agents.

DKIN v25 compliant - emits bus events for skill discovery and invocation.

Usage:
    python3 skills_scanner.py --scan                    # Discover all skills
    python3 skills_scanner.py --invoke skill-name       # Load skill instructions
    python3 skills_scanner.py --list                    # List available skills
    python3 skills_scanner.py --validate                # Validate all skill files
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

# Ensure we can import from parent
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import yaml
except ImportError:
    yaml = None


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def emit_bus_event(topic: str, data: dict, level: str = "info") -> None:
    """Emit event to Pluribus bus."""
    bus_dir = os.environ.get("PLURIBUS_BUS_DIR", "").strip()
    if not bus_dir:
        # parents[2] = /pluribus (from nucleus/tools/skills_scanner.py)
        bus_dir = str(Path(__file__).resolve().parents[2] / ".pluribus" / "bus")

    bus_path = Path(bus_dir)
    if not bus_path.exists():
        bus_path.mkdir(parents=True, exist_ok=True)

    event = {
        "id": str(uuid.uuid4()),
        "ts": time.time(),
        "iso": now_iso_utc(),
        "topic": topic,
        "kind": "event",
        "level": level,
        "actor": os.environ.get("PLURIBUS_ACTOR", "skills-scanner"),
        "data": data,
    }

    events_file = bus_path / "events.ndjson"
    with events_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


@dataclass
class SkillMetadata:
    """SKILL.md frontmatter metadata."""
    name: str
    description: str
    license: str | None = None
    author: str | None = None
    version: str | None = None
    tags: list[str] | None = None

    def validate(self) -> list[str]:
        """Validate metadata constraints per Agent Skills Protocol."""
        errors = []

        # Name: max 64 chars, lowercase, letters/numbers/hyphens
        if not self.name:
            errors.append("name is required")
        elif len(self.name) > 64:
            errors.append(f"name exceeds 64 chars: {len(self.name)}")
        elif not re.match(r'^[a-z0-9-]+$', self.name):
            errors.append(f"name must be lowercase alphanumeric with hyphens: {self.name}")

        # Description: max 1024 chars, non-empty
        if not self.description:
            errors.append("description is required")
        elif len(self.description) > 1024:
            errors.append(f"description exceeds 1024 chars: {len(self.description)}")

        return errors


@dataclass
class Skill:
    """Complete skill definition."""
    metadata: SkillMetadata
    body: str
    path: Path
    token_estimate: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": asdict(self.metadata),
            "body_preview": self.body[:200] + "..." if len(self.body) > 200 else self.body,
            "path": str(self.path),
            "token_estimate": self.token_estimate,
        }


def parse_skill_md(path: Path) -> Skill | None:
    """Parse a SKILL.md file into a Skill object."""
    try:
        content = path.read_text(encoding="utf-8")
    except Exception as e:
        print(f"[WARN] Cannot read {path}: {e}", file=sys.stderr)
        return None

    # Parse YAML frontmatter
    frontmatter = {}
    body = content

    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            fm_text = parts[1].strip()
            body = parts[2].strip()

            if yaml:
                try:
                    frontmatter = yaml.safe_load(fm_text) or {}
                except Exception:
                    # Fallback: simple key: value parsing
                    for line in fm_text.split("\n"):
                        if ":" in line:
                            k, v = line.split(":", 1)
                            frontmatter[k.strip()] = v.strip()
            else:
                # No yaml module - simple parsing
                for line in fm_text.split("\n"):
                    if ":" in line:
                        k, v = line.split(":", 1)
                        frontmatter[k.strip()] = v.strip()

    # Extract metadata fields
    metadata_dict = frontmatter.get("metadata", {})
    if isinstance(metadata_dict, str):
        metadata_dict = {}

    metadata = SkillMetadata(
        name=frontmatter.get("name", path.parent.name),
        description=frontmatter.get("description", ""),
        license=frontmatter.get("license"),
        author=metadata_dict.get("author") if isinstance(metadata_dict, dict) else None,
        version=metadata_dict.get("version") if isinstance(metadata_dict, dict) else None,
        tags=frontmatter.get("tags") if isinstance(frontmatter.get("tags"), list) else None,
    )

    # Estimate tokens (rough: ~4 chars per token)
    token_estimate = len(body) // 4

    return Skill(
        metadata=metadata,
        body=body,
        path=path,
        token_estimate=token_estimate,
    )


def discover_skills(roots: list[Path] | None = None) -> list[Skill]:
    """Discover all skills from standard locations."""
    if roots is None:
        # Standard skill locations
        roots = [
            Path("/pluribus/.pluribus/skills"),
            Path.home() / ".pluribus" / "skills",
            Path("/pluribus/nucleus/skills"),
        ]

        # Also check for .claude/skills for compatibility
        roots.extend([
            Path("/pluribus/.claude/skills"),
            Path.home() / ".claude" / "skills",
        ])

    skills: list[Skill] = []
    seen_names: set[str] = set()

    for root in roots:
        if not root.exists():
            continue

        # Find all SKILL.md files
        for skill_path in root.rglob("SKILL.md"):
            skill = parse_skill_md(skill_path)
            if skill and skill.metadata.name not in seen_names:
                skills.append(skill)
                seen_names.add(skill.metadata.name)

    return skills


def validate_skills(skills: list[Skill]) -> dict[str, list[str]]:
    """Validate all skills and return errors by skill name."""
    results: dict[str, list[str]] = {}

    for skill in skills:
        errors = skill.metadata.validate()

        # Additional validations
        if skill.token_estimate > 5000:
            errors.append(f"body exceeds recommended 5000 tokens: ~{skill.token_estimate}")

        if errors:
            results[skill.metadata.name] = errors

    return results


def invoke_skill(name: str, skills: list[Skill]) -> str | None:
    """Load and return skill body for invocation."""
    for skill in skills:
        if skill.metadata.name == name:
            # Emit invocation event
            emit_bus_event("skills.invoke", {
                "skill": name,
                "path": str(skill.path),
                "token_estimate": skill.token_estimate,
            })
            return skill.body

    return None


def create_example_skill(path: Path) -> None:
    """Create an example skill file."""
    path.mkdir(parents=True, exist_ok=True)

    skill_content = '''---
name: pluribus-bus-events
description: How to emit and consume events on the Pluribus event bus
license: MIT
metadata:
  author: pluribus
  version: "1.0"
tags:
  - bus
  - events
  - telemetry
---

# Pluribus Bus Events

## Emitting Events

Use `agent_bus.py` to emit events:

```bash
python3 nucleus/tools/agent_bus.py pub \\
    --topic "your.topic.here" \\
    --kind "event" \\
    --level "info" \\
    --data '{"key": "value"}'
```

## Event Structure

All events follow the NDJSON schema:

```json
{
  "id": "uuid",
  "ts": 1234567890.123,
  "iso": "2025-12-27T00:00:00Z",
  "topic": "namespace.action.detail",
  "kind": "event|request|response|metric",
  "level": "debug|info|warn|error",
  "actor": "agent-id",
  "data": {}
}
```

## Topic Conventions

- `strp.*` - STRp workflow events
- `qa.*` - QA and testing events
- `operator.*` - Operator actions
- `telemetry.*` - Metrics and timing
'''

    (path / "SKILL.md").write_text(skill_content, encoding="utf-8")
    print(f"Created example skill at {path / 'SKILL.md'}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Pluribus Skills Scanner")
    parser.add_argument("--scan", action="store_true", help="Discover all skills")
    parser.add_argument("--list", action="store_true", help="List available skills")
    parser.add_argument("--invoke", type=str, metavar="NAME", help="Load skill by name")
    parser.add_argument("--validate", action="store_true", help="Validate all skills")
    parser.add_argument("--init", action="store_true", help="Create example skill")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--roots", type=str, nargs="*", help="Custom skill roots")

    args = parser.parse_args()

    # Parse custom roots
    roots = [Path(r) for r in args.roots] if args.roots else None

    if args.init:
        skill_dir = Path("/pluribus/.pluribus/skills/pluribus-bus-events")
        create_example_skill(skill_dir)
        return 0

    # Discover skills
    skills = discover_skills(roots)

    # Emit discovery event
    emit_bus_event("skills.discovered", {
        "count": len(skills),
        "names": [s.metadata.name for s in skills],
    })

    if args.invoke:
        body = invoke_skill(args.invoke, skills)
        if body:
            print(body)
            return 0
        else:
            print(f"Skill not found: {args.invoke}", file=sys.stderr)
            return 1

    if args.validate:
        errors = validate_skills(skills)
        if args.json:
            print(json.dumps(errors, indent=2))
        else:
            if errors:
                for name, errs in errors.items():
                    print(f"[FAIL] {name}:")
                    for e in errs:
                        print(f"       - {e}")
                return 1
            else:
                print(f"[OK] All {len(skills)} skills valid")
        return 0 if not errors else 1

    if args.list or args.scan:
        if args.json:
            output = [s.to_dict() for s in skills]
            print(json.dumps(output, indent=2))
        else:
            print(f"Discovered {len(skills)} skills:\n")
            for skill in skills:
                print(f"  {skill.metadata.name}")
                print(f"    Description: {skill.metadata.description[:60]}...")
                print(f"    Path: {skill.path}")
                print(f"    Tokens: ~{skill.token_estimate}")
                print()
        return 0

    # Default: list
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
