#!/usr/bin/env python3
"""
skills_scanner.py - Local skill discovery/invocation fallback.

Usage:
  python3 skills_scanner.py --scan
  python3 skills_scanner.py --list
  python3 skills_scanner.py --invoke <skill-name>
  python3 skills_scanner.py --validate
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
from typing import Any, Iterable, List

sys.dont_write_bytecode = True

try:
    import yaml  # type: ignore
except Exception:
    yaml = None


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def emit_bus_event(topic: str, data: dict, level: str = "info") -> None:
    bus_dir = os.environ.get("PLURIBUS_BUS_DIR", "").strip() or "/pluribus/.pluribus/bus"
    bus_path = Path(bus_dir)
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
    with events_file.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event, ensure_ascii=False) + "\n")


@dataclass
class SkillMetadata:
    name: str
    description: str
    license: str | None = None
    author: str | None = None
    version: str | None = None
    tags: list[str] | None = None

    def validate(self) -> list[str]:
        errors: list[str] = []
        if not self.name:
            errors.append("name is required")
        elif len(self.name) > 64:
            errors.append(f"name exceeds 64 chars: {len(self.name)}")
        elif not re.match(r"^[a-z0-9-]+$", self.name):
            errors.append(f"name must be lowercase alphanumeric with hyphens: {self.name}")
        if not self.description:
            errors.append("description is required")
        elif len(self.description) > 1024:
            errors.append(f"description exceeds 1024 chars: {len(self.description)}")
        return errors


@dataclass
class Skill:
    metadata: SkillMetadata
    body: str
    path: Path
    token_estimate: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "metadata": asdict(self.metadata),
            "body_preview": self.body[:200] + ("..." if len(self.body) > 200 else ""),
            "path": str(self.path),
            "token_estimate": self.token_estimate,
        }


def parse_skill_md(path: Path) -> Skill | None:
    try:
        content = path.read_text(encoding="utf-8")
    except Exception as exc:
        print(f"[WARN] Cannot read {path}: {exc}", file=sys.stderr)
        return None

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
                    frontmatter = {}
            else:
                for line in fm_text.split("\n"):
                    if ":" in line:
                        key, value = line.split(":", 1)
                        frontmatter[key.strip()] = value.strip()

    metadata_dict = frontmatter.get("metadata", {}) if isinstance(frontmatter, dict) else {}
    if isinstance(metadata_dict, str):
        metadata_dict = {}

    metadata = SkillMetadata(
        name=frontmatter.get("name", path.parent.name) if isinstance(frontmatter, dict) else path.parent.name,
        description=frontmatter.get("description", "") if isinstance(frontmatter, dict) else "",
        license=frontmatter.get("license") if isinstance(frontmatter, dict) else None,
        author=metadata_dict.get("author") if isinstance(metadata_dict, dict) else None,
        version=metadata_dict.get("version") if isinstance(metadata_dict, dict) else None,
        tags=frontmatter.get("tags") if isinstance(frontmatter, dict) and isinstance(frontmatter.get("tags"), list) else None,
    )

    token_estimate = max(1, len(body) // 4)
    return Skill(metadata=metadata, body=body, path=path, token_estimate=token_estimate)


def _extend_roots(roots: list[Path], extra: Iterable[Path]) -> None:
    for path in extra:
        if path not in roots:
            roots.append(path)


def _agent_home_roots() -> list[Path]:
    roots: list[Path] = []
    agent_homes = Path("/pluribus/.pluribus/agent_homes")
    if not agent_homes.exists():
        return roots
    for home in agent_homes.iterdir():
        if not home.is_dir():
            continue
        _extend_roots(
            roots,
            [
                home / ".codex" / "skills",
                home / ".claude" / "skills",
                home / ".gemini" / "skills",
                home / ".qwen" / "skills",
            ],
        )
    return roots


def _default_roots() -> list[Path]:
    roots = [
        Path("/pluribus/.pluribus/skills"),
        Path("/pluribus/.agents/skills"),
        Path("/pluribus/membrane/oss-skills"),
    ]
    _extend_roots(roots, _agent_home_roots())
    extra = os.environ.get("PLURIBUS_SKILLS_ROOTS", "").strip()
    if extra:
        for part in extra.split(":"):
            if part.strip():
                roots.append(Path(part.strip()))
    return roots


def discover_skills() -> list[Skill]:
    skills: list[Skill] = []
    for root in _default_roots():
        if not root.exists():
            continue
        for path in root.rglob("SKILL.md"):
            skill = parse_skill_md(path)
            if skill:
                skills.append(skill)
    return skills


def _match_skill(skills: list[Skill], name: str) -> Skill | None:
    exact = [s for s in skills if s.metadata.name == name]
    if exact:
        return exact[0]
    lower = name.lower()
    for skill in skills:
        if skill.metadata.name.lower() == lower:
            return skill
    for skill in skills:
        if skill.path.parent.name.lower() == lower:
            return skill
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description="Local skills scanner")
    ap.add_argument("--scan", action="store_true")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--invoke", type=str, default="")
    ap.add_argument("--validate", action="store_true")
    args = ap.parse_args()

    skills = discover_skills()

    if args.scan:
        payload = {"count": len(skills), "skills": [s.to_dict() for s in skills]}
        emit_bus_event("skills.scan", {"count": len(skills)})
        print(json.dumps(payload, indent=2))
        return 0

    if args.list:
        for skill in skills:
            print(f"{skill.metadata.name}\t{skill.path}")
        emit_bus_event("skills.list", {"count": len(skills)})
        return 0

    if args.invoke:
        skill = _match_skill(skills, args.invoke)
        if not skill:
            print(f"[ERROR] Skill not found: {args.invoke}", file=sys.stderr)
            emit_bus_event("skills.invoke.missing", {"name": args.invoke}, level="warn")
            return 1
        emit_bus_event("skills.invoke", {"name": skill.metadata.name, "path": str(skill.path)})
        print(skill.body)
        return 0

    if args.validate:
        errors: list[str] = []
        for skill in skills:
            errs = skill.metadata.validate()
            for err in errs:
                errors.append(f"{skill.path}: {err}")
        if errors:
            print("\n".join(errors), file=sys.stderr)
            emit_bus_event("skills.validate.fail", {"count": len(errors)}, level="warn")
            return 2
        emit_bus_event("skills.validate.pass", {"count": len(skills)})
        print(f"ok ({len(skills)} skills)")
        return 0

    ap.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
