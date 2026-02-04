#!/usr/bin/env python3
"""R&D Workflow Integration.

Connects aux_rd intake rhizomes with central Pluribus curation and tracking.
Provides workflow stages: drop → ingest → distill → track → verify.

Usage:
    python3 rd_workflow.py status             # Show workflow status
    python3 rd_workflow.py sync               # Sync rd_aux to main rhizome
    python3 rd_workflow.py promote <sha256>   # Promote artifact to main rhizome
    python3 rd_workflow.py track <task_file>  # Track task file in discourse
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def append_ndjson(path: Path, obj: dict) -> None:
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False, separators=(",", ":")) + "\n")


def iter_ndjson(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    yield json.loads(line)
                except Exception:
                    continue


def find_rhizome_root(start: Path) -> Path | None:
    cur = start.resolve()
    for cand in [cur, *cur.parents]:
        if (cand / ".pluribus" / "rhizome.json").exists():
            return cand
    return None


@dataclass
class WorkflowStatus:
    """Status of the R&D workflow."""
    rd_aux_path: str = ""
    main_rhizome_path: str = ""
    drop_count: int = 0
    discourse_count: int = 0
    task_files: list[str] = field(default_factory=list)
    pending_tasks: int = 0
    completed_tasks: int = 0
    eval_cards: int = 0
    ocr_files: int = 0
    last_sync_iso: str = ""


class RDWorkflow:
    """Manages R&D workflow integration."""

    def __init__(self, main_root: Path, rd_aux_root: Path | None = None):
        self.main_root = main_root
        self.main_pluribus = main_root / ".pluribus"

        # Find rd_aux if not specified
        if rd_aux_root:
            self.rd_aux_root = rd_aux_root
        else:
            # Look for aux_rd in common locations
            candidates = [
                main_root / "aux_rd",
                main_root.parent / "aux_rd",
            ]
            for cand in candidates:
                if (cand / ".pluribus").is_dir():
                    self.rd_aux_root = cand
                    break
            else:
                self.rd_aux_root = main_root / "aux_rd"

        self.rd_aux_pluribus = self.rd_aux_root / ".pluribus"
        self.sync_log = self.main_pluribus / "index" / "rd_sync_log.ndjson"

    def get_status(self) -> WorkflowStatus:
        """Get current workflow status."""
        status = WorkflowStatus(
            rd_aux_path=str(self.rd_aux_root),
            main_rhizome_path=str(self.main_root),
        )

        # Count drop files
        drop_dir = self.rd_aux_root / "drop"
        if drop_dir.is_dir():
            status.drop_count = sum(1 for f in drop_dir.iterdir() if f.is_file())

        # Count discourse files
        discourse_dir = self.rd_aux_root / "discourse"
        if discourse_dir.is_dir():
            status.discourse_count = sum(1 for f in discourse_dir.rglob("*.md"))

        # Count eval cards
        eval_cards_dir = discourse_dir / "eval_cards"
        if eval_cards_dir.is_dir():
            status.eval_cards = sum(1 for f in eval_cards_dir.glob("*.md") if f.name != "README.md")

        # Count OCR files
        ocr_dir = discourse_dir / "ocr"
        if ocr_dir.is_dir():
            status.ocr_files = sum(1 for f in ocr_dir.glob("*"))

        # Count task files and status
        tasks_dir = self.rd_aux_root / "tasks"
        if tasks_dir.is_dir():
            for task_file in tasks_dir.glob("*.md"):
                status.task_files.append(task_file.name)
                content = task_file.read_text(errors="replace")
                status.pending_tasks += content.count("- [ ]")
                status.completed_tasks += content.count("- [x]")

        # Get last sync time
        for obj in iter_ndjson(self.sync_log):
            if obj.get("kind") == "rd_sync":
                status.last_sync_iso = obj.get("iso", "")

        return status

    def sync_to_main(self, promote_all: bool = False) -> dict:
        """Sync rd_aux artifacts to main rhizome."""
        synced = []
        errors = []

        rd_aux_artifacts = self.rd_aux_pluribus / "index" / "artifacts.ndjson"
        main_artifacts = self.main_pluribus / "index" / "artifacts.ndjson"

        # Get already-synced sha256s
        synced_sha256s = set()
        for obj in iter_ndjson(self.sync_log):
            if obj.get("kind") == "rd_sync_artifact":
                synced_sha256s.add(obj.get("sha256", ""))

        # Get main rhizome sha256s
        main_sha256s = set()
        for obj in iter_ndjson(main_artifacts):
            if obj.get("kind") == "artifact":
                main_sha256s.add(obj.get("sha256", ""))

        # Find new artifacts to sync
        for obj in iter_ndjson(rd_aux_artifacts):
            if obj.get("kind") != "artifact":
                continue

            sha256 = obj.get("sha256", "")
            if not sha256 or sha256 in synced_sha256s or sha256 in main_sha256s:
                continue

            # Skip if not promoted and not promote_all
            tags = obj.get("tags", [])
            if not promote_all and "promoted" not in tags:
                continue

            try:
                # Copy object file if exists
                src_path = self.rd_aux_pluribus / "objects" / sha256[:2] / sha256[2:4] / sha256
                dst_path = self.main_pluribus / "objects" / sha256[:2] / sha256[2:4] / sha256

                if src_path.exists():
                    ensure_dir(dst_path.parent)
                    if not dst_path.exists():
                        dst_path.write_bytes(src_path.read_bytes())

                    # Add to main artifacts index
                    sync_record = {
                        **obj,
                        "synced_from": "rd_aux",
                        "synced_iso": now_iso_utc(),
                    }
                    append_ndjson(main_artifacts, sync_record)

                    # Log sync
                    append_ndjson(self.sync_log, {
                        "kind": "rd_sync_artifact",
                        "sha256": sha256,
                        "filename": obj.get("filename", ""),
                        "iso": now_iso_utc(),
                    })

                    synced.append(sha256[:12])

            except Exception as e:
                errors.append(f"{sha256[:12]}: {e}")

        # Log sync operation
        append_ndjson(self.sync_log, {
            "kind": "rd_sync",
            "synced_count": len(synced),
            "error_count": len(errors),
            "iso": now_iso_utc(),
        })

        return {
            "synced": synced,
            "errors": errors,
            "count": len(synced),
        }

    def promote_artifact(self, sha256_prefix: str) -> bool:
        """Promote an artifact from rd_aux to main rhizome."""
        rd_aux_artifacts = self.rd_aux_pluribus / "index" / "artifacts.ndjson"

        # Find matching artifact
        for obj in iter_ndjson(rd_aux_artifacts):
            if obj.get("kind") != "artifact":
                continue
            sha256 = obj.get("sha256", "")
            if sha256.startswith(sha256_prefix):
                # Add promoted tag
                tags = obj.get("tags", [])
                if "promoted" not in tags:
                    tags.append("promoted")
                    obj["tags"] = tags
                    obj["promoted_iso"] = now_iso_utc()

                    # Rewrite with promotion
                    all_artifacts = list(iter_ndjson(rd_aux_artifacts))
                    for i, a in enumerate(all_artifacts):
                        if a.get("sha256") == sha256:
                            all_artifacts[i] = obj
                            break

                    ensure_dir(rd_aux_artifacts.parent)
                    with rd_aux_artifacts.open("w", encoding="utf-8") as f:
                        for a in all_artifacts:
                            f.write(json.dumps(a, ensure_ascii=False, separators=(",", ":")) + "\n")

                # Sync to main
                result = self.sync_to_main(promote_all=False)
                return sha256[:12] in result.get("synced", [])

        return False

    def track_task_file(self, task_file: Path) -> dict:
        """Track a task file in the discourse workflow."""
        if not task_file.exists():
            return {"error": f"File not found: {task_file}"}

        content = task_file.read_text(errors="replace")

        # Parse tasks
        pending = []
        completed = []
        for line in content.split("\n"):
            line = line.strip()
            if line.startswith("- [ ]"):
                pending.append(line[5:].strip())
            elif line.startswith("- [x]"):
                completed.append(line[5:].strip())

        # Log tracking
        tracking_log = self.main_pluribus / "index" / "task_tracking.ndjson"
        append_ndjson(tracking_log, {
            "kind": "task_file_track",
            "file": str(task_file),
            "filename": task_file.name,
            "pending_count": len(pending),
            "completed_count": len(completed),
            "iso": now_iso_utc(),
        })

        return {
            "file": str(task_file),
            "pending": pending,
            "completed": completed,
            "pending_count": len(pending),
            "completed_count": len(completed),
        }

    def list_eval_cards(self) -> list[dict]:
        """List all eval cards in discourse."""
        cards = []
        eval_cards_dir = self.rd_aux_root / "discourse" / "eval_cards"

        if not eval_cards_dir.is_dir():
            return cards

        for card_file in sorted(eval_cards_dir.glob("*.md")):
            if card_file.name == "README.md":
                continue

            content = card_file.read_text(errors="replace")
            lines = content.split("\n")

            # Parse front matter or first heading
            name = card_file.stem
            status = "unknown"
            verdict = "unknown"

            for line in lines[:20]:
                if line.startswith("# "):
                    name = line[2:].strip()
                elif "verdict:" in line.lower():
                    verdict = line.split(":", 1)[1].strip()
                elif "status:" in line.lower():
                    status = line.split(":", 1)[1].strip()

            cards.append({
                "name": name,
                "file": card_file.name,
                "status": status,
                "verdict": verdict,
            })

        return cards


def cmd_status(args: argparse.Namespace) -> int:
    main_root = Path(args.root).expanduser().resolve() if args.root else (find_rhizome_root(Path.cwd()) or Path.cwd())
    rd_aux = Path(args.rd_aux).expanduser().resolve() if args.rd_aux else None

    workflow = RDWorkflow(main_root, rd_aux)
    status = workflow.get_status()

    print("R&D Workflow Status")
    print("=" * 40)
    print(f"Main Rhizome:    {status.main_rhizome_path}")
    print(f"R&D Aux:         {status.rd_aux_path}")
    print(f"Last Sync:       {status.last_sync_iso or 'never'}")
    print()
    print("Drop Zone:")
    print(f"  Files pending: {status.drop_count}")
    print()
    print("Discourse:")
    print(f"  Total files:   {status.discourse_count}")
    print(f"  Eval cards:    {status.eval_cards}")
    print(f"  OCR outputs:   {status.ocr_files}")
    print()
    print("Tasks:")
    print(f"  Task files:    {', '.join(status.task_files) or 'none'}")
    print(f"  Pending:       {status.pending_tasks}")
    print(f"  Completed:     {status.completed_tasks}")

    if args.json:
        print()
        print(json.dumps({
            "rd_aux_path": status.rd_aux_path,
            "main_rhizome_path": status.main_rhizome_path,
            "drop_count": status.drop_count,
            "discourse_count": status.discourse_count,
            "eval_cards": status.eval_cards,
            "pending_tasks": status.pending_tasks,
            "completed_tasks": status.completed_tasks,
            "last_sync_iso": status.last_sync_iso,
        }, indent=2))

    return 0


def cmd_sync(args: argparse.Namespace) -> int:
    main_root = Path(args.root).expanduser().resolve() if args.root else (find_rhizome_root(Path.cwd()) or Path.cwd())
    rd_aux = Path(args.rd_aux).expanduser().resolve() if args.rd_aux else None

    workflow = RDWorkflow(main_root, rd_aux)
    result = workflow.sync_to_main(promote_all=args.all)

    print(f"Synced {result['count']} artifacts")
    if result['synced']:
        for sha in result['synced']:
            print(f"  {sha}")
    if result['errors']:
        print(f"Errors: {len(result['errors'])}")
        for err in result['errors']:
            print(f"  {err}")

    return 0 if not result['errors'] else 1


def cmd_promote(args: argparse.Namespace) -> int:
    main_root = Path(args.root).expanduser().resolve() if args.root else (find_rhizome_root(Path.cwd()) or Path.cwd())
    rd_aux = Path(args.rd_aux).expanduser().resolve() if args.rd_aux else None

    workflow = RDWorkflow(main_root, rd_aux)
    success = workflow.promote_artifact(args.sha256)

    if success:
        print(f"Promoted {args.sha256}")
        return 0
    else:
        print(f"Failed to promote {args.sha256}")
        return 1


def cmd_track(args: argparse.Namespace) -> int:
    main_root = Path(args.root).expanduser().resolve() if args.root else (find_rhizome_root(Path.cwd()) or Path.cwd())
    rd_aux = Path(args.rd_aux).expanduser().resolve() if args.rd_aux else None

    workflow = RDWorkflow(main_root, rd_aux)
    task_file = Path(args.task_file).expanduser().resolve()

    result = workflow.track_task_file(task_file)

    if "error" in result:
        print(result["error"])
        return 1

    print(f"Tracked: {result['file']}")
    print(f"  Pending:   {result['pending_count']}")
    print(f"  Completed: {result['completed_count']}")

    if args.verbose:
        print("\nPending tasks:")
        for task in result['pending'][:10]:
            print(f"  - {task[:60]}")
        if len(result['pending']) > 10:
            print(f"  ... and {len(result['pending']) - 10} more")

    return 0


def cmd_cards(args: argparse.Namespace) -> int:
    main_root = Path(args.root).expanduser().resolve() if args.root else (find_rhizome_root(Path.cwd()) or Path.cwd())
    rd_aux = Path(args.rd_aux).expanduser().resolve() if args.rd_aux else None

    workflow = RDWorkflow(main_root, rd_aux)
    cards = workflow.list_eval_cards()

    print(f"Eval Cards ({len(cards)})")
    print("-" * 40)
    for card in cards:
        print(f"  {card['name']}: {card['verdict']}")

    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="R&D Workflow Integration")
    subparsers = parser.add_subparsers(dest="command")

    # Common arguments for all subcommands
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument("--root", help="Main Pluribus root directory")
    parent_parser.add_argument("--rd-aux", help="R&D Aux root directory")
    parent_parser.add_argument("--json", action="store_true", help="Output as JSON")

    # status
    status_parser = subparsers.add_parser("status", parents=[parent_parser], help="Show workflow status")

    # sync
    sync_parser = subparsers.add_parser("sync", parents=[parent_parser], help="Sync rd_aux to main rhizome")
    sync_parser.add_argument("--all", action="store_true", help="Sync all artifacts, not just promoted")

    # promote
    promote_parser = subparsers.add_parser("promote", parents=[parent_parser], help="Promote artifact to main rhizome")
    promote_parser.add_argument("sha256", help="SHA256 prefix of artifact to promote")

    # track
    track_parser = subparsers.add_parser("track", parents=[parent_parser], help="Track task file in discourse")
    track_parser.add_argument("task_file", help="Path to task file")
    track_parser.add_argument("-v", "--verbose", action="store_true", help="Show task details")

    # cards
    cards_parser = subparsers.add_parser("cards", parents=[parent_parser], help="List eval cards")

    args = parser.parse_args()

    if not args.command:
        # Default to status with no args
        args = parser.parse_args(["status"])

    commands = {
        "status": cmd_status,
        "sync": cmd_sync,
        "promote": cmd_promote,
        "track": cmd_track,
        "cards": cmd_cards,
    }

    return commands.get(args.command, cmd_status)(args)


if __name__ == "__main__":
    sys.exit(main())
