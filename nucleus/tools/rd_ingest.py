#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tarfile
import uuid
import zipfile
from pathlib import Path

sys.dont_write_bytecode = True

def _maybe_add_pluribus_venv_site() -> None:
    """Best-effort: allow optional deps installed into /pluribus/.pluribus/venv."""
    try:
        root = Path(__file__).resolve().parents[2]
        venv_site = root / ".pluribus" / "venv" / "lib" / f"python{sys.version_info.major}.{sys.version_info.minor}" / "site-packages"
        if venv_site.exists() and str(venv_site) not in sys.path:
            sys.path.insert(0, str(venv_site))
    except Exception:
        return


_maybe_add_pluribus_venv_site()

_UNSTRUCTURED_AVAILABLE = False
_UNSTRUCTURED_PARTITION = None
_UNSTRUCTURED_VERSION = None
try:
    from unstructured.partition.auto import partition as _partition  # type: ignore

    _UNSTRUCTURED_AVAILABLE = True
    _UNSTRUCTURED_PARTITION = _partition
    try:
        import importlib.metadata as _md

        _UNSTRUCTURED_VERSION = _md.version("unstructured")
    except Exception:
        _UNSTRUCTURED_VERSION = None
except Exception:
    _UNSTRUCTURED_AVAILABLE = False


def find_rhizome_root(start: Path) -> Path | None:
    cur = start.resolve()
    for cand in [cur, *cur.parents]:
        if (cand / ".pluribus" / "rhizome.json").exists():
            return cand
    return None


def tool_path(name: str) -> Path:
    return Path(__file__).with_name(name)


def run(argv: list[str]) -> int:
    return int(subprocess.run(argv, check=False).returncode)


def which(cmd: str) -> str | None:
    from shutil import which as _which

    return _which(cmd)


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def safe_extract_zip(src: Path, dst: Path) -> list[Path]:
    extracted: list[Path] = []
    dst_resolved = dst.resolve()
    with zipfile.ZipFile(src) as zf:
        for member in zf.infolist():
            name = member.filename
            if not name or name.endswith("/"):
                continue
            out = (dst_resolved / name).resolve()
            if not str(out).startswith(str(dst_resolved) + os.sep):
                continue
            ensure_dir(out.parent)
            with zf.open(member, "r") as r, open(out, "wb") as w:
                w.write(r.read())
            extracted.append(out)
    return extracted


def safe_extract_tar(src: Path, dst: Path) -> list[Path]:
    extracted: list[Path] = []
    dst_resolved = dst.resolve()
    with tarfile.open(src) as tf:
        for member in tf.getmembers():
            if not member.isfile():
                continue
            out = (dst_resolved / member.name).resolve()
            if not str(out).startswith(str(dst_resolved) + os.sep):
                continue
            ensure_dir(out.parent)
            f = tf.extractfile(member)
            if not f:
                continue
            with f, open(out, "wb") as w:
                w.write(f.read())
            extracted.append(out)
    return extracted


def collect_files(raw_paths: list[str]) -> list[Path]:
    out: list[Path] = []
    for raw in raw_paths:
        p = Path(raw).expanduser()
        if not p.exists():
            continue
        if p.is_file():
            out.append(p)
            continue
        for f in p.rglob("*"):
            if f.is_file():
                out.append(f)
    return out


def is_archive(p: Path) -> bool:
    name = p.name.lower()
    return name.endswith(".zip") or name.endswith(".tar") or name.endswith(".tgz") or name.endswith(".tar.gz") or name.endswith(".tar.bz2") or name.endswith(".tar.xz")


def is_image(p: Path) -> bool:
    return p.suffix.lower() in {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def is_pdf(p: Path) -> bool:
    return p.suffix.lower() == ".pdf"

def unstructured_extract_text(path: Path, *, strategy: str) -> str | None:
    if not _UNSTRUCTURED_AVAILABLE or _UNSTRUCTURED_PARTITION is None:
        return None
    try:
        elements = _UNSTRUCTURED_PARTITION(filename=str(path), strategy=strategy)
        texts: list[str] = []
        for el in elements or []:
            t = getattr(el, "text", None)
            if t:
                texts.append(str(t))
        out = "\n\n".join(texts).strip()
        return out or None
    except Exception:
        return None


def emit_bus(bus_dir: str | None, topic: str, kind: str, level: str, actor: str | None, data: dict) -> None:
    if not bus_dir:
        return
    agent_bus = tool_path("agent_bus.py")
    if not agent_bus.exists():
        return
    argv = [sys.executable, str(agent_bus), "--bus-dir", bus_dir, "pub", "--topic", topic, "--kind", kind, "--level", level, "--data", json.dumps(data, ensure_ascii=False)]
    if actor:
        argv += ["--actor", actor]
    _ = subprocess.run(argv, check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def cmd_ingest(args: argparse.Namespace) -> int:
    root = Path(args.root).expanduser().resolve() if args.root else (find_rhizome_root(Path.cwd()) or Path.cwd().resolve())
    rhizome_manifest = root / ".pluribus" / "rhizome.json"
    if not rhizome_manifest.exists():
        sys.stderr.write("no rhizome manifest found; run: rhizome.py init\n")
        return 2

    bus_dir = args.bus_dir or os.environ.get("PLURIBUS_BUS_DIR")
    actor = os.environ.get("PLURIBUS_ACTOR") or None
    tags = [t for t in (args.tag or []) if t.strip()]

    rhizome_py = tool_path("rhizome.py")
    curation_py = tool_path("curation.py")
    rag_py = tool_path("rag_index.py")

    stage_dir = Path(args.stage_dir).expanduser().resolve() if args.stage_dir else (root / ".pluribus" / "staging" / "rd_ingest" / args.run_id)
    ensure_dir(stage_dir)
    ensure_dir(stage_dir / "extracted")
    ensure_dir(stage_dir / "ocr")
    ensure_dir(stage_dir / "pdf_text")
    ensure_dir(stage_dir / "unstructured")

    if args.emit_bus:
        emit_bus(bus_dir, "rd_ingest.stage.created", "log", "info", actor, {"root": str(root), "stage_dir": str(stage_dir), "run_id": args.run_id})

    input_files = collect_files(args.paths)
    derived_paths: list[Path] = []

    if args.extract_archives:
        extracted: list[Path] = []
        for f in input_files:
            if not is_archive(f):
                continue
            dst = stage_dir / "extracted" / f.name
            ensure_dir(dst)
            try:
                if f.name.lower().endswith(".zip"):
                    extracted += safe_extract_zip(f, dst)
                else:
                    extracted += safe_extract_tar(f, dst)
            except Exception:
                continue
        derived_paths += extracted
        if args.emit_bus:
            emit_bus(bus_dir, "rd_ingest.archives.extracted", "artifact", "info", actor, {"count": len(extracted), "stage_dir": str(stage_dir / "extracted")})

    if getattr(args, "unstructured", False):
        if not _UNSTRUCTURED_AVAILABLE:
            if args.emit_bus:
                emit_bus(bus_dir, "rd_ingest.unstructured.missing", "log", "warn", actor, {"missing": ["unstructured"]})
        else:
            outs: list[Path] = []
            for f in input_files:
                if not (is_pdf(f) or is_image(f)):
                    continue
                strategy = str(getattr(args, "unstructured_strategy", "fast") or "fast")
                text = unstructured_extract_text(f, strategy=strategy)
                if not text:
                    continue
                out_txt = stage_dir / "unstructured" / f"{f.name}.txt"
                try:
                    out_txt.write_text(text, encoding="utf-8")
                    outs.append(out_txt)
                except Exception:
                    continue
            derived_paths += outs
            if args.emit_bus:
                emit_bus(
                    bus_dir,
                    "rd_ingest.unstructured.done",
                    "artifact",
                    "info",
                    actor,
                    {
                        "count": len(outs),
                        "stage_dir": str(stage_dir / "unstructured"),
                        "strategy": str(getattr(args, "unstructured_strategy", "fast") or "fast"),
                        "version": _UNSTRUCTURED_VERSION,
                    },
                )

    if args.ocr_images:
        if which("tesseract") is None:
            if args.emit_bus:
                emit_bus(bus_dir, "rd_ingest.ocr.missing", "log", "warn", actor, {"missing": ["tesseract"]})
        else:
            ocr_outs: list[Path] = []
            for f in input_files:
                if not is_image(f):
                    continue
                out_base = stage_dir / "ocr" / f"{f.name}"
                try:
                    rc = subprocess.run(["tesseract", str(f), str(out_base), "-l", "eng"], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
                    out_txt = Path(str(out_base) + ".txt")
                    if rc == 0 and out_txt.exists():
                        ocr_outs.append(out_txt)
                except Exception:
                    continue
            derived_paths += ocr_outs
            if args.emit_bus:
                emit_bus(bus_dir, "rd_ingest.ocr.done", "artifact", "info", actor, {"count": len(ocr_outs), "stage_dir": str(stage_dir / "ocr")})

    if args.pdf_text:
        if which("pdftotext") is None:
            if args.emit_bus:
                emit_bus(bus_dir, "rd_ingest.pdf_text.missing", "log", "warn", actor, {"missing": ["pdftotext"]})
        else:
            pdf_outs: list[Path] = []
            for f in input_files:
                if not is_pdf(f):
                    continue
                out_txt = stage_dir / "pdf_text" / f"{f.name}.txt"
                try:
                    rc = subprocess.run(["pdftotext", str(f), str(out_txt)], check=False, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode
                    if rc == 0 and out_txt.exists():
                        pdf_outs.append(out_txt)
                except Exception:
                    continue
            derived_paths += pdf_outs
            if args.emit_bus:
                emit_bus(bus_dir, "rd_ingest.pdf_text.done", "artifact", "info", actor, {"count": len(pdf_outs), "stage_dir": str(stage_dir / "pdf_text")})

    all_ingest_paths = [*args.paths, *[str(p) for p in derived_paths]]

    # 1) Ingest blobs into rhizome (content-addressed + append-only index).
    ingest_argv = [sys.executable, str(rhizome_py), "--root", str(root), "ingest", *all_ingest_paths, "--store"]
    for t in tags:
        ingest_argv += ["--tag", t]
    if args.emit_bus:
        ingest_argv += ["--emit-bus"]
        if bus_dir:
            ingest_argv += ["--bus-dir", bus_dir]
    rc = run(ingest_argv)
    if rc != 0:
        return rc

    # 2) Import any URLs found in the provided paths into a repo-local curation index.
    curation_index = str(root / ".pluribus" / "index" / "curation.ndjson")
    for raw in all_ingest_paths:
        p = Path(raw).expanduser()
        if not p.is_file():
            continue
        if p.suffix.lower() not in {".md", ".txt"}:
            continue
        argv = [sys.executable, str(curation_py), "--index", curation_index, "import-urls", "--from", str(p), "--dedupe", "--tags", ",".join(tags)]
        if args.emit_bus and bus_dir:
            argv += ["--emit-bus", "--bus-dir", bus_dir]
        _ = run(argv)

    # 3) Sync local memory (RAG) from rhizome artifacts.
    argv = [sys.executable, str(rag_py), "--root", str(root)]
    if bus_dir:
        argv += ["--bus-dir", bus_dir]
    argv += ["sync-from-rhizome"]
    _ = run(argv)

    sys.stdout.write("ok\n")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="rd_ingest.py", description="R&D ingest wrapper: rhizome ingest + URL curation + local RAG sync.")
    p.add_argument("--root", default=None, help="Rhizome root (defaults: search upward for .pluribus/rhizome.json).")
    p.add_argument("--bus-dir", default=None, help="Bus dir (or set PLURIBUS_BUS_DIR).")
    p.add_argument("--run-id", default=None, help="Correlation id for this ingest run (default: uuid4).")
    p.add_argument("--stage-dir", default=None, help="Derived-artifact staging dir (default: <root>/.pluribus/staging/rd_ingest/<run-id>).")
    p.add_argument("--emit-bus", action="store_true", help="Emit bus events where supported.")
    sub = p.add_subparsers(dest="cmd", required=True)

    ing = sub.add_parser("ingest", help="Ingest paths (files/dirs) and update local memory indices.")
    ing.add_argument("paths", nargs="+")
    ing.add_argument("--tag", action="append", default=["rd"])
    ing.add_argument("--extract-archives", action="store_true", help="Extract .zip/.tar* inputs into the stage dir and ingest extracted files.")
    ing.add_argument("--unstructured", action="store_true", help="Extract text from PDFs/images with Unstructured (if installed) into the stage dir and ingest outputs.")
    ing.add_argument("--unstructured-strategy", default="fast", choices=["fast", "hi_res", "ocr_only"], help="Unstructured strategy (fast|hi_res|ocr_only).")
    ing.add_argument("--ocr-images", action="store_true", help="OCR common image formats via `tesseract` into the stage dir and ingest OCR .txt outputs.")
    ing.add_argument("--pdf-text", action="store_true", help="Extract PDF text via `pdftotext` into the stage dir and ingest extracted .txt outputs.")
    ing.set_defaults(func=cmd_ingest)

    return p


def main(argv: list[str]) -> int:
    args = build_parser().parse_args(argv)
    args.run_id = (args.run_id or str(uuid.uuid4())).strip()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
