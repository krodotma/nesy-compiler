#!/usr/bin/env python3
"""
SOTA Ingest: Deep Multimodal Ingestion Pipeline
===============================================

Ingests documents, URLs, and blobs into the Pluribus "SOTA" knowledge base.
This is the "Mouth" of the system, consuming external signals and converting
them into normalized, vectorized, and graphed internal state.

Capabilities:
1.  **Format Agnostic**: Consumes text, PDF, images (OCR), archives (zip/tar), and URLs.
2.  **Semantic Transduction**: Extracts text, generates summaries, and vectorizes content.
3.  **Graph Connection**: Links new artifacts to existing SOTA nodes (claims/papers).
4.  **Append-Only**: Everything lands in `artifacts.ndjson` and `rhizome/objects/`.

Usage:
    python3 sota_ingest.py --url "https://arxiv.org/..."
    python3 sota_ingest.py --file "my_paper.pdf"
"""
import sys
import time
import json
import uuid
import subprocess
from pathlib import Path

sys.dont_write_bytecode = True

# Reuse existing tools
try:
    from rd_ingest import cmd_ingest, build_parser as rd_parser
    from curation import cmd_add, build_parser as cur_parser
    from rag_vector import VectorRAG, DB_PATH
except ImportError:
    # Fallback/Bootstrap mode
    sys.path.append(str(Path(__file__).resolve().parent))
    from rd_ingest import cmd_ingest, build_parser as rd_parser
    from curation import cmd_add, build_parser as cur_parser
    # Stub RAG if missing
    class VectorRAG:
        def __init__(self, *args, **kwargs): pass
        def index_event(self, *args): pass
    DB_PATH = "/tmp/rag.sqlite3"

def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())

import re

class SotaIngestor:
    def __init__(self, root: Path):
        self.root = root
        self.rag = VectorRAG(Path(DB_PATH))

    def _clean_vtt(self, vtt_path: Path) -> str:
        """Clean WebVTT subtitles into plain text."""
        lines = []
        try:
            with open(vtt_path, 'r', encoding='utf-8') as f:
                for line in f:
                    # Skip timeline markers, headers, and empty lines
                    if '-->' in line or line.strip() == 'WEBVTT' or not line.strip():
                        continue
                    # Remove tags like <c>...</c>
                    text = re.sub(r'<[^>]+>', '', line.strip())
                    # Dedup consecutive lines (common in auto-caps)
                    if lines and lines[-1] == text:
                        continue
                    if text:
                        lines.append(text)
        except Exception:
            return ""
        return "\n".join(lines)

    def download_youtube_transcript(self, url: str) -> tuple[Path | None, dict]:
        """Download transcript and metadata using yt-dlp."""
        staging = self.root / ".pluribus" / "staging" / "sota_yt" / str(uuid.uuid4())
        ensure_dir(staging)
        
        # 1. Fetch Metadata
        meta_cmd = ["yt-dlp", "--dump-json", "--skip-download", url]
        meta = {}
        try:
            res = subprocess.run(meta_cmd, capture_output=True, text=True, check=True)
            meta = json.loads(res.stdout)
        except Exception as e:
            print(f"[SOTA] YouTube metadata fetch failed: {e}")
            return None, {}

        # 2. Download Subtitles
        # We prefer manual subs, fallback to auto
        out_tmpl = str(staging / "video")
        sub_cmd = [
            "yt-dlp", 
            "--write-sub", "--write-auto-sub", 
            "--sub-lang", "en", 
            "--skip-download", 
            "--output", out_tmpl,
            url
        ]
        try:
            subprocess.run(sub_cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"[SOTA] YouTube subtitle download failed: {e}")
            return None, meta

        # 3. Find and Clean Subtitle File
        # yt-dlp might produce video.en.vtt, video.en.ttml, etc.
        vtt_file = None
        for cand in staging.glob("video.*.vtt"):
            vtt_file = cand
            break
        
        if not vtt_file:
             print("[SOTA] No VTT subtitle file found.")
             return None, meta

        clean_text = self._clean_vtt(vtt_file)
        txt_path = staging / f"{meta.get('id', 'video')}_transcript.txt"
        txt_path.write_text(clean_text, encoding='utf-8')
        
        return txt_path, meta

    def ingest_github_repo(self, url: str) -> tuple[Path | None, dict]:
        """Deep ingest a GitHub repo (Structure + README)."""
        staging = self.root / ".pluribus" / "staging" / "sota_git" / str(uuid.uuid4())
        ensure_dir(staging)
        
        repo_name = url.split("/")[-1].replace(".git", "")
        clone_dir = staging / repo_name
        
        # 1. Shallow Clone
        try:
            subprocess.run(["git", "clone", "--depth", "1", url, str(clone_dir)], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception as e:
            print(f"[SOTA] Git clone failed: {e}")
            return None, {}

        # 2. Analyze Structure
        try:
            structure = subprocess.check_output(["find", ".", "-maxdepth", "3", "-not", "-path", "*/.*"], cwd=clone_dir, text=True)
        except:
            structure = "(find failed)"

        # 3. Read Readme
        readme = ""
        for cand in ["README.md", "README.txt", "readme.md"]:
            p = clone_dir / cand
            if p.exists():
                readme = p.read_text(errors="replace")
                break
        
        # 4. Synthesize Summary
        summary_text = f"Repo: {url}\n\n# Structure\n```\n{structure}\n```\n\n# README\n{readme}"
        out_path = staging / f"{repo_name}_summary.txt"
        out_path.write_text(summary_text, encoding="utf-8")
        
        return out_path, {"title": repo_name, "kind": "code"}

    # --- Multimodal Transducers (Semiotialysis) ---

    def _sextet_transduce(self, source_kind: str, target_kind: str, description: str) -> dict:
        """Log a Sextet Transduction event (Signal -> Symbol)."""
        return {
            "axiom": "sextet",
            "role": "transducer",
            "transformation": f"{source_kind} -> {target_kind}",
            "semantics": description,
            "iso": now_iso()
        }

    def extract_video_keyframes(self, video_path: Path) -> list[Path]:
        """Stub: Extract keyframes from video using ffmpeg."""
        # In a real implementation: ffmpeg -i video.mp4 -vf "select=eq(pict_type\,I)" -vsync vfr frame%d.png
        print(f"[SOTA] Transducing Video -> Image Sequence: {video_path}")
        self._sextet_transduce("video", "image_sequence", "Keyframe extraction")
        return []

    def transcribe_audio(self, audio_path: Path) -> Path | None:
        """Stub: Transcribe audio using Whisper."""
        print(f"[SOTA] Transducing Audio -> Text: {audio_path}")
        self._sextet_transduce("audio", "text", "Speech-to-Text transcription")
        return None

    def ingest_sensor_stream(self, path: Path, sensor_type: str = "generic") -> Path | None:
        """
        Semiotialysis: Transduce raw sensor signal (CSV/Binary) into Symbolic Description.
        """
        print(f"[SOTA] Transducing Sensor({sensor_type}) -> Symbol: {path}")
        
        # 1. Analyze Signal Stats (The "Semiosis")
        stats = {"min": 0, "max": 0, "mean": 0}
        try:
            if path.suffix == ".csv":
                import csv
                with open(path, 'r') as f:
                    reader = csv.reader(f)
                    next(reader, None)
                    values = []
                    for row in reader:
                        if row: values.append(float(row[1])) # timestamp,value
                    if values:
                        stats["min"] = min(values)
                        stats["max"] = max(values)
                        stats["mean"] = sum(values)/len(values)
        except Exception:
            pass

        # 2. Generate Symbolic Description
        description = f"Sensor: {sensor_type}\nSource: {path.name}\n"
        description += f"Signal Profile: Range[{stats['min']:.2f}, {stats['max']:.2f}] Mean:{stats['mean']:.2f}\n"
        
        if sensor_type == "lidar":
            description += "Interpretation: Point Cloud Density analysis required.\n"

        # 3. Write Artifact
        out_path = path.with_suffix(".semiosis.txt")
        out_path.write_text(description, encoding="utf-8")
        
        self._sextet_transduce(sensor_type, "symbolic_text", description)
        return out_path

    def ingest_url(self, url: str, tags: list[str] = []):
        """Ingest a URL (Paper/Code/Video)."""
        print(f"[SOTA] Ingesting URL: {url}")
        
        item_id = str(uuid.uuid4())
        title = url
        local_path = None
        
        # Deepening: YouTube Handling
        if "youtube.com" in url or "youtu.be" in url:
            print("[SOTA] Detected YouTube URL. Attempting transcript fetch...")
            path, meta = self.download_youtube_transcript(url)
            if path:
                title = meta.get('title', title)
                print(f"[SOTA] Transcript acquired: {title}")
                # Treat the transcript as a file ingest
                self.ingest_file(path, tags=["yt", "transcript"] + tags)
                local_path = str(path)

        # Deepening: GitHub Handling
        elif "github.com" in url:
            print("[SOTA] Detected GitHub URL. Attempting shallow clone & analysis...")
            path, meta = self.ingest_github_repo(url)
            if path:
                title = meta.get('title', title)
                print(f"[SOTA] Repo analyzed: {title}")
                self.ingest_file(path, tags=["code", "repo", "structure"] + tags)
                local_path = str(path)
        
        # 1. Add to Curation Index (Metadata)
        # We wrap the curation tool's logic manually here to ensure we pass the right metadata
        curation_item = {
            "id": item_id,
            "ts": time.time(),
            "iso": now_iso(),
            "kind": "yt" if local_path else "url",
            "url": url,
            "title": title,
            "tags": tags,
            "provenance": {"added_by": "sota-ingest", "local_path": local_path}
        }
        
        # Using subprocess to call curation.py is cleaner than importing its internal logic
        # given the complexity of its CLI parser
        cur_tool = Path(__file__).with_name("curation.py")
        subprocess.run([
            sys.executable, str(cur_tool), 
            "--index", str(self.root / ".pluribus/index/curation.ndjson"),
            "add", 
            "--kind", curation_item["kind"],
            "--title", curation_item["title"],
            "--url", url,
            "--tags", ",".join(tags)
        ], check=False, stdout=subprocess.DEVNULL)
        
        print(f"[SOTA] URL {url} ingested as {item_id}")
        return item_id

    def ingest_file(self, path: Path, tags: list[str] = []):
        """Ingest a local file (PDF/Text/Image/Audio/Sensor)."""
        print(f"[SOTA] Ingesting File: {path}")
        
        # Multimodal Routing
        suffix = path.suffix.lower()
        if suffix in {".mp3", ".wav", ".m4a"}:
            txt_path = self.transcribe_audio(path)
            if txt_path: path = txt_path
        elif suffix in {".mp4", ".mov", ".avi"}:
            # Video: Extract frames + audio
            _ = self.extract_video_keyframes(path)
            txt_path = self.transcribe_audio(path) # Transcribe video audio
            if txt_path: path = txt_path
        elif suffix in {".csv", ".json", ".pcd"} and ("sensor" in str(path) or "lidar" in str(path) or "imu" in str(path)):
            # Heuristic sensor detection
            sensor_type = "lidar" if "lidar" in str(path) else "generic"
            sym_path = self.ingest_sensor_stream(path, sensor_type)
            if sym_path: path = sym_path

        # 1. Use rd_ingest to handle file/archive/ocr logic
        # We construct args for the existing tool function
        # This reuses the powerful 'rd_ingest.py' we saw earlier
        parser = rd_parser()
        # Mock args
        args = parser.parse_args(["ingest", str(path), "--tag", "sota"] + [f"--tag={t}" for t in tags])
        args.root = str(self.root)
        args.emit_bus = True
        
        # Execute ingestion
        try:
            cmd_ingest(args)
            print(f"[SOTA] File {path} ingested successfully via rd_ingest.")
        except Exception as e:
            print(f"[SOTA] Ingestion failed: {e}")

    def ingest_batch(self, targets: list[str]):
        """Smart router for mixed targets."""
        for target in targets:
            if target.startswith("http"):
                self.ingest_url(target)
            else:
                self.ingest_file(Path(target))

def main():
    import argparse
    parser = argparse.ArgumentParser(description="SOTA Deep Ingest")
    parser.add_argument("--root", default="/pluribus")
    parser.add_argument("targets", nargs="+", help="URLs or Files to ingest")
    args = parser.parse_args()
    
    ingestor = SotaIngestor(Path(args.root))
    ingestor.ingest_batch(args.targets)

if __name__ == "__main__":
    main()