#!/usr/bin/env python3
"""
LSP Bridge - Language Server Protocol Integration for Pluribus
==============================================================

Bridges Language Server Protocol (LSP) features to any LLM via MCP tools.
Enables code intelligence (go-to-definition, find-references, hover) for all agents.

DKIN v25 compliant - exposes LSP operations as MCP-compatible tools.

Architecture:
- Spawns LSP servers for each supported language
- Translates LSP JSON-RPC to MCP tool responses
- Caches symbol information for performance
- Emits bus events for LSP operations

Supported Languages:
- TypeScript/JavaScript (via vtsls or typescript-language-server)
- Python (via pyright or pylsp)
- Rust (via rust-analyzer)
- Go (via gopls)

Usage:
    python3 lsp_bridge.py --start                      # Start LSP servers
    python3 lsp_bridge.py --goto-def FILE:LINE:COL     # Go to definition
    python3 lsp_bridge.py --find-refs FILE:LINE:COL    # Find references
    python3 lsp_bridge.py --hover FILE:LINE:COL        # Get hover info
    python3 lsp_bridge.py --diagnostics FILE           # Get diagnostics
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, asdict
from pathlib import Path
from threading import Thread
from typing import Any

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def emit_bus_event(topic: str, data: dict, level: str = "info") -> None:
    """Emit event to Pluribus bus."""
    bus_dir = os.environ.get("PLURIBUS_BUS_DIR", "").strip()
    if not bus_dir:
        # parents[2] = /pluribus (from nucleus/tools/lsp_bridge.py)
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
        "actor": os.environ.get("PLURIBUS_ACTOR", "lsp-bridge"),
        "data": data,
    }

    events_file = bus_path / "events.ndjson"
    with events_file.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False) + "\n")


@dataclass
class LSPServer:
    """LSP server configuration."""
    language: str
    command: list[str]
    file_patterns: list[str]
    process: subprocess.Popen | None = None
    request_id: int = 0

    def start(self, root_path: str) -> bool:
        """Start the LSP server."""
        try:
            self.process = subprocess.Popen(
                self.command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=root_path,
            )

            # Send initialize request
            init_request = {
                "jsonrpc": "2.0",
                "id": self._next_id(),
                "method": "initialize",
                "params": {
                    "processId": os.getpid(),
                    "rootUri": f"file://{root_path}",
                    "capabilities": {
                        "textDocument": {
                            "hover": {"contentFormat": ["markdown", "plaintext"]},
                            "definition": {"linkSupport": True},
                            "references": {},
                            "documentSymbol": {},
                            "publishDiagnostics": {},
                        }
                    },
                },
            }

            self._send(init_request)
            response = self._receive()

            if response and "result" in response:
                # Send initialized notification
                self._send({"jsonrpc": "2.0", "method": "initialized", "params": {}})
                emit_bus_event("lsp.server.started", {
                    "language": self.language,
                    "command": self.command[0],
                })
                return True

            return False

        except FileNotFoundError:
            emit_bus_event("lsp.server.missing", {
                "language": self.language,
                "command": self.command[0],
            }, level="warn")
            return False
        except Exception as e:
            emit_bus_event("lsp.server.error", {
                "language": self.language,
                "error": str(e),
            }, level="error")
            return False

    def stop(self) -> None:
        """Stop the LSP server."""
        if self.process:
            self._send({"jsonrpc": "2.0", "id": self._next_id(), "method": "shutdown", "params": None})
            self._send({"jsonrpc": "2.0", "method": "exit", "params": None})
            self.process.terminate()
            self.process = None

    def _next_id(self) -> int:
        self.request_id += 1
        return self.request_id

    def _send(self, message: dict) -> None:
        """Send JSON-RPC message to LSP server."""
        if not self.process or not self.process.stdin:
            return

        content = json.dumps(message)
        content_bytes = content.encode("utf-8")
        # LSP spec requires Content-Length in bytes, not characters
        header = f"Content-Length: {len(content_bytes)}\r\n\r\n"
        self.process.stdin.write(header.encode("utf-8"))
        self.process.stdin.write(content_bytes)
        self.process.stdin.flush()

    def _receive(self, timeout: float = 5.0) -> dict | None:
        """Receive JSON-RPC response from LSP server."""
        if not self.process or not self.process.stdout:
            return None

        try:
            # Read Content-Length header
            header = b""
            while b"\r\n\r\n" not in header:
                chunk = self.process.stdout.read(1)
                if not chunk:
                    return None
                header += chunk

            # Parse content length
            header_str = header.decode("utf-8")
            content_length = 0
            for line in header_str.split("\r\n"):
                if line.startswith("Content-Length:"):
                    content_length = int(line.split(":")[1].strip())
                    break

            if content_length == 0:
                return None

            # Read content
            content = self.process.stdout.read(content_length)
            return json.loads(content.decode("utf-8"))

        except Exception:
            return None

    def goto_definition(self, file_path: str, line: int, character: int) -> list[dict]:
        """Get definition location for symbol at position."""
        if not self.process:
            return []

        # Open document first
        self._open_document(file_path)

        request = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "textDocument/definition",
            "params": {
                "textDocument": {"uri": f"file://{file_path}"},
                "position": {"line": line - 1, "character": character - 1},  # LSP is 0-indexed
            },
        }

        self._send(request)
        response = self._receive()

        if response and "result" in response:
            result = response["result"]
            if isinstance(result, list):
                return [self._location_to_dict(loc) for loc in result]
            elif result:
                return [self._location_to_dict(result)]

        return []

    def find_references(self, file_path: str, line: int, character: int) -> list[dict]:
        """Find all references to symbol at position."""
        if not self.process:
            return []

        self._open_document(file_path)

        request = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "textDocument/references",
            "params": {
                "textDocument": {"uri": f"file://{file_path}"},
                "position": {"line": line - 1, "character": character - 1},
                "context": {"includeDeclaration": True},
            },
        }

        self._send(request)
        response = self._receive()

        if response and "result" in response and response["result"]:
            return [self._location_to_dict(loc) for loc in response["result"]]

        return []

    def hover(self, file_path: str, line: int, character: int) -> str | None:
        """Get hover information for symbol at position."""
        if not self.process:
            return None

        self._open_document(file_path)

        request = {
            "jsonrpc": "2.0",
            "id": self._next_id(),
            "method": "textDocument/hover",
            "params": {
                "textDocument": {"uri": f"file://{file_path}"},
                "position": {"line": line - 1, "character": character - 1},
            },
        }

        self._send(request)
        response = self._receive()

        if response and "result" in response and response["result"]:
            contents = response["result"].get("contents", "")
            if isinstance(contents, dict):
                return contents.get("value", str(contents))
            elif isinstance(contents, list):
                return "\n".join(c.get("value", str(c)) if isinstance(c, dict) else str(c) for c in contents)
            return str(contents)

        return None

    def get_diagnostics(self, file_path: str) -> list[dict]:
        """Get diagnostics for a file."""
        if not self.process:
            return []

        self._open_document(file_path)

        # Diagnostics are pushed via notifications, so we need to wait
        time.sleep(0.5)

        # For now, return empty - full implementation would track pushed diagnostics
        return []

    def _open_document(self, file_path: str) -> None:
        """Send textDocument/didOpen notification."""
        try:
            content = Path(file_path).read_text(encoding="utf-8")
        except Exception:
            content = ""

        # Determine language ID
        ext = Path(file_path).suffix.lower()
        lang_map = {
            ".py": "python",
            ".ts": "typescript",
            ".tsx": "typescriptreact",
            ".js": "javascript",
            ".jsx": "javascriptreact",
            ".rs": "rust",
            ".go": "go",
        }
        language_id = lang_map.get(ext, "plaintext")

        notification = {
            "jsonrpc": "2.0",
            "method": "textDocument/didOpen",
            "params": {
                "textDocument": {
                    "uri": f"file://{file_path}",
                    "languageId": language_id,
                    "version": 1,
                    "text": content,
                },
            },
        }

        self._send(notification)

    def _location_to_dict(self, location: dict) -> dict:
        """Convert LSP Location to simple dict."""
        uri = location.get("uri", location.get("targetUri", ""))
        file_path = uri.replace("file://", "")

        range_data = location.get("range", location.get("targetRange", {}))
        start = range_data.get("start", {})

        return {
            "file": file_path,
            "line": start.get("line", 0) + 1,  # Convert to 1-indexed
            "character": start.get("character", 0) + 1,
        }


# Predefined LSP server configurations
LSP_SERVERS: dict[str, LSPServer] = {
    "typescript": LSPServer(
        language="typescript",
        command=["typescript-language-server", "--stdio"],
        file_patterns=["*.ts", "*.tsx", "*.js", "*.jsx"],
    ),
    "python": LSPServer(
        language="python",
        command=["pyright-langserver", "--stdio"],
        file_patterns=["*.py"],
    ),
    "rust": LSPServer(
        language="rust",
        command=["rust-analyzer"],
        file_patterns=["*.rs"],
    ),
    "go": LSPServer(
        language="go",
        command=["gopls", "serve"],
        file_patterns=["*.go"],
    ),
}


class LSPBridge:
    """Bridge between LSP servers and MCP/Pluribus tools."""

    def __init__(self, root_path: str = "/pluribus"):
        self.root_path = root_path
        self.servers: dict[str, LSPServer] = {}

    def start_servers(self, languages: list[str] | None = None) -> dict[str, bool]:
        """Start LSP servers for specified languages."""
        if languages is None:
            languages = list(LSP_SERVERS.keys())

        results: dict[str, bool] = {}

        for lang in languages:
            if lang in LSP_SERVERS:
                server = LSPServer(
                    language=LSP_SERVERS[lang].language,
                    command=LSP_SERVERS[lang].command.copy(),
                    file_patterns=LSP_SERVERS[lang].file_patterns.copy(),
                )
                success = server.start(self.root_path)
                if success:
                    self.servers[lang] = server
                results[lang] = success

        return results

    def stop_servers(self) -> None:
        """Stop all LSP servers."""
        for server in self.servers.values():
            server.stop()
        self.servers.clear()

    def get_server_for_file(self, file_path: str) -> LSPServer | None:
        """Get appropriate LSP server for a file."""
        ext = Path(file_path).suffix.lower()

        for server in self.servers.values():
            for pattern in server.file_patterns:
                if pattern.endswith(ext) or (pattern == f"*{ext}"):
                    return server

        return None

    def goto_definition(self, file_path: str, line: int, character: int) -> list[dict]:
        """Go to definition via LSP."""
        server = self.get_server_for_file(file_path)
        if not server:
            return []

        results = server.goto_definition(file_path, line, character)

        emit_bus_event("lsp.operation.goto_definition", {
            "file": file_path,
            "line": line,
            "character": character,
            "results_count": len(results),
        })

        return results

    def find_references(self, file_path: str, line: int, character: int) -> list[dict]:
        """Find references via LSP."""
        server = self.get_server_for_file(file_path)
        if not server:
            return []

        results = server.find_references(file_path, line, character)

        emit_bus_event("lsp.operation.find_references", {
            "file": file_path,
            "line": line,
            "character": character,
            "results_count": len(results),
        })

        return results

    def hover(self, file_path: str, line: int, character: int) -> str | None:
        """Get hover info via LSP."""
        server = self.get_server_for_file(file_path)
        if not server:
            return None

        result = server.hover(file_path, line, character)

        emit_bus_event("lsp.operation.hover", {
            "file": file_path,
            "line": line,
            "character": character,
            "has_result": result is not None,
        })

        return result

    def diagnostics(self, file_path: str) -> list[dict]:
        """Get diagnostics via LSP."""
        server = self.get_server_for_file(file_path)
        if not server:
            return []

        return server.get_diagnostics(file_path)


def parse_position(pos_str: str) -> tuple[str, int, int]:
    """Parse FILE:LINE:COL string."""
    parts = pos_str.rsplit(":", 2)
    if len(parts) == 3:
        return parts[0], int(parts[1]), int(parts[2])
    elif len(parts) == 2:
        return parts[0], int(parts[1]), 1
    else:
        return pos_str, 1, 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Pluribus LSP Bridge")
    parser.add_argument("--start", action="store_true", help="Start LSP servers")
    parser.add_argument("--languages", nargs="*", help="Languages to start (default: all)")
    parser.add_argument("--root", type=str, default="/pluribus", help="Project root path")
    parser.add_argument("--goto-def", type=str, metavar="FILE:LINE:COL", help="Go to definition")
    parser.add_argument("--find-refs", type=str, metavar="FILE:LINE:COL", help="Find references")
    parser.add_argument("--hover", type=str, metavar="FILE:LINE:COL", help="Get hover info")
    parser.add_argument("--diagnostics", type=str, metavar="FILE", help="Get diagnostics")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--list-servers", action="store_true", help="List available LSP servers")

    args = parser.parse_args()

    if args.list_servers:
        print("Available LSP Servers:\n")
        for lang, server in LSP_SERVERS.items():
            print(f"  {lang}")
            print(f"    Command: {' '.join(server.command)}")
            print(f"    Patterns: {', '.join(server.file_patterns)}")
            print()
        return 0

    bridge = LSPBridge(args.root)

    if args.start:
        results = bridge.start_servers(args.languages)
        if args.json:
            print(json.dumps(results, indent=2))
        else:
            for lang, success in results.items():
                status = "OK" if success else "FAILED"
                print(f"  [{status}] {lang}")

        # Keep running
        print("\nLSP Bridge running. Press Ctrl+C to stop.")
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\nStopping...")
            bridge.stop_servers()
        return 0

    # For operations, start servers first
    bridge.start_servers(args.languages)

    try:
        if args.goto_def:
            file_path, line, char = parse_position(args.goto_def)
            results = bridge.goto_definition(file_path, line, char)
            if args.json:
                print(json.dumps(results, indent=2))
            else:
                for r in results:
                    print(f"{r['file']}:{r['line']}:{r['character']}")
            return 0 if results else 1

        if args.find_refs:
            file_path, line, char = parse_position(args.find_refs)
            results = bridge.find_references(file_path, line, char)
            if args.json:
                print(json.dumps(results, indent=2))
            else:
                for r in results:
                    print(f"{r['file']}:{r['line']}:{r['character']}")
            return 0 if results else 1

        if args.hover:
            file_path, line, char = parse_position(args.hover)
            result = bridge.hover(file_path, line, char)
            if result:
                print(result)
                return 0
            return 1

        if args.diagnostics:
            results = bridge.diagnostics(args.diagnostics)
            if args.json:
                print(json.dumps(results, indent=2))
            else:
                for d in results:
                    print(f"{d.get('severity', 'info')}: {d.get('message', '')}")
            return 0

    finally:
        bridge.stop_servers()

    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
