#!/usr/bin/env python3
"""
cloud_storage_daemon.py - Cloud Storage OAuth & Mount Manager

Manages OAuth flows for Google Drive (and other providers) using the VNC
browser session, captures tokens, configures rclone, and maintains FUSE mounts.

ARCHITECTURE:
  1. Dashboard requests OAuth for a provider/account
  2. This daemon navigates VNC browser to OAuth URL
  3. Local redirect server captures the auth code
  4. Exchange code for tokens, configure rclone
  5. Mount via rclone mount, expose via /mnt/gdrive/<account>

API ENDPOINTS (port 9400):
  GET  /status              - List configured remotes and mount status
  POST /auth/start          - Start OAuth flow for provider/account
  GET  /auth/callback       - OAuth redirect handler (internal)
  POST /mount/<remote>      - Mount a configured remote
  POST /unmount/<remote>    - Unmount a remote
  GET  /browse/<remote>     - List files in remote

BUS EVENTS:
  - cloud.auth.start    {provider, account}
  - cloud.auth.complete {provider, account, success}
  - cloud.mount.up      {remote, mountpoint}
  - cloud.mount.down    {remote}

Author: Claude Opus (Pluribus Core Team)
"""

import os
import sys
import json
import asyncio
import subprocess
import threading
import secrets
import hashlib
import base64
import urllib.parse
import time
from pathlib import Path
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from typing import Optional
import socket

# Best-effort import for structured bus emission.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
try:
    from nucleus.tools import agent_bus  # type: ignore
except Exception:  # pragma: no cover
    agent_bus = None  # type: ignore

# Flask for API (lightweight)
try:
    from flask import Flask, jsonify, request
    HAS_FLASK = True
except ImportError:
    HAS_FLASK = False

# Paths
MOUNT_BASE = Path("/mnt/gdrive")
RCLONE_CONFIG = Path.home() / ".config/rclone/rclone.conf"
BUS_DIR = Path(os.environ.get("PLURIBUS_BUS_DIR", "/pluribus/.pluribus/bus"))
EVENTS_FILE = BUS_DIR / "events.ndjson"

# OAuth Config
OAUTH_REDIRECT_PORT = 53682
GOOGLE_CLIENT_ID = "202264815644.apps.googleusercontent.com"
GOOGLE_CLIENT_SECRET = "X4Z3ca8xfWDb1Voo-F9a7ZxJ"  # rclone's public secret

# Provider configs
PROVIDERS = {
    "google_drive": {
        "name": "Google Drive",
        "auth_url": "https://accounts.google.com/o/oauth2/auth",
        "token_url": "https://oauth2.googleapis.com/token",
        "client_id": GOOGLE_CLIENT_ID,
        "client_secret": GOOGLE_CLIENT_SECRET,
        "scopes": ["https://www.googleapis.com/auth/drive"],
        "rclone_type": "drive",
    },
}

# Account configurations
ACCOUNTS = {
    "peter_herz": {"email": "peter.herz@gmail.com", "provider": "google_drive"},
    "peter_kroma": {"email": "peter@kro.ma", "provider": "google_drive"},
    "peter_tachy0n": {"email": "peter@tachy0n.com", "provider": "google_drive"},
}


def emit_event(topic: str, payload: dict) -> None:
    """Emit event to Pluribus bus."""
    try:
        if agent_bus is not None:
            paths = agent_bus.resolve_bus_paths(str(BUS_DIR))
            agent_bus.emit_event(
                paths,
                topic=topic,
                kind="metric",
                level="info",
                actor="cloud-storage",
                data=payload,
                trace_id=None,
                run_id=None,
                durable=False,
            )
            return

        BUS_DIR.mkdir(parents=True, exist_ok=True)
        event = {
            "id": secrets.token_urlsafe(8),
            "ts": time.time(),
            "iso": datetime.utcnow().isoformat() + "Z",
            "topic": topic,
            "kind": "metric",
            "level": "info",
            "actor": "cloud-storage",
            "data": payload,
        }
        with open(EVENTS_FILE, "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
    except Exception as e:
        print(f"[bus] Failed to emit {topic}: {e}")


def open_in_vnc_firefox(url: str) -> bool:
    """Open URL in VNC Firefox browser.

    Uses DISPLAY=:1 to target the VNC X server.
    Opens in a new tab if Firefox is already running.
    """
    try:
        env = os.environ.copy()
        env["DISPLAY"] = ":1"
        subprocess.Popen(
            ["firefox", "--new-tab", url],
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL
        )
        print(f"[vnc] Opened URL in VNC Firefox: {url[:60]}...")
        return True
    except Exception as e:
        print(f"[vnc] Failed to open Firefox: {e}")
        return False


class OAuthState:
    """Tracks pending OAuth flows."""
    pending: dict = {}  # state -> {account, verifier, provider}


class CloudStorageManager:
    """Manages cloud storage connections and mounts."""

    def __init__(self):
        self.oauth_state = OAuthState()
        self.mounts: dict[str, dict] = {}  # remote -> {mountpoint, pid}
        MOUNT_BASE.mkdir(parents=True, exist_ok=True)
        self._load_existing_mounts()

    def _load_existing_mounts(self):
        """Check for existing rclone mounts."""
        try:
            result = subprocess.run(
                ["mount", "-t", "fuse.rclone"],
                capture_output=True, text=True
            )
            for line in result.stdout.strip().split("\n"):
                if line:
                    # Parse: remote: on /mnt/gdrive/x type fuse.rclone
                    parts = line.split()
                    if len(parts) >= 3:
                        remote = parts[0].rstrip(":")
                        mountpoint = parts[2]
                        self.mounts[remote] = {"mountpoint": mountpoint, "pid": None}
        except Exception as e:
            print(f"[mount] Error loading existing mounts: {e}")

    def get_status(self) -> dict:
        """Get status of all configured remotes and mounts."""
        remotes = self._list_remotes()

        status = {
            "configured_accounts": list(ACCOUNTS.keys()),
            "remotes": {},
            "mounts": {},
            "pending_auth": list(self.oauth_state.pending.keys()),
        }

        for remote in remotes:
            status["remotes"][remote] = {
                "configured": True,
                "mounted": remote in self.mounts,
                "mountpoint": self.mounts.get(remote, {}).get("mountpoint"),
            }

        for account in ACCOUNTS:
            if account not in remotes:
                status["remotes"][account] = {
                    "configured": False,
                    "mounted": False,
                    "account_email": ACCOUNTS[account]["email"],
                }

        return status

    def _list_remotes(self) -> list[str]:
        """List configured rclone remotes."""
        try:
            result = subprocess.run(
                ["rclone", "listremotes"],
                capture_output=True, text=True
            )
            return [r.rstrip(":") for r in result.stdout.strip().split("\n") if r]
        except Exception:
            return []

    def generate_auth_url(self, account: str) -> dict:
        """Generate OAuth URL for an account."""
        if account not in ACCOUNTS:
            return {"error": f"Unknown account: {account}"}

        account_config = ACCOUNTS[account]
        provider = PROVIDERS[account_config["provider"]]

        # Generate PKCE
        code_verifier = secrets.token_urlsafe(32)
        code_challenge = base64.urlsafe_b64encode(
            hashlib.sha256(code_verifier.encode()).digest()
        ).decode().rstrip('=')
        state = secrets.token_urlsafe(16)

        # Store state
        self.oauth_state.pending[state] = {
            "account": account,
            "verifier": code_verifier,
            "provider": account_config["provider"],
        }

        params = {
            "client_id": provider["client_id"],
            "redirect_uri": f"http://localhost:{OAUTH_REDIRECT_PORT}/",
            "response_type": "code",
            "scope": " ".join(provider["scopes"]),
            "access_type": "offline",
            "state": state,
            "code_challenge": code_challenge,
            "code_challenge_method": "S256",
            "prompt": "consent",
            "login_hint": account_config["email"],
        }

        auth_url = f"{provider['auth_url']}?{urllib.parse.urlencode(params)}"

        emit_event("cloud.auth.start", {
            "account": account,
            "email": account_config["email"],
            "provider": account_config["provider"],
        })

        return {
            "auth_url": auth_url,
            "state": state,
            "account": account,
            "email": account_config["email"],
        }

    def handle_oauth_callback(self, code: str, state: str) -> dict:
        """Handle OAuth callback, exchange code for tokens."""
        if state not in self.oauth_state.pending:
            return {"error": "Invalid or expired state"}

        pending = self.oauth_state.pending.pop(state)
        account = pending["account"]
        verifier = pending["verifier"]
        provider_name = pending["provider"]
        provider = PROVIDERS[provider_name]

        # Exchange code for tokens
        token_data = {
            "client_id": provider["client_id"],
            "client_secret": provider["client_secret"],
            "code": code,
            "code_verifier": verifier,
            "grant_type": "authorization_code",
            "redirect_uri": f"http://localhost:{OAUTH_REDIRECT_PORT}/",
        }

        try:
            import urllib.request
            req = urllib.request.Request(
                provider["token_url"],
                data=urllib.parse.urlencode(token_data).encode(),
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                tokens = json.loads(resp.read().decode())
        except Exception as e:
            emit_event("cloud.auth.complete", {
                "account": account,
                "success": False,
                "error": str(e),
            })
            return {"error": f"Token exchange failed: {e}"}

        # Configure rclone
        token_json = json.dumps({
            "access_token": tokens.get("access_token"),
            "token_type": tokens.get("token_type", "Bearer"),
            "refresh_token": tokens.get("refresh_token"),
            "expiry": tokens.get("expiry", ""),
        })

        try:
            subprocess.run([
                "rclone", "config", "create", account, provider["rclone_type"],
                "scope", "drive",
                "token", token_json,
            ], check=True, capture_output=True)
        except subprocess.CalledProcessError as e:
            return {"error": f"rclone config failed: {e.stderr.decode()}"}

        emit_event("cloud.auth.complete", {
            "account": account,
            "success": True,
            "email": ACCOUNTS[account]["email"],
        })

        # Auto-mount
        self.mount_remote(account)

        return {
            "success": True,
            "account": account,
            "message": f"Successfully configured {account}",
        }

    def mount_remote(self, remote: str) -> dict:
        """Mount a remote using rclone mount."""
        mountpoint = MOUNT_BASE / remote
        mountpoint.mkdir(parents=True, exist_ok=True)

        # Check if already mounted
        if remote in self.mounts:
            return {"status": "already_mounted", "mountpoint": str(mountpoint)}

        try:
            # Start rclone mount in background
            proc = subprocess.Popen([
                "rclone", "mount",
                f"{remote}:",
                str(mountpoint),
                "--vfs-cache-mode", "writes",
                "--allow-other",
                "--daemon",
            ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

            self.mounts[remote] = {
                "mountpoint": str(mountpoint),
                "pid": proc.pid,
            }

            emit_event("cloud.mount.up", {
                "remote": remote,
                "mountpoint": str(mountpoint),
            })

            return {
                "status": "mounted",
                "remote": remote,
                "mountpoint": str(mountpoint),
            }
        except Exception as e:
            return {"error": str(e)}

    def unmount_remote(self, remote: str) -> dict:
        """Unmount a remote."""
        if remote not in self.mounts:
            return {"status": "not_mounted"}

        mountpoint = self.mounts[remote]["mountpoint"]

        try:
            subprocess.run(["fusermount", "-u", mountpoint], check=True)
            del self.mounts[remote]

            emit_event("cloud.mount.down", {"remote": remote})

            return {"status": "unmounted", "remote": remote}
        except Exception as e:
            return {"error": str(e)}

    def browse_remote(self, remote: str, path: str = "") -> dict:
        """List files in a remote."""
        try:
            result = subprocess.run(
                ["rclone", "lsjson", f"{remote}:{path}", "--max-depth", "1"],
                capture_output=True, text=True, timeout=30
            )
            if result.returncode != 0:
                return {"error": result.stderr}
            return {"files": json.loads(result.stdout)}
        except Exception as e:
            return {"error": str(e)}

    def get_file_content(self, remote: str, path: str) -> dict:
        """Fetch file content from a remote."""
        import mimetypes
        import tempfile

        try:
            # Get file info first
            info_result = subprocess.run(
                ["rclone", "lsjson", f"{remote}:{path}"],
                capture_output=True, text=True, timeout=30
            )
            if info_result.returncode != 0:
                return {"error": f"File not found: {info_result.stderr}"}

            file_info = json.loads(info_result.stdout)
            if not file_info:
                return {"error": "File not found"}

            file_meta = file_info[0]
            size = file_meta.get("Size", 0)
            mime_type = file_meta.get("MimeType", "")

            # Detect mime type if not provided
            if not mime_type:
                mime_type, _ = mimetypes.guess_type(path)
                mime_type = mime_type or "application/octet-stream"

            # Limit file size for content fetch (10MB max)
            if size > 10 * 1024 * 1024:
                return {
                    "path": path,
                    "size": size,
                    "mimeType": mime_type,
                    "error": "File too large (>10MB)",
                    "content": "",
                }

            # For images, return base64 preview
            if mime_type.startswith("image/"):
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    dl_result = subprocess.run(
                        ["rclone", "copyto", f"{remote}:{path}", tmp.name],
                        capture_output=True, timeout=60
                    )
                    if dl_result.returncode == 0:
                        with open(tmp.name, "rb") as f:
                            import base64
                            preview = base64.b64encode(f.read()).decode()
                        os.unlink(tmp.name)
                        return {
                            "path": path,
                            "size": size,
                            "mimeType": mime_type,
                            "content": "",
                            "preview": preview,
                        }

            # For text files, fetch content via rclone cat
            result = subprocess.run(
                ["rclone", "cat", f"{remote}:{path}"],
                capture_output=True, timeout=60
            )
            if result.returncode != 0:
                return {"error": f"Failed to read file: {result.stderr.decode()}"}

            # Try to decode as text
            try:
                content = result.stdout.decode("utf-8")
            except UnicodeDecodeError:
                content = result.stdout.decode("latin-1")

            return {
                "path": path,
                "size": size,
                "mimeType": mime_type,
                "content": content,
            }

        except subprocess.TimeoutExpired:
            return {"error": "Timeout fetching file"}
        except Exception as e:
            return {"error": str(e)}


# OAuth callback handler
class OAuthCallbackHandler(BaseHTTPRequestHandler):
    manager: CloudStorageManager = None

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        params = urllib.parse.parse_qs(parsed.query)

        if "code" in params and "state" in params:
            code = params["code"][0]
            state = params["state"][0]

            result = self.manager.handle_oauth_callback(code, state)

            if "error" in result:
                self.send_response(400)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(f"""
                <html><body style="font-family: sans-serif; padding: 40px; text-align: center;">
                <h1 style="color: red;">❌ Authorization Failed</h1>
                <p>{result['error']}</p>
                <p>You can close this window.</p>
                </body></html>
                """.encode())
            else:
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.end_headers()
                self.wfile.write(f"""
                <html><body style="font-family: sans-serif; padding: 40px; text-align: center;">
                <h1 style="color: green;">✅ Authorization Successful</h1>
                <p>Account <strong>{result['account']}</strong> has been configured.</p>
                <p>Drive is now mounted at <code>/mnt/gdrive/{result['account']}</code></p>
                <p>You can close this window.</p>
                </body></html>
                """.encode())
        else:
            self.send_response(400)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Missing code or state parameter")

    def log_message(self, format, *args):
        print(f"[oauth] {args[0]}")


def run_oauth_server(manager: CloudStorageManager, port: int = OAUTH_REDIRECT_PORT):
    """Run the OAuth callback server."""
    OAuthCallbackHandler.manager = manager
    server = HTTPServer(("127.0.0.1", port), OAuthCallbackHandler)
    print(f"[oauth] Callback server listening on http://127.0.0.1:{port}/")
    server.serve_forever()


# Flask API
if HAS_FLASK:
    app = Flask(__name__)
    manager: CloudStorageManager = None

    @app.route("/status")
    def api_status():
        return jsonify(manager.get_status())

    @app.route("/auth/start", methods=["POST"])
    def api_auth_start():
        data = request.get_json() or {}
        account = data.get("account")
        if not account:
            return jsonify({"error": "account required"}), 400
        result = manager.generate_auth_url(account)

        # Auto-open OAuth URL in VNC Firefox for server-side auth
        if "auth_url" in result:
            opened = open_in_vnc_firefox(result["auth_url"])
            result["opened_in_vnc"] = opened
            if opened:
                result["message"] = "OAuth page opened in VNC browser. Complete authentication there."

        return jsonify(result)

    @app.route("/mount/<remote>", methods=["POST"])
    def api_mount(remote):
        return jsonify(manager.mount_remote(remote))

    @app.route("/unmount/<remote>", methods=["POST"])
    def api_unmount(remote):
        return jsonify(manager.unmount_remote(remote))

    @app.route("/browse/<remote>")
    def api_browse(remote):
        path = request.args.get("path", "")
        return jsonify(manager.browse_remote(remote, path))

    @app.route("/file/<remote>")
    def api_file(remote):
        """Fetch file content from a remote."""
        path = request.args.get("path", "")
        if not path:
            return jsonify({"error": "path required"}), 400
        return jsonify(manager.get_file_content(remote, path))

    @app.route("/accounts")
    def api_accounts():
        return jsonify(ACCOUNTS)


def main():
    global manager

    print("=" * 60)
    print("  CLOUD STORAGE DAEMON")
    print("  OAuth + Mount Manager for Google Drive")
    print("=" * 60)

    manager = CloudStorageManager()

    # Start OAuth callback server in background
    oauth_thread = threading.Thread(
        target=run_oauth_server,
        args=(manager,),
        daemon=True
    )
    oauth_thread.start()

    if HAS_FLASK:
        print(f"[api] Starting API server on port 9400")
        emit_event("cloud.daemon.start", {"port": 9400})
        app.run(host="127.0.0.1", port=9400, debug=False, use_reloader=False)
    else:
        print("[api] Flask not installed - running OAuth server only")
        print("[api] Install with: pip install flask")
        oauth_thread.join()


if __name__ == "__main__":
    main()
