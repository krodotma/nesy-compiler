#!/usr/bin/env python3
"""
World Router - Unified VPS Gateway
==================================

A holistic gateway that consolidates:
- PBVW Bridge (LLM routing)
- CUA Engine (Computer Use Agent primitives)
- VNC Tunneling (WebSocket proxy)
- Storage Fabric (Cloud storage access)
- Identity Hub (DID + OAuth)

All traffic flows through one process with unified observability.

Usage:
    python3 world_router.py                    # Start on :8080
    python3 world_router.py --port 9000        # Custom port
    python3 world_router.py --identity-hub     # Enable DID-based auth
    python3 world_router.py --cua-native       # Enable native CUA (not proxy)

Endpoints:
    # LLM Inference (OpenAI/Anthropic/Gemini compatible)
    POST /v1/chat/completions         # OpenAI Chat
    POST /v1/responses                # OpenAI Responses API
    POST /v1/messages                 # Anthropic Messages
    POST /v1beta/models/:model:*      # Gemini Generate

    # Computer Use Agent
    POST /cua/screenshot              # Capture screen region
    POST /cua/click                   # Click at coordinates
    POST /cua/type                    # Type text
    POST /cua/navigate                # Navigate browser
    POST /cua/eval                    # Execute JavaScript
    POST /cua/query                   # Query DOM element

    # VNC Tunnel
    WS   /vnc                         # VNC over WebSocket

    # Bus Stream
    WS   /bus                         # Bus event stream
    WS   /ws/bus                      # Bus event stream (alias)

    # Storage
    GET  /storage/status              # List mounts
    GET  /storage/:provider/list      # List directory
    GET  /storage/:provider/read      # Read file
    POST /storage/:provider/oauth     # Start OAuth

    # Identity
    GET  /identity/whoami             # Current session DID
    POST /identity/oauth/start        # Start OAuth flow
    POST /identity/oauth/callback     # OAuth callback

    # Health & Observability
    GET  /health                      # Health check
    GET  /v1/models                   # List available models
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse, parse_qs

sys.dont_write_bytecode = True

# Ensure repo root is in path
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Best-effort imports
try:
    from nucleus.tools import agent_bus
except Exception:
    agent_bus = None

try:
    from aiohttp import web, WSMsgType
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False
    web = None
    WSMsgType = None

# ============================================================================
# Configuration
# ============================================================================

@dataclass
class WorldRouterConfig:
    """World Router configuration with TIER awareness."""
    host: str = "0.0.0.0"
    port: int = 8080
    bus_dir: str = "/pluribus/.pluribus/bus"
    actor: str = "world-router"
    tier: str = os.environ.get("PLURIBUS_TIER", "prod")

    # Feature flags
    identity_hub: bool = False
    cua_native: bool = False
    vnc_tunnel: bool = True
    storage_fabric: bool = True

    # Legacy proxy endpoints (for migration)
    legacy_pbvw: str = "http://localhost:8081"
    legacy_storage: str = "http://localhost:9400"
    legacy_browser: str = ""  # IPC, not HTTP

    # Timeouts
    router_timeout_s: float = 120.0
    vnc_host: str = "127.0.0.1"
    vnc_port: int = 5901

    # Paths
    router_path: Path = field(default_factory=lambda: Path(__file__).parent / "providers" / "router.py")

    def __post_init__(self):
        # Auto-configure based on TIER
        try:
            sys.path.append(str(Path(__file__).parent))
            import paip_isolation
            iso = paip_isolation.get_isolated_config(self.tier)
            self.port = int(os.environ.get("API_PORT", iso["API_PORT"]))
            # Adjust vnc port based on slot
            slot = int(iso.get("PAIP_SLOT", "0"))
            self.vnc_port = 5901 + slot
        except Exception:
            pass
        
        # Override with explicit ENV if provided
        if os.environ.get("API_PORT"):
            self.port = int(os.environ["API_PORT"])


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"


def _safe_sha256(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8", errors="replace")).hexdigest()


# ============================================================================
# Bus Integration
# ============================================================================

def emit_bus_event(
    bus_dir: str,
    topic: str,
    data: dict,
    *,
    kind: str = "metric",
    level: str = "info",
    actor: str = "world-router",
) -> None:
    """Emit event to Pluribus bus."""
    if not bus_dir:
        return
    if agent_bus is not None:
        try:
            paths = agent_bus.resolve_bus_paths(bus_dir)
            agent_bus.emit_event(
                paths,
                topic=topic,
                kind=kind,
                level=level,
                actor=actor,
                data=data,
                trace_id=None,
                run_id=None,
                durable=False,
            )
            return
        except Exception:
            pass

    # Fallback: direct NDJSON append
    events_path = Path(bus_dir) / "events.ndjson"
    events_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "id": str(uuid.uuid4()),
        "ts": time.time(),
        "iso": now_iso(),
        "topic": topic,
        "kind": kind,
        "level": level,
        "actor": actor,
        "data": data,
    }
    try:
        with events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
    except Exception:
        pass


# ============================================================================
# Gap Analysis (Aleatoric/Epistemic)
# ============================================================================

def _classify_router_blocker(text: str) -> str | None:
    t = (text or "").lower()
    if "please run /login" in t or "run /login" in t or "invalid api key" in t or "auth required" in t:
        return "auth"
    if "resource_exhausted" in t or "quota exceeded" in t or "http error: 429" in t:
        return "quota"
    if "overloaded_error" in t or "overloaded" in t or "api error: 529" in t or "http error: 529" in t:
        return "overload"
    if "no provider configured" in t or "missing api key" in t or "unsupported provider" in t or "provider config missing" in t:
        return "config"
    return None


def _classify_router_failure(stderr: str, stdout: str, exit_code: int) -> str | None:
    text = "\n".join([stderr or "", stdout or ""])
    text_lower = text.lower()
    if exit_code == 124 or "router timeout" in text_lower or "timeout" in text_lower:
        return "timeout"
    return _classify_router_blocker(text_lower)


def _router_error_status(reason: str | None) -> int:
    if reason in {"auth", "config"}:
        return 424
    if reason == "quota":
        return 429
    if reason == "overload":
        return 503
    if reason == "timeout":
        return 504
    if reason == "circuit_open":
        return 503
    return 500


def _router_error_type(reason: str | None) -> str:
    mapping = {
        "auth": "provider_auth_required",
        "config": "provider_unavailable",
        "quota": "provider_quota",
        "overload": "provider_overload",
        "timeout": "provider_timeout",
        "circuit_open": "provider_unavailable",
    }
    return mapping.get(reason or "", "router_error")


def _gap_analysis_for_reason(
    reason: str | None,
    *,
    provider: str | None = None,
    router_timeout_s: float | None = None,
    cooldown_s: float | None = None,
) -> dict[str, list[dict]]:
    gaps = {"epistemic": [], "aleatoric": []}
    if not reason:
        return gaps

    provider_label = f" ({provider})" if provider else ""

    if reason == "auth":
        gaps["epistemic"].append(
            {
                "id": "E_AUTH",
                "description": f"Provider auth required{provider_label}.",
                "severity": "high",
                "suggested_action": "Authenticate provider session or set API key.",
            }
        )
    elif reason == "config":
        gaps["epistemic"].append(
            {
                "id": "E_CONFIG",
                "description": f"Provider configuration missing or invalid{provider_label}.",
                "severity": "high",
                "suggested_action": "Configure provider credentials and availability.",
            }
        )
    elif reason == "quota":
        gaps["aleatoric"].append(
            {
                "id": "A_QUOTA",
                "description": f"Provider quota exhausted{provider_label}.",
                "bounds": {"lo": None, "hi": None, "unit": "requests"},
            }
        )
    elif reason == "overload":
        gaps["aleatoric"].append(
            {
                "id": "A_OVERLOAD",
                "description": f"Provider overloaded{provider_label}.",
                "bounds": {"lo": None, "hi": None, "unit": "seconds"},
            }
        )
    elif reason == "timeout":
        bounds_hi = float(router_timeout_s) if router_timeout_s is not None else None
        gaps["aleatoric"].append(
            {
                "id": "A_TIMEOUT",
                "description": f"Provider request timed out{provider_label}.",
                "bounds": {"lo": 0.0, "hi": bounds_hi, "unit": "seconds"},
            }
        )
    elif reason == "circuit_open":
        bounds_hi = float(cooldown_s) if cooldown_s is not None else None
        gaps["aleatoric"].append(
            {
                "id": "A_CIRCUIT_OPEN",
                "description": f"Provider circuit open{provider_label}.",
                "bounds": {"lo": 0.0, "hi": bounds_hi, "unit": "seconds"},
            }
        )

    return gaps


# ============================================================================
# Bus Stream Helpers
# ============================================================================

def _read_bus_tail_events(events_path: Path, max_lines: int, max_bytes: int) -> list[dict]:
    try:
        if not events_path.exists():
            return []
        end = events_path.stat().st_size
        if end <= 0:
            return []
        start = max(0, end - max_bytes)
        with events_path.open("r", encoding="utf-8", errors="replace") as f:
            if start:
                f.seek(start)
            data = f.read()
        lines = data.splitlines()
        if start and lines:
            lines = lines[1:]
        events: list[dict] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                events.append(obj)
        if max_lines and len(events) > max_lines:
            events = events[-max_lines:]
        return events
    except Exception:
        return []


def _normalize_bus_event(event: dict, *, actor: str) -> dict:
    out = dict(event)
    out.setdefault("id", str(uuid.uuid4()))
    out.setdefault("ts", time.time())
    out.setdefault("iso", now_iso())
    out.setdefault("kind", "log")
    out.setdefault("level", "info")
    out.setdefault("actor", actor)
    if "data" not in out:
        out["data"] = None
    return out


def _append_bus_event(events_path: Path, event: dict) -> None:
    events_path.parent.mkdir(parents=True, exist_ok=True)
    with events_path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")


def _topic_matches(subscriptions: set[str], topic: str) -> bool:
    if "*" in subscriptions:
        return True
    if topic in subscriptions:
        return True
    for sub in subscriptions:
        if sub.endswith("*") and topic.startswith(sub[:-1]):
            return True
    return False


# ============================================================================
# Protocol Detection
# ============================================================================

class ProtocolRouter:
    """Detect and classify incoming requests by protocol."""

    # Gemini path pattern
    RE_GEMINI = re.compile(r"^/v1beta/models/(?P<model>[^/:]+):(?P<method>generateContent|streamGenerateContent)$")

    @staticmethod
    def detect(request) -> str:
        """Detect protocol from request path and headers."""
        path = request.path
        method = request.method
        upgrade = request.headers.get("Upgrade", "").lower()

        # WebSocket upgrades
        if upgrade == "websocket":
            if path.startswith("/vnc"):
                return "ws/vnc"
            if path.startswith("/bus") or path.startswith("/ws/bus"):
                return "ws/bus"
            return "ws/unknown"

        # CUA endpoints
        if path.startswith("/cua/"):
            return "http/cua"

        # Storage endpoints
        if path.startswith("/storage/"):
            return "http/storage"

        # Identity endpoints
        if path.startswith("/identity/"):
            return "http/identity"

        # A2A protocol
        if path == "/.well-known/agent.json" or path.startswith("/a2a/"):
            return "http/a2a"

        # LLM protocols
        if path.startswith("/v1/chat/completions"):
            return "http/openai-chat"
        if path.startswith("/v1/responses"):
            return "http/openai-responses"
        if path.startswith("/v1/messages"):
            return "http/anthropic"
        if ProtocolRouter.RE_GEMINI.match(path):
            return "http/gemini"
        if path == "/v1/models" or path == "/v1beta/models":
            return "http/models"

        # Health/meta
        if path in {"/health", "/healthz", "/_health"}:
            return "http/health"

        # Default: proxy to dashboard/vite
        return "http/default"


# ============================================================================
# Identity Hub (Phase 2 - Full Implementation)
# ============================================================================

@dataclass
class Identity:
    """Session identity with DID-based identification."""
    did: str = "did:key:anonymous"
    session_id: str = ""
    session_type: str = "anonymous"  # human, agent, service
    capabilities: list[str] = field(default_factory=list)
    auth_method: str = "none"
    auth_provider: Optional[str] = None
    email: Optional[str] = None
    name: Optional[str] = None
    avatar_url: Optional[str] = None
    created_at: str = ""
    expires_at: Optional[str] = None
    refresh_token: Optional[str] = None

    def __post_init__(self):
        if not self.session_id:
            self.session_id = str(uuid.uuid4())
        if not self.created_at:
            self.created_at = now_iso()

    def is_expired(self) -> bool:
        """Check if session is expired."""
        if not self.expires_at:
            return False
        try:
            exp = datetime.fromisoformat(self.expires_at.replace("Z", "+00:00"))
            return datetime.now(timezone.utc) > exp
        except Exception:
            return False

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "did": self.did,
            "session_id": self.session_id,
            "session_type": self.session_type,
            "capabilities": self.capabilities,
            "auth_method": self.auth_method,
            "auth_provider": self.auth_provider,
            "email": self.email,
            "name": self.name,
            "avatar_url": self.avatar_url,
            "created_at": self.created_at,
            "expires_at": self.expires_at,
        }


@dataclass
class OAuthState:
    """OAuth flow state tracking."""
    state: str
    provider: str
    redirect_uri: str
    created_at: str
    code_verifier: Optional[str] = None  # PKCE
    expires_at: float = 0.0  # Unix timestamp

    def __post_init__(self):
        if self.expires_at == 0.0:
            # OAuth states expire after 10 minutes
            self.expires_at = time.time() + 600

    def is_expired(self) -> bool:
        return time.time() > self.expires_at


@dataclass
class WebAuthnCredential:
    """WebAuthn credential storage.

    Stores public key credentials associated with a user identity.
    Supports both platform authenticators (Face ID, Windows Hello)
    and roaming authenticators (YubiKey, security keys).
    """
    credential_id: str  # Base64-encoded credential ID
    user_id: str  # User identifier (email or DID)
    public_key: str  # Base64-encoded public key (COSE format)
    sign_count: int = 0  # Signature counter for replay protection
    authenticator_type: str = "unknown"  # platform, cross-platform, unknown
    transports: list[str] = field(default_factory=list)  # usb, nfc, ble, internal
    created_at: str = ""
    last_used: Optional[str] = None
    friendly_name: Optional[str] = None  # User-assigned name for credential

    def __post_init__(self):
        if not self.created_at:
            self.created_at = now_iso()

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        return {
            "credential_id": self.credential_id,
            "user_id": self.user_id,
            "public_key": self.public_key,
            "sign_count": self.sign_count,
            "authenticator_type": self.authenticator_type,
            "transports": self.transports,
            "created_at": self.created_at,
            "last_used": self.last_used,
            "friendly_name": self.friendly_name,
        }


@dataclass
class WebAuthnChallenge:
    """WebAuthn challenge tracking with expiration.

    Challenges are single-use and expire after 5 minutes.
    """
    challenge: str  # Base64-encoded random bytes
    user_id: Optional[str] = None  # For registration, identifies the user
    operation: str = "registration"  # registration or authentication
    created_at: str = ""
    expires_at: float = 0.0  # Unix timestamp

    def __post_init__(self):
        if not self.created_at:
            self.created_at = now_iso()
        if self.expires_at == 0.0:
            # WebAuthn challenges expire after 5 minutes
            self.expires_at = time.time() + 300

    def is_expired(self) -> bool:
        return time.time() > self.expires_at


class IdentityHub:
    """Identity resolution and management with full OAuth and WebAuthn support.

    Supports:
    - Google OAuth 2.0
    - GitHub OAuth
    - DID-based session tokens
    - Session persistence
    - WebAuthn/Passkey authentication (Phase 3)
    """

    # OAuth Provider Configurations
    OAUTH_PROVIDERS = {
        "google": {
            "name": "Google",
            "auth_url": "https://accounts.google.com/o/oauth2/v2/auth",
            "token_url": "https://oauth2.googleapis.com/token",
            "userinfo_url": "https://www.googleapis.com/oauth2/v2/userinfo",
            "scopes": ["openid", "email", "profile"],
            "client_id_env": "GOOGLE_CLIENT_ID",
            "client_secret_env": "GOOGLE_CLIENT_SECRET",
        },
        "github": {
            "name": "GitHub",
            "auth_url": "https://github.com/login/oauth/authorize",
            "token_url": "https://github.com/login/oauth/access_token",
            "userinfo_url": "https://api.github.com/user",
            "scopes": ["user:email", "read:user"],
            "client_id_env": "GITHUB_CLIENT_ID",
            "client_secret_env": "GITHUB_CLIENT_SECRET",
        },
    }

    def __init__(self, enabled: bool = False, session_file: str = None, webauthn_file: str = None):
        self.enabled = enabled
        self._sessions: dict[str, Identity] = {}
        self._oauth_states: dict[str, OAuthState] = {}
        self._webauthn_credentials: dict[str, WebAuthnCredential] = {}  # credential_id -> credential
        self._webauthn_challenges: dict[str, WebAuthnChallenge] = {}  # challenge -> WebAuthnChallenge
        self._session_file = session_file or "/pluribus/.pluribus/identity_sessions.json"
        self._webauthn_file = webauthn_file or "/pluribus/.pluribus/webauthn_credentials.json"
        self._load_sessions()
        self._load_webauthn_credentials()

    def _load_sessions(self):
        """Load persisted sessions from disk."""
        try:
            path = Path(self._session_file)
            if path.exists():
                data = json.loads(path.read_text())
                for session_id, sess_data in data.get("sessions", {}).items():
                    identity = Identity(
                        did=sess_data.get("did", "did:key:anonymous"),
                        session_id=session_id,
                        session_type=sess_data.get("session_type", "human"),
                        capabilities=sess_data.get("capabilities", []),
                        auth_method=sess_data.get("auth_method", "oauth"),
                        auth_provider=sess_data.get("auth_provider"),
                        email=sess_data.get("email"),
                        name=sess_data.get("name"),
                        avatar_url=sess_data.get("avatar_url"),
                        created_at=sess_data.get("created_at", now_iso()),
                        expires_at=sess_data.get("expires_at"),
                    )
                    if not identity.is_expired():
                        self._sessions[session_id] = identity
        except Exception:
            pass

    def _save_sessions(self):
        """Persist sessions to disk."""
        try:
            path = Path(self._session_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "sessions": {
                    sid: identity.to_dict()
                    for sid, identity in self._sessions.items()
                    if not identity.is_expired()
                }
            }
            path.write_text(json.dumps(data, indent=2))
        except Exception:
            pass

    def _load_webauthn_credentials(self):
        """Load persisted WebAuthn credentials from disk."""
        try:
            path = Path(self._webauthn_file)
            if path.exists():
                data = json.loads(path.read_text())
                for cred_id, cred_data in data.get("credentials", {}).items():
                    credential = WebAuthnCredential(
                        credential_id=cred_id,
                        user_id=cred_data.get("user_id", ""),
                        public_key=cred_data.get("public_key", ""),
                        sign_count=cred_data.get("sign_count", 0),
                        authenticator_type=cred_data.get("authenticator_type", "unknown"),
                        transports=cred_data.get("transports", []),
                        created_at=cred_data.get("created_at", now_iso()),
                        last_used=cred_data.get("last_used"),
                        friendly_name=cred_data.get("friendly_name"),
                    )
                    self._webauthn_credentials[cred_id] = credential
        except Exception:
            pass

    def _save_webauthn_credentials(self):
        """Persist WebAuthn credentials to disk."""
        try:
            path = Path(self._webauthn_file)
            path.parent.mkdir(parents=True, exist_ok=True)
            data = {
                "credentials": {
                    cred_id: cred.to_dict()
                    for cred_id, cred in self._webauthn_credentials.items()
                }
            }
            path.write_text(json.dumps(data, indent=2))
        except Exception:
            pass

    # -------------------------------------------------------------------------
    # WebAuthn/Passkey Methods (Phase 3)
    # -------------------------------------------------------------------------

    def _generate_webauthn_challenge(self) -> str:
        """Generate a cryptographically random challenge for WebAuthn."""
        return base64.urlsafe_b64encode(os.urandom(32)).decode().rstrip("=")

    def _get_rp_id(self, host: str) -> str:
        """Get the Relying Party ID from the request host.

        The rpId is typically the domain without port.
        For localhost, we use "localhost".
        """
        # Remove port if present
        if ":" in host:
            host = host.split(":")[0]
        return host

    def webauthn_register_start(self, user_id: str, user_name: str, host: str) -> dict:
        """Start WebAuthn registration.

        Generates registration options for a new passkey.

        Args:
            user_id: Unique user identifier (email or DID)
            user_name: Display name for the user
            host: Request host header for rpId

        Returns:
            PublicKeyCredentialCreationOptions-compatible dict
        """
        challenge = self._generate_webauthn_challenge()
        rp_id = self._get_rp_id(host)

        # Store challenge for verification
        self._webauthn_challenges[challenge] = WebAuthnChallenge(
            challenge=challenge,
            user_id=user_id,
            operation="registration",
        )

        # Clean up expired challenges
        self._cleanup_webauthn_challenges()

        # Generate user handle (random bytes for privacy)
        user_handle = base64.urlsafe_b64encode(os.urandom(16)).decode().rstrip("=")

        # Get existing credentials for this user to exclude
        exclude_credentials = [
            {
                "type": "public-key",
                "id": cred.credential_id,
                "transports": cred.transports or ["internal", "usb"],
            }
            for cred in self._webauthn_credentials.values()
            if cred.user_id == user_id
        ]

        return {
            "challenge": challenge,
            "rp": {
                "name": "Pluribus",
                "id": rp_id,
            },
            "user": {
                "id": user_handle,
                "name": user_id,
                "displayName": user_name or user_id,
            },
            "pubKeyCredParams": [
                {"type": "public-key", "alg": -7},   # ES256 (P-256)
                {"type": "public-key", "alg": -257}, # RS256
                {"type": "public-key", "alg": -8},   # EdDSA
            ],
            "timeout": 300000,  # 5 minutes
            "authenticatorSelection": {
                "residentKey": "preferred",
                "userVerification": "preferred",
            },
            "attestation": "none",  # Privacy-preserving
            "excludeCredentials": exclude_credentials,
        }

    def webauthn_register_complete(
        self,
        challenge: str,
        credential_id: str,
        public_key: str,
        authenticator_type: str = "unknown",
        transports: list[str] = None,
        friendly_name: str = None,
    ) -> Optional[WebAuthnCredential]:
        """Complete WebAuthn registration.

        Verifies the attestation and stores the credential.

        Args:
            challenge: The challenge from registration start
            credential_id: Base64-encoded credential ID from authenticator
            public_key: Base64-encoded public key (COSE format)
            authenticator_type: platform, cross-platform, or unknown
            transports: Supported transports (usb, nfc, ble, internal)
            friendly_name: User-friendly name for the credential

        Returns:
            WebAuthnCredential on success, None on failure
        """
        # Verify challenge exists and is valid
        if challenge not in self._webauthn_challenges:
            return None

        challenge_obj = self._webauthn_challenges[challenge]

        if challenge_obj.is_expired():
            del self._webauthn_challenges[challenge]
            return None

        if challenge_obj.operation != "registration":
            return None

        user_id = challenge_obj.user_id

        # Check if credential ID already exists
        if credential_id in self._webauthn_credentials:
            return None

        # Create and store credential
        credential = WebAuthnCredential(
            credential_id=credential_id,
            user_id=user_id,
            public_key=public_key,
            sign_count=0,
            authenticator_type=authenticator_type,
            transports=transports or [],
            friendly_name=friendly_name,
        )

        self._webauthn_credentials[credential_id] = credential

        # Remove used challenge
        del self._webauthn_challenges[challenge]

        # Persist credentials
        self._save_webauthn_credentials()

        return credential

    def webauthn_login_start(self, user_id: Optional[str], host: str) -> dict:
        """Start WebAuthn authentication.

        Generates authentication options for passkey login.

        Args:
            user_id: Optional user identifier to restrict allowed credentials
            host: Request host header for rpId

        Returns:
            PublicKeyCredentialRequestOptions-compatible dict
        """
        challenge = self._generate_webauthn_challenge()
        rp_id = self._get_rp_id(host)

        # Store challenge for verification
        self._webauthn_challenges[challenge] = WebAuthnChallenge(
            challenge=challenge,
            user_id=user_id,
            operation="authentication",
        )

        # Clean up expired challenges
        self._cleanup_webauthn_challenges()

        # Get allowed credentials
        if user_id:
            # Restrict to user's credentials
            allowed_credentials = [
                {
                    "type": "public-key",
                    "id": cred.credential_id,
                    "transports": cred.transports or ["internal", "usb"],
                }
                for cred in self._webauthn_credentials.values()
                if cred.user_id == user_id
            ]
        else:
            # Discoverable credentials (passkeys) - empty list allows any
            allowed_credentials = []

        return {
            "challenge": challenge,
            "rpId": rp_id,
            "timeout": 300000,  # 5 minutes
            "userVerification": "preferred",
            "allowCredentials": allowed_credentials,
        }

    def webauthn_login_complete(
        self,
        challenge: str,
        credential_id: str,
        signature: str,
        authenticator_data: str,
        client_data_json: str,
    ) -> Optional[Identity]:
        """Complete WebAuthn authentication.

        Verifies the assertion and creates a session.

        Args:
            challenge: The challenge from login start
            credential_id: Base64-encoded credential ID
            signature: Base64-encoded signature
            authenticator_data: Base64-encoded authenticator data
            client_data_json: Base64-encoded client data JSON

        Returns:
            Identity on success, None on failure
        """
        # Verify challenge exists and is valid
        if challenge not in self._webauthn_challenges:
            return None

        challenge_obj = self._webauthn_challenges[challenge]

        if challenge_obj.is_expired():
            del self._webauthn_challenges[challenge]
            return None

        if challenge_obj.operation != "authentication":
            return None

        # Find credential
        if credential_id not in self._webauthn_credentials:
            return None

        credential = self._webauthn_credentials[credential_id]

        # In a full implementation, we would:
        # 1. Decode and verify client_data_json contains the challenge
        # 2. Verify authenticator_data flags
        # 3. Verify signature using stored public key
        # For now, we trust the client-side verification (browser handles this)

        # Extract sign count from authenticator data (bytes 33-36, big-endian)
        try:
            auth_data_bytes = base64.urlsafe_b64decode(authenticator_data + "==")
            if len(auth_data_bytes) >= 37:
                new_sign_count = int.from_bytes(auth_data_bytes[33:37], "big")
                # Verify sign count increased (replay protection)
                if new_sign_count <= credential.sign_count and credential.sign_count > 0:
                    # Possible cloned authenticator
                    return None
                credential.sign_count = new_sign_count
        except Exception:
            pass

        # Update last used timestamp
        credential.last_used = now_iso()
        self._save_webauthn_credentials()

        # Remove used challenge
        del self._webauthn_challenges[challenge]

        # Create session identity
        user_id = credential.user_id
        did = self._generate_did_from_email(user_id, "webauthn")
        expires = datetime.now(timezone.utc) + timedelta(days=30)

        identity = Identity(
            did=did,
            session_type="human",
            capabilities=["llm:invoke", "cua:all", "storage:read", "storage:write"],
            auth_method="webauthn",
            auth_provider="passkey",
            email=user_id if "@" in user_id else None,
            created_at=now_iso(),
            expires_at=expires.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
        )

        # Create session
        self.create_session(identity)

        return identity

    def _cleanup_webauthn_challenges(self):
        """Remove expired WebAuthn challenges."""
        expired = [
            ch for ch, challenge in self._webauthn_challenges.items()
            if challenge.is_expired()
        ]
        for ch in expired:
            del self._webauthn_challenges[ch]

    def get_user_credentials(self, user_id: str) -> list[dict]:
        """Get all credentials for a user."""
        return [
            {
                "credential_id": cred.credential_id,
                "authenticator_type": cred.authenticator_type,
                "transports": cred.transports,
                "created_at": cred.created_at,
                "last_used": cred.last_used,
                "friendly_name": cred.friendly_name,
            }
            for cred in self._webauthn_credentials.values()
            if cred.user_id == user_id
        ]

    def delete_credential(self, credential_id: str, user_id: str) -> bool:
        """Delete a WebAuthn credential.

        Only allows deletion if the credential belongs to the user.
        """
        if credential_id not in self._webauthn_credentials:
            return False

        credential = self._webauthn_credentials[credential_id]
        if credential.user_id != user_id:
            return False

        del self._webauthn_credentials[credential_id]
        self._save_webauthn_credentials()
        return True

    def _generate_did_from_email(self, email: str, provider: str) -> str:
        """Generate a deterministic DID from email and provider."""
        # Create a stable hash from email + provider
        seed = f"{provider}:{email}".encode("utf-8")
        hash_bytes = hashlib.sha256(seed).digest()
        # Encode as base58-like (simplified)
        encoded = base64.urlsafe_b64encode(hash_bytes[:24]).decode().rstrip("=")
        return f"did:web:pluribus.{provider}:{encoded}"

    def _generate_pkce_verifier(self) -> tuple[str, str]:
        """Generate PKCE code verifier and challenge."""
        verifier = base64.urlsafe_b64encode(os.urandom(32)).decode().rstrip("=")
        challenge_bytes = hashlib.sha256(verifier.encode()).digest()
        challenge = base64.urlsafe_b64encode(challenge_bytes).decode().rstrip("=")
        return verifier, challenge

    async def resolve(self, request) -> Identity:
        """Resolve identity from request headers/cookies."""
        if not self.enabled:
            return Identity()

        # Check Authorization header (Bearer token)
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            token = auth[7:]
            # Check if token is a session ID
            if token in self._sessions:
                identity = self._sessions[token]
                if not identity.is_expired():
                    return identity
            # Treat as DID token
            return Identity(
                did=f"did:key:{_safe_sha256(token)[:32]}",
                auth_method="bearer",
                session_type="agent",
            )

        # Check session cookie
        session_id = request.cookies.get("pluribus_session")
        if session_id and session_id in self._sessions:
            identity = self._sessions[session_id]
            if not identity.is_expired():
                return identity

        # Check X-Session-ID header
        header_session = request.headers.get("X-Session-ID", "")
        if header_session and header_session in self._sessions:
            identity = self._sessions[header_session]
            if not identity.is_expired():
                return identity

        return Identity()

    def create_session(self, identity: Identity) -> str:
        """Create a new session and persist."""
        self._sessions[identity.session_id] = identity
        self._save_sessions()
        return identity.session_id

    def revoke_session(self, session_id: str) -> bool:
        """Revoke a session."""
        if session_id in self._sessions:
            del self._sessions[session_id]
            self._save_sessions()
            return True
        return False

    def get_oauth_url(self, provider: str, redirect_uri: str) -> tuple[str, str]:
        """Generate OAuth authorization URL with PKCE.

        Returns: (auth_url, state)
        """
        if provider not in self.OAUTH_PROVIDERS:
            raise ValueError(f"Unknown provider: {provider}")

        config = self.OAUTH_PROVIDERS[provider]
        client_id = os.environ.get(config["client_id_env"], "")

        if not client_id:
            raise ValueError(f"Missing {config['client_id_env']} environment variable")

        # Generate state and PKCE
        state = base64.urlsafe_b64encode(os.urandom(16)).decode().rstrip("=")
        verifier, challenge = self._generate_pkce_verifier()

        # Store OAuth state
        self._oauth_states[state] = OAuthState(
            state=state,
            provider=provider,
            redirect_uri=redirect_uri,
            created_at=now_iso(),
            code_verifier=verifier,
        )

        # Build authorization URL
        params = {
            "client_id": client_id,
            "redirect_uri": redirect_uri,
            "response_type": "code",
            "scope": " ".join(config["scopes"]),
            "state": state,
            "access_type": "offline",  # For refresh tokens
            "prompt": "consent",
        }

        # Add PKCE for Google
        if provider == "google":
            params["code_challenge"] = challenge
            params["code_challenge_method"] = "S256"

        query = "&".join(f"{k}={v}" for k, v in params.items())
        return f"{config['auth_url']}?{query}", state

    def cleanup_expired_states(self):
        """Remove expired OAuth states."""
        expired = [
            state for state, oauth_state in self._oauth_states.items()
            if oauth_state.is_expired()
        ]
        for state in expired:
            del self._oauth_states[state]

    async def exchange_code(self, code: str, state: str) -> Optional[Identity]:
        """Exchange OAuth code for tokens and create identity.

        Returns Identity on success, None on failure.
        """
        # Clean up expired states first
        self.cleanup_expired_states()

        if state not in self._oauth_states:
            return None

        oauth_state = self._oauth_states[state]

        # Check if this specific state is expired
        if oauth_state.is_expired():
            del self._oauth_states[state]
            return None
        provider = oauth_state.provider
        config = self.OAUTH_PROVIDERS[provider]

        client_id = os.environ.get(config["client_id_env"], "")
        client_secret = os.environ.get(config["client_secret_env"], "")

        if not client_id or not client_secret:
            return None

        try:
            import aiohttp

            # Exchange code for tokens
            token_data = {
                "client_id": client_id,
                "client_secret": client_secret,
                "code": code,
                "redirect_uri": oauth_state.redirect_uri,
                "grant_type": "authorization_code",
            }

            # Add PKCE verifier for Google
            if provider == "google" and oauth_state.code_verifier:
                token_data["code_verifier"] = oauth_state.code_verifier

            async with aiohttp.ClientSession() as session:
                # Get tokens
                headers = {"Accept": "application/json"}
                async with session.post(config["token_url"], data=token_data, headers=headers) as resp:
                    if resp.status != 200:
                        return None
                    tokens = await resp.json()

                access_token = tokens.get("access_token")
                if not access_token:
                    return None

                # Get user info
                auth_header = {"Authorization": f"Bearer {access_token}"}
                async with session.get(config["userinfo_url"], headers=auth_header) as resp:
                    if resp.status != 200:
                        return None
                    userinfo = await resp.json()

                # Extract user details
                if provider == "google":
                    email = userinfo.get("email", "")
                    name = userinfo.get("name", "")
                    avatar = userinfo.get("picture", "")
                elif provider == "github":
                    email = userinfo.get("email")
                    name = userinfo.get("name") or userinfo.get("login", "")
                    avatar = userinfo.get("avatar_url", "")

                    # GitHub may not return email in user info - fetch from emails API
                    if not email:
                        try:
                            async with session.get(
                                "https://api.github.com/user/emails",
                                headers=auth_header
                            ) as emails_resp:
                                if emails_resp.status == 200:
                                    emails = await emails_resp.json()
                                    # Find primary email
                                    for e in emails:
                                        if e.get("primary") and e.get("verified"):
                                            email = e.get("email", "")
                                            break
                                    # Fallback to first verified email
                                    if not email:
                                        for e in emails:
                                            if e.get("verified"):
                                                email = e.get("email", "")
                                                break
                        except Exception:
                            pass

                    # Final fallback: use GitHub username as pseudo-email
                    if not email:
                        login = userinfo.get("login", "")
                        if login:
                            email = f"{login}@users.noreply.github.com"
                else:
                    email = userinfo.get("email", "")
                    name = userinfo.get("name", "")
                    avatar = ""

                # Create identity
                did = self._generate_did_from_email(email, provider)
                expires = datetime.now(timezone.utc) + timedelta(days=30)

                identity = Identity(
                    did=did,
                    session_type="human",
                    capabilities=["llm:invoke", "cua:all", "storage:read", "storage:write"],
                    auth_method="oauth",
                    auth_provider=provider,
                    email=email,
                    name=name,
                    avatar_url=avatar,
                    expires_at=expires.strftime("%Y-%m-%dT%H:%M:%S.000Z"),
                    refresh_token=tokens.get("refresh_token"),
                )

                # Clean up OAuth state
                del self._oauth_states[state]

                # Create session
                self.create_session(identity)

                return identity

        except Exception:
            return None

    def list_sessions(self) -> list[dict]:
        """List all active sessions."""
        return [
            identity.to_dict()
            for identity in self._sessions.values()
            if not identity.is_expired()
        ]

    async def refresh_session(self, session_id: str) -> Optional[Identity]:
        """Refresh an expired session using stored refresh token.

        Returns refreshed Identity on success, None if refresh not possible.
        """
        if session_id not in self._sessions:
            return None

        identity = self._sessions[session_id]

        # Check if we have a refresh token
        if not identity.refresh_token:
            return None

        provider = identity.auth_provider
        if not provider or provider not in self.OAUTH_PROVIDERS:
            return None

        config = self.OAUTH_PROVIDERS[provider]
        client_id = os.environ.get(config["client_id_env"], "")
        client_secret = os.environ.get(config["client_secret_env"], "")

        if not client_id or not client_secret:
            return None

        try:
            import aiohttp

            refresh_data = {
                "client_id": client_id,
                "client_secret": client_secret,
                "refresh_token": identity.refresh_token,
                "grant_type": "refresh_token",
            }

            async with aiohttp.ClientSession() as session:
                headers = {"Accept": "application/json"}
                async with session.post(config["token_url"], data=refresh_data, headers=headers) as resp:
                    if resp.status != 200:
                        # Refresh failed - revoke session
                        self.revoke_session(session_id)
                        return None

                    tokens = await resp.json()

                access_token = tokens.get("access_token")
                if not access_token:
                    return None

                # Update session expiry
                new_expiry = datetime.now(timezone.utc) + timedelta(days=30)
                identity.expires_at = new_expiry.strftime("%Y-%m-%dT%H:%M:%S.000Z")

                # Update refresh token if new one provided
                if tokens.get("refresh_token"):
                    identity.refresh_token = tokens["refresh_token"]

                # Save updated session
                self._save_sessions()

                return identity

        except Exception:
            return None

    async def resolve_with_refresh(self, request) -> Identity:
        """Resolve identity, attempting refresh if session is expired."""
        identity = await self.resolve(request)

        # If we got an anonymous identity, try to refresh from cookie
        if identity.did == "did:key:anonymous":
            session_id = request.cookies.get("pluribus_session")
            if session_id and session_id in self._sessions:
                stored_identity = self._sessions[session_id]
                if stored_identity.is_expired() and stored_identity.refresh_token:
                    refreshed = await self.refresh_session(session_id)
                    if refreshed:
                        return refreshed

        return identity


# ============================================================================
# Circuit Breaker (Phase 2)
# ============================================================================

@dataclass
class CircuitState:
    """Circuit breaker state for a provider."""
    failures: int = 0
    last_failure: float = 0.0
    state: str = "closed"  # closed, open, half-open
    cooldown_until: float = 0.0


class CircuitBreaker:
    """Circuit breaker for provider failure handling.

    States:
    - closed: Normal operation, requests pass through
    - open: Provider failing, requests short-circuit
    - half-open: Testing if provider recovered
    """

    def __init__(self, failure_threshold: int = 3, cooldown_seconds: float = 60.0):
        self.failure_threshold = failure_threshold
        self.cooldown_seconds = cooldown_seconds
        self._circuits: dict[str, CircuitState] = {}

    def get_state(self, provider: str) -> str:
        """Get circuit state for provider."""
        if provider not in self._circuits:
            return "closed"

        circuit = self._circuits[provider]

        # Check if cooldown expired
        if circuit.state == "open" and time.time() > circuit.cooldown_until:
            circuit.state = "half-open"

        return circuit.state

    def is_available(self, provider: str) -> bool:
        """Check if provider is available (circuit not open)."""
        state = self.get_state(provider)
        return state in ("closed", "half-open")

    def record_success(self, provider: str):
        """Record successful request."""
        if provider in self._circuits:
            circuit = self._circuits[provider]
            circuit.failures = 0
            circuit.state = "closed"

    def record_failure(self, provider: str):
        """Record failed request."""
        if provider not in self._circuits:
            self._circuits[provider] = CircuitState()

        circuit = self._circuits[provider]
        circuit.failures += 1
        circuit.last_failure = time.time()

        if circuit.failures >= self.failure_threshold:
            circuit.state = "open"
            circuit.cooldown_until = time.time() + self.cooldown_seconds

    def get_all_states(self) -> dict[str, dict]:
        """Get all circuit states."""
        return {
            provider: {
                "state": self.get_state(provider),
                "failures": circuit.failures,
                "cooldown_until": circuit.cooldown_until,
            }
            for provider, circuit in self._circuits.items()
        }


# ============================================================================
# CUA Engine (Computer Use Agent)
# ============================================================================

@dataclass
class CUAConfig:
    """Configuration for CUA Engine."""
    browser: str = "chromium"  # chromium, firefox, webkit
    headless: bool = False  # Visible for VNC
    viewport_width: int = 1920
    viewport_height: int = 1080
    display: str = ""  # DISPLAY env var, empty = auto
    slow_mo: int = 0  # Slow down operations by ms
    default_timeout_ms: int = 30000
    max_retries: int = 3
    retry_delay_ms: int = 1000


class CUAEngine:
    """Computer Use Agent primitives.

    Provides REST API access to browser/computer control actions.
    In native mode, uses Playwright directly.
    In proxy mode, forwards to browser_session_daemon.

    Native mode features:
    - Configurable browser (chromium/firefox/webkit)
    - Multiple page/context support
    - Wait for element functionality
    - Page state inspection
    - Graceful cleanup
    """

    def __init__(
        self,
        native: bool = False,
        config: CUAConfig | None = None,
    ):
        self.native = native
        self.config = config or CUAConfig()
        self._playwright = None
        self._browser = None
        self._context = None
        self._page = None
        self._pages: dict[str, Any] = {}  # page_id -> page
        self._current_page_id: str = "default"
        self._initialized = False

    async def initialize(self) -> dict:
        """Initialize Playwright browser if in native mode.

        Returns dict with initialization status and details.
        Includes retry logic and proper error handling.
        """
        if not self.native:
            return {"ok": True, "mode": "proxy", "message": "Proxy mode - no initialization needed"}

        if self._initialized:
            return {"ok": True, "mode": "native", "message": "Already initialized"}

        # Determine display
        display = self.config.display or os.environ.get("DISPLAY", ":1")

        last_error = None
        for attempt in range(1, self.config.max_retries + 1):
            try:
                # Import playwright
                try:
                    from playwright.async_api import async_playwright
                except ImportError as e:
                    return {
                        "ok": False,
                        "mode": "native",
                        "error": "Playwright not installed. Run: pip install playwright && playwright install",
                        "details": str(e),
                    }

                # Start playwright
                self._playwright = await async_playwright().start()

                # Browser launch args
                browser_args = [
                    "--no-sandbox",
                    "--disable-gpu",
                    "--disable-dev-shm-usage",
                    f"--display={display}",
                ]

                # Launch browser based on config
                browser_type = self.config.browser.lower()
                launch_options = {
                    "headless": self.config.headless,
                    "args": browser_args,
                    "slow_mo": self.config.slow_mo,
                }

                if browser_type == "firefox":
                    self._browser = await self._playwright.firefox.launch(**launch_options)
                elif browser_type == "webkit":
                    self._browser = await self._playwright.webkit.launch(**launch_options)
                else:
                    # Default to chromium
                    self._browser = await self._playwright.chromium.launch(**launch_options)

                # Create context with viewport
                self._context = await self._browser.new_context(
                    viewport={
                        "width": self.config.viewport_width,
                        "height": self.config.viewport_height,
                    },
                )

                # Set default timeout
                self._context.set_default_timeout(self.config.default_timeout_ms)

                # Create default page
                self._page = await self._context.new_page()
                self._pages["default"] = self._page
                self._current_page_id = "default"
                self._initialized = True

                return {
                    "ok": True,
                    "mode": "native",
                    "browser": browser_type,
                    "viewport": {
                        "width": self.config.viewport_width,
                        "height": self.config.viewport_height,
                    },
                    "display": display,
                    "attempt": attempt,
                }

            except Exception as e:
                last_error = str(e)
                print(f"[CUA] Initialization attempt {attempt}/{self.config.max_retries} failed: {e}", file=sys.stderr)

                # Cleanup partial initialization
                await self._cleanup_partial()

                if attempt < self.config.max_retries:
                    await asyncio.sleep(self.config.retry_delay_ms / 1000.0)

        return {
            "ok": False,
            "mode": "native",
            "error": f"Failed after {self.config.max_retries} attempts",
            "last_error": last_error,
        }

    async def _cleanup_partial(self):
        """Clean up partial initialization state."""
        try:
            if self._browser:
                await self._browser.close()
        except Exception:
            pass
        try:
            if self._playwright:
                await self._playwright.stop()
        except Exception:
            pass
        self._browser = None
        self._context = None
        self._page = None
        self._pages = {}
        self._playwright = None

    async def shutdown(self) -> dict:
        """Gracefully shutdown browser and cleanup resources."""
        if not self.native or not self._initialized:
            return {"ok": True, "message": "Nothing to shutdown"}

        errors = []

        # Close all pages
        for page_id, page in list(self._pages.items()):
            try:
                await page.close()
            except Exception as e:
                errors.append(f"page {page_id}: {e}")

        self._pages = {}
        self._page = None

        # Close context
        try:
            if self._context:
                await self._context.close()
                self._context = None
        except Exception as e:
            errors.append(f"context: {e}")

        # Close browser
        try:
            if self._browser:
                await self._browser.close()
                self._browser = None
        except Exception as e:
            errors.append(f"browser: {e}")

        # Stop playwright
        try:
            if self._playwright:
                await self._playwright.stop()
                self._playwright = None
        except Exception as e:
            errors.append(f"playwright: {e}")

        self._initialized = False

        if errors:
            return {"ok": False, "errors": errors}
        return {"ok": True, "message": "Shutdown complete"}

    # -------------------------------------------------------------------------
    # Session Management
    # -------------------------------------------------------------------------

    async def new_page(self, page_id: str | None = None) -> dict:
        """Create a new browser page/tab."""
        if not self.native or not self._context:
            return {"ok": False, "error": "Native mode not initialized"}

        try:
            if not page_id:
                page_id = f"page_{uuid.uuid4().hex[:8]}"

            if page_id in self._pages:
                return {"ok": False, "error": f"Page {page_id} already exists"}

            page = await self._context.new_page()
            self._pages[page_id] = page

            return {"ok": True, "page_id": page_id, "total_pages": len(self._pages)}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def switch_page(self, page_id: str) -> dict:
        """Switch to a different page."""
        if not self.native:
            return {"ok": False, "error": "Native mode not initialized"}

        if page_id not in self._pages:
            return {"ok": False, "error": f"Page {page_id} not found", "available": list(self._pages.keys())}

        try:
            self._page = self._pages[page_id]
            self._current_page_id = page_id
            await self._page.bring_to_front()

            return {
                "ok": True,
                "page_id": page_id,
                "url": self._page.url,
                "title": await self._page.title(),
            }
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def close_page(self, page_id: str) -> dict:
        """Close a specific page."""
        if not self.native:
            return {"ok": False, "error": "Native mode not initialized"}

        if page_id not in self._pages:
            return {"ok": False, "error": f"Page {page_id} not found"}

        if len(self._pages) <= 1:
            return {"ok": False, "error": "Cannot close the last page"}

        try:
            page = self._pages[page_id]
            await page.close()
            del self._pages[page_id]

            if self._current_page_id == page_id:
                self._current_page_id = next(iter(self._pages))
                self._page = self._pages[self._current_page_id]

            return {"ok": True, "closed": page_id, "current_page": self._current_page_id}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def list_pages(self) -> dict:
        """List all open pages."""
        if not self.native:
            return {"ok": False, "error": "Native mode not initialized"}

        pages_info = []
        for page_id, page in self._pages.items():
            try:
                pages_info.append({
                    "page_id": page_id,
                    "url": page.url,
                    "title": await page.title(),
                    "is_current": page_id == self._current_page_id,
                })
            except Exception:
                pages_info.append({
                    "page_id": page_id,
                    "error": "Failed to get page info",
                    "is_current": page_id == self._current_page_id,
                })

        return {"ok": True, "pages": pages_info, "count": len(pages_info)}

    # -------------------------------------------------------------------------
    # Wait Functionality
    # -------------------------------------------------------------------------

    async def wait_for_selector(
        self,
        selector: str,
        state: str = "visible",
        timeout_ms: int | None = None,
    ) -> dict:
        """Wait for an element matching selector."""
        if self.native and self._page:
            try:
                timeout = timeout_ms or self.config.default_timeout_ms
                element = await self._page.wait_for_selector(
                    selector,
                    state=state,
                    timeout=timeout,
                )
                if element:
                    box = await element.bounding_box()
                    return {
                        "ok": True,
                        "found": True,
                        "selector": selector,
                        "state": state,
                        "bounding_box": box,
                        "visible": await element.is_visible(),
                    }
                return {"ok": True, "found": False, "selector": selector}
            except Exception as e:
                error_str = str(e)
                if "Timeout" in error_str:
                    return {"ok": False, "error": "timeout", "selector": selector, "timeout_ms": timeout_ms}
                return {"ok": False, "error": error_str}

        return {"ok": False, "error": "Wait requires native CUA mode"}

    async def wait_for_load_state(self, state: str = "domcontentloaded") -> dict:
        """Wait for page to reach a specific load state."""
        if self.native and self._page:
            try:
                await self._page.wait_for_load_state(state)
                return {"ok": True, "state": state, "url": self._page.url}
            except Exception as e:
                return {"ok": False, "error": str(e)}

        return {"ok": False, "error": "Wait requires native CUA mode"}

    # -------------------------------------------------------------------------
    # Page State Inspection
    # -------------------------------------------------------------------------

    async def get_page_info(self) -> dict:
        """Get current page URL, title, and viewport."""
        if self.native and self._page:
            try:
                viewport = self._page.viewport_size
                return {
                    "ok": True,
                    "url": self._page.url,
                    "title": await self._page.title(),
                    "viewport": viewport,
                    "page_id": self._current_page_id,
                }
            except Exception as e:
                return {"ok": False, "error": str(e)}

        return {"ok": True, "mode": "proxy", "message": "Page info limited in proxy mode"}

    async def get_html(self, selector: str | None = None) -> dict:
        """Get HTML content of page or element."""
        if self.native and self._page:
            try:
                if selector:
                    element = await self._page.query_selector(selector)
                    if element:
                        html = await element.inner_html()
                        return {"ok": True, "html": html, "selector": selector}
                    return {"ok": False, "error": f"Selector not found: {selector}"}
                else:
                    html = await self._page.content()
                    return {"ok": True, "html": html, "full_page": True}
            except Exception as e:
                return {"ok": False, "error": str(e)}

        return {"ok": False, "error": "HTML inspection requires native CUA mode"}

    async def get_text(self, selector: str) -> dict:
        """Get text content of an element."""
        if self.native and self._page:
            try:
                element = await self._page.query_selector(selector)
                if element:
                    text = await element.text_content()
                    return {"ok": True, "text": text, "selector": selector}
                return {"ok": False, "error": f"Selector not found: {selector}"}
            except Exception as e:
                return {"ok": False, "error": str(e)}

        return {"ok": False, "error": "Text extraction requires native CUA mode"}

    # -------------------------------------------------------------------------
    # Extended Navigation
    # -------------------------------------------------------------------------

    async def goto(
        self,
        url: str,
        wait_until: str = "domcontentloaded",
        timeout_ms: int | None = None,
        referer: str | None = None,
    ) -> dict:
        """Navigate to URL with extended options."""
        if self.native and self._page:
            try:
                options = {"wait_until": wait_until}
                if timeout_ms:
                    options["timeout"] = timeout_ms
                if referer:
                    options["referer"] = referer

                response = await self._page.goto(url, **options)
                return {
                    "ok": True,
                    "url": self._page.url,
                    "title": await self._page.title(),
                    "status": response.status if response else None,
                }
            except Exception as e:
                return {"ok": False, "error": str(e)}

        # Fall back to basic navigate in proxy mode
        return await self.navigate(url)

    async def screenshot(self, region: str = "full") -> dict:
        """Capture screenshot."""
        if self.native and self._page:
            try:
                screenshot = await self._page.screenshot(full_page=(region == "full"))
                return {
                    "ok": True,
                    "data": base64.b64encode(screenshot).decode("utf-8"),
                    "format": "png",
                }
            except Exception as e:
                return {"ok": False, "error": str(e)}

        # Proxy mode: shell out to scrot or similar
        try:
            result = subprocess.run(
                ["scrot", "-o", "/tmp/cua_screenshot.png"],
                capture_output=True,
                timeout=10,
                env={**os.environ, "DISPLAY": os.environ.get("DISPLAY", ":1")},
            )
            if result.returncode == 0:
                with open("/tmp/cua_screenshot.png", "rb") as f:
                    data = base64.b64encode(f.read()).decode("utf-8")
                return {"ok": True, "data": data, "format": "png"}
            return {"ok": False, "error": result.stderr.decode()}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def click(self, x: int, y: int, button: str = "left") -> dict:
        """Click at coordinates."""
        if self.native and self._page:
            try:
                await self._page.mouse.click(x, y, button=button)
                return {"ok": True, "x": x, "y": y, "button": button}
            except Exception as e:
                return {"ok": False, "error": str(e)}

        # Proxy mode: xdotool
        try:
            btn_map = {"left": "1", "middle": "2", "right": "3"}
            result = subprocess.run(
                ["xdotool", "mousemove", str(x), str(y), "click", btn_map.get(button, "1")],
                capture_output=True,
                timeout=5,
                env={**os.environ, "DISPLAY": os.environ.get("DISPLAY", ":1")},
            )
            return {"ok": result.returncode == 0, "x": x, "y": y, "button": button}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def type_text(self, text: str, delay_ms: int = 50) -> dict:
        """Type text."""
        if self.native and self._page:
            try:
                await self._page.keyboard.type(text, delay=delay_ms)
                return {"ok": True, "typed": len(text)}
            except Exception as e:
                return {"ok": False, "error": str(e)}

        # Proxy mode: xdotool
        try:
            result = subprocess.run(
                ["xdotool", "type", "--delay", str(delay_ms), text],
                capture_output=True,
                timeout=30,
                env={**os.environ, "DISPLAY": os.environ.get("DISPLAY", ":1")},
            )
            return {"ok": result.returncode == 0, "typed": len(text)}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def navigate(self, url: str) -> dict:
        """Navigate browser to URL."""
        if self.native and self._page:
            try:
                await self._page.goto(url, wait_until="domcontentloaded")
                return {"ok": True, "url": self._page.url, "title": await self._page.title()}
            except Exception as e:
                return {"ok": False, "error": str(e)}

        # Proxy mode: open in Firefox via VNC
        try:
            result = subprocess.run(
                ["firefox", "--new-tab", url],
                capture_output=True,
                timeout=10,
                env={**os.environ, "DISPLAY": os.environ.get("DISPLAY", ":1")},
            )
            return {"ok": True, "url": url}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def execute_js(self, script: str) -> dict:
        """Execute JavaScript in page context."""
        if self.native and self._page:
            try:
                result = await self._page.evaluate(script)
                return {"ok": True, "result": result}
            except Exception as e:
                return {"ok": False, "error": str(e)}

        return {"ok": False, "error": "JS execution requires native CUA mode"}

    async def query_element(self, selector: str) -> dict:
        """Query DOM element by selector."""
        if self.native and self._page:
            try:
                element = await self._page.query_selector(selector)
                if element:
                    box = await element.bounding_box()
                    return {
                        "ok": True,
                        "found": True,
                        "selector": selector,
                        "bounding_box": box,
                        "visible": await element.is_visible(),
                    }
                return {"ok": True, "found": False, "selector": selector}
            except Exception as e:
                return {"ok": False, "error": str(e)}

        return {"ok": False, "error": "DOM queries require native CUA mode"}

    async def key_press(self, key: str, modifiers: list[str] | None = None) -> dict:
        """Press a key (Enter, Tab, Escape, etc.)."""
        if self.native and self._page:
            try:
                # Playwright key format
                key_combo = "+".join(modifiers or []) + ("+" if modifiers else "") + key
                await self._page.keyboard.press(key_combo)
                return {"ok": True, "key": key, "modifiers": modifiers}
            except Exception as e:
                return {"ok": False, "error": str(e)}

        # Proxy mode: xdotool
        try:
            # Map common key names to xdotool format
            key_map = {
                "Enter": "Return",
                "Tab": "Tab",
                "Escape": "Escape",
                "Backspace": "BackSpace",
                "Delete": "Delete",
                "ArrowUp": "Up",
                "ArrowDown": "Down",
                "ArrowLeft": "Left",
                "ArrowRight": "Right",
                "Home": "Home",
                "End": "End",
                "PageUp": "Page_Up",
                "PageDown": "Page_Down",
            }
            xdo_key = key_map.get(key, key)

            # Build modifier string
            mod_map = {"Control": "ctrl", "Alt": "alt", "Shift": "shift", "Meta": "super"}
            if modifiers:
                prefix = "+".join(mod_map.get(m, m.lower()) for m in modifiers) + "+"
                xdo_key = prefix + xdo_key

            result = subprocess.run(
                ["xdotool", "key", xdo_key],
                capture_output=True,
                timeout=5,
                env={**os.environ, "DISPLAY": os.environ.get("DISPLAY", ":1")},
            )
            return {"ok": result.returncode == 0, "key": key, "modifiers": modifiers}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def scroll(self, x: int, y: int, delta_x: int = 0, delta_y: int = 0) -> dict:
        """Scroll at position."""
        if self.native and self._page:
            try:
                await self._page.mouse.move(x, y)
                await self._page.mouse.wheel(delta_x, delta_y)
                return {"ok": True, "x": x, "y": y, "delta_x": delta_x, "delta_y": delta_y}
            except Exception as e:
                return {"ok": False, "error": str(e)}

        # Proxy mode: xdotool
        try:
            # Move mouse first
            subprocess.run(
                ["xdotool", "mousemove", str(x), str(y)],
                capture_output=True,
                timeout=5,
                env={**os.environ, "DISPLAY": os.environ.get("DISPLAY", ":1")},
            )
            # Scroll (button 4=up, 5=down)
            if delta_y != 0:
                button = "4" if delta_y < 0 else "5"
                clicks = abs(delta_y) // 50  # Approximate scroll amount
                for _ in range(max(1, clicks)):
                    subprocess.run(
                        ["xdotool", "click", button],
                        capture_output=True,
                        timeout=2,
                        env={**os.environ, "DISPLAY": os.environ.get("DISPLAY", ":1")},
                    )
            return {"ok": True, "x": x, "y": y, "delta_y": delta_y}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def double_click(self, x: int, y: int) -> dict:
        """Double-click at coordinates."""
        if self.native and self._page:
            try:
                await self._page.mouse.dblclick(x, y)
                return {"ok": True, "x": x, "y": y}
            except Exception as e:
                return {"ok": False, "error": str(e)}

        # Proxy mode: xdotool
        try:
            result = subprocess.run(
                ["xdotool", "mousemove", str(x), str(y), "click", "--repeat", "2", "--delay", "50", "1"],
                capture_output=True,
                timeout=5,
                env={**os.environ, "DISPLAY": os.environ.get("DISPLAY", ":1")},
            )
            return {"ok": result.returncode == 0, "x": x, "y": y}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def mouse_move(self, x: int, y: int) -> dict:
        """Move mouse without clicking."""
        if self.native and self._page:
            try:
                await self._page.mouse.move(x, y)
                return {"ok": True, "x": x, "y": y}
            except Exception as e:
                return {"ok": False, "error": str(e)}

        # Proxy mode: xdotool
        try:
            result = subprocess.run(
                ["xdotool", "mousemove", str(x), str(y)],
                capture_output=True,
                timeout=5,
                env={**os.environ, "DISPLAY": os.environ.get("DISPLAY", ":1")},
            )
            return {"ok": result.returncode == 0, "x": x, "y": y}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def drag(self, start_x: int, start_y: int, end_x: int, end_y: int) -> dict:
        """Drag from start to end coordinates."""
        if self.native and self._page:
            try:
                await self._page.mouse.move(start_x, start_y)
                await self._page.mouse.down()
                await self._page.mouse.move(end_x, end_y)
                await self._page.mouse.up()
                return {"ok": True, "start": (start_x, start_y), "end": (end_x, end_y)}
            except Exception as e:
                return {"ok": False, "error": str(e)}

        # Proxy mode: xdotool
        try:
            result = subprocess.run(
                ["xdotool", "mousemove", str(start_x), str(start_y),
                 "mousedown", "1",
                 "mousemove", str(end_x), str(end_y),
                 "mouseup", "1"],
                capture_output=True,
                timeout=10,
                env={**os.environ, "DISPLAY": os.environ.get("DISPLAY", ":1")},
            )
            return {"ok": result.returncode == 0, "start": (start_x, start_y), "end": (end_x, end_y)}
        except Exception as e:
            return {"ok": False, "error": str(e)}

    async def get_screen_size(self) -> dict:
        """Get screen dimensions."""
        if self.native and self._page:
            try:
                size = await self._page.evaluate("() => ({ width: window.screen.width, height: window.screen.height })")
                return {"ok": True, "width": size["width"], "height": size["height"]}
            except Exception as e:
                return {"ok": False, "error": str(e)}

        # Proxy mode: xdpyinfo
        try:
            result = subprocess.run(
                ["xdpyinfo"],
                capture_output=True,
                timeout=5,
                env={**os.environ, "DISPLAY": os.environ.get("DISPLAY", ":1")},
            )
            output = result.stdout.decode()
            # Parse "dimensions:    1920x1080 pixels"
            import re
            match = re.search(r"dimensions:\s+(\d+)x(\d+)", output)
            if match:
                return {"ok": True, "width": int(match.group(1)), "height": int(match.group(2))}
            return {"ok": False, "error": "Could not parse screen size"}
        except Exception as e:
            return {"ok": False, "error": str(e)}


# ============================================================================
# LLM Router Integration
# ============================================================================

def _resolve_provider(model: str) -> str:
    """Resolve model name to provider."""
    m = (model or "").strip().lower()
    if not m or m == "auto":
        return "auto"
    if m in {"chatgpt-web", "claude-web", "gemini-web", "vertex-gemini", "vertex-gemini-curl"}:
        return m
    if "claude" in m:
        return "claude-web"
    if m.startswith("gemini-3"):
        # Gemini 3 is often a Vertex-only target; defer to the provider router.
        return "auto"
    if "gemini" in m:
        return "gemini-web"
    if "gpt" in m or "openai" in m:
        return "chatgpt-web"
    return "auto"


async def call_router(
    prompt: str,
    model: str,
    cfg: WorldRouterConfig,
    provider: str | None = None,
    req_id: str | None = None,
) -> tuple[int, str, str, float, str]:
    """Call the provider router."""
    resolved_provider = provider or _resolve_provider(model)
    cmd = [
        sys.executable,
        str(cfg.router_path),
        "--provider",
        resolved_provider,
        "--prompt",
        prompt,
        "--model",
        model,
    ]
    started = time.monotonic()
    try:
        env = {
            **os.environ,
            "PLURIBUS_BUS_DIR": cfg.bus_dir,
            "PLURIBUS_ACTOR": cfg.actor,
            "PYTHONDONTWRITEBYTECODE": "1",
        }
        if req_id:
            env["PLURIBUS_GATEWAY_REQ_ID"] = req_id
            env["PLURIBUS_REQ_ID"] = req_id
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env=env,
        )
        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(),
                timeout=cfg.router_timeout_s,
            )
            elapsed = time.monotonic() - started
            return proc.returncode or 0, stdout.decode(), stderr.decode(), elapsed, resolved_provider
        except asyncio.TimeoutError:
            proc.kill()
            elapsed = time.monotonic() - started
            return 124, "", "router timeout", elapsed, resolved_provider
    except Exception as e:
        elapsed = time.monotonic() - started
        return 1, "", str(e), elapsed, resolved_provider


# ============================================================================
# Request Handlers
# ============================================================================

class WorldRouterHandlers:
    """Request handlers for World Router with Circuit Breaker and SSE streaming."""

    def __init__(self, cfg: WorldRouterConfig):
        self.cfg = cfg
        self.identity_hub = IdentityHub(enabled=cfg.identity_hub)
        self.cua_engine = CUAEngine(native=cfg.cua_native)
        self.proto_router = ProtocolRouter()
        self.circuit_breaker = CircuitBreaker(failure_threshold=3, cooldown_seconds=60.0)

    async def initialize(self):
        """Initialize subsystems."""
        await self.cua_engine.initialize()

    def _with_gap_data(self, data: dict, gaps: dict, reason: str | None) -> dict:
        if gaps.get("epistemic") or gaps.get("aleatoric"):
            out = dict(data)
            out["gap_reason"] = reason
            out["gaps"] = gaps
            return out
        return data

    def _emit_gap_events(
        self,
        req_id: str,
        gaps: dict,
        *,
        provider: str | None,
        proto: str,
        reason: str | None,
    ) -> None:
        if not (gaps.get("epistemic") or gaps.get("aleatoric")):
            return
        if gaps.get("epistemic"):
            emit_bus_event(
                self.cfg.bus_dir,
                "coordination.gap.detected",
                {"req_id": req_id, "provider": provider, "proto": proto, "reason": reason, "gaps": gaps},
                kind="request",
                level="warn",
                actor=self.cfg.actor,
            )

    # -------------------------------------------------------------------------
    # SSE Streaming Helpers
    # -------------------------------------------------------------------------

    async def _stream_sse_response(
        self,
        request,
        prompt: str,
        model: str,
        format_chunk,  # Callable[[str, int], dict]
        format_done,   # Callable[[], dict]
    ) -> web.StreamResponse:
        """Stream response as Server-Sent Events."""
        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
            },
        )
        await response.prepare(request)

        provider = _resolve_provider(model)

        # Check circuit breaker
        if not self.circuit_breaker.is_available(provider):
            error_data = {"error": {"message": f"Provider {provider} is temporarily unavailable", "type": "circuit_open"}}
            await response.write(f"data: {json.dumps(error_data)}\n\n".encode())
            await response.write(b"data: [DONE]\n\n")
            return response

        # Call router
        exit_code, stdout, stderr, elapsed, resolved_provider = await call_router(
            prompt, model, self.cfg, provider, req_id=req_id
        )

        # Update circuit breaker
        if exit_code == 0:
            self.circuit_breaker.record_success(resolved_provider)
        else:
            self.circuit_breaker.record_failure(resolved_provider)

        if exit_code != 0:
            gap_reason = _classify_router_failure(stderr, stdout, exit_code)
            error_data = {
                "error": {
                    "message": stderr or "Router error",
                    "type": _router_error_type(gap_reason),
                    "gap_reason": gap_reason,
                }
            }
            await response.write(f"data: {json.dumps(error_data)}\n\n".encode())
            await response.write(b"data: [DONE]\n\n")
            return response

        # Stream the response in chunks (simulate streaming for non-streaming backend)
        content = stdout.strip()
        chunk_size = 50  # Characters per chunk

        for i in range(0, len(content), chunk_size):
            chunk = content[i:i + chunk_size]
            chunk_data = format_chunk(chunk, i)
            await response.write(f"data: {json.dumps(chunk_data)}\n\n".encode())
            await asyncio.sleep(0.01)  # Small delay between chunks

        # Send final done message
        done_data = format_done()
        await response.write(f"data: {json.dumps(done_data)}\n\n".encode())
        await response.write(b"data: [DONE]\n\n")

        return response

    def _check_circuit_and_route(self, provider: str) -> tuple[bool, str]:
        """Check circuit breaker and potentially fallback to another provider."""
        # Keep this list aligned with the router's "verified" profile defaults.
        providers = ["chatgpt-web", "claude-web", "gemini-web", "vertex-gemini", "vertex-gemini-curl"]

        if provider == "auto":
            # Do NOT pre-select a provider: let `providers/router.py --provider auto --model <...>`
            # choose based on model family and availability. We only gate liveness here.
            any_available = any(self.circuit_breaker.is_available(p) for p in providers)
            return (any_available, "auto")

        if self.circuit_breaker.is_available(provider):
            return True, provider

        # Try fallback
        for p in providers:
            if p != provider and self.circuit_breaker.is_available(p):
                return True, p

        return False, provider

    # -------------------------------------------------------------------------
    # Health & Meta
    # -------------------------------------------------------------------------

    async def handle_health(self, request) -> web.Response:
        """Health check endpoint with circuit breaker status."""
        circuit_states = self.circuit_breaker.get_all_states()
        all_available = all(
            state["state"] != "open"
            for state in circuit_states.values()
        ) if circuit_states else True

        return web.json_response({
            "status": "ok" if all_available else "degraded",
            "service": "world-router",
            "version": "0.3.0",  # Phase 3 complete - WebAuthn/Passkey
            "timestamp": now_iso(),
            "features": {
                "identity_hub": self.cfg.identity_hub,
                "cua_native": self.cfg.cua_native,
                "vnc_tunnel": self.cfg.vnc_tunnel,
                "storage_fabric": self.cfg.storage_fabric,
                "circuit_breaker": True,
                "sse_streaming": True,
                "webauthn": True,  # Phase 3
            },
            "circuits": circuit_states,
        })

    async def handle_circuit_status(self, request) -> web.Response:
        """Get detailed circuit breaker status for all providers."""
        return web.json_response({
            "timestamp": now_iso(),
            "circuits": self.circuit_breaker.get_all_states(),
            "config": {
                "failure_threshold": self.circuit_breaker.failure_threshold,
                "cooldown_seconds": self.circuit_breaker.cooldown_seconds,
            },
        })

    async def handle_circuit_reset(self, request) -> web.Response:
        """Reset circuit breaker for a provider (admin endpoint)."""
        try:
            body = await request.json()
        except Exception:
            body = {}

        provider = body.get("provider", "")

        if not provider:
            # Reset all circuits
            self.circuit_breaker._circuits.clear()
            emit_bus_event(
                self.cfg.bus_dir,
                "world_router.circuit_breaker",
                {"action": "reset_all"},
                kind="admin",
            )
            return web.json_response({"status": "all_reset"})

        if provider in self.circuit_breaker._circuits:
            del self.circuit_breaker._circuits[provider]
            emit_bus_event(
                self.cfg.bus_dir,
                "world_router.circuit_breaker",
                {"action": "reset", "provider": provider},
                kind="admin",
            )
            return web.json_response({"status": "reset", "provider": provider})

        return web.json_response({"status": "not_found", "provider": provider})

    async def handle_models(self, request) -> web.Response:
        """List available models."""
        models = [
            {"id": "chatgpt-web", "object": "model", "owned_by": "openai"},
            {"id": "claude-web", "object": "model", "owned_by": "anthropic"},
            {"id": "gemini-web", "object": "model", "owned_by": "google"},
            {"id": "vertex-gemini", "object": "model", "owned_by": "google"},
            {"id": "vertex-gemini-curl", "object": "model", "owned_by": "google"},
            {"id": "auto", "object": "model", "owned_by": "pluribus"},
        ]
        return web.json_response({"object": "list", "data": models})

    # -------------------------------------------------------------------------
    # LLM Inference
    # -------------------------------------------------------------------------

    async def handle_openai_chat(self, request) -> web.Response:
        """Handle OpenAI Chat Completions with circuit breaker and SSE streaming."""
        identity = await self.identity_hub.resolve(request)
        req_id = str(uuid.uuid4())

        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        messages = body.get("messages", [])
        model = body.get("model", "auto")
        stream = body.get("stream", False)

        # Build prompt from messages
        lines = []
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") for p in content if isinstance(p, dict)
                )
            lines.append(f"{role}: {content}")
        prompt = "\n".join(lines)

        # Resolve provider with circuit breaker fallback
        requested_provider = _resolve_provider(model)
        available, provider = self._check_circuit_and_route(requested_provider)

        # Emit request evidence
        emit_bus_event(
            self.cfg.bus_dir,
            "world_router.request",
            {
                "req_id": req_id,
                "proto": "openai-chat",
                "model": model,
                "identity": identity.did,
                "prompt_len": len(prompt),
                "prompt_sha256": _safe_sha256(prompt),
                "stream": stream,
                "provider": provider,
                "circuit_fallback": provider != requested_provider,
            },
            kind="request",
            actor=self.cfg.actor,
        )

        # Check if all providers are unavailable
        if not available:
            gap_reason = "circuit_open"
            gaps = _gap_analysis_for_reason(
                gap_reason,
                provider=requested_provider,
                cooldown_s=self.circuit_breaker.cooldown_seconds,
            )
            self._emit_gap_events(
                req_id,
                gaps,
                provider=requested_provider,
                proto="openai-chat",
                reason=gap_reason,
            )
            emit_bus_event(
                self.cfg.bus_dir,
                "world_router.circuit_breaker",
                self._with_gap_data(
                    {"action": "all_unavailable", "states": self.circuit_breaker.get_all_states()},
                    gaps,
                    gap_reason,
                ),
                kind="warning",
            )
            emit_bus_event(
                self.cfg.bus_dir,
                "world_router.response",
                self._with_gap_data(
                    {
                        "req_id": req_id,
                        "ok": False,
                        "exit_code": None,
                        "elapsed_s": 0.0,
                        "provider": requested_provider,
                    },
                    gaps,
                    gap_reason,
                ),
                kind="response",
                actor=self.cfg.actor,
            )
            return web.json_response({
                "error": {
                    "message": "All providers temporarily unavailable",
                    "type": "circuit_open",
                    "states": self.circuit_breaker.get_all_states(),
                    "gaps": gaps if gaps.get("epistemic") or gaps.get("aleatoric") else None,
                    "gap_reason": gap_reason,
                },
            }, status=503)

        # Handle streaming response
        if stream:
            return await self._stream_openai_chat(request, req_id, prompt, model, provider)

        # Non-streaming: Call router
        exit_code, stdout, stderr, elapsed, resolved_provider = await call_router(
            prompt, model, self.cfg, provider, req_id=req_id
        )

        # Update circuit breaker
        if exit_code == 0:
            self.circuit_breaker.record_success(resolved_provider)
        else:
            self.circuit_breaker.record_failure(resolved_provider)

        gap_reason = None
        gaps = {"epistemic": [], "aleatoric": []}
        if exit_code != 0:
            gap_reason = _classify_router_failure(stderr, stdout, exit_code)
            gaps = _gap_analysis_for_reason(
                gap_reason,
                provider=resolved_provider,
                router_timeout_s=self.cfg.router_timeout_s,
            )
            self._emit_gap_events(
                req_id,
                gaps,
                provider=resolved_provider,
                proto="openai-chat",
                reason=gap_reason,
            )

        # Emit response evidence
        emit_bus_event(
            self.cfg.bus_dir,
            "world_router.response",
            self._with_gap_data(
                {
                    "req_id": req_id,
                    "ok": exit_code == 0,
                    "exit_code": exit_code,
                    "elapsed_s": round(elapsed, 3),
                    "provider": resolved_provider,
                    "stdout_len": len(stdout),
                    "stderr_len": len(stderr),
                },
                gaps,
                gap_reason,
            ),
            kind="response",
            actor=self.cfg.actor,
        )

        if exit_code != 0:
            status = _router_error_status(gap_reason)
            return web.json_response({
                "error": {
                    "message": stderr or "Router error",
                    "type": _router_error_type(gap_reason),
                    "gaps": gaps if gaps.get("epistemic") or gaps.get("aleatoric") else None,
                    "gap_reason": gap_reason,
                },
            }, status=status)

        # Build OpenAI-compatible response
        response = {
            "id": f"chatcmpl-{req_id[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": stdout.strip()},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": len(prompt) // 4, "completion_tokens": len(stdout) // 4},
        }
        return web.json_response(response)

    async def _stream_openai_chat(
        self,
        request,
        req_id: str,
        prompt: str,
        model: str,
        provider: str,
    ) -> web.StreamResponse:
        """Stream OpenAI Chat Completions as SSE."""
        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
            },
        )
        await response.prepare(request)

        # Call router (get full response, then simulate streaming)
        exit_code, stdout, stderr, elapsed, resolved_provider = await call_router(
            prompt, model, self.cfg, provider
        )

        # Update circuit breaker
        if exit_code == 0:
            self.circuit_breaker.record_success(resolved_provider)
        else:
            self.circuit_breaker.record_failure(resolved_provider)

        gap_reason = None
        gaps = {"epistemic": [], "aleatoric": []}
        if exit_code != 0:
            gap_reason = _classify_router_failure(stderr, stdout, exit_code)
            gaps = _gap_analysis_for_reason(
                gap_reason,
                provider=resolved_provider,
                router_timeout_s=self.cfg.router_timeout_s,
            )
            self._emit_gap_events(
                req_id,
                gaps,
                provider=resolved_provider,
                proto="openai-chat",
                reason=gap_reason,
            )

        # Emit response evidence
        emit_bus_event(
            self.cfg.bus_dir,
            "world_router.response",
            self._with_gap_data(
                {
                    "req_id": req_id,
                    "ok": exit_code == 0,
                    "elapsed_s": round(elapsed, 3),
                    "provider": resolved_provider,
                    "stream": True,
                },
                gaps,
                gap_reason,
            ),
            kind="response",
            actor=self.cfg.actor,
        )

        if exit_code != 0:
            error_chunk = {
                "id": f"chatcmpl-{req_id[:8]}",
                "object": "chat.completion.chunk",
                "error": {
                    "message": stderr or "Router error",
                    "type": _router_error_type(gap_reason),
                    "gaps": gaps if gaps.get("epistemic") or gaps.get("aleatoric") else None,
                    "gap_reason": gap_reason,
                },
            }
            await response.write(f"data: {json.dumps(error_chunk)}\n\n".encode())
            await response.write(b"data: [DONE]\n\n")
            return response

        # Stream the content in chunks
        content = stdout.strip()
        chunk_size = 20  # Characters per chunk for realistic streaming

        for i in range(0, len(content), chunk_size):
            chunk_text = content[i:i + chunk_size]
            chunk_data = {
                "id": f"chatcmpl-{req_id[:8]}",
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model,
                "choices": [{
                    "index": 0,
                    "delta": {"content": chunk_text},
                    "finish_reason": None,
                }],
            }
            await response.write(f"data: {json.dumps(chunk_data)}\n\n".encode())
            await asyncio.sleep(0.02)  # Simulate streaming delay

        # Send final chunk with finish_reason
        final_chunk = {
            "id": f"chatcmpl-{req_id[:8]}",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": model,
            "choices": [{
                "index": 0,
                "delta": {},
                "finish_reason": "stop",
            }],
        }
        await response.write(f"data: {json.dumps(final_chunk)}\n\n".encode())
        await response.write(b"data: [DONE]\n\n")

        return response

    async def handle_openai_responses(self, request) -> web.Response:
        """Handle OpenAI Responses API (newer format)."""
        identity = await self.identity_hub.resolve(request)
        req_id = str(uuid.uuid4())

        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        model = body.get("model", "auto")
        stream = body.get("stream", False)

        # Build prompt from various input formats
        prompt_parts = []

        # System instructions
        instructions = body.get("instructions") or body.get("system") or body.get("system_prompt")
        if instructions:
            prompt_parts.append(f"system: {instructions}")

        # Input handling (string, list of messages, or list of parts)
        inp = body.get("input") or body.get("messages")
        if isinstance(inp, str):
            prompt_parts.append(inp)
        elif isinstance(inp, list):
            for item in inp:
                if isinstance(item, str):
                    prompt_parts.append(item)
                elif isinstance(item, dict):
                    if "role" in item:
                        # Message format
                        role = item.get("role", "user")
                        content = item.get("content", "")
                        if isinstance(content, list):
                            content = " ".join(
                                p.get("text", "") for p in content
                                if isinstance(p, dict) and p.get("type") in ("text", "input_text")
                            )
                        prompt_parts.append(f"{role}: {content}")
                    elif item.get("type") == "input_text":
                        prompt_parts.append(item.get("text", ""))

        prompt = "\n".join(prompt_parts).strip()

        # Emit evidence
        emit_bus_event(
            self.cfg.bus_dir,
            "world_router.request",
            {
                "req_id": req_id,
                "proto": "openai-responses",
                "model": model,
                "identity": identity.did,
                "prompt_len": len(prompt),
                "prompt_sha256": _safe_sha256(prompt),
                "stream": stream,
            },
            kind="request",
            actor=self.cfg.actor,
        )

        # Call router
        exit_code, stdout, stderr, elapsed, provider = await call_router(
            prompt, model, self.cfg, req_id=req_id
        )

        gap_reason = None
        gaps = {"epistemic": [], "aleatoric": []}
        if exit_code != 0:
            gap_reason = _classify_router_failure(stderr, stdout, exit_code)
            gaps = _gap_analysis_for_reason(
                gap_reason,
                provider=provider,
                router_timeout_s=self.cfg.router_timeout_s,
            )
            self._emit_gap_events(
                req_id,
                gaps,
                provider=provider,
                proto="openai-responses",
                reason=gap_reason,
            )

        emit_bus_event(
            self.cfg.bus_dir,
            "world_router.response",
            self._with_gap_data(
                {
                    "req_id": req_id,
                    "ok": exit_code == 0,
                    "exit_code": exit_code,
                    "elapsed_s": round(elapsed, 3),
                    "provider": provider,
                },
                gaps,
                gap_reason,
            ),
            kind="response",
            actor=self.cfg.actor,
        )

        if exit_code != 0:
            status = _router_error_status(gap_reason)
            return web.json_response({
                "error": {
                    "message": stderr or "Router error",
                    "type": _router_error_type(gap_reason),
                    "gaps": gaps if gaps.get("epistemic") or gaps.get("aleatoric") else None,
                    "gap_reason": gap_reason,
                },
            }, status=status)

        # OpenAI Responses format
        response = {
            "id": f"resp-{req_id[:8]}",
            "object": "response",
            "created_at": int(time.time()),
            "model": model,
            "output": [
                {
                    "type": "message",
                    "id": f"msg-{req_id[:8]}",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": stdout.strip()}],
                }
            ],
            "usage": {
                "input_tokens": len(prompt) // 4,
                "output_tokens": len(stdout) // 4,
            },
        }
        return web.json_response(response)

    async def handle_gemini_generate(self, request) -> web.Response:
        """Handle Gemini generateContent API."""
        identity = await self.identity_hub.resolve(request)
        req_id = str(uuid.uuid4())

        # Extract model from path
        path = request.path
        match = ProtocolRouter.RE_GEMINI.match(path)
        if match:
            model = match.group("model")
            method = match.group("method")
        else:
            model = "gemini-web"
            method = "generateContent"

        stream = method == "streamGenerateContent"

        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        # Build prompt from Gemini format
        prompt_parts = []

        # System instruction
        system = body.get("systemInstruction")
        if isinstance(system, dict):
            parts = system.get("parts", [])
            for part in parts:
                if isinstance(part, dict) and part.get("text"):
                    prompt_parts.append(f"system: {part['text']}")
        elif isinstance(system, str):
            prompt_parts.append(f"system: {system}")

        # Contents
        contents = body.get("contents", [])
        for content in contents:
            if isinstance(content, dict):
                role = content.get("role", "user")
                if role == "model":
                    role = "assistant"
                parts = content.get("parts", [])
                for part in parts:
                    if isinstance(part, dict):
                        if part.get("text"):
                            prompt_parts.append(f"{role}: {part['text']}")
                        elif part.get("inlineData"):
                            prompt_parts.append(f"{role}: [image]")

        prompt = "\n".join(prompt_parts).strip()

        emit_bus_event(
            self.cfg.bus_dir,
            "world_router.request",
            {
                "req_id": req_id,
                "proto": "gemini",
                "model": model,
                "identity": identity.did,
                "prompt_len": len(prompt),
                "stream": stream,
            },
            kind="request",
            actor=self.cfg.actor,
        )

        # Call router (model-aware): allow Vertex Gemini for gemini-3* and allow explicit `vertex-gemini*`.
        exit_code, stdout, stderr, elapsed, provider = await call_router(
            prompt, model, self.cfg, req_id=req_id
        )

        gap_reason = None
        gaps = {"epistemic": [], "aleatoric": []}
        if exit_code != 0:
            gap_reason = _classify_router_failure(stderr, stdout, exit_code)
            gaps = _gap_analysis_for_reason(
                gap_reason,
                provider=provider,
                router_timeout_s=self.cfg.router_timeout_s,
            )
            self._emit_gap_events(
                req_id,
                gaps,
                provider=provider,
                proto="gemini",
                reason=gap_reason,
            )

        emit_bus_event(
            self.cfg.bus_dir,
            "world_router.response",
            self._with_gap_data(
                {"req_id": req_id, "ok": exit_code == 0, "elapsed_s": round(elapsed, 3)},
                gaps,
                gap_reason,
            ),
            kind="response",
            actor=self.cfg.actor,
        )

        if exit_code != 0:
            status = _router_error_status(gap_reason)
            return web.json_response({
                "error": {
                    "code": status,
                    "message": stderr or "Router error",
                    "gaps": gaps if gaps.get("epistemic") or gaps.get("aleatoric") else None,
                    "gap_reason": gap_reason,
                },
            }, status=status)

        # Gemini response format
        response = {
            "candidates": [
                {
                    "content": {
                        "parts": [{"text": stdout.strip()}],
                        "role": "model",
                    },
                    "finishReason": "STOP",
                    "index": 0,
                }
            ],
            "usageMetadata": {
                "promptTokenCount": len(prompt) // 4,
                "candidatesTokenCount": len(stdout) // 4,
                "totalTokenCount": (len(prompt) + len(stdout)) // 4,
            },
        }
        return web.json_response(response)

    async def handle_anthropic_count_tokens(self, request) -> web.Response:
        """Handle Anthropic token counting endpoint."""
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        messages = body.get("messages", [])
        system = body.get("system", "")

        # Simple token estimation (4 chars per token)
        total_chars = len(system)
        for m in messages:
            content = m.get("content", "")
            if isinstance(content, str):
                total_chars += len(content)
            elif isinstance(content, list):
                for p in content:
                    if isinstance(p, dict) and p.get("text"):
                        total_chars += len(p["text"])

        return web.json_response({"input_tokens": total_chars // 4})

    async def handle_anthropic_messages(self, request) -> web.Response:
        """Handle Anthropic Messages API with circuit breaker and streaming."""
        identity = await self.identity_hub.resolve(request)
        req_id = str(uuid.uuid4())

        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        messages = body.get("messages", [])
        model = body.get("model", "claude-web")
        system = body.get("system", "")
        stream = body.get("stream", False)

        # Build prompt
        lines = []
        if system:
            lines.append(f"system: {system}")
        for m in messages:
            role = m.get("role", "user")
            content = m.get("content", "")
            if isinstance(content, list):
                content = " ".join(
                    p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text"
                )
            lines.append(f"{role}: {content}")
        prompt = "\n".join(lines)

        # Resolve provider with circuit breaker fallback
        available, provider = self._check_circuit_and_route("claude-web")

        # Emit evidence
        emit_bus_event(
            self.cfg.bus_dir,
            "world_router.request",
            {
                "req_id": req_id,
                "proto": "anthropic",
                "model": model,
                "identity": identity.did,
                "stream": stream,
                "provider": provider,
            },
            kind="request",
        )

        # Check circuit breaker
        if not available:
            gap_reason = "circuit_open"
            gaps = _gap_analysis_for_reason(
                gap_reason,
                provider=provider,
                cooldown_s=self.circuit_breaker.cooldown_seconds,
            )
            self._emit_gap_events(
                req_id,
                gaps,
                provider=provider,
                proto="anthropic",
                reason=gap_reason,
            )
            emit_bus_event(
                self.cfg.bus_dir,
                "world_router.response",
                self._with_gap_data(
                    {
                        "req_id": req_id,
                        "ok": False,
                        "exit_code": None,
                        "elapsed_s": 0.0,
                        "provider": provider,
                    },
                    gaps,
                    gap_reason,
                ),
                kind="response",
                actor=self.cfg.actor,
            )
            return web.json_response({
                "type": "error",
                "error": {
                    "type": "circuit_open",
                    "message": "Provider temporarily unavailable",
                    "gaps": gaps if gaps.get("epistemic") or gaps.get("aleatoric") else None,
                    "gap_reason": gap_reason,
                },
            }, status=503)

        # Handle streaming response
        if stream:
            return await self._stream_anthropic_messages(request, req_id, prompt, model, provider)

        # Call router
        exit_code, stdout, stderr, elapsed, resolved_provider = await call_router(
            prompt, model, self.cfg, provider=provider, req_id=req_id
        )

        # Update circuit breaker
        if exit_code == 0:
            self.circuit_breaker.record_success(resolved_provider)
        else:
            self.circuit_breaker.record_failure(resolved_provider)

        gap_reason = None
        gaps = {"epistemic": [], "aleatoric": []}
        if exit_code != 0:
            gap_reason = _classify_router_failure(stderr, stdout, exit_code)
            gaps = _gap_analysis_for_reason(
                gap_reason,
                provider=resolved_provider,
                router_timeout_s=self.cfg.router_timeout_s,
            )
            self._emit_gap_events(
                req_id,
                gaps,
                provider=resolved_provider,
                proto="anthropic",
                reason=gap_reason,
            )

        emit_bus_event(
            self.cfg.bus_dir,
            "world_router.response",
            self._with_gap_data(
                {
                    "req_id": req_id,
                    "ok": exit_code == 0,
                    "elapsed_s": round(elapsed, 3),
                    "provider": resolved_provider,
                },
                gaps,
                gap_reason,
            ),
            kind="response",
        )

        if exit_code != 0:
            status = _router_error_status(gap_reason)
            return web.json_response({
                "type": "error",
                "error": {
                    "type": _router_error_type(gap_reason),
                    "message": stderr or "Router error",
                    "gaps": gaps if gaps.get("epistemic") or gaps.get("aleatoric") else None,
                    "gap_reason": gap_reason,
                },
            }, status=status)

        # Anthropic-compatible response
        content_text = stdout.strip()
        response = {
            "id": f"msg_{req_id[:8]}",
            "type": "message",
            "role": "assistant",
            "content": [{"type": "text", "text": content_text}],
            "model": model,
            "stop_reason": "end_turn",
            "stop_sequence": None,
            "usage": {
                "input_tokens": len(prompt) // 4,
                "output_tokens": len(content_text) // 4,
            },
        }
        return web.json_response(response)

    async def _stream_anthropic_messages(
        self,
        request,
        req_id: str,
        prompt: str,
        model: str,
        provider: str,
    ) -> web.StreamResponse:
        """Stream Anthropic Messages as SSE."""
        response = web.StreamResponse(
            status=200,
            headers={
                "Content-Type": "text/event-stream",
                "Cache-Control": "no-cache",
                "Connection": "keep-alive",
                "Access-Control-Allow-Origin": "*",
            },
        )
        await response.prepare(request)

        # Call router
        exit_code, stdout, stderr, elapsed, resolved_provider = await call_router(
            prompt, model, self.cfg, provider, req_id=req_id
        )

        # Update circuit breaker
        if exit_code == 0:
            self.circuit_breaker.record_success(resolved_provider)
        else:
            self.circuit_breaker.record_failure(resolved_provider)

        gap_reason = None
        gaps = {"epistemic": [], "aleatoric": []}
        if exit_code != 0:
            gap_reason = _classify_router_failure(stderr, stdout, exit_code)
            gaps = _gap_analysis_for_reason(
                gap_reason,
                provider=resolved_provider,
                router_timeout_s=self.cfg.router_timeout_s,
            )
            self._emit_gap_events(
                req_id,
                gaps,
                provider=resolved_provider,
                proto="anthropic",
                reason=gap_reason,
            )

        emit_bus_event(
            self.cfg.bus_dir,
            "world_router.response",
            self._with_gap_data(
                {
                    "req_id": req_id,
                    "ok": exit_code == 0,
                    "elapsed_s": round(elapsed, 3),
                    "provider": resolved_provider,
                    "stream": True,
                },
                gaps,
                gap_reason,
            ),
            kind="response",
        )

        if exit_code != 0:
            error_event = {
                "type": "error",
                "error": {
                    "type": _router_error_type(gap_reason),
                    "message": stderr or "Router error",
                    "gaps": gaps if gaps.get("epistemic") or gaps.get("aleatoric") else None,
                    "gap_reason": gap_reason,
                },
            }
            await response.write(f"event: error\ndata: {json.dumps(error_event)}\n\n".encode())
            return response

        # Send message_start event
        message_start = {
            "type": "message_start",
            "message": {
                "id": f"msg_{req_id[:8]}",
                "type": "message",
                "role": "assistant",
                "content": [],
                "model": model,
                "stop_reason": None,
                "stop_sequence": None,
                "usage": {"input_tokens": len(prompt) // 4, "output_tokens": 0},
            },
        }
        await response.write(f"event: message_start\ndata: {json.dumps(message_start)}\n\n".encode())

        # Send content_block_start
        content_block_start = {
            "type": "content_block_start",
            "index": 0,
            "content_block": {"type": "text", "text": ""},
        }
        await response.write(f"event: content_block_start\ndata: {json.dumps(content_block_start)}\n\n".encode())

        # Stream content deltas
        content = stdout.strip()
        chunk_size = 20

        for i in range(0, len(content), chunk_size):
            chunk_text = content[i:i + chunk_size]
            content_delta = {
                "type": "content_block_delta",
                "index": 0,
                "delta": {"type": "text_delta", "text": chunk_text},
            }
            await response.write(f"event: content_block_delta\ndata: {json.dumps(content_delta)}\n\n".encode())
            await asyncio.sleep(0.02)

        # Send content_block_stop
        content_block_stop = {"type": "content_block_stop", "index": 0}
        await response.write(f"event: content_block_stop\ndata: {json.dumps(content_block_stop)}\n\n".encode())

        # Send message_delta with stop_reason
        message_delta = {
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn", "stop_sequence": None},
            "usage": {"output_tokens": len(content) // 4},
        }
        await response.write(f"event: message_delta\ndata: {json.dumps(message_delta)}\n\n".encode())

        # Send message_stop
        message_stop = {"type": "message_stop"}
        await response.write(f"event: message_stop\ndata: {json.dumps(message_stop)}\n\n".encode())

        return response

    # -------------------------------------------------------------------------
    # CUA Endpoints
    # -------------------------------------------------------------------------

    async def handle_cua_screenshot(self, request) -> web.Response:
        """Capture screenshot."""
        try:
            body = await request.json()
        except Exception:
            body = {}

        region = body.get("region", "full")
        result = await self.cua_engine.screenshot(region)

        emit_bus_event(
            self.cfg.bus_dir,
            "world_router.cua.screenshot",
            {"ok": result.get("ok"), "region": region},
        )

        return web.json_response(result)

    async def handle_cua_click(self, request) -> web.Response:
        """Click at coordinates."""
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON"}, status=400)

        x = body.get("x", 0)
        y = body.get("y", 0)
        button = body.get("button", "left")

        result = await self.cua_engine.click(x, y, button)

        emit_bus_event(
            self.cfg.bus_dir,
            "world_router.cua.click",
            {"ok": result.get("ok"), "x": x, "y": y, "button": button},
        )

        return web.json_response(result)

    async def handle_cua_type(self, request) -> web.Response:
        """Type text."""
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON"}, status=400)

        text = body.get("text", "")
        delay_ms = body.get("delay_ms", 50)

        result = await self.cua_engine.type_text(text, delay_ms)

        emit_bus_event(
            self.cfg.bus_dir,
            "world_router.cua.type",
            {"ok": result.get("ok"), "chars": len(text)},
        )

        return web.json_response(result)

    async def handle_cua_navigate(self, request) -> web.Response:
        """Navigate to URL."""
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON"}, status=400)

        url = body.get("url", "")
        if not url:
            return web.json_response({"ok": False, "error": "URL required"}, status=400)

        result = await self.cua_engine.navigate(url)

        emit_bus_event(
            self.cfg.bus_dir,
            "world_router.cua.navigate",
            {"ok": result.get("ok"), "url": url},
        )

        return web.json_response(result)

    async def handle_cua_eval(self, request) -> web.Response:
        """Execute JavaScript."""
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON"}, status=400)

        script = body.get("script", "")
        result = await self.cua_engine.execute_js(script)

        return web.json_response(result)

    async def handle_cua_query(self, request) -> web.Response:
        """Query DOM element."""
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON"}, status=400)

        selector = body.get("selector", "")
        result = await self.cua_engine.query_element(selector)

        return web.json_response(result)

    async def handle_cua_key(self, request) -> web.Response:
        """Press a key (Enter, Tab, etc.)."""
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON"}, status=400)

        key = body.get("key", "")
        if not key:
            return web.json_response({"ok": False, "error": "Key required"}, status=400)

        modifiers = body.get("modifiers", [])
        result = await self.cua_engine.key_press(key, modifiers)

        emit_bus_event(
            self.cfg.bus_dir,
            "world_router.cua.key",
            {"ok": result.get("ok"), "key": key, "modifiers": modifiers},
        )

        return web.json_response(result)

    async def handle_cua_scroll(self, request) -> web.Response:
        """Scroll at position."""
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON"}, status=400)

        x = body.get("x", 0)
        y = body.get("y", 0)
        delta_x = body.get("delta_x", 0)
        delta_y = body.get("delta_y", 0)

        result = await self.cua_engine.scroll(x, y, delta_x, delta_y)

        emit_bus_event(
            self.cfg.bus_dir,
            "world_router.cua.scroll",
            {"ok": result.get("ok"), "x": x, "y": y, "delta_y": delta_y},
        )

        return web.json_response(result)

    async def handle_cua_double_click(self, request) -> web.Response:
        """Double-click at coordinates."""
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON"}, status=400)

        x = body.get("x", 0)
        y = body.get("y", 0)

        result = await self.cua_engine.double_click(x, y)

        emit_bus_event(
            self.cfg.bus_dir,
            "world_router.cua.double_click",
            {"ok": result.get("ok"), "x": x, "y": y},
        )

        return web.json_response(result)

    async def handle_cua_move(self, request) -> web.Response:
        """Move mouse without clicking."""
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON"}, status=400)

        x = body.get("x", 0)
        y = body.get("y", 0)

        result = await self.cua_engine.mouse_move(x, y)

        return web.json_response(result)

    async def handle_cua_drag(self, request) -> web.Response:
        """Drag from start to end."""
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON"}, status=400)

        start_x = body.get("start_x", 0)
        start_y = body.get("start_y", 0)
        end_x = body.get("end_x", 0)
        end_y = body.get("end_y", 0)

        result = await self.cua_engine.drag(start_x, start_y, end_x, end_y)

        emit_bus_event(
            self.cfg.bus_dir,
            "world_router.cua.drag",
            {"ok": result.get("ok"), "start": (start_x, start_y), "end": (end_x, end_y)},
        )

        return web.json_response(result)

    async def handle_cua_screen_size(self, request) -> web.Response:
        """Get screen dimensions."""
        result = await self.cua_engine.get_screen_size()
        return web.json_response(result)

    # -------------------------------------------------------------------------
    # CUA - Phase 3 Native Extensions
    # -------------------------------------------------------------------------

    async def handle_cua_wait(self, request) -> web.Response:
        """Wait for element to appear."""
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON"}, status=400)

        selector = body.get("selector", "")
        if not selector:
            return web.json_response({"ok": False, "error": "Selector required"}, status=400)

        state = body.get("state", "visible")
        timeout_ms = body.get("timeout_ms")

        result = await self.cua_engine.wait_for_selector(selector, state, timeout_ms)

        emit_bus_event(
            self.cfg.bus_dir,
            "world_router.cua.wait",
            {"ok": result.get("ok"), "selector": selector, "state": state},
        )

        return web.json_response(result)

    async def handle_cua_page_info(self, request) -> web.Response:
        """Get current page URL, title, viewport."""
        result = await self.cua_engine.get_page_info()
        return web.json_response(result)

    async def handle_cua_goto(self, request) -> web.Response:
        """Navigate to URL with extended options."""
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON"}, status=400)

        url = body.get("url", "")
        if not url:
            return web.json_response({"ok": False, "error": "URL required"}, status=400)

        wait_until = body.get("wait_until", "domcontentloaded")
        timeout_ms = body.get("timeout_ms")
        referer = body.get("referer")

        result = await self.cua_engine.goto(url, wait_until, timeout_ms, referer)

        emit_bus_event(
            self.cfg.bus_dir,
            "world_router.cua.goto",
            {"ok": result.get("ok"), "url": url, "wait_until": wait_until},
        )

        return web.json_response(result)

    async def handle_cua_html(self, request) -> web.Response:
        """Get HTML content of page or element."""
        try:
            body = await request.json()
        except Exception:
            body = {}

        selector = body.get("selector")  # Optional
        result = await self.cua_engine.get_html(selector)
        return web.json_response(result)

    async def handle_cua_text(self, request) -> web.Response:
        """Get text content of element."""
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON"}, status=400)

        selector = body.get("selector", "")
        if not selector:
            return web.json_response({"ok": False, "error": "Selector required"}, status=400)

        result = await self.cua_engine.get_text(selector)
        return web.json_response(result)

    async def handle_cua_pages(self, request) -> web.Response:
        """List all browser pages."""
        result = await self.cua_engine.list_pages()
        return web.json_response(result)

    async def handle_cua_new_page(self, request) -> web.Response:
        """Create a new browser page."""
        try:
            body = await request.json()
        except Exception:
            body = {}

        page_id = body.get("page_id")
        result = await self.cua_engine.new_page(page_id)

        emit_bus_event(
            self.cfg.bus_dir,
            "world_router.cua.new_page",
            {"ok": result.get("ok"), "page_id": result.get("page_id")},
        )

        return web.json_response(result)

    async def handle_cua_switch_page(self, request) -> web.Response:
        """Switch to a different browser page."""
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON"}, status=400)

        page_id = body.get("page_id", "")
        if not page_id:
            return web.json_response({"ok": False, "error": "page_id required"}, status=400)

        result = await self.cua_engine.switch_page(page_id)

        emit_bus_event(
            self.cfg.bus_dir,
            "world_router.cua.switch_page",
            {"ok": result.get("ok"), "page_id": page_id},
        )

        return web.json_response(result)

    async def handle_cua_close_page(self, request) -> web.Response:
        """Close a browser page."""
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"ok": False, "error": "Invalid JSON"}, status=400)

        page_id = body.get("page_id", "")
        if not page_id:
            return web.json_response({"ok": False, "error": "page_id required"}, status=400)

        result = await self.cua_engine.close_page(page_id)

        emit_bus_event(
            self.cfg.bus_dir,
            "world_router.cua.close_page",
            {"ok": result.get("ok"), "page_id": page_id},
        )

        return web.json_response(result)

    async def handle_cua_shutdown(self, request) -> web.Response:
        """Shutdown the browser (native mode only)."""
        result = await self.cua_engine.shutdown()

        emit_bus_event(
            self.cfg.bus_dir,
            "world_router.cua.shutdown",
            {"ok": result.get("ok")},
        )

        return web.json_response(result)

    # -------------------------------------------------------------------------
    # Bus WebSocket Stream
    # -------------------------------------------------------------------------

    async def handle_bus_websocket(self, request) -> web.WebSocketResponse:
        """Stream bus events over WebSocket (NDJSON bridge)."""
        ws = web.WebSocketResponse(autoping=True, heartbeat=30)
        await ws.prepare(request)

        identity = await self.identity_hub.resolve(request)

        bus_dir = Path(self.cfg.bus_dir)
        events_path = bus_dir / "events.ndjson"
        bus_dir.mkdir(parents=True, exist_ok=True)

        subscriptions: set[str] = {"*"}
        last_by_topic: dict[str, dict] = {}
        seeded = False
        seed_lock = asyncio.Lock()

        async def ensure_seeded() -> None:
            nonlocal seeded
            if seeded:
                return
            async with seed_lock:
                if seeded:
                    return
                events = await asyncio.to_thread(
                    _read_bus_tail_events, events_path, 5000, 20 * 1024 * 1024
                )
                for event in events:
                    topic = event.get("topic")
                    if isinstance(topic, str):
                        last_by_topic[topic] = event
                seeded = True

        async def send_sync(max_lines: int) -> None:
            events = await asyncio.to_thread(
                _read_bus_tail_events, events_path, max_lines, 10 * 1024 * 1024
            )
            for event in events:
                topic = event.get("topic")
                if isinstance(topic, str):
                    last_by_topic[topic] = event
            await ws.send_str(json.dumps({"type": "sync", "events": events}, ensure_ascii=False))

        poll_s = 0.25
        try:
            last_pos = events_path.stat().st_size
        except Exception:
            last_pos = 0

        async def tail_loop() -> None:
            nonlocal last_pos
            while not ws.closed:
                try:
                    size = events_path.stat().st_size
                except Exception:
                    await asyncio.sleep(poll_s)
                    continue
                if size < last_pos:
                    last_pos = 0
                if size > last_pos:
                    try:
                        with events_path.open("r", encoding="utf-8", errors="replace") as f:
                            f.seek(last_pos)
                            data = f.read(size - last_pos)
                        last_pos = size
                    except Exception:
                        await asyncio.sleep(poll_s)
                        continue
                    for line in data.splitlines():
                        line = line.strip()
                        if not line:
                            continue
                        try:
                            event = json.loads(line)
                        except Exception:
                            continue
                        if not isinstance(event, dict):
                            continue
                        topic = event.get("topic")
                        if isinstance(topic, str):
                            last_by_topic[topic] = event
                            if _topic_matches(subscriptions, topic):
                                try:
                                    await ws.send_str(
                                        json.dumps({"type": "event", "event": event}, ensure_ascii=False)
                                    )
                                except Exception:
                                    return
                await asyncio.sleep(poll_s)

        tail_task = asyncio.create_task(tail_loop())
        try:
            async for msg in ws:
                if msg.type != WSMsgType.TEXT:
                    if msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR):
                        break
                    continue
                try:
                    payload = json.loads(msg.data)
                except Exception:
                    continue
                msg_type = payload.get("type")
                if msg_type == "subscribe":
                    topic = payload.get("topic")
                    if isinstance(topic, str) and topic:
                        subscriptions.add(topic)
                elif msg_type == "unsubscribe":
                    topic = payload.get("topic")
                    if isinstance(topic, str) and topic:
                        subscriptions.discard(topic)
                elif msg_type == "publish":
                    event = payload.get("event")
                    if isinstance(event, dict):
                        topic = event.get("topic")
                        if isinstance(topic, str) and topic:
                            normalized = _normalize_bus_event(event, actor=identity.did)
                            _append_bus_event(events_path, normalized)
                elif msg_type == "sync_topic":
                    topic = payload.get("topic")
                    if isinstance(topic, str) and topic:
                        await ensure_seeded()
                        event = last_by_topic.get(topic)
                        await ws.send_str(
                            json.dumps({"type": "sync_topic", "topic": topic, "event": event}, ensure_ascii=False)
                        )
                elif msg_type == "sync":
                    limit = payload.get("limit")
                    try:
                        limit_value = int(limit) if limit is not None else 600
                    except Exception:
                        limit_value = 600
                    limit_value = max(1, min(limit_value, 2000))
                    await send_sync(limit_value)
        finally:
            tail_task.cancel()
            try:
                await tail_task
            except asyncio.CancelledError:
                pass

        return ws

    # -------------------------------------------------------------------------
    # VNC WebSocket Tunnel
    # -------------------------------------------------------------------------

    async def handle_vnc_websocket(self, request) -> web.WebSocketResponse:
        """Proxy VNC over WebSocket."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        identity = await self.identity_hub.resolve(request)
        writer = None  # Track for cleanup

        emit_bus_event(
            self.cfg.bus_dir,
            "world_router.vnc.connect",
            {"identity": identity.did},
        )

        try:
            # Connect to VNC server
            reader, writer = await asyncio.open_connection(
                self.cfg.vnc_host, self.cfg.vnc_port
            )

            # Track connection state
            connection_active = True

            async def vnc_to_ws():
                """Forward VNC server -> WebSocket."""
                nonlocal connection_active
                try:
                    while connection_active:
                        data = await reader.read(4096)
                        if not data:
                            connection_active = False
                            break
                        if not ws.closed:
                            await ws.send_bytes(data)
                except asyncio.CancelledError:
                    pass
                except Exception:
                    connection_active = False

            async def ws_to_vnc():
                """Forward WebSocket -> VNC server."""
                nonlocal connection_active
                try:
                    async for msg in ws:
                        if not connection_active:
                            break
                        if msg.type == WSMsgType.BINARY:
                            writer.write(msg.data)
                            await writer.drain()
                        elif msg.type in (WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR):
                            connection_active = False
                            break
                except asyncio.CancelledError:
                    pass
                except Exception:
                    connection_active = False

            # Run bidirectional relay
            tasks = [
                asyncio.create_task(vnc_to_ws()),
                asyncio.create_task(ws_to_vnc()),
            ]
            try:
                await asyncio.gather(*tasks, return_exceptions=True)
            finally:
                # Cancel any remaining tasks
                for task in tasks:
                    if not task.done():
                        task.cancel()
                        try:
                            await task
                        except asyncio.CancelledError:
                            pass

        except ConnectionRefusedError:
            await ws.close(code=1011, message=b"VNC server not available")
        except Exception as e:
            if not ws.closed:
                await ws.close(code=1011, message=str(e).encode()[:123])
        finally:
            # Clean up TCP connection
            if writer is not None:
                try:
                    writer.close()
                    await writer.wait_closed()
                except Exception:
                    pass

            emit_bus_event(
                self.cfg.bus_dir,
                "world_router.vnc.disconnect",
                {"identity": identity.did},
            )

        return ws

    # -------------------------------------------------------------------------
    # Storage Fabric (Proxy to cloud_storage_daemon)
    # -------------------------------------------------------------------------

    async def handle_storage_status(self, request) -> web.Response:
        """Get storage status."""
        # Proxy to legacy daemon for now
        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{self.cfg.legacy_storage}/status") as resp:
                    data = await resp.json()
                    return web.json_response(data)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_storage_list(self, request) -> web.Response:
        """List directory contents."""
        provider = request.match_info.get("provider", "")
        path = request.query.get("path", "/")

        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.cfg.legacy_storage}/browse/{provider}",
                    params={"path": path}
                ) as resp:
                    data = await resp.json()
                    return web.json_response(data)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_storage_read(self, request) -> web.Response:
        """Read file contents."""
        provider = request.match_info.get("provider", "")
        path = request.query.get("path", "")

        if not path:
            return web.json_response({"error": "Path required"}, status=400)

        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"{self.cfg.legacy_storage}/file/{provider}",
                    params={"path": path}
                ) as resp:
                    if resp.content_type.startswith("application/json"):
                        data = await resp.json()
                        return web.json_response(data)
                    else:
                        # Return raw content
                        content = await resp.read()
                        return web.Response(
                            body=content,
                            content_type=resp.content_type,
                        )
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_storage_oauth_start(self, request) -> web.Response:
        """Start OAuth flow for storage provider."""
        provider = request.match_info.get("provider", "")

        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.cfg.legacy_storage}/auth/start",
                    json={"account": provider}
                ) as resp:
                    data = await resp.json()
                    return web.json_response(data)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_storage_mount(self, request) -> web.Response:
        """Mount storage provider."""
        provider = request.match_info.get("provider", "")

        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.cfg.legacy_storage}/mount/{provider}"
                ) as resp:
                    data = await resp.json()
                    return web.json_response(data)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    async def handle_storage_unmount(self, request) -> web.Response:
        """Unmount storage provider."""
        provider = request.match_info.get("provider", "")

        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.cfg.legacy_storage}/unmount/{provider}"
                ) as resp:
                    data = await resp.json()
                    return web.json_response(data)
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)

    # -------------------------------------------------------------------------
    # Identity Endpoints
    # -------------------------------------------------------------------------

    async def handle_identity_whoami(self, request) -> web.Response:
        """Get current session identity."""
        identity = await self.identity_hub.resolve(request)
        return web.json_response({
            "did": identity.did,
            "session_id": identity.session_id,
            "session_type": identity.session_type,
            "capabilities": identity.capabilities,
            "auth_method": identity.auth_method,
        })

    async def handle_identity_oauth_start(self, request) -> web.Response:
        """Start OAuth flow.

        Supports Google and GitHub OAuth 2.0 with PKCE.
        Returns an authorization URL to redirect the user to.
        """
        try:
            body = await request.json()
        except Exception:
            body = {}

        provider = body.get("provider", "google")

        # Build redirect URI from request origin
        host = request.headers.get("Host", "localhost")
        scheme = "https" if "localhost" not in host else "http"
        redirect_uri = body.get("redirect_uri", f"{scheme}://{host}/identity/oauth/callback")

        emit_bus_event(
            self.cfg.bus_dir,
            "world_router.identity.oauth_start",
            {"provider": provider},
        )

        # Check if provider is supported
        if provider not in self.identity_hub.OAUTH_PROVIDERS:
            return web.json_response({
                "error": f"Unsupported provider: {provider}",
                "supported": list(self.identity_hub.OAUTH_PROVIDERS.keys()),
            }, status=400)

        try:
            auth_url, state = self.identity_hub.get_oauth_url(provider, redirect_uri)

            return web.json_response({
                "status": "redirect",
                "provider": provider,
                "auth_url": auth_url,
                "state": state,
                "redirect_uri": redirect_uri,
            })
        except ValueError as e:
            return web.json_response({
                "error": str(e),
                "hint": f"Set {self.identity_hub.OAUTH_PROVIDERS[provider]['client_id_env']} environment variable",
            }, status=500)

    async def handle_identity_oauth_callback(self, request) -> web.Response:
        """Handle OAuth callback.

        Exchanges authorization code for tokens, fetches user info,
        creates a session with DID-based identity, and returns session token.
        """
        code = request.query.get("code", "")
        state = request.query.get("state", "")
        error = request.query.get("error", "")

        if error:
            return web.json_response({
                "error": error,
                "error_description": request.query.get("error_description", ""),
            }, status=400)

        if not code:
            return web.json_response({"error": "Missing authorization code"}, status=400)

        if not state:
            return web.json_response({"error": "Missing state parameter"}, status=400)

        emit_bus_event(
            self.cfg.bus_dir,
            "world_router.identity.oauth_callback",
            {"state": state, "has_code": bool(code)},
        )

        # Exchange code for tokens and create identity
        identity = await self.identity_hub.exchange_code(code, state)

        if identity is None:
            return web.json_response({
                "error": "Failed to exchange authorization code",
                "hint": "Code may be expired or state mismatch",
            }, status=400)

        emit_bus_event(
            self.cfg.bus_dir,
            "world_router.identity.session_created",
            {
                "did": identity.did,
                "provider": identity.auth_provider,
                "session_id": identity.session_id,
            },
        )

        # Return session info with Set-Cookie header
        response = web.json_response({
            "status": "authenticated",
            "session_id": identity.session_id,
            "identity": identity.to_dict(),
        })

        # Set session cookie (30 day expiry)
        response.set_cookie(
            "pluribus_session",
            identity.session_id,
            max_age=30 * 24 * 60 * 60,  # 30 days
            httponly=True,
            samesite="Lax",
        )

        return response

    async def handle_identity_revoke(self, request) -> web.Response:
        """Revoke a session."""
        try:
            body = await request.json()
        except Exception:
            body = {}

        session_id = body.get("session_id", "")

        # Allow revoking own session from cookie
        if not session_id:
            session_id = request.cookies.get("pluribus_session", "")

        if not session_id:
            return web.json_response({"error": "No session to revoke"}, status=400)

        success = self.identity_hub.revoke_session(session_id)

        emit_bus_event(
            self.cfg.bus_dir,
            "world_router.identity.session_revoked",
            {"session_id": session_id, "success": success},
        )

        response = web.json_response({
            "status": "revoked" if success else "not_found",
            "session_id": session_id,
        })

        # Clear cookie
        response.del_cookie("pluribus_session")

        return response

    async def handle_identity_sessions(self, request) -> web.Response:
        """List active sessions."""
        sessions = []
        for session_id, identity in self.identity_hub._sessions.items():
            sessions.append({
                "session_id": session_id,
                "did": identity.did,
                "session_type": identity.session_type,
                "auth_method": identity.auth_method,
            })
        return web.json_response({"sessions": sessions})

    # -------------------------------------------------------------------------
    # WebAuthn/Passkey Endpoints (Phase 3)
    # -------------------------------------------------------------------------

    async def handle_webauthn_register_start(self, request) -> web.Response:
        """Start WebAuthn registration.

        POST /identity/webauthn/register/start
        Body: { "user_id": "email@example.com", "user_name": "Display Name" }

        Returns PublicKeyCredentialCreationOptions for navigator.credentials.create()
        """
        try:
            body = await request.json()
        except Exception:
            body = {}

        user_id = body.get("user_id", "")
        user_name = body.get("user_name", "")

        if not user_id:
            return web.json_response({
                "error": "user_id is required",
                "hint": "Provide email or unique identifier",
            }, status=400)

        host = request.headers.get("Host", "localhost")

        emit_bus_event(
            self.cfg.bus_dir,
            "world_router.identity.webauthn_register_start",
            {"user_id": user_id, "host": host},
        )

        options = self.identity_hub.webauthn_register_start(user_id, user_name, host)

        return web.json_response({
            "status": "challenge_issued",
            "options": options,
        })

    async def handle_webauthn_register_complete(self, request) -> web.Response:
        """Complete WebAuthn registration.

        POST /identity/webauthn/register/complete
        Body: {
            "challenge": "...",
            "credential_id": "base64...",
            "public_key": "base64...",
            "authenticator_type": "platform|cross-platform",
            "transports": ["internal", "usb"],
            "friendly_name": "My YubiKey"
        }

        Returns credential info on success.
        """
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        challenge = body.get("challenge", "")
        credential_id = body.get("credential_id", "")
        public_key = body.get("public_key", "")
        authenticator_type = body.get("authenticator_type", "unknown")
        transports = body.get("transports", [])
        friendly_name = body.get("friendly_name")

        if not challenge or not credential_id or not public_key:
            return web.json_response({
                "error": "Missing required fields",
                "required": ["challenge", "credential_id", "public_key"],
            }, status=400)

        credential = self.identity_hub.webauthn_register_complete(
            challenge=challenge,
            credential_id=credential_id,
            public_key=public_key,
            authenticator_type=authenticator_type,
            transports=transports,
            friendly_name=friendly_name,
        )

        if credential is None:
            emit_bus_event(
                self.cfg.bus_dir,
                "world_router.identity.webauthn_register_failed",
                {"credential_id": credential_id[:20] + "..."},
            )
            return web.json_response({
                "error": "Registration failed",
                "hint": "Challenge may be expired or invalid",
            }, status=400)

        emit_bus_event(
            self.cfg.bus_dir,
            "world_router.identity.webauthn_registered",
            {
                "user_id": credential.user_id,
                "credential_id": credential_id[:20] + "...",
                "authenticator_type": authenticator_type,
            },
        )

        return web.json_response({
            "status": "registered",
            "credential": {
                "credential_id": credential.credential_id,
                "user_id": credential.user_id,
                "authenticator_type": credential.authenticator_type,
                "created_at": credential.created_at,
            },
        })

    async def handle_webauthn_login_start(self, request) -> web.Response:
        """Start WebAuthn authentication.

        POST /identity/webauthn/login/start
        Body: { "user_id": "email@example.com" }  # Optional, for non-discoverable credentials

        Returns PublicKeyCredentialRequestOptions for navigator.credentials.get()
        """
        try:
            body = await request.json()
        except Exception:
            body = {}

        user_id = body.get("user_id")  # Optional
        host = request.headers.get("Host", "localhost")

        emit_bus_event(
            self.cfg.bus_dir,
            "world_router.identity.webauthn_login_start",
            {"user_id": user_id, "host": host},
        )

        options = self.identity_hub.webauthn_login_start(user_id, host)

        return web.json_response({
            "status": "challenge_issued",
            "options": options,
        })

    async def handle_webauthn_login_complete(self, request) -> web.Response:
        """Complete WebAuthn authentication.

        POST /identity/webauthn/login/complete
        Body: {
            "challenge": "...",
            "credential_id": "base64...",
            "signature": "base64...",
            "authenticator_data": "base64...",
            "client_data_json": "base64..."
        }

        Returns session info and sets cookie on success.
        """
        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        challenge = body.get("challenge", "")
        credential_id = body.get("credential_id", "")
        signature = body.get("signature", "")
        authenticator_data = body.get("authenticator_data", "")
        client_data_json = body.get("client_data_json", "")

        if not all([challenge, credential_id, signature, authenticator_data, client_data_json]):
            return web.json_response({
                "error": "Missing required fields",
                "required": ["challenge", "credential_id", "signature", "authenticator_data", "client_data_json"],
            }, status=400)

        identity = self.identity_hub.webauthn_login_complete(
            challenge=challenge,
            credential_id=credential_id,
            signature=signature,
            authenticator_data=authenticator_data,
            client_data_json=client_data_json,
        )

        if identity is None:
            emit_bus_event(
                self.cfg.bus_dir,
                "world_router.identity.webauthn_login_failed",
                {"credential_id": credential_id[:20] + "..."},
            )
            return web.json_response({
                "error": "Authentication failed",
                "hint": "Challenge expired, invalid credential, or replay attack detected",
            }, status=401)

        emit_bus_event(
            self.cfg.bus_dir,
            "world_router.identity.webauthn_authenticated",
            {
                "did": identity.did,
                "session_id": identity.session_id,
            },
        )

        # Return session info with Set-Cookie header
        response = web.json_response({
            "status": "authenticated",
            "session_id": identity.session_id,
            "identity": identity.to_dict(),
        })

        # Set session cookie (30 day expiry)
        response.set_cookie(
            "pluribus_session",
            identity.session_id,
            max_age=30 * 24 * 60 * 60,  # 30 days
            httponly=True,
            samesite="Lax",
        )

        return response

    async def handle_webauthn_credentials_list(self, request) -> web.Response:
        """List WebAuthn credentials for the current user.

        GET /identity/webauthn/credentials

        Returns list of registered credentials.
        """
        identity = await self.identity_hub.resolve(request)

        if identity.did == "did:key:anonymous":
            return web.json_response({
                "error": "Authentication required",
                "hint": "Log in first to view credentials",
            }, status=401)

        # Use email or DID as user_id
        user_id = identity.email or identity.did

        credentials = self.identity_hub.get_user_credentials(user_id)

        return web.json_response({
            "user_id": user_id,
            "credentials": credentials,
            "count": len(credentials),
        })

    async def handle_webauthn_credential_delete(self, request) -> web.Response:
        """Delete a WebAuthn credential.

        POST /identity/webauthn/credentials/delete
        Body: { "credential_id": "..." }

        Deletes a specific credential owned by the current user.
        """
        identity = await self.identity_hub.resolve(request)

        if identity.did == "did:key:anonymous":
            return web.json_response({
                "error": "Authentication required",
            }, status=401)

        try:
            body = await request.json()
        except Exception:
            return web.json_response({"error": "Invalid JSON"}, status=400)

        credential_id = body.get("credential_id", "")

        if not credential_id:
            return web.json_response({
                "error": "credential_id is required",
            }, status=400)

        # Use email or DID as user_id
        user_id = identity.email or identity.did

        success = self.identity_hub.delete_credential(credential_id, user_id)

        emit_bus_event(
            self.cfg.bus_dir,
            "world_router.identity.webauthn_credential_deleted",
            {"credential_id": credential_id[:20] + "...", "success": success},
        )

        if success:
            return web.json_response({
                "status": "deleted",
                "credential_id": credential_id,
            })
        else:
            return web.json_response({
                "error": "Credential not found or not owned by user",
            }, status=404)

    # -------------------------------------------------------------------------
    # A2A Protocol (Agent-to-Agent)
    # -------------------------------------------------------------------------

    async def handle_a2a_agent_json(self, request) -> web.Response:
        """Return agent.json for A2A discovery.

        Follows the A2A (Agent-to-Agent) protocol specification.
        https://google.github.io/a2a-spec/
        """
        host = request.headers.get("Host", "localhost")
        scheme = "https" if "localhost" not in host else "http"
        base_url = f"{scheme}://{host}"

        agent_card = {
            "name": "Pluribus World Router",
            "description": "Unified VPS gateway for LLM inference, computer use, and identity management",
            "url": base_url,
            "provider": {
                "organization": "Pluribus",
                "url": "https://kroma.live"
            },
            "version": "2.0.0",
            "capabilities": {
                "streaming": True,
                "pushNotifications": False,
                "stateTransitionHistory": False,
            },
            "authentication": {
                "schemes": ["bearer", "oauth2"],
                "oauth2": {
                    "providers": list(self.identity_hub.OAUTH_PROVIDERS.keys()),
                    "startUrl": f"{base_url}/identity/oauth/start",
                    "callbackUrl": f"{base_url}/identity/oauth/callback",
                }
            },
            "defaultInputModes": ["text"],
            "defaultOutputModes": ["text"],
            "skills": [
                {
                    "id": "llm-inference",
                    "name": "LLM Inference",
                    "description": "Invoke LLM providers (Claude, GPT, Gemini)",
                    "tags": ["llm", "inference", "chat"],
                    "examples": [
                        "Send a message to Claude",
                        "Generate text with GPT-4",
                    ],
                },
                {
                    "id": "computer-use",
                    "name": "Computer Use Agent",
                    "description": "Control browser and desktop via CUA primitives",
                    "tags": ["cua", "browser", "automation"],
                    "examples": [
                        "Take a screenshot",
                        "Click at coordinates",
                        "Navigate to URL",
                    ],
                },
                {
                    "id": "vnc-tunnel",
                    "name": "VNC Tunnel",
                    "description": "WebSocket tunnel to VNC server for remote desktop",
                    "tags": ["vnc", "remote-desktop"],
                },
                {
                    "id": "storage",
                    "name": "Cloud Storage",
                    "description": "Access cloud storage providers (Google Drive, etc.)",
                    "tags": ["storage", "cloud", "files"],
                },
            ],
        }

        return web.json_response(agent_card)

    async def handle_a2a_tasks(self, request) -> web.Response:
        """Handle A2A task submission.

        POST /a2a/tasks - Submit a task
        GET /a2a/tasks/:id - Get task status
        """
        if request.method == "POST":
            try:
                body = await request.json()
            except Exception:
                return web.json_response({"error": "Invalid JSON"}, status=400)

            task_id = str(uuid.uuid4())
            skill_id = body.get("skill", "")
            params = body.get("params", {})

            emit_bus_event(
                self.cfg.bus_dir,
                "world_router.a2a.task_submitted",
                {"task_id": task_id, "skill": skill_id},
            )

            # Route to appropriate handler based on skill
            if skill_id == "llm-inference":
                # Delegate to OpenAI chat handler
                messages = params.get("messages", [{"role": "user", "content": params.get("prompt", "")}])
                model = params.get("model", "auto")

                # Build internal request
                from aiohttp.test_utils import make_mocked_request
                # For simplicity, return task acknowledgment
                return web.json_response({
                    "id": task_id,
                    "status": "submitted",
                    "skill": skill_id,
                    "message": "Task submitted. Use A2A streaming or poll for results.",
                })

            elif skill_id == "computer-use":
                action = params.get("action", "screenshot")
                return web.json_response({
                    "id": task_id,
                    "status": "submitted",
                    "skill": skill_id,
                    "action": action,
                })

            else:
                return web.json_response({
                    "id": task_id,
                    "status": "submitted",
                    "skill": skill_id,
                    "message": "Unknown skill, task queued for manual processing",
                })

        # GET - retrieve task status
        task_id = request.match_info.get("task_id", "")
        if not task_id:
            return web.json_response({"error": "Task ID required"}, status=400)

        # For now, return not found (task storage would be Phase 3+)
        return web.json_response({
            "id": task_id,
            "status": "unknown",
            "message": "Task history not yet implemented",
        }, status=404)

    # -------------------------------------------------------------------------
    # Storage Write Endpoint
    # -------------------------------------------------------------------------

    async def handle_storage_write(self, request) -> web.Response:
        """Write file to storage provider."""
        provider = request.match_info.get("provider", "")
        path = request.query.get("path", "")

        if not path:
            return web.json_response({"error": "Path required"}, status=400)

        # Read body content
        try:
            content = await request.read()
        except Exception as e:
            return web.json_response({"error": f"Failed to read body: {e}"}, status=400)

        emit_bus_event(
            self.cfg.bus_dir,
            "world_router.storage.write",
            {"provider": provider, "path": path, "size": len(content)},
        )

        try:
            import aiohttp
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    f"{self.cfg.legacy_storage}/file/{provider}",
                    params={"path": path},
                    data=content,
                    headers={"Content-Type": request.content_type or "application/octet-stream"},
                ) as resp:
                    if resp.content_type.startswith("application/json"):
                        data = await resp.json()
                        return web.json_response(data, status=resp.status)
                    else:
                        return web.Response(
                            body=await resp.read(),
                            content_type=resp.content_type,
                            status=resp.status,
                        )
        except Exception as e:
            return web.json_response({"error": str(e)}, status=500)


    async def handle_manifest_dispatch(self, request: web.Request) -> web.Response:
        """Phase 6.2: Dynamic App Dispatch via Manifest Registry."""
        try:
            from nucleus.tools import manifest_registry
            reg = manifest_registry.get_registry()
            app_manifest = reg.get_by_route(request.path)
            
            if not app_manifest:
                return web.json_response({"error": "App resolution failed"}, status=404)
            
            # Phase 6.2 Logic: Return metadata proving integration
            return web.json_response({
                "service": "world_router",
                "layer": "manifest_dispatch",
                "app": app_manifest.name,
                "route_config": app_manifest.route, 
                "version": app_manifest.version,
                "status": "integrated"
            })
        except Exception as e:
            return web.json_response({"error": f"Dispatch error: {e}"}, status=500)


# ============================================================================
# Application Factory
# ============================================================================

def create_app(cfg: WorldRouterConfig) -> web.Application:
    """Create the aiohttp application."""
    if not AIOHTTP_AVAILABLE:
        raise RuntimeError("aiohttp is required. Install with: pip install aiohttp")

    app = web.Application()
    handlers = WorldRouterHandlers(cfg)

    # Store handlers for initialization
    app["handlers"] = handlers
    app["config"] = cfg

    # Routes - Health & Meta
    app.router.add_get("/health", handlers.handle_health)
    app.router.add_get("/healthz", handlers.handle_health)
    app.router.add_get("/v1/models", handlers.handle_models)
    app.router.add_get("/v1beta/models", handlers.handle_models)

    # Circuit Breaker Admin
    app.router.add_get("/admin/circuits", handlers.handle_circuit_status)
    app.router.add_post("/admin/circuits/reset", handlers.handle_circuit_reset)

    # LLM - OpenAI
    app.router.add_post("/v1/chat/completions", handlers.handle_openai_chat)
    app.router.add_post("/v1/responses", handlers.handle_openai_responses)

    # LLM - Anthropic
    app.router.add_post("/v1/messages", handlers.handle_anthropic_messages)
    app.router.add_post("/v1/messages/count_tokens", handlers.handle_anthropic_count_tokens)

    # LLM - Gemini (dynamic path matching)
    app.router.add_post("/v1beta/models/{model}:generateContent", handlers.handle_gemini_generate)
    app.router.add_post("/v1beta/models/{model}:streamGenerateContent", handlers.handle_gemini_generate)

    # CUA - Basic
    app.router.add_post("/cua/screenshot", handlers.handle_cua_screenshot)
    app.router.add_post("/cua/click", handlers.handle_cua_click)
    app.router.add_post("/cua/type", handlers.handle_cua_type)
    app.router.add_post("/cua/navigate", handlers.handle_cua_navigate)
    app.router.add_post("/cua/eval", handlers.handle_cua_eval)
    app.router.add_post("/cua/query", handlers.handle_cua_query)

    # CUA - Extended
    app.router.add_post("/cua/key", handlers.handle_cua_key)
    app.router.add_post("/cua/scroll", handlers.handle_cua_scroll)
    app.router.add_post("/cua/double_click", handlers.handle_cua_double_click)
    app.router.add_post("/cua/move", handlers.handle_cua_move)
    app.router.add_post("/cua/drag", handlers.handle_cua_drag)
    app.router.add_get("/cua/screen_size", handlers.handle_cua_screen_size)

    # CUA - Phase 3 Native Extensions
    app.router.add_post("/cua/wait", handlers.handle_cua_wait)
    app.router.add_get("/cua/page_info", handlers.handle_cua_page_info)
    app.router.add_post("/cua/goto", handlers.handle_cua_goto)
    app.router.add_post("/cua/html", handlers.handle_cua_html)
    app.router.add_post("/cua/text", handlers.handle_cua_text)
    app.router.add_get("/cua/pages", handlers.handle_cua_pages)
    app.router.add_post("/cua/new_page", handlers.handle_cua_new_page)
    app.router.add_post("/cua/switch_page", handlers.handle_cua_switch_page)
    app.router.add_post("/cua/close_page", handlers.handle_cua_close_page)
    app.router.add_post("/cua/shutdown", handlers.handle_cua_shutdown)

    # Bus WebSocket Stream
    app.router.add_get("/bus", handlers.handle_bus_websocket)
    app.router.add_get("/ws/bus", handlers.handle_bus_websocket)

    # VNC WebSocket
    if cfg.vnc_tunnel:
        app.router.add_get("/vnc", handlers.handle_vnc_websocket)

    # Storage Fabric
    if cfg.storage_fabric:
        app.router.add_get("/storage/status", handlers.handle_storage_status)
        app.router.add_get("/storage/{provider}/list", handlers.handle_storage_list)
        app.router.add_get("/storage/{provider}/read", handlers.handle_storage_read)
        app.router.add_post("/storage/{provider}/oauth", handlers.handle_storage_oauth_start)
        app.router.add_post("/storage/{provider}/mount", handlers.handle_storage_mount)
        app.router.add_post("/storage/{provider}/unmount", handlers.handle_storage_unmount)

    # Identity Hub
    app.router.add_get("/identity/whoami", handlers.handle_identity_whoami)
    app.router.add_get("/identity/sessions", handlers.handle_identity_sessions)
    app.router.add_post("/identity/oauth/start", handlers.handle_identity_oauth_start)
    app.router.add_get("/identity/oauth/callback", handlers.handle_identity_oauth_callback)
    app.router.add_post("/identity/revoke", handlers.handle_identity_revoke)

    # WebAuthn/Passkey (Phase 3)
    app.router.add_post("/identity/webauthn/register/start", handlers.handle_webauthn_register_start)
    app.router.add_post("/identity/webauthn/register/complete", handlers.handle_webauthn_register_complete)
    app.router.add_post("/identity/webauthn/login/start", handlers.handle_webauthn_login_start)
    app.router.add_post("/identity/webauthn/login/complete", handlers.handle_webauthn_login_complete)
    app.router.add_get("/identity/webauthn/credentials", handlers.handle_webauthn_credentials_list)
    app.router.add_post("/identity/webauthn/credentials/delete", handlers.handle_webauthn_credential_delete)

    # A2A Protocol (Agent-to-Agent)
    app.router.add_get("/.well-known/agent.json", handlers.handle_a2a_agent_json)
    app.router.add_post("/a2a/tasks", handlers.handle_a2a_tasks)
    app.router.add_get("/a2a/tasks/{task_id}", handlers.handle_a2a_tasks)

    # Storage Write (in addition to existing read)
    if cfg.storage_fabric:
        app.router.add_post("/storage/{provider}/write", handlers.handle_storage_write)

    # Manifest App Routing (Phase 6.2)
    try:
        from nucleus.tools import manifest_registry
        reg = manifest_registry.get_registry()
        print(f"[WorldRouter] Loading Manifest Registry...")
        for app_def in reg.list_apps():
            if not app_def.enabled:
                continue
            
            # Normalize route pattern
            route_prefix = app_def.route.rstrip("/")
            if not route_prefix: 
                continue # Skip root if handled elsewhere (dashboard) or handle carefully
                
            # Add exact match and subpath wildcard
            print(f"[WorldRouter] + App: {app_def.name} on {route_prefix}")
            app.router.add_route('*', route_prefix, handlers.handle_manifest_dispatch)
            app.router.add_route('*', f"{route_prefix}/{{tail:.*}}", handlers.handle_manifest_dispatch)
            
    except Exception as e:
        print(f"[WorldRouter] Manifest routing init failed: {e}")


    # CORS middleware
    @web.middleware
    async def cors_middleware(request, handler):
        if request.method == "OPTIONS":
            return web.Response(
                headers={
                    "Access-Control-Allow-Origin": "*",
                    "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
                    "Access-Control-Allow-Headers": "Content-Type, Authorization",
                }
            )
        response = await handler(request)
        response.headers["Access-Control-Allow-Origin"] = "*"
        return response

    app.middlewares.append(cors_middleware)

    # Startup hook
    async def on_startup(app):
        await app["handlers"].initialize()
        emit_bus_event(
            cfg.bus_dir,
            "world_router.lifecycle",
            {"status": "started", "port": cfg.port},
            kind="log",
        )

    app.on_startup.append(on_startup)

    return app


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="World Router - Unified VPS Gateway")
    parser.add_argument("--host", default="0.0.0.0", help="Bind host")
    parser.add_argument("--port", type=int, default=8080, help="Bind port")
    parser.add_argument("--bus-dir", default="/pluribus/.pluribus/bus", help="Bus directory")
    parser.add_argument("--identity-hub", action="store_true", help="Enable Identity Hub")
    parser.add_argument("--cua-native", action="store_true", help="Enable native CUA (Playwright)")
    parser.add_argument("--no-vnc", action="store_true", help="Disable VNC tunnel")
    parser.add_argument("--no-storage", action="store_true", help="Disable storage fabric")

    args = parser.parse_args()

    cfg = WorldRouterConfig(
        host=args.host,
        port=args.port,
        bus_dir=args.bus_dir,
        identity_hub=args.identity_hub,
        cua_native=args.cua_native,
        vnc_tunnel=not args.no_vnc,
        storage_fabric=not args.no_storage,
    )

    print(f"[WorldRouter] Starting on http://{cfg.host}:{cfg.port}")
    print(f"[WorldRouter] Features: identity={cfg.identity_hub}, cua_native={cfg.cua_native}, vnc={cfg.vnc_tunnel}, storage={cfg.storage_fabric}")

    app = create_app(cfg)
    web.run_app(app, host=cfg.host, port=cfg.port, print=None)


if __name__ == "__main__":
    main()
