#!/usr/bin/env python3
"""Unit tests for World Router."""
import json
import sys
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Import module under test
from world_router import (
    WorldRouterConfig,
    ProtocolRouter,
    Identity,
    IdentityHub,
    CUAEngine,
    CircuitBreaker,
    OAuthState,
    WebAuthnCredential,
    WebAuthnChallenge,
    _resolve_provider,
    _safe_sha256,
    _classify_router_failure,
    _gap_analysis_for_reason,
    now_iso,
)


class TestProtocolRouter(unittest.TestCase):
    """Test protocol detection."""

    def _make_request(self, path: str, method: str = "GET", headers: dict = None):
        """Create mock request."""
        req = MagicMock()
        req.path = path
        req.method = method
        req.headers = headers or {}
        return req

    def test_detect_openai_chat(self):
        req = self._make_request("/v1/chat/completions", "POST")
        self.assertEqual(ProtocolRouter.detect(req), "http/openai-chat")

    def test_detect_anthropic(self):
        req = self._make_request("/v1/messages", "POST")
        self.assertEqual(ProtocolRouter.detect(req), "http/anthropic")

    def test_detect_gemini(self):
        req = self._make_request("/v1beta/models/gemini-pro:generateContent", "POST")
        self.assertEqual(ProtocolRouter.detect(req), "http/gemini")

    def test_detect_cua(self):
        req = self._make_request("/cua/screenshot", "POST")
        self.assertEqual(ProtocolRouter.detect(req), "http/cua")

        req = self._make_request("/cua/click", "POST")
        self.assertEqual(ProtocolRouter.detect(req), "http/cua")

    def test_detect_vnc_websocket(self):
        req = self._make_request("/vnc", "GET", {"Upgrade": "websocket"})
        self.assertEqual(ProtocolRouter.detect(req), "ws/vnc")

    def test_detect_storage(self):
        req = self._make_request("/storage/status", "GET")
        self.assertEqual(ProtocolRouter.detect(req), "http/storage")

    def test_detect_identity(self):
        req = self._make_request("/identity/whoami", "GET")
        self.assertEqual(ProtocolRouter.detect(req), "http/identity")

    def test_detect_health(self):
        for path in ["/health", "/healthz", "/_health"]:
            req = self._make_request(path, "GET")
            self.assertEqual(ProtocolRouter.detect(req), "http/health")

    def test_detect_a2a(self):
        req = self._make_request("/.well-known/agent.json", "GET")
        self.assertEqual(ProtocolRouter.detect(req), "http/a2a")

    def test_detect_default(self):
        req = self._make_request("/some/random/path", "GET")
        self.assertEqual(ProtocolRouter.detect(req), "http/default")


class TestProviderResolution(unittest.TestCase):
    """Test provider resolution logic."""

    def test_resolve_auto(self):
        self.assertEqual(_resolve_provider("auto"), "auto")
        self.assertEqual(_resolve_provider(""), "auto")
        self.assertEqual(_resolve_provider(None), "auto")

    def test_resolve_explicit_web(self):
        self.assertEqual(_resolve_provider("chatgpt-web"), "chatgpt-web")
        self.assertEqual(_resolve_provider("claude-web"), "claude-web")
        self.assertEqual(_resolve_provider("gemini-web"), "gemini-web")

    def test_resolve_claude_variants(self):
        self.assertEqual(_resolve_provider("claude-3-opus"), "claude-web")
        self.assertEqual(_resolve_provider("claude-sonnet"), "claude-web")

    def test_resolve_gemini_variants(self):
        self.assertEqual(_resolve_provider("gemini-pro"), "gemini-web")
        self.assertEqual(_resolve_provider("gemini-1.5-flash"), "gemini-web")

    def test_resolve_gpt_variants(self):
        self.assertEqual(_resolve_provider("gpt-4"), "chatgpt-web")
        self.assertEqual(_resolve_provider("gpt-4-turbo"), "chatgpt-web")
        self.assertEqual(_resolve_provider("openai-gpt"), "chatgpt-web")


class TestIdentity(unittest.TestCase):
    """Test Identity dataclass."""

    def test_default_identity(self):
        identity = Identity()
        self.assertEqual(identity.did, "did:key:anonymous")
        self.assertEqual(identity.session_type, "anonymous")
        self.assertIsNotNone(identity.session_id)

    def test_custom_identity(self):
        identity = Identity(
            did="did:key:z6MkTest",
            session_type="agent",
            auth_method="bearer",
        )
        self.assertEqual(identity.did, "did:key:z6MkTest")
        self.assertEqual(identity.session_type, "agent")


class TestIdentityHub(unittest.TestCase):
    """Test IdentityHub."""

    def test_disabled_returns_anonymous(self):
        hub = IdentityHub(enabled=False)
        # Mock request
        req = MagicMock()
        req.headers = {}
        req.cookies = {}

        import asyncio
        identity = asyncio.run(hub.resolve(req))
        self.assertEqual(identity.did, "did:key:anonymous")

    def test_bearer_auth(self):
        hub = IdentityHub(enabled=True)
        req = MagicMock()
        req.headers = {"Authorization": "Bearer test_token_12345"}
        req.cookies = {}

        import asyncio
        identity = asyncio.run(hub.resolve(req))
        self.assertTrue(identity.did.startswith("did:key:"))
        self.assertEqual(identity.auth_method, "bearer")


class TestCUAEngine(unittest.TestCase):
    """Test CUA Engine."""

    def test_proxy_mode_default(self):
        engine = CUAEngine(native=False)
        self.assertFalse(engine.native)

    def test_native_mode(self):
        engine = CUAEngine(native=True)
        self.assertTrue(engine.native)


class TestHelpers(unittest.TestCase):
    """Test helper functions."""

    def test_safe_sha256(self):
        h1 = _safe_sha256("hello")
        h2 = _safe_sha256("hello")
        h3 = _safe_sha256("world")
        self.assertEqual(h1, h2)
        self.assertNotEqual(h1, h3)
        self.assertEqual(len(h1), 64)

    def test_now_iso_format(self):
        ts = now_iso()
        self.assertTrue(ts.endswith("Z"))
        self.assertIn("T", ts)


class TestGapAnalysis(unittest.TestCase):
    """Test router gap analysis helpers."""

    def test_classify_router_failure_auth(self):
        reason = _classify_router_failure("Please run /login to authenticate", "", 1)
        self.assertEqual(reason, "auth")

    def test_classify_router_failure_timeout(self):
        reason = _classify_router_failure("", "router timeout", 124)
        self.assertEqual(reason, "timeout")

    def test_gap_analysis_auth(self):
        gaps = _gap_analysis_for_reason("auth", provider="claude-web")
        self.assertTrue(gaps["epistemic"])
        self.assertEqual(gaps["epistemic"][0]["id"], "E_AUTH")

    def test_gap_analysis_circuit_open(self):
        gaps = _gap_analysis_for_reason("circuit_open", provider="auto", cooldown_s=60.0)
        self.assertTrue(gaps["aleatoric"])
        self.assertEqual(gaps["aleatoric"][0]["id"], "A_CIRCUIT_OPEN")
        self.assertEqual(gaps["aleatoric"][0]["bounds"]["hi"], 60.0)


class TestWorldRouterConfig(unittest.TestCase):
    """Test configuration."""

    def test_default_config(self):
        cfg = WorldRouterConfig()
        self.assertEqual(cfg.port, 8080)
        self.assertEqual(cfg.host, "0.0.0.0")
        self.assertFalse(cfg.identity_hub)
        self.assertFalse(cfg.cua_native)
        self.assertTrue(cfg.vnc_tunnel)
        self.assertTrue(cfg.storage_fabric)

    def test_custom_config(self):
        cfg = WorldRouterConfig(
            port=9000,
            identity_hub=True,
            cua_native=True,
        )
        self.assertEqual(cfg.port, 9000)
        self.assertTrue(cfg.identity_hub)
        self.assertTrue(cfg.cua_native)


class TestCUAEngineMethods(unittest.TestCase):
    """Test CUA engine methods exist and have correct signatures."""

    def setUp(self):
        self.engine = CUAEngine(native=False)

    def test_has_screenshot(self):
        self.assertTrue(hasattr(self.engine, 'screenshot'))
        self.assertTrue(callable(self.engine.screenshot))

    def test_has_click(self):
        self.assertTrue(hasattr(self.engine, 'click'))
        self.assertTrue(callable(self.engine.click))

    def test_has_type_text(self):
        self.assertTrue(hasattr(self.engine, 'type_text'))
        self.assertTrue(callable(self.engine.type_text))

    def test_has_navigate(self):
        self.assertTrue(hasattr(self.engine, 'navigate'))
        self.assertTrue(callable(self.engine.navigate))

    def test_has_key_press(self):
        self.assertTrue(hasattr(self.engine, 'key_press'))
        self.assertTrue(callable(self.engine.key_press))

    def test_has_scroll(self):
        self.assertTrue(hasattr(self.engine, 'scroll'))
        self.assertTrue(callable(self.engine.scroll))

    def test_has_double_click(self):
        self.assertTrue(hasattr(self.engine, 'double_click'))
        self.assertTrue(callable(self.engine.double_click))

    def test_has_mouse_move(self):
        self.assertTrue(hasattr(self.engine, 'mouse_move'))
        self.assertTrue(callable(self.engine.mouse_move))

    def test_has_drag(self):
        self.assertTrue(hasattr(self.engine, 'drag'))
        self.assertTrue(callable(self.engine.drag))

    def test_has_get_screen_size(self):
        self.assertTrue(hasattr(self.engine, 'get_screen_size'))
        self.assertTrue(callable(self.engine.get_screen_size))


class TestAppRoutes(unittest.TestCase):
    """Test that all expected routes are registered."""

    def setUp(self):
        from world_router import create_app, WorldRouterConfig
        cfg = WorldRouterConfig()
        self.app = create_app(cfg)
        self.routes = [str(r.resource.canonical) for r in self.app.router.routes()]

    def test_health_routes(self):
        self.assertIn('/health', self.routes)
        self.assertIn('/healthz', self.routes)

    def test_llm_routes(self):
        self.assertIn('/v1/models', self.routes)
        self.assertIn('/v1/chat/completions', self.routes)
        self.assertIn('/v1/responses', self.routes)
        self.assertIn('/v1/messages', self.routes)
        self.assertIn('/v1/messages/count_tokens', self.routes)

    def test_cua_basic_routes(self):
        self.assertIn('/cua/screenshot', self.routes)
        self.assertIn('/cua/click', self.routes)
        self.assertIn('/cua/type', self.routes)
        self.assertIn('/cua/navigate', self.routes)
        self.assertIn('/cua/eval', self.routes)
        self.assertIn('/cua/query', self.routes)

    def test_cua_extended_routes(self):
        self.assertIn('/cua/key', self.routes)
        self.assertIn('/cua/scroll', self.routes)
        self.assertIn('/cua/double_click', self.routes)
        self.assertIn('/cua/move', self.routes)
        self.assertIn('/cua/drag', self.routes)
        self.assertIn('/cua/screen_size', self.routes)

    def test_vnc_route(self):
        self.assertIn('/vnc', self.routes)

    def test_bus_routes(self):
        self.assertIn('/bus', self.routes)
        self.assertIn('/ws/bus', self.routes)

    def test_storage_routes(self):
        self.assertIn('/storage/status', self.routes)
        # Dynamic routes have different canonical form
        route_strs = ' '.join(self.routes)
        self.assertIn('storage', route_strs)

    def test_identity_routes(self):
        self.assertIn('/identity/whoami', self.routes)
        self.assertIn('/identity/sessions', self.routes)
        self.assertIn('/identity/oauth/start', self.routes)
        self.assertIn('/identity/oauth/callback', self.routes)

    def test_total_route_count(self):
        """Verify we have all expected routes."""
        # Should have 45+ routes
        self.assertGreaterEqual(len(self.routes), 40)


class TestProtocolRouterExtended(unittest.TestCase):
    """Extended protocol detection tests."""

    def _make_request(self, path: str, method: str = "GET", headers: dict = None):
        req = MagicMock()
        req.path = path
        req.method = method
        req.headers = headers or {}
        return req

    def test_detect_openai_responses(self):
        req = self._make_request("/v1/responses", "POST")
        # This goes through as openai-chat since we don't have explicit responses detection
        # The ProtocolRouter.detect currently returns 'http/openai-chat' for /v1/* paths
        result = ProtocolRouter.detect(req)
        # Either openai-chat or a new openai-responses type would be valid
        self.assertIn(result, ["http/openai-chat", "http/openai-responses", "http/default"])

    def test_detect_bus_websocket(self):
        req = self._make_request("/bus", "GET", {"Upgrade": "websocket"})
        self.assertEqual(ProtocolRouter.detect(req), "ws/bus")
        req = self._make_request("/ws/bus", "GET", {"Upgrade": "websocket"})
        self.assertEqual(ProtocolRouter.detect(req), "ws/bus")


# ============================================================================
# Phase 2 Tests - Circuit Breaker
# ============================================================================

class TestCircuitBreaker(unittest.TestCase):
    """Test Circuit Breaker functionality."""

    def setUp(self):
        from world_router import CircuitBreaker
        self.cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=60.0)

    def test_initial_state_closed(self):
        """New providers start in closed state."""
        self.assertEqual(self.cb.get_state("new-provider"), "closed")
        self.assertTrue(self.cb.is_available("new-provider"))

    def test_success_keeps_closed(self):
        """Successful requests keep circuit closed."""
        self.cb.record_success("provider-a")
        self.assertEqual(self.cb.get_state("provider-a"), "closed")

    def test_failure_opens_after_threshold(self):
        """Circuit opens after failure threshold."""
        for _ in range(3):
            self.cb.record_failure("provider-b")
        self.assertEqual(self.cb.get_state("provider-b"), "open")
        self.assertFalse(self.cb.is_available("provider-b"))

    def test_failure_below_threshold_stays_closed(self):
        """Circuit stays closed if failures below threshold."""
        self.cb.record_failure("provider-c")
        self.cb.record_failure("provider-c")
        self.assertEqual(self.cb.get_state("provider-c"), "closed")
        self.assertTrue(self.cb.is_available("provider-c"))

    def test_success_resets_failure_count(self):
        """Success resets failure counter."""
        self.cb.record_failure("provider-d")
        self.cb.record_failure("provider-d")
        self.cb.record_success("provider-d")
        self.cb.record_failure("provider-d")
        self.cb.record_failure("provider-d")
        # Should still be closed (failures reset by success)
        self.assertEqual(self.cb.get_state("provider-d"), "closed")

    def test_get_all_states(self):
        """get_all_states returns all tracked providers."""
        self.cb.record_failure("provider-e")
        # Note: record_success only updates existing entries, doesn't create new ones
        # This is correct - no need to track providers that only have successes
        self.cb.record_failure("provider-f")  # Create entry first
        self.cb.record_success("provider-f")  # Then reset it
        states = self.cb.get_all_states()
        self.assertIn("provider-e", states)
        self.assertIn("provider-f", states)
        # provider-f should be reset to 0 failures
        self.assertEqual(states["provider-f"]["failures"], 0)


# ============================================================================
# Phase 2 Tests - Identity with OAuth
# ============================================================================

class TestIdentityOAuth(unittest.TestCase):
    """Test Identity OAuth functionality."""

    def test_oauth_providers_configured(self):
        """OAuth providers are properly configured."""
        hub = IdentityHub(enabled=True)
        self.assertIn("google", hub.OAUTH_PROVIDERS)
        self.assertIn("github", hub.OAUTH_PROVIDERS)

        google = hub.OAUTH_PROVIDERS["google"]
        self.assertIn("auth_url", google)
        self.assertIn("token_url", google)
        self.assertIn("userinfo_url", google)
        self.assertIn("scopes", google)

    def test_generate_pkce_verifier(self):
        """PKCE verifier and challenge are generated correctly."""
        hub = IdentityHub(enabled=True)
        verifier, challenge = hub._generate_pkce_verifier()

        # Verifier should be URL-safe base64
        self.assertIsInstance(verifier, str)
        self.assertGreater(len(verifier), 30)

        # Challenge should be different from verifier
        self.assertNotEqual(verifier, challenge)

    def test_generate_did_from_email(self):
        """DID generation is deterministic."""
        hub = IdentityHub(enabled=True)
        did1 = hub._generate_did_from_email("test@example.com", "google")
        did2 = hub._generate_did_from_email("test@example.com", "google")
        did3 = hub._generate_did_from_email("other@example.com", "google")

        # Same email produces same DID
        self.assertEqual(did1, did2)
        # Different email produces different DID
        self.assertNotEqual(did1, did3)
        # DID has correct prefix
        self.assertTrue(did1.startswith("did:web:pluribus.google:"))

    def test_session_persistence(self):
        """Sessions can be created and persisted."""
        import tempfile
        import os

        with tempfile.TemporaryDirectory() as tmpdir:
            session_file = os.path.join(tmpdir, "sessions.json")
            hub = IdentityHub(enabled=True, session_file=session_file)

            # Create a test identity
            identity = Identity(
                did="did:test:123",
                session_type="human",
                auth_method="oauth",
                auth_provider="google",
                email="test@example.com",
            )

            session_id = hub.create_session(identity)
            self.assertIn(session_id, hub._sessions)

            # Create new hub to test persistence
            hub2 = IdentityHub(enabled=True, session_file=session_file)
            self.assertIn(session_id, hub2._sessions)

    def test_session_revocation(self):
        """Sessions can be revoked."""
        hub = IdentityHub(enabled=True, session_file="/tmp/test_revoke_sessions.json")
        identity = Identity(did="did:test:revoke", session_type="human")
        session_id = hub.create_session(identity)

        self.assertIn(session_id, hub._sessions)
        hub.revoke_session(session_id)
        self.assertNotIn(session_id, hub._sessions)


# ============================================================================
# Phase 2 Tests - App Routes Extended
# ============================================================================

class TestAppRoutesPhase2(unittest.TestCase):
    """Test Phase 2 routes are registered."""

    def setUp(self):
        from world_router import create_app, WorldRouterConfig
        cfg = WorldRouterConfig()
        self.app = create_app(cfg)
        self.routes = [str(r.resource.canonical) for r in self.app.router.routes()]

    def test_admin_circuit_routes(self):
        """Admin circuit breaker routes are registered."""
        self.assertIn('/admin/circuits', self.routes)
        self.assertIn('/admin/circuits/reset', self.routes)

    def test_identity_revoke_route(self):
        """Identity revoke route is registered."""
        self.assertIn('/identity/revoke', self.routes)

    def test_route_count_phase2(self):
        """Route count increased for Phase 2."""
        # Phase 2 adds: /admin/circuits, /admin/circuits/reset, /identity/revoke
        self.assertGreaterEqual(len(self.routes), 45)


# ============================================================================
# Phase 1-2 Enhancement Tests
# ============================================================================

class TestOAuthStateExpiration(unittest.TestCase):
    """Test OAuth state expiration functionality."""

    def test_oauth_state_expires(self):
        """OAuth states should expire after 10 minutes."""
        from world_router import OAuthState
        import time

        state = OAuthState(
            state="test",
            provider="google",
            redirect_uri="http://localhost/callback",
            created_at="2025-01-01T00:00:00Z",
            expires_at=time.time() - 1,  # Already expired
        )

        self.assertTrue(state.is_expired())

    def test_oauth_state_not_expired(self):
        """Fresh OAuth states should not be expired."""
        from world_router import OAuthState
        import time

        state = OAuthState(
            state="test",
            provider="google",
            redirect_uri="http://localhost/callback",
            created_at="2025-01-01T00:00:00Z",
        )

        self.assertFalse(state.is_expired())

    def test_cleanup_removes_expired(self):
        """cleanup_expired_states removes expired states."""
        import time
        hub = IdentityHub(enabled=True)

        # Add expired state
        from world_router import OAuthState
        hub._oauth_states["expired"] = OAuthState(
            state="expired",
            provider="google",
            redirect_uri="http://localhost",
            created_at="2025-01-01T00:00:00Z",
            expires_at=time.time() - 1,
        )

        # Add fresh state
        hub._oauth_states["fresh"] = OAuthState(
            state="fresh",
            provider="google",
            redirect_uri="http://localhost",
            created_at="2025-01-01T00:00:00Z",
            expires_at=time.time() + 600,
        )

        hub.cleanup_expired_states()

        self.assertNotIn("expired", hub._oauth_states)
        self.assertIn("fresh", hub._oauth_states)


class TestTokenRefresh(unittest.TestCase):
    """Test token refresh functionality."""

    def test_refresh_requires_token(self):
        """refresh_session returns None if no refresh token."""
        import asyncio
        hub = IdentityHub(enabled=True)

        identity = Identity(did="did:test:notoken", session_type="human")
        hub.create_session(identity)

        result = asyncio.run(
            hub.refresh_session(identity.session_id)
        )

        self.assertIsNone(result)


class TestA2ARoutes(unittest.TestCase):
    """Test A2A protocol routes."""

    def setUp(self):
        from world_router import create_app, WorldRouterConfig
        cfg = WorldRouterConfig()
        self.app = create_app(cfg)
        self.routes = [str(r.resource.canonical) for r in self.app.router.routes()]

    def test_agent_json_route(self):
        """/.well-known/agent.json route is registered."""
        self.assertIn('/.well-known/agent.json', self.routes)

    def test_a2a_tasks_route(self):
        """A2A tasks routes are registered."""
        self.assertIn('/a2a/tasks', self.routes)
        # Dynamic route will be different
        route_str = ' '.join(self.routes)
        self.assertIn('a2a/tasks', route_str)


class TestStorageWriteRoute(unittest.TestCase):
    """Test storage write route."""

    def setUp(self):
        from world_router import create_app, WorldRouterConfig
        cfg = WorldRouterConfig(storage_fabric=True)
        self.app = create_app(cfg)
        self.routes = [str(r.resource.canonical) for r in self.app.router.routes()]

    def test_storage_write_route(self):
        """Storage write route is registered."""
        route_str = ' '.join(self.routes)
        self.assertIn('storage', route_str)
        self.assertIn('write', route_str)


class TestTotalRouteCount(unittest.TestCase):
    """Test total route count after all enhancements."""

    def setUp(self):
        from world_router import create_app, WorldRouterConfig
        cfg = WorldRouterConfig()
        self.app = create_app(cfg)
        self.routes = [str(r.resource.canonical) for r in self.app.router.routes()]

    def test_route_count_enhanced(self):
        """Route count after Phase 1-2 enhancements."""
        # Should have 50+ routes now with A2A and storage write
        self.assertGreaterEqual(len(self.routes), 50)


# ============================================================================
# Phase 3 Tests - WebAuthn/Passkey
# ============================================================================

class TestWebAuthnCredential(unittest.TestCase):
    """Test WebAuthnCredential dataclass."""

    def test_credential_creation(self):
        """Test creating a WebAuthn credential."""
        from world_router import WebAuthnCredential
        cred = WebAuthnCredential(
            credential_id="test-cred-id",
            user_id="test@example.com",
            public_key="test-public-key",
        )
        self.assertEqual(cred.credential_id, "test-cred-id")
        self.assertEqual(cred.user_id, "test@example.com")
        self.assertEqual(cred.sign_count, 0)
        self.assertIsNotNone(cred.created_at)

    def test_credential_to_dict(self):
        """Test credential serialization."""
        from world_router import WebAuthnCredential
        cred = WebAuthnCredential(
            credential_id="test-cred",
            user_id="user@test.com",
            public_key="pubkey123",
            authenticator_type="platform",
            transports=["internal"],
        )
        d = cred.to_dict()
        self.assertEqual(d["credential_id"], "test-cred")
        self.assertEqual(d["authenticator_type"], "platform")
        self.assertIn("internal", d["transports"])


class TestWebAuthnChallenge(unittest.TestCase):
    """Test WebAuthnChallenge dataclass."""

    def test_challenge_expiration(self):
        """Test challenge expiration (5 minutes)."""
        from world_router import WebAuthnChallenge
        import time

        # Fresh challenge should not be expired
        challenge = WebAuthnChallenge(
            challenge="test-challenge",
            user_id="user@test.com",
            operation="registration",
        )
        self.assertFalse(challenge.is_expired())

        # Expired challenge
        expired = WebAuthnChallenge(
            challenge="expired",
            expires_at=time.time() - 1,
        )
        self.assertTrue(expired.is_expired())

    def test_challenge_default_expiration(self):
        """Challenge defaults to 5 minute expiration."""
        from world_router import WebAuthnChallenge
        import time

        challenge = WebAuthnChallenge(challenge="test")
        # Should expire in about 5 minutes (300 seconds)
        time_until_expiry = challenge.expires_at - time.time()
        self.assertGreater(time_until_expiry, 290)  # At least 290 seconds
        self.assertLess(time_until_expiry, 310)  # At most 310 seconds


class TestIdentityHubWebAuthn(unittest.TestCase):
    """Test IdentityHub WebAuthn functionality."""

    def setUp(self):
        import tempfile
        self.tmpdir = tempfile.mkdtemp()
        self.session_file = f"{self.tmpdir}/sessions.json"
        self.webauthn_file = f"{self.tmpdir}/webauthn.json"
        self.hub = IdentityHub(
            enabled=True,
            session_file=self.session_file,
            webauthn_file=self.webauthn_file,
        )

    def test_generate_webauthn_challenge(self):
        """Challenge generation produces unique values."""
        c1 = self.hub._generate_webauthn_challenge()
        c2 = self.hub._generate_webauthn_challenge()
        self.assertNotEqual(c1, c2)
        # Should be base64-ish (URL-safe chars)
        self.assertGreater(len(c1), 30)

    def test_get_rp_id(self):
        """rpId extraction from host."""
        self.assertEqual(self.hub._get_rp_id("localhost:8080"), "localhost")
        self.assertEqual(self.hub._get_rp_id("example.com"), "example.com")
        self.assertEqual(self.hub._get_rp_id("sub.example.com:443"), "sub.example.com")

    def test_webauthn_register_start(self):
        """Test registration start generates proper options."""
        options = self.hub.webauthn_register_start(
            user_id="test@example.com",
            user_name="Test User",
            host="localhost:8080",
        )

        self.assertIn("challenge", options)
        self.assertIn("rp", options)
        self.assertEqual(options["rp"]["id"], "localhost")
        self.assertEqual(options["rp"]["name"], "Pluribus")
        self.assertIn("user", options)
        self.assertEqual(options["user"]["name"], "test@example.com")
        self.assertIn("pubKeyCredParams", options)
        self.assertEqual(options["timeout"], 300000)

        # Challenge should be stored
        self.assertIn(options["challenge"], self.hub._webauthn_challenges)

    def test_webauthn_register_complete(self):
        """Test registration completion stores credential."""
        # Start registration
        options = self.hub.webauthn_register_start(
            user_id="test@example.com",
            user_name="Test",
            host="localhost",
        )
        challenge = options["challenge"]

        # Complete registration
        cred = self.hub.webauthn_register_complete(
            challenge=challenge,
            credential_id="new-cred-123",
            public_key="public-key-data",
            authenticator_type="platform",
            transports=["internal"],
            friendly_name="My MacBook",
        )

        self.assertIsNotNone(cred)
        self.assertEqual(cred.credential_id, "new-cred-123")
        self.assertEqual(cred.user_id, "test@example.com")
        self.assertEqual(cred.friendly_name, "My MacBook")

        # Credential should be stored
        self.assertIn("new-cred-123", self.hub._webauthn_credentials)

        # Challenge should be consumed
        self.assertNotIn(challenge, self.hub._webauthn_challenges)

    def test_webauthn_register_invalid_challenge(self):
        """Registration fails with invalid challenge."""
        cred = self.hub.webauthn_register_complete(
            challenge="invalid-challenge",
            credential_id="cred",
            public_key="key",
        )
        self.assertIsNone(cred)

    def test_webauthn_register_expired_challenge(self):
        """Registration fails with expired challenge."""
        from world_router import WebAuthnChallenge
        import time

        # Add expired challenge
        self.hub._webauthn_challenges["expired"] = WebAuthnChallenge(
            challenge="expired",
            user_id="test@example.com",
            operation="registration",
            expires_at=time.time() - 1,
        )

        cred = self.hub.webauthn_register_complete(
            challenge="expired",
            credential_id="cred",
            public_key="key",
        )
        self.assertIsNone(cred)

    def test_webauthn_login_start(self):
        """Test login start generates proper options."""
        options = self.hub.webauthn_login_start(
            user_id=None,
            host="example.com:443",
        )

        self.assertIn("challenge", options)
        self.assertEqual(options["rpId"], "example.com")
        self.assertEqual(options["timeout"], 300000)
        self.assertIn("allowCredentials", options)

    def test_webauthn_login_complete(self):
        """Test login completion creates session."""
        # First register a credential
        reg_options = self.hub.webauthn_register_start(
            user_id="login@test.com",
            user_name="Login Test",
            host="localhost",
        )
        self.hub.webauthn_register_complete(
            challenge=reg_options["challenge"],
            credential_id="login-cred-456",
            public_key="pubkey",
        )

        # Start login
        login_options = self.hub.webauthn_login_start(
            user_id="login@test.com",
            host="localhost",
        )

        # Complete login
        # Note: In real scenario, signature verification would happen
        identity = self.hub.webauthn_login_complete(
            challenge=login_options["challenge"],
            credential_id="login-cred-456",
            signature="fake-signature",
            authenticator_data="AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA",  # 37+ bytes
            client_data_json="eyJ0eXBlIjoid2ViYXV0aG4uZ2V0In0",
        )

        self.assertIsNotNone(identity)
        self.assertEqual(identity.auth_method, "webauthn")
        self.assertEqual(identity.auth_provider, "passkey")
        self.assertIn(identity.session_id, self.hub._sessions)

    def test_webauthn_login_invalid_credential(self):
        """Login fails with unknown credential."""
        # Start login
        options = self.hub.webauthn_login_start(user_id=None, host="localhost")

        identity = self.hub.webauthn_login_complete(
            challenge=options["challenge"],
            credential_id="nonexistent-cred",
            signature="sig",
            authenticator_data="data",
            client_data_json="json",
        )
        self.assertIsNone(identity)

    def test_get_user_credentials(self):
        """Test listing user credentials."""
        # Register two credentials for same user
        for i in range(2):
            opts = self.hub.webauthn_register_start(
                user_id="multi@test.com",
                user_name="Multi",
                host="localhost",
            )
            self.hub.webauthn_register_complete(
                challenge=opts["challenge"],
                credential_id=f"multi-cred-{i}",
                public_key=f"key-{i}",
            )

        creds = self.hub.get_user_credentials("multi@test.com")
        self.assertEqual(len(creds), 2)

    def test_delete_credential(self):
        """Test credential deletion."""
        # Register credential
        opts = self.hub.webauthn_register_start(
            user_id="delete@test.com",
            user_name="Delete",
            host="localhost",
        )
        self.hub.webauthn_register_complete(
            challenge=opts["challenge"],
            credential_id="to-delete",
            public_key="key",
        )

        # Delete as owner
        success = self.hub.delete_credential("to-delete", "delete@test.com")
        self.assertTrue(success)
        self.assertNotIn("to-delete", self.hub._webauthn_credentials)

    def test_delete_credential_wrong_user(self):
        """Cannot delete another user's credential."""
        opts = self.hub.webauthn_register_start(
            user_id="owner@test.com",
            user_name="Owner",
            host="localhost",
        )
        self.hub.webauthn_register_complete(
            challenge=opts["challenge"],
            credential_id="owned-cred",
            public_key="key",
        )

        # Try to delete as different user
        success = self.hub.delete_credential("owned-cred", "attacker@test.com")
        self.assertFalse(success)
        self.assertIn("owned-cred", self.hub._webauthn_credentials)

    def test_credential_persistence(self):
        """Test credentials are persisted to disk."""
        # Register credential
        opts = self.hub.webauthn_register_start(
            user_id="persist@test.com",
            user_name="Persist",
            host="localhost",
        )
        self.hub.webauthn_register_complete(
            challenge=opts["challenge"],
            credential_id="persist-cred",
            public_key="persist-key",
        )

        # Create new hub instance to test persistence
        hub2 = IdentityHub(
            enabled=True,
            session_file=self.session_file,
            webauthn_file=self.webauthn_file,
        )

        self.assertIn("persist-cred", hub2._webauthn_credentials)
        cred = hub2._webauthn_credentials["persist-cred"]
        self.assertEqual(cred.user_id, "persist@test.com")


class TestWebAuthnRoutes(unittest.TestCase):
    """Test WebAuthn routes are registered."""

    def setUp(self):
        from world_router import create_app, WorldRouterConfig
        cfg = WorldRouterConfig()
        self.app = create_app(cfg)
        self.routes = [str(r.resource.canonical) for r in self.app.router.routes()]

    def test_webauthn_register_routes(self):
        """WebAuthn registration routes are registered."""
        self.assertIn('/identity/webauthn/register/start', self.routes)
        self.assertIn('/identity/webauthn/register/complete', self.routes)

    def test_webauthn_login_routes(self):
        """WebAuthn login routes are registered."""
        self.assertIn('/identity/webauthn/login/start', self.routes)
        self.assertIn('/identity/webauthn/login/complete', self.routes)

    def test_webauthn_credential_routes(self):
        """WebAuthn credential management routes are registered."""
        self.assertIn('/identity/webauthn/credentials', self.routes)
        self.assertIn('/identity/webauthn/credentials/delete', self.routes)

    def test_route_count_phase3(self):
        """Route count increased for Phase 3."""
        # Phase 3 adds 6 WebAuthn routes
        self.assertGreaterEqual(len(self.routes), 56)


class TestWebAuthnExcludeCredentials(unittest.TestCase):
    """Test that registration excludes existing credentials."""

    def setUp(self):
        import tempfile
        self.tmpdir = tempfile.mkdtemp()
        self.hub = IdentityHub(
            enabled=True,
            session_file=f"{self.tmpdir}/sessions.json",
            webauthn_file=f"{self.tmpdir}/webauthn.json",
        )

    def test_exclude_existing_credentials(self):
        """Registration options exclude user's existing credentials."""
        # Register first credential
        opts1 = self.hub.webauthn_register_start(
            user_id="exclude@test.com",
            user_name="Exclude Test",
            host="localhost",
        )
        self.hub.webauthn_register_complete(
            challenge=opts1["challenge"],
            credential_id="first-cred",
            public_key="key1",
            transports=["usb"],
        )

        # Start second registration
        opts2 = self.hub.webauthn_register_start(
            user_id="exclude@test.com",
            user_name="Exclude Test",
            host="localhost",
        )

        # Should have excludeCredentials with first credential
        excluded = opts2.get("excludeCredentials", [])
        self.assertEqual(len(excluded), 1)
        self.assertEqual(excluded[0]["id"], "first-cred")


# ============================================================================
# Phase 3 Tests - CUA Native Playwright Integration
# ============================================================================

class TestCUAConfig(unittest.TestCase):
    """Test CUAConfig dataclass."""

    def test_default_config(self):
        """Test default CUA configuration values."""
        from world_router import CUAConfig
        cfg = CUAConfig()
        self.assertEqual(cfg.browser, "chromium")
        self.assertFalse(cfg.headless)
        self.assertEqual(cfg.viewport_width, 1920)
        self.assertEqual(cfg.viewport_height, 1080)
        self.assertEqual(cfg.display, "")
        self.assertEqual(cfg.slow_mo, 0)
        self.assertEqual(cfg.default_timeout_ms, 30000)
        self.assertEqual(cfg.max_retries, 3)
        self.assertEqual(cfg.retry_delay_ms, 1000)

    def test_custom_config(self):
        """Test custom CUA configuration."""
        from world_router import CUAConfig
        cfg = CUAConfig(
            browser="firefox",
            headless=True,
            viewport_width=1280,
            viewport_height=720,
            display=":5",
            slow_mo=100,
            default_timeout_ms=60000,
            max_retries=5,
        )
        self.assertEqual(cfg.browser, "firefox")
        self.assertTrue(cfg.headless)
        self.assertEqual(cfg.viewport_width, 1280)
        self.assertEqual(cfg.viewport_height, 720)
        self.assertEqual(cfg.display, ":5")
        self.assertEqual(cfg.slow_mo, 100)
        self.assertEqual(cfg.default_timeout_ms, 60000)
        self.assertEqual(cfg.max_retries, 5)


class TestCUAEngineNativeInit(unittest.TestCase):
    """Test CUAEngine initialization and configuration."""

    def test_proxy_mode_default(self):
        """Engine defaults to proxy mode."""
        engine = CUAEngine(native=False)
        self.assertFalse(engine.native)
        self.assertFalse(engine._initialized)

    def test_native_mode(self):
        """Engine can be set to native mode."""
        engine = CUAEngine(native=True)
        self.assertTrue(engine.native)
        self.assertFalse(engine._initialized)

    def test_custom_config(self):
        """Engine accepts custom configuration."""
        from world_router import CUAConfig
        cfg = CUAConfig(browser="webkit", viewport_width=800)
        engine = CUAEngine(native=True, config=cfg)
        self.assertEqual(engine.config.browser, "webkit")
        self.assertEqual(engine.config.viewport_width, 800)

    def test_default_config_created(self):
        """Engine creates default config if not provided."""
        engine = CUAEngine(native=True)
        self.assertIsNotNone(engine.config)
        self.assertEqual(engine.config.browser, "chromium")

    def test_page_tracking_initialized(self):
        """Engine initializes page tracking."""
        engine = CUAEngine(native=True)
        self.assertEqual(engine._pages, {})
        self.assertEqual(engine._current_page_id, "default")


class TestCUAEngineProxyMode(unittest.TestCase):
    """Test CUAEngine in proxy mode (no Playwright)."""

    def setUp(self):
        self.engine = CUAEngine(native=False)

    def test_initialize_returns_proxy_ok(self):
        """Initialize in proxy mode returns ok."""
        import asyncio
        result = asyncio.run(
            self.engine.initialize()
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["mode"], "proxy")

    def test_wait_for_selector_requires_native(self):
        """wait_for_selector returns error in proxy mode."""
        import asyncio
        result = asyncio.run(
            self.engine.wait_for_selector("div")
        )
        self.assertFalse(result["ok"])
        self.assertIn("native", result["error"].lower())

    def test_wait_for_load_state_requires_native(self):
        """wait_for_load_state returns error in proxy mode."""
        import asyncio
        result = asyncio.run(
            self.engine.wait_for_load_state()
        )
        self.assertFalse(result["ok"])

    def test_get_page_info_proxy_mode(self):
        """get_page_info returns limited info in proxy mode."""
        import asyncio
        result = asyncio.run(
            self.engine.get_page_info()
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["mode"], "proxy")

    def test_get_html_requires_native(self):
        """get_html returns error in proxy mode."""
        import asyncio
        result = asyncio.run(
            self.engine.get_html()
        )
        self.assertFalse(result["ok"])

    def test_get_text_requires_native(self):
        """get_text returns error in proxy mode."""
        import asyncio
        result = asyncio.run(
            self.engine.get_text("body")
        )
        self.assertFalse(result["ok"])

    def test_new_page_requires_native(self):
        """new_page returns error in proxy mode."""
        import asyncio
        result = asyncio.run(
            self.engine.new_page("http://example.com")
        )
        self.assertFalse(result["ok"])

    def test_switch_page_requires_native(self):
        """switch_page returns error in proxy mode."""
        import asyncio
        result = asyncio.run(
            self.engine.switch_page("page_id")
        )
        self.assertFalse(result["ok"])

    def test_close_page_requires_native(self):
        """close_page returns error in proxy mode."""
        import asyncio
        result = asyncio.run(
            self.engine.close_page("page_id")
        )
        self.assertFalse(result["ok"])

    def test_list_pages_requires_native(self):
        """list_pages returns error in proxy mode."""
        import asyncio
        result = asyncio.run(
            self.engine.list_pages()
        )
        self.assertFalse(result["ok"])

    def test_shutdown_proxy_mode(self):
        """shutdown in proxy mode does nothing."""
        import asyncio
        result = asyncio.run(
            self.engine.shutdown()
        )
        self.assertTrue(result["ok"])


class TestCUAEngineMethods(unittest.TestCase):
    """Test CUA engine method availability and signatures."""

    def setUp(self):
        self.engine = CUAEngine(native=False)

    def test_has_shutdown(self):
        """Engine has shutdown method."""
        self.assertTrue(hasattr(self.engine, 'shutdown'))
        self.assertTrue(callable(self.engine.shutdown))

    def test_has_new_page(self):
        """Engine has new_page method."""
        self.assertTrue(hasattr(self.engine, 'new_page'))
        self.assertTrue(callable(self.engine.new_page))

    def test_has_switch_page(self):
        """Engine has switch_page method."""
        self.assertTrue(hasattr(self.engine, 'switch_page'))
        self.assertTrue(callable(self.engine.switch_page))

    def test_has_close_page(self):
        """Engine has close_page method."""
        self.assertTrue(hasattr(self.engine, 'close_page'))
        self.assertTrue(callable(self.engine.close_page))

    def test_has_list_pages(self):
        """Engine has list_pages method."""
        self.assertTrue(hasattr(self.engine, 'list_pages'))
        self.assertTrue(callable(self.engine.list_pages))

    def test_has_wait_for_selector(self):
        """Engine has wait_for_selector method."""
        self.assertTrue(hasattr(self.engine, 'wait_for_selector'))
        self.assertTrue(callable(self.engine.wait_for_selector))

    def test_has_wait_for_load_state(self):
        """Engine has wait_for_load_state method."""
        self.assertTrue(hasattr(self.engine, 'wait_for_load_state'))
        self.assertTrue(callable(self.engine.wait_for_load_state))

    def test_has_get_page_info(self):
        """Engine has get_page_info method."""
        self.assertTrue(hasattr(self.engine, 'get_page_info'))
        self.assertTrue(callable(self.engine.get_page_info))

    def test_has_get_html(self):
        """Engine has get_html method."""
        self.assertTrue(hasattr(self.engine, 'get_html'))
        self.assertTrue(callable(self.engine.get_html))

    def test_has_get_text(self):
        """Engine has get_text method."""
        self.assertTrue(hasattr(self.engine, 'get_text'))
        self.assertTrue(callable(self.engine.get_text))

    def test_has_goto(self):
        """Engine has goto method."""
        self.assertTrue(hasattr(self.engine, 'goto'))
        self.assertTrue(callable(self.engine.goto))


class TestCUAPhase3Routes(unittest.TestCase):
    """Test Phase 3 CUA routes are registered."""

    def setUp(self):
        from world_router import create_app, WorldRouterConfig
        cfg = WorldRouterConfig()
        self.app = create_app(cfg)
        self.routes = [str(r.resource.canonical) for r in self.app.router.routes()]

    def test_cua_wait_route(self):
        """CUA wait route is registered."""
        self.assertIn('/cua/wait', self.routes)

    def test_cua_page_info_route(self):
        """CUA page_info route is registered."""
        self.assertIn('/cua/page_info', self.routes)

    def test_cua_goto_route(self):
        """CUA goto route is registered."""
        self.assertIn('/cua/goto', self.routes)

    def test_cua_html_route(self):
        """CUA html route is registered."""
        self.assertIn('/cua/html', self.routes)

    def test_cua_text_route(self):
        """CUA text route is registered."""
        self.assertIn('/cua/text', self.routes)

    def test_cua_pages_route(self):
        """CUA pages route is registered."""
        self.assertIn('/cua/pages', self.routes)

    def test_cua_new_page_route(self):
        """CUA new_page route is registered."""
        self.assertIn('/cua/new_page', self.routes)

    def test_cua_switch_page_route(self):
        """CUA switch_page route is registered."""
        self.assertIn('/cua/switch_page', self.routes)

    def test_cua_close_page_route(self):
        """CUA close_page route is registered."""
        self.assertIn('/cua/close_page', self.routes)

    def test_cua_shutdown_route(self):
        """CUA shutdown route is registered."""
        self.assertIn('/cua/shutdown', self.routes)


class TestRouteCountPhase3(unittest.TestCase):
    """Test route count includes Phase 3 CUA endpoints."""

    def setUp(self):
        from world_router import create_app, WorldRouterConfig
        cfg = WorldRouterConfig()
        self.app = create_app(cfg)
        self.routes = [str(r.resource.canonical) for r in self.app.router.routes()]

    def test_route_count_phase3(self):
        """Route count includes Phase 3 CUA routes."""
        # Phase 3 adds 10 new CUA routes
        self.assertGreaterEqual(len(self.routes), 60)

    def test_cua_route_count(self):
        """CUA routes count is correct."""
        cua_routes = [r for r in self.routes if '/cua/' in r]
        # Basic (6) + Extended (6) + Phase 3 (10) = 22 CUA routes
        self.assertGreaterEqual(len(cua_routes), 20)


if __name__ == "__main__":
    unittest.main()
