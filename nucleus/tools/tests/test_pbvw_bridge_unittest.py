import json
import os
import socket
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from urllib.request import Request, urlopen


def _free_port() -> int:
    try:
        s = socket.socket()
    except PermissionError:
        raise unittest.SkipTest("socket creation not permitted in this environment")
    s.bind(("127.0.0.1", 0))
    port = int(s.getsockname()[1])
    s.close()
    return port


def _wait_http_ok(url: str, *, deadline_s: float = 5.0) -> None:
    deadline = time.time() + deadline_s
    last_err: Exception | None = None
    while time.time() < deadline:
        try:
            with urlopen(url, timeout=0.5) as resp:
                body = resp.read(200).decode("utf-8", errors="replace")
            if body:
                return
        except Exception as e:
            last_err = e
            time.sleep(0.1)
    raise AssertionError(f"timeout waiting for {url} ({last_err})")


def _get_json(url: str) -> dict:
    with urlopen(url, timeout=5) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _post_json(url: str, payload: dict) -> tuple[int, dict]:
    req = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urlopen(req, timeout=15) as resp:
            return int(resp.status), json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        # urllib raises on non-2xx; unwrap the body if present.
        status = getattr(getattr(e, "fp", None), "status", None)
        raw = getattr(getattr(e, "fp", None), "read", lambda: b"")()
        try:
            body = json.loads(raw.decode("utf-8", errors="replace") or "{}")
        except Exception:
            body = {}
        return int(status or 0), body


def _post_raw(url: str, payload: dict, *, headers: dict[str, str] | None = None) -> tuple[int, bytes, dict[str, str]]:
    req = Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json", **(headers or {})},
        method="POST",
    )
    try:
        with urlopen(req, timeout=15) as resp:
            hdrs = {k.lower(): v for k, v in resp.headers.items()}
            return int(resp.status), resp.read(), hdrs
    except Exception as e:
        status = getattr(getattr(e, "fp", None), "status", None)
        raw = getattr(getattr(e, "fp", None), "read", lambda: b"")()
        hdrs = {k.lower(): v for k, v in getattr(getattr(e, "fp", None), "headers", {}).items()}
        return int(status or 0), raw, hdrs


class TestPBVWBridge(unittest.TestCase):
    def test_chat_completions_roundtrip_and_bus_evidence(self):
        tools_dir = Path(__file__).resolve().parents[1]
        bridge = tools_dir / "pbvw_bridge.py"
        port = _free_port()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bus_dir = root / "bus"
            bus_dir.mkdir(parents=True, exist_ok=True)
            events_path = bus_dir / "events.ndjson"
            events_path.write_text("", encoding="utf-8")

            # Stub router: echoes OK unless prompt contains FAIL.
            router = root / "router_stub.py"
            router.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env python3",
                        "import argparse,json,sys",
                        "p=argparse.ArgumentParser()",
                        "p.add_argument('--provider',default='auto')",
                        "p.add_argument('--prompt',required=True)",
                        "p.add_argument('--model',default=None)",
                        "p.add_argument('--format',default='text')",
                        "a=p.parse_args()",
                        "if 'FAIL' in a.prompt:",
                        "    payload={'ok':False,'provider':a.provider,'model':a.model,'text':'','stderr':'router fail'}",
                        "    sys.stdout.write(json.dumps(payload)+'\\n')",
                        "    raise SystemExit(2)",
                        "payload={'ok':True,'provider':a.provider,'model':a.model,'text':'OK\\n','stderr':''}",
                        "sys.stdout.write(json.dumps(payload)+'\\n')",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            os.chmod(router, 0o755)

            env = {**os.environ, "PLURIBUS_BUS_DIR": str(bus_dir), "PYTHONDONTWRITEBYTECODE": "1"}
            proc = subprocess.Popen(
                [
                    "python3",
                    str(bridge),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--bus-dir",
                    str(bus_dir),
                    "--actor",
                    "test-pbvw",
                    "--router-path",
                    str(router),
                    "--router-timeout-s",
                    "10",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
            )
            try:
                _wait_http_ok(f"http://127.0.0.1:{port}/health")

                models = _get_json(f"http://127.0.0.1:{port}/v1/models")
                model_ids = [m.get("id") for m in models.get("data") or []]
                self.assertIn("chatgpt-web", model_ids)
                self.assertIn("claude-web", model_ids)
                self.assertIn("gemini-web", model_ids)

                status, out = _post_json(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    {
                        "model": "gpt-5.2-turbo",
                        "messages": [{"role": "user", "content": "PLURIBUS(kind=apply,effects=file): Reply with exactly: OK"}],
                    },
                )
                self.assertEqual(status, 200)
                self.assertEqual(out["choices"][0]["message"]["content"].strip(), "OK")

                # Verify bus evidence contains one request + one response with matching req_id.
                req = None
                resp = None
                detected = None
                for line in events_path.read_text(encoding="utf-8").splitlines():
                    obj = json.loads(line)
                    if obj.get("topic") == "pbvw.request":
                        req = obj
                    if obj.get("topic") == "pbvw.response":
                        resp = obj
                    if obj.get("topic") == "pluribus.directive.detected":
                        detected = obj
                self.assertIsNotNone(req)
                self.assertIsNotNone(resp)
                self.assertEqual(req["data"]["req_id"], resp["data"]["req_id"])
                self.assertTrue(resp["data"]["ok"])
                self.assertIsNotNone(detected)
                self.assertEqual(req["data"]["req_id"], detected["data"]["req_id"])
                self.assertTrue(req["data"].get("pluribus"))
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except Exception:
                    proc.kill()

    def test_error_response_when_router_fails(self):
        tools_dir = Path(__file__).resolve().parents[1]
        bridge = tools_dir / "pbvw_bridge.py"
        port = _free_port()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bus_dir = root / "bus"
            bus_dir.mkdir(parents=True, exist_ok=True)
            (bus_dir / "events.ndjson").write_text("", encoding="utf-8")

            router = root / "router_stub.py"
            router.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env python3",
                        "import argparse,json,sys",
                        "p=argparse.ArgumentParser()",
                        "p.add_argument('--provider',default='auto')",
                        "p.add_argument('--prompt',required=True)",
                        "p.add_argument('--model',default=None)",
                        "p.add_argument('--format',default='text')",
                        "a=p.parse_args()",
                        "payload={'ok':False,'provider':a.provider,'model':a.model,'text':'','stderr':'router fail'}",
                        "sys.stdout.write(json.dumps(payload)+'\\n')",
                        "raise SystemExit(2)",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            os.chmod(router, 0o755)

            env = {**os.environ, "PLURIBUS_BUS_DIR": str(bus_dir), "PYTHONDONTWRITEBYTECODE": "1"}
            proc = subprocess.Popen(
                [
                    "python3",
                    str(bridge),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--bus-dir",
                    str(bus_dir),
                    "--actor",
                    "test-pbvw",
                    "--router-path",
                    str(router),
                    "--router-timeout-s",
                    "10",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
            )
            try:
                _wait_http_ok(f"http://127.0.0.1:{port}/health")

                status, out = _post_json(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    {
                        "model": "claude-web",
                        "messages": [{"role": "user", "content": "FAIL"}],
                    },
                )
                self.assertEqual(status, 500)
                self.assertIn("error", out)
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except Exception:
                    proc.kill()

    def test_multi_protocol_endpoints(self):
        tools_dir = Path(__file__).resolve().parents[1]
        bridge = tools_dir / "pbvw_bridge.py"
        port = _free_port()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bus_dir = root / "bus"
            bus_dir.mkdir(parents=True, exist_ok=True)
            (bus_dir / "events.ndjson").write_text("", encoding="utf-8")

            router = root / "router_stub.py"
            router.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env python3",
                        "import argparse,json,sys",
                        "p=argparse.ArgumentParser()",
                        "p.add_argument('--provider',default='auto')",
                        "p.add_argument('--prompt',required=True)",
                        "p.add_argument('--model',default=None)",
                        "p.add_argument('--format',default='text')",
                        "a=p.parse_args()",
                        "payload={'ok':True,'provider':a.provider,'model':a.model,'text':'OK\\n','stderr':''}",
                        "sys.stdout.write(json.dumps(payload)+'\\n')",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            os.chmod(router, 0o755)

            env = {**os.environ, "PLURIBUS_BUS_DIR": str(bus_dir), "PYTHONDONTWRITEBYTECODE": "1"}
            proc = subprocess.Popen(
                [
                    "python3",
                    str(bridge),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--bus-dir",
                    str(bus_dir),
                    "--actor",
                    "test-gateway",
                    "--router-path",
                    str(router),
                    "--router-timeout-s",
                    "10",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
            )
            try:
                _wait_http_ok(f"http://127.0.0.1:{port}/health")

                status, out = _post_json(
                    f"http://127.0.0.1:{port}/v1/responses",
                    {"model": "gpt-5.2", "input": "Say OK"},
                )
                self.assertEqual(status, 200)
                self.assertEqual(out["output"][0]["content"][0]["text"].strip(), "OK")

                status, out = _post_json(
                    f"http://127.0.0.1:{port}/v1/messages",
                    {"model": "claude-3-5-sonnet", "messages": [{"role": "user", "content": "Say OK"}]},
                )
                self.assertEqual(status, 200)
                self.assertEqual(out["content"][0]["text"].strip(), "OK")

                status, out = _post_json(
                    f"http://127.0.0.1:{port}/v1/messages/count_tokens",
                    {"model": "claude-3-5-sonnet", "messages": [{"role": "user", "content": "Count me"}]},
                )
                self.assertEqual(status, 200)
                self.assertIn("input_tokens", out)

                status, out = _post_json(
                    f"http://127.0.0.1:{port}/v1beta/models/gemini-2.0-flash:generateContent?key=test",
                    {"contents": [{"role": "user", "parts": [{"text": "Say OK"}]}]},
                )
                self.assertEqual(status, 200)
                self.assertEqual(out["candidates"][0]["content"]["parts"][0]["text"].strip(), "OK")

                status, raw, _ = _post_raw(
                    f"http://127.0.0.1:{port}/api/event_logging/batch",
                    {"events": []},
                )
                self.assertEqual(status, 204)

                status, raw, hdrs = _post_raw(
                    f"http://127.0.0.1:{port}/v1/responses",
                    {"model": "gpt-5.2", "input": "Say OK", "stream": True},
                )
                self.assertEqual(status, 200)
                self.assertIn("text/event-stream", hdrs.get("content-type", ""))
                self.assertIn(b"response.output_text.delta", raw)
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except Exception:
                    proc.kill()

    def test_model_alias_applies_and_headers_are_honest(self):
        tools_dir = Path(__file__).resolve().parents[1]
        bridge = tools_dir / "pbvw_bridge.py"
        port = _free_port()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bus_dir = root / "bus"
            bus_dir.mkdir(parents=True, exist_ok=True)
            (bus_dir / "events.ndjson").write_text("", encoding="utf-8")

            router = root / "router_stub.py"
            router.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env python3",
                        "import argparse,json,sys",
                        "p=argparse.ArgumentParser()",
                        "p.add_argument('--provider',default='auto')",
                        "p.add_argument('--prompt',required=True)",
                        "p.add_argument('--model',default=None)",
                        "p.add_argument('--format',default='text')",
                        "a=p.parse_args()",
                        "payload={'ok':True,'provider':a.provider,'model':a.model,'text':f'MODEL={a.model}\\n','stderr':''}",
                        "sys.stdout.write(json.dumps(payload)+'\\n')",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            os.chmod(router, 0o755)

            env = {
                **os.environ,
                "PLURIBUS_BUS_DIR": str(bus_dir),
                "PLURIBUS_GATEWAY_MODEL_ALIAS": json.dumps({"gemini-2.5-pro": "gemini-3-pro-preview"}),
                "PYTHONDONTWRITEBYTECODE": "1",
            }
            proc = subprocess.Popen(
                [
                    "python3",
                    str(bridge),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--bus-dir",
                    str(bus_dir),
                    "--actor",
                    "test-alias",
                    "--router-path",
                    str(router),
                    "--router-timeout-s",
                    "10",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
            )
            try:
                _wait_http_ok(f"http://127.0.0.1:{port}/health")

                status, raw, hdrs = _post_raw(
                    f"http://127.0.0.1:{port}/v1beta/models/gemini-2.5-pro:generateContent?key=test",
                    {"contents": [{"role": "user", "parts": [{"text": "Say OK"}]}]},
                )
                self.assertEqual(status, 200)
                out = json.loads(raw.decode("utf-8"))
                txt = out["candidates"][0]["content"]["parts"][0]["text"].strip()
                self.assertEqual(txt, "MODEL=gemini-3-pro-preview")
                self.assertEqual(hdrs.get("x-requested-model"), "gemini-2.5-pro")
                self.assertEqual(hdrs.get("x-served-by-model"), "gemini-3-pro-preview")
                self.assertEqual(hdrs.get("x-model-alias-hit"), "1")
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except Exception:
                    proc.kill()

    def test_api_key_enforced_when_configured(self):
        tools_dir = Path(__file__).resolve().parents[1]
        bridge = tools_dir / "pbvw_bridge.py"
        port = _free_port()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bus_dir = root / "bus"
            bus_dir.mkdir(parents=True, exist_ok=True)
            (bus_dir / "events.ndjson").write_text("", encoding="utf-8")

            router = root / "router_stub.py"
            router.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env python3",
                        "import argparse,json,sys",
                        "p=argparse.ArgumentParser()",
                        "p.add_argument('--provider',default='auto')",
                        "p.add_argument('--prompt',required=True)",
                        "p.add_argument('--model',default=None)",
                        "p.add_argument('--format',default='text')",
                        "a=p.parse_args()",
                        "payload={'ok':True,'provider':a.provider,'model':a.model,'text':'OK\\n','stderr':''}",
                        "sys.stdout.write(json.dumps(payload)+'\\n')",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            os.chmod(router, 0o755)

            env = {**os.environ, "PLURIBUS_BUS_DIR": str(bus_dir), "PLURIBUS_GATEWAY_API_KEY": "test-key", "PYTHONDONTWRITEBYTECODE": "1"}
            proc = subprocess.Popen(
                [
                    "python3",
                    str(bridge),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--bus-dir",
                    str(bus_dir),
                    "--actor",
                    "test-auth",
                    "--router-path",
                    str(router),
                    "--router-timeout-s",
                    "10",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
            )
            try:
                _wait_http_ok(f"http://127.0.0.1:{port}/health")

                status, _ = _post_json(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    {"model": "gpt-5.2", "messages": [{"role": "user", "content": "Say OK"}]},
                )
                self.assertEqual(status, 401)

                status, out = _post_json(
                    f"http://127.0.0.1:{port}/v1beta/models/gemini-2.0-flash:generateContent?key=test-key",
                    {"contents": [{"role": "user", "parts": [{"text": "Say OK"}]}]},
                )
                self.assertEqual(status, 200)
                self.assertEqual(out["candidates"][0]["content"]["parts"][0]["text"].strip(), "OK")
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except Exception:
                    proc.kill()

    def test_policy_denylist_blocks_request(self):
        tools_dir = Path(__file__).resolve().parents[1]
        bridge = tools_dir / "pbvw_bridge.py"
        port = _free_port()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bus_dir = root / "bus"
            bus_dir.mkdir(parents=True, exist_ok=True)
            (bus_dir / "events.ndjson").write_text("", encoding="utf-8")

            router = root / "router_stub.py"
            router.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env python3",
                        "import argparse,json,sys",
                        "p=argparse.ArgumentParser()",
                        "p.add_argument('--provider',default='auto')",
                        "p.add_argument('--prompt',required=True)",
                        "p.add_argument('--model',default=None)",
                        "p.add_argument('--format',default='text')",
                        "a=p.parse_args()",
                        "payload={'ok':True,'provider':a.provider,'model':a.model,'text':'OK\\n','stderr':''}",
                        "sys.stdout.write(json.dumps(payload)+'\\n')",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            os.chmod(router, 0o755)

            env = {
                **os.environ,
                "PLURIBUS_BUS_DIR": str(bus_dir),
                "PLURIBUS_GATEWAY_POLICY": json.dumps({"policy_id": "test", "denylist": ["BLOCKME"]}),
                "PYTHONDONTWRITEBYTECODE": "1",
            }
            proc = subprocess.Popen(
                [
                    "python3",
                    str(bridge),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--bus-dir",
                    str(bus_dir),
                    "--actor",
                    "test-policy",
                    "--router-path",
                    str(router),
                    "--router-timeout-s",
                    "10",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
            )
            try:
                _wait_http_ok(f"http://127.0.0.1:{port}/health")

                status, out = _post_json(
                    f"http://127.0.0.1:{port}/v1/chat/completions",
                    {"model": "gpt-5.2", "messages": [{"role": "user", "content": "BLOCKME"}]},
                )
                self.assertEqual(status, 403)
                self.assertIn("error", out)
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except Exception:
                    proc.kill()

    def test_policy_require_local_passes_env_to_router(self):
        tools_dir = Path(__file__).resolve().parents[1]
        bridge = tools_dir / "pbvw_bridge.py"
        port = _free_port()

        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            bus_dir = root / "bus"
            bus_dir.mkdir(parents=True, exist_ok=True)
            (bus_dir / "events.ndjson").write_text("", encoding="utf-8")

            router = root / "router_stub.py"
            router.write_text(
                "\n".join(
                    [
                        "#!/usr/bin/env python3",
                        "import argparse,json,os,sys",
                        "p=argparse.ArgumentParser()",
                        "p.add_argument('--provider',default='auto')",
                        "p.add_argument('--prompt',required=True)",
                        "p.add_argument('--model',default=None)",
                        "p.add_argument('--format',default='text')",
                        "a=p.parse_args()",
                        "local=os.environ.get('PLURIBUS_ROUTER_REQUIRE_LOCAL','0')",
                        "payload={'ok':True,'provider':a.provider,'model':a.model,'text':f'LOCAL={local}\\n','stderr':''}",
                        "sys.stdout.write(json.dumps(payload)+'\\n')",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            os.chmod(router, 0o755)

            env = {
                **os.environ,
                "PLURIBUS_BUS_DIR": str(bus_dir),
                "PLURIBUS_GATEWAY_POLICY": json.dumps({"policy_id": "test", "require_local_only": ["LOCALONLY"]}),
                "PYTHONDONTWRITEBYTECODE": "1",
            }
            proc = subprocess.Popen(
                [
                    "python3",
                    str(bridge),
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--bus-dir",
                    str(bus_dir),
                    "--actor",
                    "test-policy",
                    "--router-path",
                    str(router),
                    "--router-timeout-s",
                    "10",
                ],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                env=env,
            )
            try:
                _wait_http_ok(f"http://127.0.0.1:{port}/health")

                status, out = _post_json(
                    f"http://127.0.0.1:{port}/v1/responses",
                    {"model": "gpt-5.2", "input": "LOCALONLY"},
                )
                self.assertEqual(status, 200)
                self.assertEqual(out["output"][0]["content"][0]["text"].strip(), "LOCAL=1")
            finally:
                proc.terminate()
                try:
                    proc.wait(timeout=2)
                except Exception:
                    proc.kill()


if __name__ == "__main__":
    unittest.main()
