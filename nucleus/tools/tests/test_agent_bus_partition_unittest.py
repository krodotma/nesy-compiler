import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory


def _set_env(key, value, env):
    if value is None:
        env.pop(key, None)
    else:
        env[key] = value


def test_topic_bucket_mapping() -> None:
    from nucleus.tools import agent_bus

    cases = {
        "omega.heartbeat": "omega",
        "qa.verdict.pass": "qa",
        "telemetry.client.error": "telemetry",
        "browser.chat.request": "browser",
        "dashboard.vps.provider_status": "dashboard",
        "agent.codex.task": "agent",
        "operator.pbtest.request": "operator",
        "rd.tasks.dispatch": "task",
        "task.dispatch": "task",
        "a2a.negotiate.request": "a2a",
        "lens.collimator.decision": "lens",
        "dialogos.submit": "dialogos",
        "infer_sync.request": "infer_sync",
        "providers.incident": "providers",
        "unknown.topic": "other",
    }
    for topic, bucket in cases.items():
        assert agent_bus._topic_bucket(topic) == bucket


def test_topic_bucket_override_from_config() -> None:
    from nucleus.tools import agent_bus

    orig_env = dict(os.environ)
    env = dict(os.environ)
    try:
        with TemporaryDirectory() as td:
            config_path = Path(td) / "bus_partition_policy.json"
            config_path.write_text(
                json.dumps({"bucket_overrides": {"operator.pbtest": "pbtest"}}),
                encoding="utf-8",
            )
            env["PLURIBUS_BUS_PARTITION_CONFIG"] = str(config_path)
            os.environ.clear()
            os.environ.update(env)
            agent_bus._PARTITION_CONFIG_CACHE = None
            assert agent_bus._topic_bucket("operator.pbtest.request") == "pbtest"
    finally:
        os.environ.clear()
        os.environ.update(orig_env)


def test_emit_event_writes_partition() -> None:
    from nucleus.tools import agent_bus

    orig_env = dict(os.environ)
    env = dict(os.environ)
    try:
        _set_env("PLURIBUS_BUS_PARTITION", "1", env)
        _set_env("PLURIBUS_BUS_ROTATE", "0", env)
        _set_env("PLURIBUS_BUS_PARTITION_FANOUT", "topic,type", env)
        _set_env("PLURIBUS_BUS_PARTITION_SHARDS", "1", env)
        _set_env("PLURIBUS_BUS_PARTITION_LEGACY", "0", env)
        os.environ.clear()
        os.environ.update(env)

        with TemporaryDirectory() as td:
            paths = agent_bus.resolve_bus_paths(td)
            agent_bus.emit_event(
                paths,
                topic="omega.heartbeat",
                kind="metric",
                level="info",
                actor="tester",
                data={"ok": True},
                trace_id=None,
                run_id=None,
                durable=False,
            )

            events_path = Path(td) / "events.ndjson"
            part_path = Path(td) / "topics" / "topic" / "omega" / "heartbeat" / "events.ndjson"
            type_path = Path(td) / "topics" / "eventtypes" / "metric" / "info" / "omega" / "heartbeat" / "events.ndjson"
            assert events_path.exists()
            assert part_path.exists()
            assert type_path.exists()

            line = part_path.read_text(encoding="utf-8").strip().splitlines()[-1]
            payload = json.loads(line)
            assert payload.get("topic") == "omega.heartbeat"
    finally:
        os.environ.clear()
        os.environ.update(orig_env)


def test_emit_event_writes_frequency_partition() -> None:
    from nucleus.tools import agent_bus

    orig_env = dict(os.environ)
    env = dict(os.environ)
    try:
        _set_env("PLURIBUS_BUS_PARTITION", "1", env)
        _set_env("PLURIBUS_BUS_ROTATE", "0", env)
        _set_env("PLURIBUS_BUS_PARTITION_FANOUT", "frequency", env)
        _set_env("PLURIBUS_BUS_PARTITION_SHARDS", "1", env)
        _set_env("PLURIBUS_BUS_PARTITION_LEGACY", "0", env)
        os.environ.clear()
        os.environ.update(env)

        with TemporaryDirectory() as td:
            paths = agent_bus.resolve_bus_paths(td)
            agent_bus.emit_event(
                paths,
                topic="omega.heartbeat",
                kind="metric",
                level="info",
                actor="tester",
                data={"ok": True},
                trace_id=None,
                run_id=None,
                durable=False,
            )

            freq_path = Path(td) / "topics" / "frequency" / "hot" / "omega" / "heartbeat" / "events.ndjson"
            assert freq_path.exists()
    finally:
        os.environ.clear()
        os.environ.update(orig_env)


def test_rotate_log_tail_retains_bytes() -> None:
    from nucleus.tools import agent_bus

    with TemporaryDirectory() as td:
        path = Path(td) / "events.ndjson"
        path.write_bytes(b"x" * 4096)
        archive_dir = Path(td) / "archive"

        result = agent_bus.rotate_log_tail(
            str(path),
            retain_bytes=1024,
            archive_dir=str(archive_dir),
            durable=False,
        )

        assert result is not None
        assert path.stat().st_size <= 1024
        assert archive_dir.exists()
        assert any(p.suffix == ".gz" for p in archive_dir.iterdir())
