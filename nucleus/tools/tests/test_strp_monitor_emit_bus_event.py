import inspect


def test_emit_bus_event_signature(temp_rhizome_dir):
    from strp_monitor import STRpControlCenter

    app = STRpControlCenter(temp_rhizome_dir / ".pluribus")
    sig = inspect.signature(app.emit_bus_event)
    assert list(sig.parameters)[:3] == ["topic", "kind", "data"]


def test_parse_command_normalizes_kind(temp_rhizome_dir):
    from strp_monitor import STRpControlCenter

    app = STRpControlCenter(temp_rhizome_dir / ".pluribus")

    assert app.parse_command("howdy") == ("howdy", "request", {})
    assert app.parse_command("topic:command:{}") == ("topic", "request", {})
    assert app.parse_command("topic:control:{}") == ("topic", "request", {})
    assert app.parse_command("topic:event:{}") == ("topic", "log", {})


def test_emit_bus_event_normalizes_kind_and_sets_bus_dir(monkeypatch, temp_rhizome_dir):
    from strp_monitor import STRpControlCenter

    app = STRpControlCenter(temp_rhizome_dir / ".pluribus")

    calls = []

    class _Result:
        def __init__(self, returncode=0, stderr=""):
            self.returncode = returncode
            self.stderr = stderr

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return _Result(0)

    monkeypatch.setattr("strp_monitor.subprocess.run", fake_run)

    assert app.emit_bus_event("t", "command", {"x": 1}) is True

    (cmd, kwargs) = calls[-1]
    assert "--kind" in cmd
    assert cmd[cmd.index("--kind") + 1] == "request"
    assert kwargs["env"]["PLURIBUS_BUS_DIR"].endswith("/.pluribus/bus")

