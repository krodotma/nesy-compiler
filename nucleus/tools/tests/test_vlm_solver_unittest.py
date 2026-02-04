import importlib.util
import sys
from pathlib import Path


def _load_vlm_solver():
    tools_dir = Path(__file__).resolve().parents[1]
    target = tools_dir / "vlm_solver.py"
    spec = importlib.util.spec_from_file_location("vlm_solver", target)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_extract_json_object_exact() -> None:
    m = _load_vlm_solver()
    obj = m._extract_json_object('{"a":1,"b":"x"}')
    assert obj == {"a": 1, "b": "x"}


def test_extract_json_object_embedded() -> None:
    m = _load_vlm_solver()
    obj = m._extract_json_object('prefix {"a": 2} suffix')
    assert obj == {"a": 2}


def test_decide_escalates_on_captcha_html(tmp_path) -> None:
    m = _load_vlm_solver()
    html_path = tmp_path / "captcha.html"
    html_path.write_text("<html>recaptcha</html>", encoding="utf-8")

    decision = m.decide(
        provider="gemini-web",
        screenshot_path=str(tmp_path / "missing.png"),
        html_path=str(html_path),
        reason="login",
        backend="heuristic",
        model="llava",
        timeout_s=1.0,
        max_image_bytes=1024,
    )

    assert decision.action == "escalate"
    assert decision.reason == "captcha_detected"
    assert decision.confidence == 1.0


def test_decide_none_backend_noop(tmp_path) -> None:
    m = _load_vlm_solver()
    html_path = tmp_path / "plain.html"
    html_path.write_text("<html>hello</html>", encoding="utf-8")

    decision = m.decide(
        provider="gemini-web",
        screenshot_path=str(tmp_path / "missing.png"),
        html_path=str(html_path),
        reason="challenge",
        backend="none",
        model="llava",
        timeout_s=1.0,
        max_image_bytes=1024,
    )

    assert decision.action in {"noop", "escalate"}
