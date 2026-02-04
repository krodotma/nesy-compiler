from pathlib import Path

from agent_conductor import (
    append_plan_update,
    collect_open_tasks,
    render_plan_update,
)


def test_collect_open_tasks_uses_latest_status():
    entries = [
        {"req_id": "req-a", "status": "planned", "ts": 1, "meta": {"desc": "seed"}},
        {"req_id": "req-a", "status": "completed", "ts": 2, "meta": {"desc": "done"}},
        {"req_id": "req-b", "status": "blocked", "ts": 3, "meta": {"note": "needs input"}},
    ]

    open_tasks, counts = collect_open_tasks(entries, max_tasks=10)

    assert len(open_tasks) == 1
    assert open_tasks[0]["req_id"] == "req-b"
    assert open_tasks[0]["status"] == "blocked"
    assert open_tasks[0]["desc"] == "needs input"
    assert counts["blocked"] == 1


def test_render_plan_update_includes_tasks():
    update = render_plan_update(
        now_iso="2025-12-22T00:00:00Z",
        note="sync",
        open_tasks=[{"req_id": "req-1", "status": "in_progress", "desc": "work"}],
        status_counts={"in_progress": 1},
        total_open=1,
    )

    assert "2025-12-22T00:00:00Z" in update
    assert "sync" in update
    assert "req-1" in update
    assert "in_progress" in update
    assert "work" in update


def test_collect_open_tasks_filters_statuses():
    entries = [
        {"req_id": "req-a", "status": "planned", "ts": 1},
        {"req_id": "req-b", "status": "blocked", "ts": 2},
    ]

    open_tasks, counts = collect_open_tasks(entries, statuses={"planned"})

    assert len(open_tasks) == 1
    assert open_tasks[0]["req_id"] == "req-a"
    assert counts == {"planned": 1}


def test_collect_open_tasks_truncates_desc():
    entries = [
        {"req_id": "req-a", "status": "planned", "ts": 1, "meta": {"desc": "a" * 40}},
    ]

    open_tasks, _ = collect_open_tasks(entries, max_desc_chars=12)

    assert open_tasks[0]["desc"].endswith("...")
    assert len(open_tasks[0]["desc"]) == 12


def test_append_plan_update_appends(tmp_path: Path):
    plan_path = tmp_path / "plan.md"
    plan_path.write_text("# Header\n\n## Progress Log (append-only)\n", encoding="utf-8")

    append_plan_update(plan_path, "- 2025-12-22T00:00:00Z: ok\n")

    content = plan_path.read_text(encoding="utf-8")
    assert content.startswith("# Header")
    assert "Progress Log" in content
    assert "2025-12-22T00:00:00Z: ok" in content


def test_append_plan_update_adds_progress_section(tmp_path: Path):
    plan_path = tmp_path / "plan.md"
    plan_path.write_text("# Header\n", encoding="utf-8")

    append_plan_update(plan_path, "- 2025-12-22T00:00:00Z: ok\n")

    content = plan_path.read_text(encoding="utf-8")
    assert "## Progress Log (append-only)" in content


def test_collect_open_tasks_sanitizes_desc():
    entries = [
        {"req_id": "req-a", "status": "planned", "ts": 1, "meta": {"desc": "line1\nline2\tline3"}},
    ]

    open_tasks, _ = collect_open_tasks(entries, max_desc_chars=200)

    assert "\n" not in open_tasks[0]["desc"]
    assert "\t" not in open_tasks[0]["desc"]


def test_render_plan_update_no_tasks():
    update = render_plan_update(
        now_iso="2025-12-22T00:00:00Z",
        note="sync",
        open_tasks=[],
        status_counts={},
        total_open=0,
    )

    assert "Open tasks: none" in update
