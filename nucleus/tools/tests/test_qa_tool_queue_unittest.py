from nucleus.tools import qa_tool_queue


def test_build_state_counts():
    events = [
        {"id": "job-1", "status": "queued", "ts": 1},
        {"id": "job-2", "status": "queued", "ts": 2},
        {"id": "job-1", "status": "started", "ts": 3},
        {"id": "job-1", "status": "completed", "ts": 4},
    ]
    state = qa_tool_queue.build_state(events)
    assert state["counts"]["queued"] == 1
    assert state["counts"]["started"] == 0
    assert state["counts"]["completed"] == 1
    assert state["active"] == 0


def test_select_next_job_respects_active_limit():
    events = [
        {"id": "job-1", "status": "queued", "ts": 1},
        {"id": "job-2", "status": "queued", "ts": 2},
        {"id": "job-3", "status": "started", "ts": 3},
    ]
    state = qa_tool_queue.build_state(events)
    assert qa_tool_queue.select_next_job(state, max_active=0) is None
    assert qa_tool_queue.select_next_job(state, max_active=1) is None
    assert qa_tool_queue.select_next_job(state, max_active=2)["id"] == "job-1"
