from nucleus.tools.dashboard_guard import (
    extract_ts_errors,
    error_counts,
    decide_revert,
)


def test_extract_ts_errors_parses_paths_and_codes():
    output = """
src/components/Widget.tsx(12,3): error TS2322: Type 'string' is not assignable to type 'number'.
../nucleus/dashboard/src/lib/foo.ts(1,1): error TS7006: Parameter 'x' implicitly has an 'any' type.
random noise line
"""
    errors = extract_ts_errors(output)
    counts = error_counts(errors)
    assert counts[('src/components/Widget.tsx', 'TS2322')] == 1
    # Normalize path fallback should preserve relative segments
    assert counts[('../nucleus/dashboard/src/lib/foo.ts', 'TS7006')] == 1


def test_decide_revert_on_regression_from_clean():
    should_revert, reason, new_only = decide_revert(
        True,
        {},
        {('src/a.ts', 'TS1234'): 1},
    )
    assert should_revert is True
    assert reason == 'regression_from_clean'
    assert new_only == {('src/a.ts', 'TS1234'): 1}


def test_decide_revert_allows_unchanged_baseline():
    baseline = {('src/a.ts', 'TS1234'): 2}
    should_revert, reason, new_only = decide_revert(
        False,
        baseline,
        {('src/a.ts', 'TS1234'): 2},
    )
    assert should_revert is False
    assert reason == 'baseline_unchanged'
    assert new_only == {}


def test_decide_revert_when_baseline_worsens():
    baseline = {('src/a.ts', 'TS1234'): 1}
    should_revert, reason, new_only = decide_revert(
        False,
        baseline,
        {('src/a.ts', 'TS1234'): 1, ('src/b.ts', 'TS9999'): 1},
    )
    assert should_revert is True
    assert reason == 'baseline_worsened'
    assert new_only == {('src/b.ts', 'TS9999'): 1}


def test_decide_revert_when_baseline_improves():
    baseline = {('src/a.ts', 'TS1234'): 2}
    should_revert, reason, new_only = decide_revert(
        False,
        baseline,
        {('src/a.ts', 'TS1234'): 1},
    )
    assert should_revert is False
    assert reason == 'baseline_improved'
    assert new_only == {}
