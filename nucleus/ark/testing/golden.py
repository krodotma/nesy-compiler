#!/usr/bin/env python3
"""
golden.py - Golden Test Management for ARK

P2-078: Add golden test management
P2-079: Create `ark golden` command
P2-077: Implement test oracle synthesis

Golden tests: snapshot tests with auto-updating.
"""

import json
import hashlib
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from pathlib import Path

logger = logging.getLogger("ARK.Testing.Golden")


@dataclass
class GoldenTest:
    """A golden (snapshot) test."""
    id: str
    name: str
    inputs: Dict[str, Any]
    expected: Any
    actual: Optional[Any] = None
    passed: bool = True
    diff: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return {"id": self.id, "name": self.name, "inputs": self.inputs,
                "expected": self.expected, "passed": self.passed}


class GoldenTestManager:
    """
    Manages golden (snapshot) tests.
    
    P2-078, P2-079: Golden test management
    """
    
    def __init__(self, golden_dir: Optional[str] = None):
        self.golden_dir = Path(golden_dir) if golden_dir else Path("~/.ark/golden").expanduser()
        self.golden_dir.mkdir(parents=True, exist_ok=True)
        self.tests: Dict[str, GoldenTest] = {}
        self._load()
    
    def _load(self) -> None:
        """Load golden tests from storage."""
        index_path = self.golden_dir / "index.json"
        if index_path.exists():
            try:
                data = json.loads(index_path.read_text())
                for t in data.get("tests", []):
                    self.tests[t["id"]] = GoldenTest(**t)
            except Exception as e:
                logger.warning("Failed to load golden tests: %s", e)
    
    def _save(self) -> None:
        """Save golden tests."""
        try:
            data = {"tests": [t.to_dict() for t in self.tests.values()]}
            (self.golden_dir / "index.json").write_text(json.dumps(data, indent=2))
        except Exception as e:
            logger.warning("Failed to save: %s", e)
    
    def record(self, name: str, inputs: Dict, expected: Any) -> GoldenTest:
        """Record a new golden test."""
        test_id = hashlib.md5(f"{name}{inputs}".encode()).hexdigest()[:8]
        test = GoldenTest(id=test_id, name=name, inputs=inputs, expected=expected)
        self.tests[test_id] = test
        self._save()
        return test
    
    def check(self, name: str, inputs: Dict, actual: Any) -> GoldenTest:
        """Check against golden expected value."""
        test_id = hashlib.md5(f"{name}{inputs}".encode()).hexdigest()[:8]
        
        if test_id in self.tests:
            test = self.tests[test_id]
            test.actual = actual
            test.passed = self._compare(test.expected, actual)
            if not test.passed:
                test.diff = f"Expected: {test.expected}, Got: {actual}"
            return test
        else:
            # Auto-record new golden
            return self.record(name, inputs, actual)
    
    def update(self, test_id: str, new_expected: Any) -> bool:
        """Update golden expected value."""
        if test_id in self.tests:
            self.tests[test_id].expected = new_expected
            self.tests[test_id].passed = True
            self._save()
            return True
        return False
    
    def update_all_failing(self) -> int:
        """Update all failing tests to current actual values."""
        count = 0
        for test in self.tests.values():
            if not test.passed and test.actual is not None:
                test.expected = test.actual
                test.passed = True
                count += 1
        self._save()
        return count
    
    def _compare(self, expected: Any, actual: Any) -> bool:
        """Compare expected and actual values."""
        if isinstance(expected, float) and isinstance(actual, float):
            return abs(expected - actual) < 0.001
        return expected == actual
    
    def list_tests(self) -> List[Dict]:
        """List all golden tests."""
        return [t.to_dict() for t in self.tests.values()]
    
    def get_statistics(self) -> Dict:
        """Get test statistics."""
        total = len(self.tests)
        passed = sum(1 for t in self.tests.values() if t.passed)
        return {"total": total, "passed": passed, "failed": total - passed}
