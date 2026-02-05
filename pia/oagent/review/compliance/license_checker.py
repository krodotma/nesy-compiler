#!/usr/bin/env python3
"""
License Compliance Checker (Step 159)

Checks project dependencies for license compatibility.

PBTSO Phase: VERIFY
Bus Topics: review.license.check, review.license.violations

License categories:
- Permissive: MIT, BSD, Apache 2.0, ISC
- Copyleft: GPL, LGPL, AGPL
- Weak Copyleft: MPL, EPL
- Public Domain: CC0, Unlicense
- Proprietary: Commercial licenses
- Unknown: License not detected

Protocol: DKIN v30, CITIZEN v2, PAIP v16
"""

from __future__ import annotations

import json
import os
import re
import time
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Set


# ============================================================================
# Types
# ============================================================================

class LicenseCategory(Enum):
    """Categories of open source licenses."""
    PERMISSIVE = "permissive"         # MIT, BSD, Apache
    COPYLEFT = "copyleft"             # GPL family
    WEAK_COPYLEFT = "weak_copyleft"   # MPL, LGPL
    PUBLIC_DOMAIN = "public_domain"   # CC0, Unlicense
    PROPRIETARY = "proprietary"       # Commercial
    UNKNOWN = "unknown"


class ComplianceLevel(Enum):
    """Compliance levels for license combinations."""
    COMPLIANT = "compliant"           # No issues
    WARNING = "warning"               # Potential issues
    VIOLATION = "violation"           # License conflict
    UNKNOWN = "unknown"               # Cannot determine


@dataclass
class LicenseInfo:
    """Information about a detected license."""
    name: str
    spdx_id: str
    category: LicenseCategory
    url: Optional[str] = None
    is_osi_approved: bool = False
    allows_commercial: bool = True
    requires_attribution: bool = False
    requires_source_disclosure: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["category"] = self.category.value
        return result


@dataclass
class LicenseViolation:
    """
    A license compliance violation.

    Attributes:
        dependency_name: Name of the dependency
        dependency_version: Version of the dependency
        dependency_license: License of the dependency
        project_license: License of the project
        violation_type: Type of violation
        description: Human-readable description
        suggestion: How to resolve the violation
    """
    dependency_name: str
    dependency_version: str
    dependency_license: LicenseInfo
    project_license: Optional[LicenseInfo]
    violation_type: str
    description: str
    suggestion: str

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dependency_name": self.dependency_name,
            "dependency_version": self.dependency_version,
            "dependency_license": self.dependency_license.to_dict(),
            "project_license": self.project_license.to_dict() if self.project_license else None,
            "violation_type": self.violation_type,
            "description": self.description,
            "suggestion": self.suggestion,
        }


@dataclass
class LicenseCheckResult:
    """Result from license checking."""
    dependencies_checked: int = 0
    violations: List[LicenseViolation] = field(default_factory=list)
    warnings: List[LicenseViolation] = field(default_factory=list)
    duration_ms: float = 0
    licenses_found: Dict[str, int] = field(default_factory=dict)
    project_license: Optional[LicenseInfo] = None
    compliance_level: ComplianceLevel = ComplianceLevel.UNKNOWN

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dependencies_checked": self.dependencies_checked,
            "violations": [v.to_dict() for v in self.violations],
            "warnings": [w.to_dict() for w in self.warnings],
            "duration_ms": self.duration_ms,
            "licenses_found": self.licenses_found,
            "project_license": self.project_license.to_dict() if self.project_license else None,
            "compliance_level": self.compliance_level.value,
        }


# ============================================================================
# License Database
# ============================================================================

# Common license definitions
LICENSE_DB: Dict[str, LicenseInfo] = {
    "MIT": LicenseInfo(
        name="MIT License",
        spdx_id="MIT",
        category=LicenseCategory.PERMISSIVE,
        url="https://opensource.org/licenses/MIT",
        is_osi_approved=True,
        allows_commercial=True,
        requires_attribution=True,
        requires_source_disclosure=False,
    ),
    "BSD-3-Clause": LicenseInfo(
        name="BSD 3-Clause License",
        spdx_id="BSD-3-Clause",
        category=LicenseCategory.PERMISSIVE,
        url="https://opensource.org/licenses/BSD-3-Clause",
        is_osi_approved=True,
        allows_commercial=True,
        requires_attribution=True,
        requires_source_disclosure=False,
    ),
    "BSD-2-Clause": LicenseInfo(
        name="BSD 2-Clause License",
        spdx_id="BSD-2-Clause",
        category=LicenseCategory.PERMISSIVE,
        url="https://opensource.org/licenses/BSD-2-Clause",
        is_osi_approved=True,
        allows_commercial=True,
        requires_attribution=True,
        requires_source_disclosure=False,
    ),
    "Apache-2.0": LicenseInfo(
        name="Apache License 2.0",
        spdx_id="Apache-2.0",
        category=LicenseCategory.PERMISSIVE,
        url="https://opensource.org/licenses/Apache-2.0",
        is_osi_approved=True,
        allows_commercial=True,
        requires_attribution=True,
        requires_source_disclosure=False,
    ),
    "ISC": LicenseInfo(
        name="ISC License",
        spdx_id="ISC",
        category=LicenseCategory.PERMISSIVE,
        url="https://opensource.org/licenses/ISC",
        is_osi_approved=True,
        allows_commercial=True,
        requires_attribution=True,
        requires_source_disclosure=False,
    ),
    "GPL-2.0": LicenseInfo(
        name="GNU General Public License v2.0",
        spdx_id="GPL-2.0",
        category=LicenseCategory.COPYLEFT,
        url="https://www.gnu.org/licenses/old-licenses/gpl-2.0.html",
        is_osi_approved=True,
        allows_commercial=True,
        requires_attribution=True,
        requires_source_disclosure=True,
    ),
    "GPL-3.0": LicenseInfo(
        name="GNU General Public License v3.0",
        spdx_id="GPL-3.0",
        category=LicenseCategory.COPYLEFT,
        url="https://www.gnu.org/licenses/gpl-3.0.html",
        is_osi_approved=True,
        allows_commercial=True,
        requires_attribution=True,
        requires_source_disclosure=True,
    ),
    "AGPL-3.0": LicenseInfo(
        name="GNU Affero General Public License v3.0",
        spdx_id="AGPL-3.0",
        category=LicenseCategory.COPYLEFT,
        url="https://www.gnu.org/licenses/agpl-3.0.html",
        is_osi_approved=True,
        allows_commercial=True,
        requires_attribution=True,
        requires_source_disclosure=True,
    ),
    "LGPL-2.1": LicenseInfo(
        name="GNU Lesser General Public License v2.1",
        spdx_id="LGPL-2.1",
        category=LicenseCategory.WEAK_COPYLEFT,
        url="https://www.gnu.org/licenses/old-licenses/lgpl-2.1.html",
        is_osi_approved=True,
        allows_commercial=True,
        requires_attribution=True,
        requires_source_disclosure=False,
    ),
    "LGPL-3.0": LicenseInfo(
        name="GNU Lesser General Public License v3.0",
        spdx_id="LGPL-3.0",
        category=LicenseCategory.WEAK_COPYLEFT,
        url="https://www.gnu.org/licenses/lgpl-3.0.html",
        is_osi_approved=True,
        allows_commercial=True,
        requires_attribution=True,
        requires_source_disclosure=False,
    ),
    "MPL-2.0": LicenseInfo(
        name="Mozilla Public License 2.0",
        spdx_id="MPL-2.0",
        category=LicenseCategory.WEAK_COPYLEFT,
        url="https://opensource.org/licenses/MPL-2.0",
        is_osi_approved=True,
        allows_commercial=True,
        requires_attribution=True,
        requires_source_disclosure=False,
    ),
    "CC0-1.0": LicenseInfo(
        name="Creative Commons Zero v1.0 Universal",
        spdx_id="CC0-1.0",
        category=LicenseCategory.PUBLIC_DOMAIN,
        url="https://creativecommons.org/publicdomain/zero/1.0/",
        is_osi_approved=False,
        allows_commercial=True,
        requires_attribution=False,
        requires_source_disclosure=False,
    ),
    "Unlicense": LicenseInfo(
        name="The Unlicense",
        spdx_id="Unlicense",
        category=LicenseCategory.PUBLIC_DOMAIN,
        url="https://unlicense.org/",
        is_osi_approved=True,
        allows_commercial=True,
        requires_attribution=False,
        requires_source_disclosure=False,
    ),
}

# License compatibility matrix
# True = compatible, False = incompatible, None = check context
COMPATIBILITY_MATRIX: Dict[str, Dict[str, Optional[bool]]] = {
    "MIT": {
        "MIT": True, "BSD-3-Clause": True, "Apache-2.0": True, "ISC": True,
        "GPL-2.0": True, "GPL-3.0": True, "AGPL-3.0": True,
        "LGPL-2.1": True, "LGPL-3.0": True, "MPL-2.0": True,
    },
    "Apache-2.0": {
        "MIT": True, "BSD-3-Clause": True, "Apache-2.0": True, "ISC": True,
        "GPL-2.0": False, "GPL-3.0": True, "AGPL-3.0": True,
        "LGPL-2.1": True, "LGPL-3.0": True, "MPL-2.0": True,
    },
    "GPL-3.0": {
        "MIT": True, "BSD-3-Clause": True, "Apache-2.0": True, "ISC": True,
        "GPL-2.0": None, "GPL-3.0": True, "AGPL-3.0": None,
        "LGPL-2.1": True, "LGPL-3.0": True, "MPL-2.0": True,
    },
    # Proprietary project constraints
    "PROPRIETARY": {
        "MIT": True, "BSD-3-Clause": True, "Apache-2.0": True, "ISC": True,
        "GPL-2.0": False, "GPL-3.0": False, "AGPL-3.0": False,
        "LGPL-2.1": True, "LGPL-3.0": True, "MPL-2.0": True,
    },
}

# License detection patterns
LICENSE_PATTERNS: Dict[str, List[str]] = {
    "MIT": [r"MIT\s+License", r"Expat\s+License", r'"MIT"', r"'MIT'"],
    "BSD-3-Clause": [r"BSD[\s-]*3[\s-]*Clause", r'"BSD-3-Clause"'],
    "BSD-2-Clause": [r"BSD[\s-]*2[\s-]*Clause", r'"BSD-2-Clause"'],
    "Apache-2.0": [r"Apache[\s-]*(?:License[\s,]*)?(?:Version\s*)?2\.0?", r'"Apache-2.0"'],
    "ISC": [r"ISC\s+License", r'"ISC"'],
    "GPL-2.0": [r"GNU\s+General\s+Public\s+License.*?(?:version\s*)?2", r'"GPL-2.0"'],
    "GPL-3.0": [r"GNU\s+General\s+Public\s+License.*?(?:version\s*)?3", r'"GPL-3.0"'],
    "AGPL-3.0": [r"GNU\s+Affero\s+General\s+Public", r'"AGPL-3.0"'],
    "LGPL-2.1": [r"GNU\s+Lesser\s+General\s+Public.*?2\.1", r'"LGPL-2.1"'],
    "LGPL-3.0": [r"GNU\s+Lesser\s+General\s+Public.*?3", r'"LGPL-3.0"'],
    "MPL-2.0": [r"Mozilla\s+Public\s+License.*?2\.0", r'"MPL-2.0"'],
    "CC0-1.0": [r"CC0.*1\.0", r"Creative\s+Commons\s+Zero"],
    "Unlicense": [r"The\s+Unlicense", r'"Unlicense"'],
}


# ============================================================================
# License Checker
# ============================================================================

class LicenseChecker:
    """
    Checks project dependencies for license compliance.

    Validates that dependency licenses are compatible with the
    project license and detects potential licensing conflicts.

    Example:
        checker = LicenseChecker(project_license="MIT")
        result = checker.check("/path/to/project")
        for violation in result.violations:
            print(f"{violation.dependency_name}: {violation.description}")
    """

    def __init__(
        self,
        project_license: Optional[str] = None,
        allowed_licenses: Optional[List[str]] = None,
        forbidden_licenses: Optional[List[str]] = None,
        bus_path: Optional[Path] = None,
    ):
        """
        Initialize the license checker.

        Args:
            project_license: SPDX ID of project license (e.g., "MIT")
            allowed_licenses: List of allowed license SPDX IDs
            forbidden_licenses: List of forbidden license SPDX IDs
            bus_path: Path to event bus file
        """
        self.project_license = LICENSE_DB.get(project_license) if project_license else None
        self.allowed_licenses = set(allowed_licenses) if allowed_licenses else None
        self.forbidden_licenses = set(forbidden_licenses) if forbidden_licenses else set()
        self.bus_path = bus_path or self._get_bus_path()

    def _get_bus_path(self) -> Path:
        """Get path to bus events file."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _emit_event(self, topic: str, data: Dict[str, Any]) -> None:
        """Emit event to bus."""
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

        event = {
            "id": str(uuid.uuid4()),
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": "license",
            "actor": "license-checker",
            "data": data,
        }

        with open(self.bus_path, "a") as f:
            f.write(json.dumps(event) + "\n")

    def detect_license(self, content: str) -> Optional[LicenseInfo]:
        """Detect license from file content."""
        content_upper = content.upper()

        for spdx_id, patterns in LICENSE_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, content, re.IGNORECASE):
                    return LICENSE_DB.get(spdx_id)

        return None

    def check_compatibility(
        self,
        dependency_license: LicenseInfo,
        project_license: Optional[LicenseInfo],
    ) -> Optional[bool]:
        """
        Check if dependency license is compatible with project license.

        Returns:
            True if compatible, False if incompatible, None if uncertain
        """
        if not project_license:
            return True  # No project license specified

        proj_id = project_license.spdx_id
        dep_id = dependency_license.spdx_id

        # Check compatibility matrix
        if proj_id in COMPATIBILITY_MATRIX:
            return COMPATIBILITY_MATRIX[proj_id].get(dep_id, None)

        # Fallback: permissive licenses are generally compatible
        if dependency_license.category == LicenseCategory.PERMISSIVE:
            return True

        return None

    def check(
        self,
        dependencies: List[Dict[str, str]],
        license_map: Optional[Dict[str, str]] = None,
    ) -> LicenseCheckResult:
        """
        Check dependencies for license compliance.

        Args:
            dependencies: List of dicts with 'name' and 'version' keys
            license_map: Optional mapping of package names to SPDX IDs

        Returns:
            LicenseCheckResult with violations found

        Emits:
            review.license.check (start)
            review.license.violations (per violation batch)
        """
        start_time = time.time()
        license_map = license_map or {}

        # Emit start event
        self._emit_event("review.license.check", {
            "dependency_count": len(dependencies),
            "project_license": self.project_license.spdx_id if self.project_license else None,
            "status": "started",
        })

        result = LicenseCheckResult(
            dependencies_checked=len(dependencies),
            project_license=self.project_license,
        )

        licenses_found: Dict[str, int] = {}

        for dep in dependencies:
            dep_name = dep.get("name", "")
            dep_version = dep.get("version", "")

            # Get license from map or try to detect
            license_id = license_map.get(dep_name)
            if license_id:
                dep_license = LICENSE_DB.get(license_id)
            else:
                dep_license = None

            if not dep_license:
                # Mark as unknown
                dep_license = LicenseInfo(
                    name="Unknown",
                    spdx_id="UNKNOWN",
                    category=LicenseCategory.UNKNOWN,
                )

            # Track license counts
            license_name = dep_license.spdx_id
            licenses_found[license_name] = licenses_found.get(license_name, 0) + 1

            # Check against forbidden list
            if dep_license.spdx_id in self.forbidden_licenses:
                result.violations.append(LicenseViolation(
                    dependency_name=dep_name,
                    dependency_version=dep_version,
                    dependency_license=dep_license,
                    project_license=self.project_license,
                    violation_type="forbidden_license",
                    description=f"License '{dep_license.name}' is on the forbidden list",
                    suggestion=f"Replace {dep_name} with an alternative using an allowed license",
                ))
                continue

            # Check against allowed list if specified
            if self.allowed_licenses and dep_license.spdx_id not in self.allowed_licenses:
                result.violations.append(LicenseViolation(
                    dependency_name=dep_name,
                    dependency_version=dep_version,
                    dependency_license=dep_license,
                    project_license=self.project_license,
                    violation_type="unlisted_license",
                    description=f"License '{dep_license.name}' is not on the allowed list",
                    suggestion=f"Review {dep_name} and add its license to allowed list if acceptable",
                ))
                continue

            # Check compatibility with project license
            compatible = self.check_compatibility(dep_license, self.project_license)

            if compatible is False:
                result.violations.append(LicenseViolation(
                    dependency_name=dep_name,
                    dependency_version=dep_version,
                    dependency_license=dep_license,
                    project_license=self.project_license,
                    violation_type="incompatible_license",
                    description=(
                        f"License '{dep_license.name}' is incompatible with "
                        f"project license '{self.project_license.name if self.project_license else 'unknown'}'"
                    ),
                    suggestion=f"Replace {dep_name} with an alternative using a compatible license",
                ))
            elif compatible is None:
                result.warnings.append(LicenseViolation(
                    dependency_name=dep_name,
                    dependency_version=dep_version,
                    dependency_license=dep_license,
                    project_license=self.project_license,
                    violation_type="uncertain_compatibility",
                    description=f"License compatibility for '{dep_license.name}' is uncertain",
                    suggestion=f"Review license terms for {dep_name} to ensure compatibility",
                ))

            # Check for copyleft in commercial projects
            if dep_license.requires_source_disclosure and self.project_license:
                if self.project_license.spdx_id == "PROPRIETARY":
                    result.violations.append(LicenseViolation(
                        dependency_name=dep_name,
                        dependency_version=dep_version,
                        dependency_license=dep_license,
                        project_license=self.project_license,
                        violation_type="copyleft_in_proprietary",
                        description=f"Copyleft license '{dep_license.name}' requires source disclosure",
                        suggestion=f"Replace {dep_name} with an alternative using a permissive license",
                    ))

        result.licenses_found = licenses_found

        # Determine overall compliance level
        if result.violations:
            result.compliance_level = ComplianceLevel.VIOLATION
        elif result.warnings:
            result.compliance_level = ComplianceLevel.WARNING
        else:
            result.compliance_level = ComplianceLevel.COMPLIANT

        result.duration_ms = (time.time() - start_time) * 1000

        # Emit violations found
        if result.violations:
            self._emit_event("review.license.violations", {
                "violation_count": len(result.violations),
                "warning_count": len(result.warnings),
                "compliance_level": result.compliance_level.value,
                "violations": [v.to_dict() for v in result.violations[:10]],
            })

        # Emit completion
        self._emit_event("review.license.check", {
            "status": "completed",
            "dependencies_checked": result.dependencies_checked,
            "violation_count": len(result.violations),
            "compliance_level": result.compliance_level.value,
            "duration_ms": result.duration_ms,
        })

        return result


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    """CLI entry point for License Checker."""
    import argparse

    parser = argparse.ArgumentParser(description="License Compliance Checker (Step 159)")
    parser.add_argument("--project-license", help="Project license SPDX ID (e.g., MIT)")
    parser.add_argument("--allowed", nargs="+", help="Allowed license SPDX IDs")
    parser.add_argument("--forbidden", nargs="+", help="Forbidden license SPDX IDs")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--summary", action="store_true", help="Show summary only")

    args = parser.parse_args()

    checker = LicenseChecker(
        project_license=args.project_license,
        allowed_licenses=args.allowed,
        forbidden_licenses=args.forbidden,
    )

    # Example dependencies (in practice, would be extracted from project)
    sample_deps = [
        {"name": "requests", "version": "2.28.0"},
        {"name": "flask", "version": "2.0.0"},
        {"name": "django", "version": "4.0.0"},
    ]

    sample_licenses = {
        "requests": "Apache-2.0",
        "flask": "BSD-3-Clause",
        "django": "BSD-3-Clause",
    }

    result = checker.check(sample_deps, sample_licenses)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    elif args.summary:
        print(f"License Check Summary:")
        print(f"  Dependencies checked: {result.dependencies_checked}")
        print(f"  Violations: {len(result.violations)}")
        print(f"  Warnings: {len(result.warnings)}")
        print(f"  Compliance: {result.compliance_level.value}")
        print(f"  Duration: {result.duration_ms:.1f}ms")
        print(f"\nLicenses found:")
        for license_id, count in sorted(result.licenses_found.items(), key=lambda x: -x[1]):
            print(f"    {license_id}: {count}")
    else:
        if result.violations:
            print("Violations:")
            for v in result.violations:
                print(f"  [ERROR] {v.dependency_name}@{v.dependency_version}")
                print(f"    {v.description}")
                print(f"    Suggestion: {v.suggestion}")
                print()

        if result.warnings:
            print("Warnings:")
            for w in result.warnings:
                print(f"  [WARN] {w.dependency_name}@{w.dependency_version}")
                print(f"    {w.description}")
                print()

        if not result.violations and not result.warnings:
            print("All licenses are compliant!")

    return 1 if result.violations else 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
