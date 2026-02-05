#!/usr/bin/env python3
"""
License Compliance Checker Package (Step 159)

Provides checking for license compatibility in project dependencies.
"""

from .license_checker import (
    LicenseChecker,
    LicenseInfo,
    LicenseViolation,
    LicenseCheckResult,
    LicenseCategory,
)

__all__ = [
    "LicenseChecker",
    "LicenseInfo",
    "LicenseViolation",
    "LicenseCheckResult",
    "LicenseCategory",
]
