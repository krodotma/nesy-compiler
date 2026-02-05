#!/usr/bin/env python3
"""
Security Scanner (Step 153)

Scans code for security vulnerabilities using pattern matching and AST analysis.
Based on OWASP Top 10 and CWE vulnerability patterns.

PBTSO Phase: VERIFY
Bus Topics: review.security.scan, review.security.vulnerabilities

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
from typing import Any, Dict, List, Optional, Pattern, Tuple


# ============================================================================
# Types
# ============================================================================

class SeverityLevel(Enum):
    """Severity levels aligned with CVSS."""
    CRITICAL = "critical"  # CVSS 9.0-10.0
    HIGH = "high"          # CVSS 7.0-8.9
    MEDIUM = "medium"      # CVSS 4.0-6.9
    LOW = "low"            # CVSS 0.1-3.9
    INFO = "info"          # Informational

    @property
    def priority(self) -> int:
        """Get numeric priority (higher = more severe)."""
        return {
            self.CRITICAL: 5,
            self.HIGH: 4,
            self.MEDIUM: 3,
            self.LOW: 2,
            self.INFO: 1,
        }.get(self, 0)


class VulnerabilityCategory(Enum):
    """OWASP Top 10 2021 categories plus additional common vulnerabilities."""
    # OWASP Top 10 2021
    A01_BROKEN_ACCESS_CONTROL = "A01:2021 Broken Access Control"
    A02_CRYPTOGRAPHIC_FAILURES = "A02:2021 Cryptographic Failures"
    A03_INJECTION = "A03:2021 Injection"
    A04_INSECURE_DESIGN = "A04:2021 Insecure Design"
    A05_SECURITY_MISCONFIGURATION = "A05:2021 Security Misconfiguration"
    A06_VULNERABLE_COMPONENTS = "A06:2021 Vulnerable and Outdated Components"
    A07_AUTH_FAILURES = "A07:2021 Identification and Authentication Failures"
    A08_DATA_INTEGRITY_FAILURES = "A08:2021 Software and Data Integrity Failures"
    A09_LOGGING_FAILURES = "A09:2021 Security Logging and Monitoring Failures"
    A10_SSRF = "A10:2021 Server-Side Request Forgery"

    # Additional common vulnerabilities
    HARDCODED_SECRETS = "Hardcoded Secrets"
    COMMAND_INJECTION = "Command Injection"
    PATH_TRAVERSAL = "Path Traversal"
    INSECURE_DESERIALIZATION = "Insecure Deserialization"
    XXE = "XML External Entity"
    SENSITIVE_DATA_EXPOSURE = "Sensitive Data Exposure"


@dataclass
class SecurityVulnerability:
    """
    Represents a security vulnerability finding.

    Attributes:
        id: Unique identifier for this vulnerability
        severity: Severity level (critical, high, medium, low, info)
        category: OWASP category
        file: File path where vulnerability was found
        line: Line number (1-indexed)
        column: Column number (1-indexed)
        description: Human-readable description
        cwe: Common Weakness Enumeration ID (e.g., "CWE-89")
        owasp: OWASP category reference
        remediation: Suggested fix
        confidence: Confidence level (high, medium, low)
        snippet: Code snippet containing the vulnerability
    """
    id: str
    severity: SeverityLevel
    category: VulnerabilityCategory
    file: str
    line: int
    column: int
    description: str
    cwe: Optional[str] = None
    owasp: Optional[str] = None
    remediation: Optional[str] = None
    confidence: str = "medium"
    snippet: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        result = asdict(self)
        result["severity"] = self.severity.value
        result["category"] = self.category.value
        return result

    @property
    def location(self) -> str:
        """Get human-readable location string."""
        return f"{self.file}:{self.line}:{self.column}"


@dataclass
class SecurityScanResult:
    """Result from a security scan."""
    files_scanned: int = 0
    vulnerabilities: List[SecurityVulnerability] = field(default_factory=list)
    duration_ms: float = 0
    critical_count: int = 0
    high_count: int = 0
    medium_count: int = 0
    low_count: int = 0
    patterns_checked: int = 0
    scan_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "scan_id": self.scan_id,
            "files_scanned": self.files_scanned,
            "vulnerabilities": [v.to_dict() for v in self.vulnerabilities],
            "duration_ms": self.duration_ms,
            "critical_count": self.critical_count,
            "high_count": self.high_count,
            "medium_count": self.medium_count,
            "low_count": self.low_count,
            "patterns_checked": self.patterns_checked,
        }

    @property
    def has_blocking_issues(self) -> bool:
        """Check if there are critical/high issues that should block merge."""
        return self.critical_count > 0 or self.high_count > 0


# ============================================================================
# Vulnerability Patterns
# ============================================================================

@dataclass
class VulnerabilityPattern:
    """Definition of a vulnerability detection pattern."""
    name: str
    pattern: str
    severity: SeverityLevel
    category: VulnerabilityCategory
    cwe: str
    description: str
    remediation: str
    confidence: str = "medium"
    languages: List[str] = field(default_factory=lambda: ["python", "javascript", "typescript"])

    def compile(self) -> Pattern:
        """Compile the regex pattern."""
        return re.compile(self.pattern, re.IGNORECASE | re.MULTILINE)


# OWASP-based vulnerability patterns
VULNERABILITY_PATTERNS: List[VulnerabilityPattern] = [
    # ========================================================================
    # A02: Cryptographic Failures
    # ========================================================================
    VulnerabilityPattern(
        name="hardcoded_password",
        pattern=r"(?:password|passwd|pwd|secret|api_key|apikey|api_secret|auth_token|access_token)\s*[=:]\s*['\"][^'\"]{4,}['\"]",
        severity=SeverityLevel.CRITICAL,
        category=VulnerabilityCategory.HARDCODED_SECRETS,
        cwe="CWE-798",
        description="Hardcoded credentials detected. Secrets should be stored in environment variables or secure vaults.",
        remediation="Move secrets to environment variables or a secrets management system.",
        confidence="high",
    ),
    VulnerabilityPattern(
        name="hardcoded_aws_key",
        pattern=r"(?:AKIA|A3T|AGPA|AIDA|AROA|AIPA|ANPA|ANVA|ASIA)[A-Z0-9]{16}",
        severity=SeverityLevel.CRITICAL,
        category=VulnerabilityCategory.HARDCODED_SECRETS,
        cwe="CWE-798",
        description="AWS access key ID detected in code.",
        remediation="Remove the key and rotate credentials immediately. Use IAM roles or environment variables.",
        confidence="high",
    ),
    VulnerabilityPattern(
        name="hardcoded_private_key",
        pattern=r"-----BEGIN (?:RSA |DSA |EC |OPENSSH )?PRIVATE KEY-----",
        severity=SeverityLevel.CRITICAL,
        category=VulnerabilityCategory.HARDCODED_SECRETS,
        cwe="CWE-321",
        description="Private key embedded in code.",
        remediation="Remove private key from code. Store in secure key management system.",
        confidence="high",
    ),
    VulnerabilityPattern(
        name="weak_crypto_md5",
        pattern=r"(?:hashlib\.md5|MD5\.Create|crypto\.createHash\(['\"]md5['\"]\)|MessageDigest\.getInstance\(['\"]MD5['\"]\))",
        severity=SeverityLevel.MEDIUM,
        category=VulnerabilityCategory.A02_CRYPTOGRAPHIC_FAILURES,
        cwe="CWE-327",
        description="MD5 hash algorithm is cryptographically broken.",
        remediation="Use SHA-256 or stronger hash algorithms.",
        confidence="high",
    ),
    VulnerabilityPattern(
        name="weak_crypto_sha1",
        pattern=r"(?:hashlib\.sha1|SHA1\.Create|crypto\.createHash\(['\"]sha1['\"]\)|MessageDigest\.getInstance\(['\"]SHA-?1['\"]\))",
        severity=SeverityLevel.LOW,
        category=VulnerabilityCategory.A02_CRYPTOGRAPHIC_FAILURES,
        cwe="CWE-327",
        description="SHA-1 hash algorithm is deprecated for security purposes.",
        remediation="Use SHA-256 or stronger hash algorithms.",
        confidence="medium",
    ),

    # ========================================================================
    # A03: Injection
    # ========================================================================
    VulnerabilityPattern(
        name="sql_injection_format",
        pattern=r"(?:execute|cursor\.execute|query|raw)\s*\(\s*(?:f['\"]|['\"].*%|['\"].*\.format)",
        severity=SeverityLevel.HIGH,
        category=VulnerabilityCategory.A03_INJECTION,
        cwe="CWE-89",
        description="Potential SQL injection via string formatting.",
        remediation="Use parameterized queries or prepared statements.",
        confidence="medium",
    ),
    VulnerabilityPattern(
        name="sql_injection_concat",
        pattern=r"(?:SELECT|INSERT|UPDATE|DELETE|DROP|UNION|WHERE).*\+\s*(?:request\.|params\.|query\.|user_input|input)",
        severity=SeverityLevel.HIGH,
        category=VulnerabilityCategory.A03_INJECTION,
        cwe="CWE-89",
        description="Potential SQL injection via string concatenation.",
        remediation="Use parameterized queries or prepared statements.",
        confidence="medium",
    ),
    VulnerabilityPattern(
        name="command_injection_shell",
        pattern=r"(?:subprocess\.(?:call|run|Popen)|os\.system|os\.popen|exec|eval)\s*\([^)]*(?:shell\s*=\s*True|f['\"]|\.format|%)",
        severity=SeverityLevel.CRITICAL,
        category=VulnerabilityCategory.COMMAND_INJECTION,
        cwe="CWE-78",
        description="Potential command injection vulnerability.",
        remediation="Avoid shell=True. Use subprocess with list arguments and validate all inputs.",
        confidence="high",
    ),
    VulnerabilityPattern(
        name="eval_usage",
        pattern=r"(?<!#.*)(?:^|\s)eval\s*\(",
        severity=SeverityLevel.HIGH,
        category=VulnerabilityCategory.A03_INJECTION,
        cwe="CWE-94",
        description="Use of eval() can lead to code injection.",
        remediation="Avoid eval(). Use ast.literal_eval() for safe literal evaluation or proper parsers.",
        confidence="high",
    ),
    VulnerabilityPattern(
        name="exec_usage",
        pattern=r"(?<!#.*)(?:^|\s)exec\s*\(",
        severity=SeverityLevel.HIGH,
        category=VulnerabilityCategory.A03_INJECTION,
        cwe="CWE-94",
        description="Use of exec() can lead to code injection.",
        remediation="Avoid exec(). Use safer alternatives based on the specific use case.",
        confidence="high",
    ),
    VulnerabilityPattern(
        name="xss_innerhtml",
        pattern=r"\.innerHTML\s*=\s*(?!['\"]\s*['\"])",
        severity=SeverityLevel.MEDIUM,
        category=VulnerabilityCategory.A03_INJECTION,
        cwe="CWE-79",
        description="Potential XSS vulnerability via innerHTML.",
        remediation="Use textContent or sanitize HTML input. Consider using DOMPurify.",
        confidence="medium",
        languages=["javascript", "typescript"],
    ),
    VulnerabilityPattern(
        name="xss_document_write",
        pattern=r"document\.write\s*\(",
        severity=SeverityLevel.MEDIUM,
        category=VulnerabilityCategory.A03_INJECTION,
        cwe="CWE-79",
        description="document.write can introduce XSS vulnerabilities.",
        remediation="Use DOM manipulation methods instead of document.write.",
        confidence="medium",
        languages=["javascript", "typescript"],
    ),

    # ========================================================================
    # A05: Security Misconfiguration
    # ========================================================================
    VulnerabilityPattern(
        name="debug_enabled",
        pattern=r"(?:DEBUG\s*=\s*True|app\.debug\s*=\s*True|debug:\s*true)",
        severity=SeverityLevel.MEDIUM,
        category=VulnerabilityCategory.A05_SECURITY_MISCONFIGURATION,
        cwe="CWE-489",
        description="Debug mode enabled in configuration.",
        remediation="Disable debug mode in production environments.",
        confidence="high",
    ),
    VulnerabilityPattern(
        name="cors_wildcard",
        pattern=r"(?:Access-Control-Allow-Origin|cors.*origin)['\"]?\s*[:=]\s*['\"]?\*",
        severity=SeverityLevel.MEDIUM,
        category=VulnerabilityCategory.A05_SECURITY_MISCONFIGURATION,
        cwe="CWE-942",
        description="CORS allows all origins, potentially exposing APIs to malicious sites.",
        remediation="Specify explicit allowed origins instead of wildcard.",
        confidence="high",
    ),
    VulnerabilityPattern(
        name="insecure_ssl_verify",
        pattern=r"(?:verify\s*=\s*False|ssl_verify\s*=\s*False|CERT_NONE|rejectUnauthorized:\s*false)",
        severity=SeverityLevel.HIGH,
        category=VulnerabilityCategory.A05_SECURITY_MISCONFIGURATION,
        cwe="CWE-295",
        description="SSL certificate verification disabled.",
        remediation="Enable SSL certificate verification for all connections.",
        confidence="high",
    ),

    # ========================================================================
    # A07: Identification and Authentication Failures
    # ========================================================================
    VulnerabilityPattern(
        name="jwt_none_algorithm",
        pattern=r"(?:algorithm\s*[=:]\s*['\"]none['\"]|alg['\"]?\s*:\s*['\"]none['\"])",
        severity=SeverityLevel.CRITICAL,
        category=VulnerabilityCategory.A07_AUTH_FAILURES,
        cwe="CWE-327",
        description="JWT with 'none' algorithm allows token forgery.",
        remediation="Use strong algorithms like RS256 or HS256 and verify algorithm in validation.",
        confidence="high",
    ),
    VulnerabilityPattern(
        name="weak_jwt_secret",
        pattern=r"(?:jwt\.encode|sign)\s*\([^)]*['\"](?:secret|password|key)['\"]",
        severity=SeverityLevel.HIGH,
        category=VulnerabilityCategory.A07_AUTH_FAILURES,
        cwe="CWE-798",
        description="Weak or predictable JWT secret.",
        remediation="Use strong, randomly generated secrets for JWT signing.",
        confidence="medium",
    ),

    # ========================================================================
    # A08: Software and Data Integrity Failures
    # ========================================================================
    VulnerabilityPattern(
        name="insecure_deserialization_pickle",
        pattern=r"pickle\.(?:loads?|Unpickler)",
        severity=SeverityLevel.HIGH,
        category=VulnerabilityCategory.INSECURE_DESERIALIZATION,
        cwe="CWE-502",
        description="Pickle deserialization can execute arbitrary code.",
        remediation="Use JSON or other safe serialization formats. Never unpickle untrusted data.",
        confidence="high",
        languages=["python"],
    ),
    VulnerabilityPattern(
        name="insecure_deserialization_yaml",
        pattern=r"yaml\.(?:load|unsafe_load)\s*\([^)]*(?!Loader)",
        severity=SeverityLevel.HIGH,
        category=VulnerabilityCategory.INSECURE_DESERIALIZATION,
        cwe="CWE-502",
        description="Unsafe YAML loading can execute arbitrary code.",
        remediation="Use yaml.safe_load() instead of yaml.load().",
        confidence="high",
        languages=["python"],
    ),

    # ========================================================================
    # A10: Server-Side Request Forgery (SSRF)
    # ========================================================================
    VulnerabilityPattern(
        name="ssrf_potential",
        pattern=r"(?:requests\.get|urllib\.request\.urlopen|fetch|http\.get)\s*\([^)]*(?:request\.|params\.|query\.|user_input|input|url_param)",
        severity=SeverityLevel.MEDIUM,
        category=VulnerabilityCategory.A10_SSRF,
        cwe="CWE-918",
        description="Potential SSRF vulnerability - URL from user input.",
        remediation="Validate and sanitize URLs. Use allowlists for permitted domains.",
        confidence="medium",
    ),

    # ========================================================================
    # Path Traversal
    # ========================================================================
    VulnerabilityPattern(
        name="path_traversal",
        pattern=r"(?:open|Path|os\.path\.join|readFile|writeFile)\s*\([^)]*(?:request\.|params\.|query\.|user_input|input)",
        severity=SeverityLevel.HIGH,
        category=VulnerabilityCategory.PATH_TRAVERSAL,
        cwe="CWE-22",
        description="Potential path traversal vulnerability.",
        remediation="Validate and sanitize file paths. Use os.path.realpath() and verify against base path.",
        confidence="medium",
    ),

    # ========================================================================
    # Sensitive Data Exposure
    # ========================================================================
    VulnerabilityPattern(
        name="sensitive_log",
        pattern=r"(?:log(?:ger)?\.(?:info|debug|warn|error)|print|console\.log)\s*\([^)]*(?:password|secret|token|api_key|credit_card|ssn)",
        severity=SeverityLevel.MEDIUM,
        category=VulnerabilityCategory.SENSITIVE_DATA_EXPOSURE,
        cwe="CWE-532",
        description="Sensitive data may be logged.",
        remediation="Remove sensitive data from log statements. Use redaction for necessary logging.",
        confidence="medium",
    ),
]


# ============================================================================
# Security Scanner
# ============================================================================

class SecurityScanner:
    """
    Scans code for security vulnerabilities.

    Uses regex patterns based on OWASP Top 10 and CWE to detect
    common security issues in source code.

    Example:
        scanner = SecurityScanner()
        result = scanner.scan(["/path/to/file.py"])
        for vuln in result.vulnerabilities:
            print(f"[{vuln.severity.value}] {vuln.location}: {vuln.description}")
    """

    def __init__(
        self,
        patterns: Optional[List[VulnerabilityPattern]] = None,
        bus_path: Optional[Path] = None,
    ):
        """
        Initialize the security scanner.

        Args:
            patterns: Custom vulnerability patterns (defaults to OWASP patterns)
            bus_path: Path to event bus file
        """
        self.patterns = patterns or VULNERABILITY_PATTERNS
        self.bus_path = bus_path or self._get_bus_path()
        self._compiled_patterns: List[Tuple[VulnerabilityPattern, Pattern]] = []
        self._compile_patterns()

    def _get_bus_path(self) -> Path:
        """Get path to bus events file."""
        pluribus_root = Path(os.environ.get("PLURIBUS_ROOT", "/pluribus"))
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", str(pluribus_root / ".pluribus" / "bus"))
        return Path(bus_dir) / "events.ndjson"

    def _compile_patterns(self) -> None:
        """Compile all regex patterns."""
        self._compiled_patterns = [
            (pattern, pattern.compile())
            for pattern in self.patterns
        ]

    def _emit_event(self, topic: str, data: Dict[str, Any]) -> None:
        """Emit event to bus."""
        self.bus_path.parent.mkdir(parents=True, exist_ok=True)

        event = {
            "id": str(uuid.uuid4()),
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat() + "Z",
            "topic": topic,
            "kind": "security",
            "actor": "security-scanner",
            "data": data,
        }

        with open(self.bus_path, "a") as f:
            f.write(json.dumps(event) + "\n")

    def _detect_language(self, file_path: str) -> str:
        """Detect file language from extension."""
        ext_map = {
            ".py": "python",
            ".pyi": "python",
            ".ts": "typescript",
            ".tsx": "typescript",
            ".js": "javascript",
            ".jsx": "javascript",
            ".go": "go",
            ".rs": "rust",
            ".java": "java",
            ".rb": "ruby",
        }
        ext = Path(file_path).suffix.lower()
        return ext_map.get(ext, "unknown")

    def scan(
        self,
        files: List[str],
        content_map: Optional[Dict[str, str]] = None,
    ) -> SecurityScanResult:
        """
        Scan files for security vulnerabilities.

        Args:
            files: List of file paths to scan
            content_map: Optional pre-loaded file contents

        Returns:
            SecurityScanResult with all vulnerabilities found

        Emits:
            review.security.scan (start)
            review.security.vulnerabilities (per vulnerability batch)
        """
        start_time = time.time()
        scan_id = str(uuid.uuid4())[:8]

        # Emit start event
        self._emit_event("review.security.scan", {
            "scan_id": scan_id,
            "files": files[:20],  # First 20 only
            "file_count": len(files),
            "pattern_count": len(self.patterns),
            "status": "started",
        })

        result = SecurityScanResult(
            scan_id=scan_id,
            files_scanned=len(files),
            patterns_checked=len(self.patterns),
        )

        for file_path in files:
            # Get file content
            if content_map and file_path in content_map:
                content = content_map[file_path]
            else:
                try:
                    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                        content = f.read()
                except (IOError, OSError):
                    continue

            # Detect language
            language = self._detect_language(file_path)

            # Check each pattern
            vulns = self._scan_content(file_path, content, language)
            result.vulnerabilities.extend(vulns)

        # Calculate severity counts
        for vuln in result.vulnerabilities:
            if vuln.severity == SeverityLevel.CRITICAL:
                result.critical_count += 1
            elif vuln.severity == SeverityLevel.HIGH:
                result.high_count += 1
            elif vuln.severity == SeverityLevel.MEDIUM:
                result.medium_count += 1
            elif vuln.severity == SeverityLevel.LOW:
                result.low_count += 1

        result.duration_ms = (time.time() - start_time) * 1000

        # Emit vulnerabilities found
        if result.vulnerabilities:
            self._emit_event("review.security.vulnerabilities", {
                "scan_id": scan_id,
                "vulnerability_count": len(result.vulnerabilities),
                "critical_count": result.critical_count,
                "high_count": result.high_count,
                "blocking": result.has_blocking_issues,
                "vulnerabilities": [v.to_dict() for v in result.vulnerabilities[:10]],
            })

        # Emit completion
        self._emit_event("review.security.scan", {
            "scan_id": scan_id,
            "status": "completed",
            "files_scanned": result.files_scanned,
            "vulnerability_count": len(result.vulnerabilities),
            "critical_count": result.critical_count,
            "high_count": result.high_count,
            "duration_ms": result.duration_ms,
        })

        return result

    def _scan_content(
        self,
        file_path: str,
        content: str,
        language: str,
    ) -> List[SecurityVulnerability]:
        """Scan content for vulnerabilities."""
        vulnerabilities = []
        lines = content.split("\n")

        for pattern_def, compiled_pattern in self._compiled_patterns:
            # Skip patterns not applicable to this language
            if pattern_def.languages and language not in pattern_def.languages:
                continue

            for match in compiled_pattern.finditer(content):
                # Calculate line and column
                line_num = content[:match.start()].count("\n") + 1
                line_start = content.rfind("\n", 0, match.start()) + 1
                column = match.start() - line_start + 1

                # Get code snippet
                snippet = lines[line_num - 1] if line_num <= len(lines) else ""

                vuln_id = f"{pattern_def.name}-{file_path}-{line_num}"

                vulnerabilities.append(SecurityVulnerability(
                    id=vuln_id,
                    severity=pattern_def.severity,
                    category=pattern_def.category,
                    file=file_path,
                    line=line_num,
                    column=column,
                    description=pattern_def.description,
                    cwe=pattern_def.cwe,
                    owasp=pattern_def.category.value if "A0" in pattern_def.category.value else None,
                    remediation=pattern_def.remediation,
                    confidence=pattern_def.confidence,
                    snippet=snippet.strip()[:200],
                ))

        return vulnerabilities

    def scan_content(
        self,
        content: str,
        file_path: str = "<stdin>",
    ) -> List[SecurityVulnerability]:
        """
        Scan a single piece of content.

        Args:
            content: Code content to scan
            file_path: Virtual file path for reporting

        Returns:
            List of vulnerabilities found
        """
        language = self._detect_language(file_path)
        return self._scan_content(file_path, content, language)


# ============================================================================
# CLI
# ============================================================================

def main() -> int:
    """CLI entry point for Security Scanner."""
    import argparse
    import sys

    parser = argparse.ArgumentParser(description="Security Scanner (Step 153)")
    parser.add_argument("files", nargs="*", help="Files to scan")
    parser.add_argument("--stdin", action="store_true", help="Read from stdin")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    parser.add_argument("--summary", action="store_true", help="Show summary only")
    parser.add_argument("--severity", choices=["critical", "high", "medium", "low", "info"],
                        default="low", help="Minimum severity to report")

    args = parser.parse_args()

    scanner = SecurityScanner()

    if args.stdin:
        content = sys.stdin.read()
        vulns = scanner.scan_content(content)
        result = SecurityScanResult(
            files_scanned=1,
            vulnerabilities=vulns,
            critical_count=sum(1 for v in vulns if v.severity == SeverityLevel.CRITICAL),
            high_count=sum(1 for v in vulns if v.severity == SeverityLevel.HIGH),
            medium_count=sum(1 for v in vulns if v.severity == SeverityLevel.MEDIUM),
            low_count=sum(1 for v in vulns if v.severity == SeverityLevel.LOW),
        )
    elif args.files:
        result = scanner.scan(args.files)
    else:
        parser.print_help()
        return 1

    # Filter by severity
    min_severity = SeverityLevel[args.severity.upper()]
    result.vulnerabilities = [
        v for v in result.vulnerabilities
        if v.severity.priority >= min_severity.priority
    ]

    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    elif args.summary:
        print(f"Security Scan Summary:")
        print(f"  Files scanned: {result.files_scanned}")
        print(f"  Vulnerabilities: {len(result.vulnerabilities)}")
        print(f"  Critical: {result.critical_count}")
        print(f"  High: {result.high_count}")
        print(f"  Medium: {result.medium_count}")
        print(f"  Low: {result.low_count}")
        print(f"  Duration: {result.duration_ms:.1f}ms")
        if result.has_blocking_issues:
            print(f"  ** BLOCKING ISSUES FOUND **")
    else:
        for vuln in result.vulnerabilities:
            severity_color = {
                SeverityLevel.CRITICAL: "\033[91m",  # Red
                SeverityLevel.HIGH: "\033[93m",      # Yellow
                SeverityLevel.MEDIUM: "\033[94m",    # Blue
                SeverityLevel.LOW: "\033[90m",       # Gray
                SeverityLevel.INFO: "\033[37m",      # White
            }.get(vuln.severity, "")
            reset = "\033[0m"

            print(f"{severity_color}[{vuln.severity.value.upper()}]{reset} {vuln.location}")
            print(f"  {vuln.description}")
            if vuln.cwe:
                print(f"  {vuln.cwe} | {vuln.category.value}")
            if vuln.snippet:
                print(f"  > {vuln.snippet[:100]}")
            print()

    # Return non-zero if blocking issues found
    return 1 if result.has_blocking_issues else 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
