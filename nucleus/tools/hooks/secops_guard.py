#!/usr/bin/env python3
"""SecOps Pre-Commit Guard Hook.

Pre-commit hook that scans staged changes for security issues:
- Secret patterns (API keys, tokens, passwords)
- Sensitive file additions
- Permission anomalies

Install: Link or copy to .git/hooks/pre-commit

Effects: R(git staging area)
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import Path


# Secret detection patterns
SECRET_PATTERNS = [
    # API Keys
    (r'(?i)(api[_-]?key|apikey)\s*[=:]\s*["\']?([a-zA-Z0-9_\-]{20,})["\']?', "API Key"),
    (r'(?i)(secret[_-]?key|secretkey)\s*[=:]\s*["\']?([a-zA-Z0-9_\-]{20,})["\']?', "Secret Key"),

    # AWS
    (r'AKIA[0-9A-Z]{16}', "AWS Access Key ID"),
    (r'(?i)aws[_-]?secret[_-]?access[_-]?key\s*[=:]\s*["\']?([a-zA-Z0-9/+=]{40})["\']?', "AWS Secret Key"),

    # GitHub
    (r'ghp_[a-zA-Z0-9]{36}', "GitHub Personal Access Token"),
    (r'gho_[a-zA-Z0-9]{36}', "GitHub OAuth Token"),
    (r'ghu_[a-zA-Z0-9]{36}', "GitHub User Token"),
    (r'ghs_[a-zA-Z0-9]{36}', "GitHub Server Token"),
    (r'ghr_[a-zA-Z0-9]{36}', "GitHub Refresh Token"),

    # Google/GCP
    (r'AIza[0-9A-Za-z_\-]{35}', "Google API Key"),

    # Generic Tokens
    (r'(?i)(bearer|authorization)\s*[=:]\s*["\']?([a-zA-Z0-9_\-\.]{30,})["\']?', "Bearer Token"),
    (r'(?i)token\s*[=:]\s*["\']?([a-zA-Z0-9_\-]{30,})["\']?', "Generic Token"),

    # Passwords
    (r'(?i)(password|passwd|pwd)\s*[=:]\s*["\']?([^\s"\'\n]{8,})["\']?', "Password"),

    # Private Keys
    (r'-----BEGIN (RSA|DSA|EC|OPENSSH|PGP) PRIVATE KEY-----', "Private Key"),

    # Database URLs
    (r'(?i)(postgres|mysql|mongodb|redis)://[^\s]+:[^\s]+@', "Database URL with credentials"),

    # Slack
    (r'xox[baprs]-[0-9a-zA-Z]{10,48}', "Slack Token"),

    # Stripe
    (r'sk_live_[0-9a-zA-Z]{24,}', "Stripe Secret Key"),
    (r'rk_live_[0-9a-zA-Z]{24,}', "Stripe Restricted Key"),

    # Twilio
    (r'SK[0-9a-fA-F]{32}', "Twilio API Key"),

    # OpenAI
    (r'sk-[a-zA-Z0-9]{48}', "OpenAI API Key"),

    # Anthropic
    (r'sk-ant-[a-zA-Z0-9\-]{93}', "Anthropic API Key"),
]

# Sensitive file patterns
SENSITIVE_FILES = [
    r'\.env$',
    r'\.env\.local$',
    r'\.env\.production$',
    r'credentials\.json$',
    r'service[_-]?account\.json$',
    r'private[_-]?key\.pem$',
    r'id_rsa$',
    r'id_ed25519$',
    r'\.pem$',
    r'\.key$',
    r'secrets\.ya?ml$',
    r'\.htpasswd$',
]


def get_staged_files() -> list[str]:
    """Get list of staged files."""
    try:
        result = subprocess.run(
            ["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR"],
            capture_output=True,
            text=True,
            check=True,
        )
        return [f.strip() for f in result.stdout.splitlines() if f.strip()]
    except subprocess.CalledProcessError:
        return []


def get_staged_content(filepath: str) -> str:
    """Get staged content of a file."""
    try:
        result = subprocess.run(
            ["git", "show", f":{filepath}"],
            capture_output=True,
            text=True,
            check=True,
        )
        return result.stdout
    except subprocess.CalledProcessError:
        return ""


def scan_for_secrets(content: str, filepath: str) -> list[dict]:
    """Scan content for secret patterns."""
    findings = []

    for pattern, secret_type in SECRET_PATTERNS:
        matches = re.finditer(pattern, content)
        for match in matches:
            # Get line number
            line_num = content[:match.start()].count('\n') + 1
            findings.append({
                "type": "secret",
                "secret_type": secret_type,
                "file": filepath,
                "line": line_num,
                "match": match.group(0)[:50] + "..." if len(match.group(0)) > 50 else match.group(0),
            })

    return findings


def check_sensitive_files(filepath: str) -> dict | None:
    """Check if file matches sensitive patterns."""
    for pattern in SENSITIVE_FILES:
        if re.search(pattern, filepath, re.IGNORECASE):
            return {
                "type": "sensitive_file",
                "file": filepath,
                "pattern": pattern,
                "message": f"Sensitive file detected: {filepath}",
            }
    return None


def should_skip_file(filepath: str) -> bool:
    """Check if file should be skipped (binary, large, etc)."""
    # Skip common binary extensions
    binary_extensions = {
        '.png', '.jpg', '.jpeg', '.gif', '.ico', '.pdf',
        '.zip', '.tar', '.gz', '.rar', '.7z',
        '.exe', '.dll', '.so', '.dylib',
        '.woff', '.woff2', '.ttf', '.eot',
        '.mp3', '.mp4', '.wav', '.avi',
        '.pyc', '.pyo', '.class',
    }

    ext = Path(filepath).suffix.lower()
    if ext in binary_extensions:
        return True

    # Skip lock files
    if filepath.endswith('.lock') or filepath.endswith('-lock.json'):
        return True

    return False


def main() -> int:
    """Run the pre-commit security check."""
    findings = []
    staged_files = get_staged_files()

    if not staged_files:
        return 0

    print("SecOps Guard: Scanning staged files for security issues...", file=sys.stderr)

    for filepath in staged_files:
        # Check for sensitive files
        sensitive = check_sensitive_files(filepath)
        if sensitive:
            findings.append(sensitive)

        # Skip binary files for content scanning
        if should_skip_file(filepath):
            continue

        # Scan content for secrets
        content = get_staged_content(filepath)
        if content:
            secrets = scan_for_secrets(content, filepath)
            findings.extend(secrets)

    if not findings:
        print("SecOps Guard: No security issues found.", file=sys.stderr)
        return 0

    # Report findings
    print("\n" + "=" * 60, file=sys.stderr)
    print("SECOPS GUARD: SECURITY ISSUES DETECTED", file=sys.stderr)
    print("=" * 60, file=sys.stderr)

    for finding in findings:
        if finding["type"] == "secret":
            print(f"\n[SECRET] {finding['secret_type']}", file=sys.stderr)
            print(f"  File: {finding['file']}:{finding['line']}", file=sys.stderr)
            print(f"  Match: {finding['match']}", file=sys.stderr)
        elif finding["type"] == "sensitive_file":
            print(f"\n[SENSITIVE FILE] {finding['file']}", file=sys.stderr)
            print(f"  {finding['message']}", file=sys.stderr)

    print("\n" + "=" * 60, file=sys.stderr)
    print(f"Total issues: {len(findings)}", file=sys.stderr)
    print("Commit blocked. Please fix these issues before committing.", file=sys.stderr)
    print("To bypass (not recommended): git commit --no-verify", file=sys.stderr)
    print("=" * 60 + "\n", file=sys.stderr)

    return 1


if __name__ == "__main__":
    raise SystemExit(main())
