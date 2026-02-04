#!/usr/bin/env python3
"""
Alert Notifier - Send critical alerts via email with rate limiting.

Prevents flooding by:
- Cooldown period between alerts (default 30 min)
- Deduplication of same alert within window
- Max alerts per hour limit

Protocol: DKIN v28
"""
import json
import os
import smtplib
import subprocess
import sys
import time
from datetime import datetime, timezone
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from pathlib import Path
from typing import Optional

# Configuration
ALERT_EMAIL = os.environ.get("ALERT_EMAIL", "peter@kro.ma")
ALERT_FROM = os.environ.get("ALERT_FROM", "ohm@kroma.live")
SMTP_HOST = os.environ.get("SMTP_HOST", "localhost")
SMTP_PORT = int(os.environ.get("SMTP_PORT", "25"))

# Rate limiting
COOLDOWN_SECONDS = int(os.environ.get("ALERT_COOLDOWN_S", "1800"))  # 30 min default
MAX_ALERTS_PER_HOUR = int(os.environ.get("MAX_ALERTS_PER_HOUR", "4"))
DEDUP_WINDOW_S = int(os.environ.get("ALERT_DEDUP_WINDOW_S", "3600"))  # 1 hour

# State file for rate limiting
STATE_FILE = Path("/pluribus/.pluribus/alerts/state.json")
ALERT_LOG = Path("/pluribus/.pluribus/alerts/history.ndjson")


def load_state() -> dict:
    """Load alert state for rate limiting."""
    try:
        if STATE_FILE.exists():
            return json.loads(STATE_FILE.read_text())
    except Exception:
        pass
    return {
        "last_alert_time": 0,
        "alerts_this_hour": [],
        "recent_signatures": {}
    }


def save_state(state: dict):
    """Save alert state."""
    try:
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(state, indent=2))
    except Exception as e:
        print(f"[ALERT] Warning: Could not save state: {e}", file=sys.stderr)


def log_alert(level: str, title: str, message: str, sent: bool):
    """Log alert to history file."""
    try:
        ALERT_LOG.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "ts": time.time(),
            "iso": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "level": level,
            "title": title,
            "message": message[:500],
            "sent": sent,
            "recipient": ALERT_EMAIL if sent else None
        }
        with open(ALERT_LOG, "a") as f:
            f.write(json.dumps(record) + "\n")
    except Exception:
        pass


def can_send_alert(state: dict, signature: str) -> tuple[bool, str]:
    """Check if we can send an alert (rate limiting)."""
    now = time.time()

    # Clean old entries from alerts_this_hour
    hour_ago = now - 3600
    state["alerts_this_hour"] = [t for t in state["alerts_this_hour"] if t > hour_ago]

    # Clean old signatures
    state["recent_signatures"] = {
        sig: ts for sig, ts in state["recent_signatures"].items()
        if now - ts < DEDUP_WINDOW_S
    }

    # Check cooldown
    if now - state["last_alert_time"] < COOLDOWN_SECONDS:
        remaining = int(COOLDOWN_SECONDS - (now - state["last_alert_time"]))
        return False, f"Cooldown active ({remaining}s remaining)"

    # Check hourly limit
    if len(state["alerts_this_hour"]) >= MAX_ALERTS_PER_HOUR:
        return False, f"Hourly limit reached ({MAX_ALERTS_PER_HOUR}/hour)"

    # Check deduplication
    if signature in state["recent_signatures"]:
        return False, "Duplicate alert suppressed"

    return True, "OK"


def send_email(subject: str, body: str) -> tuple[bool, str]:
    """Try to send email via various methods."""

    # Method 1: Try local SMTP
    try:
        msg = MIMEMultipart()
        msg['Subject'] = subject
        msg['From'] = ALERT_FROM
        msg['To'] = ALERT_EMAIL
        msg.attach(MIMEText(body, 'plain'))

        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=10) as s:
            s.sendmail(ALERT_FROM, [ALERT_EMAIL], msg.as_string())
        return True, "Sent via SMTP"
    except Exception as e:
        smtp_error = str(e)

    # Method 2: Try sendmail command
    try:
        result = subprocess.run(
            ["/usr/sbin/sendmail", "-t"],
            input=f"To: {ALERT_EMAIL}\nFrom: {ALERT_FROM}\nSubject: {subject}\n\n{body}",
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            return True, "Sent via sendmail"
    except Exception:
        pass

    # Method 3: Try mail command
    try:
        result = subprocess.run(
            ["mail", "-s", subject, ALERT_EMAIL],
            input=body,
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            return True, "Sent via mail"
    except Exception:
        pass

    return False, f"All methods failed (SMTP: {smtp_error})"


def send_alert(level: str, title: str, message: str, force: bool = False) -> dict:
    """
    Send an alert with rate limiting.

    Args:
        level: critical, error, warn, info
        title: Short alert title
        message: Detailed message
        force: Bypass rate limiting (use sparingly!)

    Returns:
        dict with status, sent, reason
    """
    signature = f"{level}:{title}"
    state = load_state()

    # Check rate limits unless forced
    if not force:
        can_send, reason = can_send_alert(state, signature)
        if not can_send:
            log_alert(level, title, message, sent=False)
            return {"status": "suppressed", "sent": False, "reason": reason}

    # Build email
    subject = f"[PLURIBUS {level.upper()}] {title}"
    body = f"""Pluribus Alert Notification
{'=' * 40}

Level: {level.upper()}
Time: {datetime.now(timezone.utc).isoformat().replace('+00:00', 'Z')}
Host: kroma.live

{title}
{'-' * 40}

{message}

{'=' * 40}
This is an automated alert from Pluribus OHM.
To adjust alert frequency, modify ALERT_COOLDOWN_S (current: {COOLDOWN_SECONDS}s)
"""

    # Try to send
    sent, method = send_email(subject, body)

    # Update state
    now = time.time()
    if sent:
        state["last_alert_time"] = now
        state["alerts_this_hour"].append(now)
        state["recent_signatures"][signature] = now

    save_state(state)
    log_alert(level, title, message, sent=sent)

    return {
        "status": "sent" if sent else "failed",
        "sent": sent,
        "method": method if sent else None,
        "reason": method
    }


def main():
    """CLI interface."""
    import argparse
    parser = argparse.ArgumentParser(description="Send alert notification")
    parser.add_argument("--level", "-l", default="error", choices=["critical", "error", "warn", "info"])
    parser.add_argument("--title", "-t", required=True, help="Alert title")
    parser.add_argument("--message", "-m", required=True, help="Alert message")
    parser.add_argument("--force", "-f", action="store_true", help="Bypass rate limiting")
    parser.add_argument("--test", action="store_true", help="Send test alert")

    args = parser.parse_args()

    if args.test:
        result = send_alert(
            "info",
            "Alert System Test",
            "This is a test alert to verify the notification system is working.\n\nIf you received this, alerts are configured correctly.",
            force=True
        )
    else:
        result = send_alert(args.level, args.title, args.message, args.force)

    print(json.dumps(result, indent=2))
    sys.exit(0 if result["sent"] else 1)


if __name__ == "__main__":
    main()
