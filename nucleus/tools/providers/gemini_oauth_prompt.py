from __future__ import annotations

import re

AUTH_URL_RE = re.compile(r"(https://accounts\.google\.com/o/oauth2/v2/auth\?[^\s]+)")


def extract_google_oauth_url(text: str) -> str | None:
    """
    Extract the OAuth URL printed by @google/gemini-cli when a manual login is required.

    The URL is not a secret, but it is time-bound and contains long query params.
    """
    if not text:
        return None
    m = AUTH_URL_RE.search(text)
    if m:
        return m.group(1)
    return None


def looks_like_oauth_prompt(text: str) -> bool:
    if not text:
        return False
    t = text.lower()
    if "please visit the following url to authorize the application" in t:
        return True
    if "enter the authorization code" in t:
        return True
    if "codeassist.google.com/authcode" in t:
        return True
    return extract_google_oauth_url(text) is not None

