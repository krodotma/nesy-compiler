from __future__ import annotations

from nucleus.tools.providers.gemini_oauth_prompt import extract_google_oauth_url, looks_like_oauth_prompt


def test_extract_google_oauth_url_finds_accounts_url() -> None:
    txt = (
        "Please visit the following URL to authorize the application:\n\n"
        "https://accounts.google.com/o/oauth2/v2/auth?redirect_uri=https%3A%2F%2Fcodeassist.google.com%2Fauthcode&access_type=offline&scope=a%20b&code_challenge_method=S256&code_challenge=abc&state=deadbeef&response_type=code&client_id=123\n\n"
        "Enter the authorization code: "
    )
    url = extract_google_oauth_url(txt)
    assert url is not None
    assert url.startswith("https://accounts.google.com/o/oauth2/v2/auth?")


def test_looks_like_oauth_prompt_detects_prompt_without_url() -> None:
    txt = "Enter the authorization code:"
    assert looks_like_oauth_prompt(txt) is True

