import pathlib
import sys
import unittest


TOOLS_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

import browser_session_daemon  # noqa: E402


class TestChatGPTNoAuthGateDetection(unittest.TestCase):
    def test_detects_modal_by_data_testid(self) -> None:
        html = '<div data-testid="modal-no-auth-rate-limit">Thanks for trying ChatGPT</div>'
        self.assertTrue(browser_session_daemon._chatgpt_noauth_gate(html))

    def test_detects_modal_by_copy(self) -> None:
        html = "<p>Thanks for trying ChatGPT</p><a>Stay logged out</a>"
        self.assertTrue(browser_session_daemon._chatgpt_noauth_gate(html))

    def test_false_for_normal_page(self) -> None:
        html = "<html><head><title>ChatGPT</title></head><body><div id='app'></div></body></html>"
        self.assertFalse(browser_session_daemon._chatgpt_noauth_gate(html))


class TestBrowserSessionDaemonGuardrails(unittest.TestCase):
    def test_url_host_parses_hostname(self) -> None:
        self.assertEqual(browser_session_daemon._url_host("https://claude.ai/new"), "claude.ai")
        self.assertEqual(browser_session_daemon._url_host("not a url"), "")
        self.assertEqual(browser_session_daemon._url_host(""), "")

    def test_allowed_hosts_prefers_config(self) -> None:
        config = {"allowed_hosts": ["EXAMPLE.com", "  Foo.Bar "]}
        self.assertEqual(
            browser_session_daemon._allowed_hosts_for_config(config),
            {"example.com", "foo.bar"},
        )

    def test_allowed_hosts_falls_back_to_url(self) -> None:
        config = {"url": "https://chat.openai.com/"}
        self.assertEqual(browser_session_daemon._allowed_hosts_for_config(config), {"chat.openai.com"})

    def test_claude_service_disruption_detection(self) -> None:
        html = "<div>Claude will return soon</div>"
        self.assertTrue(browser_session_daemon._looks_like_claude_service_disruption(html))
        html = "<div>Temporary service disruption</div>"
        self.assertTrue(browser_session_daemon._looks_like_claude_service_disruption(html))
        html = "<div>Welcome to Claude</div>"
        self.assertFalse(browser_session_daemon._looks_like_claude_service_disruption(html))


if __name__ == "__main__":
    unittest.main()
