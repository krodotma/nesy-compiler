"""
Theia Browser Module — Browser automation via agent-browser (L0).

Uses vercel-labs/agent-browser CLI for lightweight headless automation.
Falls back to browser_session_daemon for VNC-backed persistent sessions.

Architecture:
    - Rust CLI + Node.js daemon (agent-browser)
    - Bearer auth via --headers
    - A11y tree snapshots for AI consumption
"""

import subprocess
import os
import json
from dataclasses import dataclass
from typing import Optional, Tuple, List


@dataclass
class BrowserConfig:
    """Browser session configuration."""
    display: str = ":1"
    timeout: int = 60
    headless: bool = True


class AgentBrowser:
    """
    Wrapper for vercel-labs/agent-browser CLI.
    
    This is the lightweight alternative to browser_session_daemon.
    Supports Bearer auth via --headers, snapshot for AI output.
    
    Usage:
        ab = AgentBrowser()
        ab.open("https://claude.ai", headers={"Authorization": "Bearer token"})
        ab.fill("textarea", "Hello world")
        ab.click("button[type=submit]")
        snapshot = ab.snapshot()
        ab.close()
    """
    
    # Webchat URLs
    WEBCHAT_URLS = {
        "claude": "https://claude.ai/new",
        "chatgpt": "https://chatgpt.com/",
        "gemini": "https://aistudio.google.com/",
    }
    
    # CSS selectors for chat input
    CHAT_SELECTORS = {
        "claude": "div[contenteditable='true'], textarea",
        "chatgpt": "#prompt-textarea",
        "gemini": "textarea",
    }
    
    def __init__(self, config: Optional[BrowserConfig] = None):
        self.config = config or BrowserConfig()
        self._session_active = False
    
    def _run(self, *args: str) -> Tuple[str, int]:
        """Run agent-browser CLI command."""
        env = {
            **os.environ,
            "DISPLAY": self.config.display,
            "PATH": f"/usr/bin:/usr/local/bin:{os.environ.get('PATH', '')}",
        }
        
        cmd = ["agent-browser"] + list(args)
        
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.config.timeout,
                env=env,
            )
            return result.stdout.strip(), result.returncode
        except subprocess.TimeoutExpired:
            return "Timeout", 1
        except FileNotFoundError:
            return "agent-browser not installed", 127
        except Exception as e:
            return str(e), 1
    
    def is_available(self) -> bool:
        """Check if agent-browser CLI is installed."""
        output, code = self._run("--version")
        return code == 0
    
    def open(self, url: str, headers: Optional[dict] = None) -> Tuple[bool, str]:
        """
        Navigate to URL with optional auth headers.
        
        Args:
            url: Target URL
            headers: Optional headers (e.g., {"Authorization": "Bearer token"})
            
        Returns:
            (success, message)
        """
        args = ["open", url]
        if headers:
            args.extend(["--headers", json.dumps(headers)])
        
        output, code = self._run(*args)
        self._session_active = code == 0
        return code == 0, output
    
    def fill(self, selector: str, text: str) -> Tuple[bool, str]:
        """Fill text into element matching selector."""
        output, code = self._run("fill", selector, text)
        return code == 0, output
    
    def click(self, selector: str) -> Tuple[bool, str]:
        """Click element matching selector."""
        output, code = self._run("click", selector)
        return code == 0, output
    
    def press(self, key: str) -> Tuple[bool, str]:
        """Press keyboard key (e.g., 'Enter', 'Tab')."""
        output, code = self._run("press", key)
        return code == 0, output
    
    def snapshot(self) -> str:
        """
        Get accessibility tree snapshot.
        
        This is the AI-optimized output format from agent-browser.
        Returns structured representation of visible elements.
        """
        output, code = self._run("snapshot")
        return output if code == 0 else ""
    
    def screenshot(self, path: Optional[str] = None) -> str:
        """Take screenshot, returns base64 PNG if no path given."""
        args = ["screenshot"]
        if path:
            args.append(path)
        output, code = self._run(*args)
        return output if code == 0 else ""
    
    def get_url(self) -> str:
        """Get current page URL."""
        output, code = self._run("get", "url")
        return output if code == 0 else ""
    
    def get_title(self) -> str:
        """Get current page title."""
        output, code = self._run("get", "title")
        return output if code == 0 else ""
    
    def wait(self, selector_or_ms: str) -> Tuple[bool, str]:
        """Wait for selector or milliseconds."""
        output, code = self._run("wait", selector_or_ms)
        return code == 0, output
    
    def close(self) -> Tuple[bool, str]:
        """Close browser session."""
        output, code = self._run("close")
        self._session_active = False
        return code == 0, output


class BrowserModule:
    """
    Main browser module interface.
    
    Provides unified access to agent-browser and webchat providers.
    
    Usage:
        browser = BrowserModule()
        browser.open("https://claude.ai")
        browser.fill("textarea", "Hello")
        snapshot = browser.snapshot()
    """
    
    def __init__(self):
        self._agent = AgentBrowser()
    
    def status(self) -> dict:
        """Return browser module status."""
        return {
            "agent_browser_available": self._agent.is_available(),
            "session_active": self._agent._session_active,
            "display": self._agent.config.display,
        }
    
    def open(self, url: str, headers: Optional[dict] = None) -> Tuple[bool, str]:
        """Open URL with optional auth headers."""
        return self._agent.open(url, headers)
    
    def open_provider(self, provider: str, auth_token: Optional[str] = None) -> Tuple[bool, str]:
        """
        Open webchat provider with optional auth.
        
        Args:
            provider: 'claude', 'chatgpt', or 'gemini'
            auth_token: Optional Bearer token
        """
        url = AgentBrowser.WEBCHAT_URLS.get(provider)
        if not url:
            return False, f"Unknown provider: {provider}"
        
        headers = {"Authorization": f"Bearer {auth_token}"} if auth_token else None
        return self._agent.open(url, headers)
    
    def chat(self, provider: str, message: str, wait_seconds: float = 3.0) -> str:
        """
        Send message to webchat provider and return snapshot.
        
        Args:
            provider: 'claude', 'chatgpt', or 'gemini'
            message: Message to send
            wait_seconds: Seconds to wait for response
            
        Returns:
            Accessibility tree snapshot of response
        """
        import time
        
        selector = AgentBrowser.CHAT_SELECTORS.get(provider, "textarea")
        
        self._agent.fill(selector, message)
        self._agent.press("Enter")
        time.sleep(wait_seconds)
        
        return self._agent.snapshot()
    
    def fill(self, selector: str, text: str) -> Tuple[bool, str]:
        """Fill text into element."""
        return self._agent.fill(selector, text)
    
    def click(self, selector: str) -> Tuple[bool, str]:
        """Click element."""
        return self._agent.click(selector)
    
    def snapshot(self) -> str:
        """Get accessibility tree snapshot."""
        return self._agent.snapshot()
    
    def close(self) -> Tuple[bool, str]:
        """Close browser."""
        return self._agent.close()


__all__ = ["BrowserModule", "AgentBrowser", "BrowserConfig"]
