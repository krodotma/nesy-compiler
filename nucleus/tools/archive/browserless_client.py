#!/usr/bin/env python3
"""Browserless Client - HTTP Chrome API wrapper for web scraping.

Provides a Pythonic interface to Browserless.io's HTTP Chrome API for:
- Screenshot capture
- PDF generation
- Content extraction / web scraping

Emits bus events for all actions to enable observability and audit trails.

Usage:
    from browserless_client import BrowserlessClient

    client = BrowserlessClient("http://localhost:3000")

    # Screenshot
    img_bytes = await client.screenshot("https://example.com")

    # PDF
    pdf_bytes = await client.pdf("https://example.com")

    # Scrape content
    content = await client.scrape("https://example.com", elements=["h1", "p"])
"""
from __future__ import annotations

import argparse
import asyncio
import base64
import json
import os
import sys
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

sys.dont_write_bytecode = True

# Optional httpx import (fallback to urllib for basic operations)
try:
    import httpx

    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False
    import urllib.request
    import urllib.error


def now_iso_utc() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def emit_bus_event(topic: str, data: dict[str, Any], level: str = "info") -> None:
    """Emit a bus event for observability."""
    try:
        # Dynamically import agent_bus to avoid circular deps
        tools_dir = Path(__file__).parent
        sys.path.insert(0, str(tools_dir))
        import agent_bus

        paths = agent_bus.resolve_bus_paths(None)
        agent_bus.emit_event(
            paths,
            topic=topic,
            kind="metric",
            level=level,
            actor="browserless-client",
            data=data,
            trace_id=None,
            run_id=None,
            durable=False,
        )
    except Exception:
        # Bus emission is best-effort
        pass


@dataclass
class BrowserlessConfig:
    """Configuration for Browserless client."""

    base_url: str = "http://localhost:3000"
    token: str | None = None  # API token if using Browserless.io cloud
    timeout: float = 30.0  # Request timeout in seconds
    default_viewport: dict = field(
        default_factory=lambda: {"width": 1920, "height": 1080}
    )
    default_wait_for: str | None = None  # CSS selector to wait for
    user_agent: str | None = None


@dataclass
class ScrapeResult:
    """Result of a scrape operation."""

    url: str
    html: str | None = None
    text: str | None = None
    elements: dict[str, list[str]] = field(default_factory=dict)
    screenshot_b64: str | None = None
    pdf_b64: str | None = None
    metadata: dict = field(default_factory=dict)
    error: str | None = None
    duration_ms: float = 0.0


class BrowserlessClient:
    """HTTP client for Browserless Chrome API."""

    def __init__(self, config: BrowserlessConfig | str | None = None):
        if config is None:
            config = BrowserlessConfig()
        elif isinstance(config, str):
            config = BrowserlessConfig(base_url=config)
        self.config = config
        self._client: httpx.AsyncClient | None = None

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create httpx client."""
        if not HTTPX_AVAILABLE:
            raise RuntimeError(
                "httpx is required for async operations. Install with: pip install httpx"
            )
        if self._client is None:
            headers = {"Content-Type": "application/json"}
            if self.config.token:
                headers["Authorization"] = f"Bearer {self.config.token}"
            self._client = httpx.AsyncClient(
                base_url=self.config.base_url,
                headers=headers,
                timeout=self.config.timeout,
            )
        return self._client

    async def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            await self._client.aclose()
            self._client = None

    async def screenshot(
        self,
        url: str,
        *,
        full_page: bool = False,
        viewport: dict | None = None,
        wait_for: str | None = None,
        format: str = "png",
    ) -> bytes:
        """Capture a screenshot of a URL.

        Args:
            url: Target URL to screenshot
            full_page: Capture full scrollable page
            viewport: Custom viewport dimensions
            wait_for: CSS selector to wait for before capture
            format: Image format (png, jpeg, webp)

        Returns:
            Screenshot bytes
        """
        start_time = time.time()
        client = await self._get_client()

        payload = {
            "url": url,
            "options": {
                "type": format,
                "fullPage": full_page,
            },
            "viewport": viewport or self.config.default_viewport,
        }

        if wait_for or self.config.default_wait_for:
            payload["waitForSelector"] = wait_for or self.config.default_wait_for

        if self.config.user_agent:
            payload["userAgent"] = self.config.user_agent

        try:
            response = await client.post("/screenshot", json=payload)
            response.raise_for_status()
            screenshot_bytes = response.content
            duration_ms = (time.time() - start_time) * 1000

            emit_bus_event(
                "browserless.screenshot",
                {
                    "url": url,
                    "format": format,
                    "full_page": full_page,
                    "size_bytes": len(screenshot_bytes),
                    "duration_ms": duration_ms,
                    "success": True,
                },
            )

            return screenshot_bytes

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            emit_bus_event(
                "browserless.screenshot",
                {
                    "url": url,
                    "format": format,
                    "error": str(e),
                    "duration_ms": duration_ms,
                    "success": False,
                },
                level="error",
            )
            raise

    async def pdf(
        self,
        url: str,
        *,
        viewport: dict | None = None,
        wait_for: str | None = None,
        print_background: bool = True,
        format: str = "A4",
        landscape: bool = False,
        margin: dict | None = None,
    ) -> bytes:
        """Generate a PDF of a URL.

        Args:
            url: Target URL to render
            viewport: Custom viewport dimensions
            wait_for: CSS selector to wait for
            print_background: Include background graphics
            format: Paper format (A4, Letter, etc.)
            landscape: Landscape orientation
            margin: Page margins

        Returns:
            PDF bytes
        """
        start_time = time.time()
        client = await self._get_client()

        payload = {
            "url": url,
            "options": {
                "printBackground": print_background,
                "format": format,
                "landscape": landscape,
            },
            "viewport": viewport or self.config.default_viewport,
        }

        if margin:
            payload["options"]["margin"] = margin

        if wait_for or self.config.default_wait_for:
            payload["waitForSelector"] = wait_for or self.config.default_wait_for

        if self.config.user_agent:
            payload["userAgent"] = self.config.user_agent

        try:
            response = await client.post("/pdf", json=payload)
            response.raise_for_status()
            pdf_bytes = response.content
            duration_ms = (time.time() - start_time) * 1000

            emit_bus_event(
                "browserless.pdf",
                {
                    "url": url,
                    "format": format,
                    "landscape": landscape,
                    "size_bytes": len(pdf_bytes),
                    "duration_ms": duration_ms,
                    "success": True,
                },
            )

            return pdf_bytes

        except Exception as e:
            duration_ms = (time.time() - start_time) * 1000
            emit_bus_event(
                "browserless.pdf",
                {
                    "url": url,
                    "error": str(e),
                    "duration_ms": duration_ms,
                    "success": False,
                },
                level="error",
            )
            raise

    async def scrape(
        self,
        url: str,
        *,
        elements: list[str] | None = None,
        wait_for: str | None = None,
        viewport: dict | None = None,
        include_html: bool = False,
        include_screenshot: bool = False,
    ) -> ScrapeResult:
        """Scrape content from a URL.

        Args:
            url: Target URL to scrape
            elements: CSS selectors to extract text from
            wait_for: CSS selector to wait for
            viewport: Custom viewport dimensions
            include_html: Include full HTML in result
            include_screenshot: Include screenshot in result

        Returns:
            ScrapeResult with extracted content
        """
        start_time = time.time()
        client = await self._get_client()

        # Build scrape function payload
        payload = {
            "url": url,
            "viewport": viewport or self.config.default_viewport,
        }

        if wait_for or self.config.default_wait_for:
            payload["waitForSelector"] = wait_for or self.config.default_wait_for

        if self.config.user_agent:
            payload["userAgent"] = self.config.user_agent

        # Use /content endpoint for HTML
        result = ScrapeResult(url=url)

        try:
            # Get HTML content
            response = await client.post("/content", json=payload)
            response.raise_for_status()

            content_data = response.json() if response.headers.get(
                "content-type", ""
            ).startswith("application/json") else {"html": response.text}

            if include_html:
                result.html = content_data.get("html", response.text)

            # Extract text from specified elements using /scrape endpoint
            if elements:
                scrape_payload = {
                    **payload,
                    "elements": [
                        {"selector": sel, "timeout": 5000} for sel in elements
                    ],
                }
                try:
                    scrape_response = await client.post("/scrape", json=scrape_payload)
                    scrape_response.raise_for_status()
                    scrape_data = scrape_response.json()

                    for item in scrape_data.get("data", []):
                        selector = item.get("selector", "unknown")
                        texts = item.get("results", [])
                        result.elements[selector] = [
                            r.get("text", "") for r in texts if isinstance(r, dict)
                        ]
                except Exception:
                    # Scrape endpoint may not be available, continue without elements
                    pass

            # Get screenshot if requested
            if include_screenshot:
                try:
                    screenshot_bytes = await self.screenshot(url, viewport=viewport)
                    result.screenshot_b64 = base64.b64encode(screenshot_bytes).decode(
                        "utf-8"
                    )
                except Exception:
                    pass

            result.duration_ms = (time.time() - start_time) * 1000

            # Extract plain text from HTML if available
            if result.html:
                # Simple text extraction (strip tags)
                import re

                result.text = re.sub(r"<[^>]+>", " ", result.html)
                result.text = re.sub(r"\s+", " ", result.text).strip()[:5000]

            emit_bus_event(
                "browserless.scrape",
                {
                    "url": url,
                    "elements_requested": elements or [],
                    "elements_found": list(result.elements.keys()),
                    "has_html": result.html is not None,
                    "has_screenshot": result.screenshot_b64 is not None,
                    "text_length": len(result.text) if result.text else 0,
                    "duration_ms": result.duration_ms,
                    "success": True,
                },
            )

            return result

        except Exception as e:
            result.error = str(e)
            result.duration_ms = (time.time() - start_time) * 1000

            emit_bus_event(
                "browserless.scrape",
                {
                    "url": url,
                    "error": str(e),
                    "duration_ms": result.duration_ms,
                    "success": False,
                },
                level="error",
            )

            return result

    async def health_check(self) -> dict:
        """Check Browserless service health.

        Returns:
            Health status dict
        """
        client = await self._get_client()
        try:
            response = await client.get("/health")
            response.raise_for_status()
            return {"status": "healthy", "data": response.json()}
        except Exception as e:
            return {"status": "unhealthy", "error": str(e)}


# Synchronous wrapper for CLI usage
def screenshot_sync(
    url: str, output: str, base_url: str = "http://localhost:3000"
) -> None:
    """Synchronous screenshot capture."""
    if not HTTPX_AVAILABLE:
        # Fallback to urllib
        payload = json.dumps(
            {
                "url": url,
                "options": {"type": "png", "fullPage": False},
                "viewport": {"width": 1920, "height": 1080},
            }
        ).encode("utf-8")
        req = urllib.request.Request(
            f"{base_url}/screenshot",
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
            Path(output).write_bytes(data)
            print(f"Screenshot saved to {output} ({len(data)} bytes)")
        return

    async def _run():
        client = BrowserlessClient(base_url)
        try:
            data = await client.screenshot(url)
            Path(output).write_bytes(data)
            print(f"Screenshot saved to {output} ({len(data)} bytes)")
        finally:
            await client.close()

    asyncio.run(_run())


def scrape_sync(
    url: str, elements: list[str] | None = None, base_url: str = "http://localhost:3000"
) -> dict:
    """Synchronous scrape."""

    async def _run():
        client = BrowserlessClient(base_url)
        try:
            result = await client.scrape(url, elements=elements, include_html=True)
            return {
                "url": result.url,
                "text": result.text,
                "elements": result.elements,
                "error": result.error,
                "duration_ms": result.duration_ms,
            }
        finally:
            await client.close()

    return asyncio.run(_run())


def cmd_screenshot(args: argparse.Namespace) -> int:
    """CLI: screenshot command."""
    try:
        screenshot_sync(args.url, args.output, args.base_url)
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_scrape(args: argparse.Namespace) -> int:
    """CLI: scrape command."""
    try:
        elements = args.elements.split(",") if args.elements else None
        result = scrape_sync(args.url, elements, args.base_url)
        if args.json:
            print(json.dumps(result, indent=2, ensure_ascii=False))
        else:
            if result.get("error"):
                print(f"Error: {result['error']}")
                return 1
            print(f"URL: {result['url']}")
            print(f"Duration: {result['duration_ms']:.1f}ms")
            if result.get("elements"):
                for sel, texts in result["elements"].items():
                    print(f"\n{sel}:")
                    for t in texts[:5]:
                        print(f"  - {t[:100]}")
            if result.get("text"):
                print(f"\nText preview:\n{result['text'][:500]}...")
        return 0
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def cmd_health(args: argparse.Namespace) -> int:
    """CLI: health check command."""

    async def _run():
        client = BrowserlessClient(args.base_url)
        try:
            result = await client.health_check()
            print(json.dumps(result, indent=2))
            return 0 if result["status"] == "healthy" else 1
        finally:
            await client.close()

    try:
        return asyncio.run(_run())
    except Exception as e:
        print(json.dumps({"status": "unhealthy", "error": str(e)}, indent=2))
        return 1


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="browserless_client.py",
        description="Browserless HTTP Chrome API client for web scraping",
    )
    p.add_argument(
        "--base-url",
        default=os.environ.get("BROWSERLESS_URL", "http://localhost:3000"),
        help="Browserless service URL",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    # screenshot
    ss = sub.add_parser("screenshot", help="Capture a screenshot")
    ss.add_argument("url", help="URL to capture")
    ss.add_argument("-o", "--output", default="screenshot.png", help="Output file")
    ss.set_defaults(func=cmd_screenshot)

    # scrape
    sc = sub.add_parser("scrape", help="Scrape content from a URL")
    sc.add_argument("url", help="URL to scrape")
    sc.add_argument(
        "-e", "--elements", help="Comma-separated CSS selectors to extract"
    )
    sc.add_argument("--json", action="store_true", help="Output as JSON")
    sc.set_defaults(func=cmd_scrape)

    # health
    h = sub.add_parser("health", help="Check Browserless service health")
    h.set_defaults(func=cmd_health)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
