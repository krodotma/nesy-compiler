#!/usr/bin/env python3
from __future__ import annotations

import curses
import json
import os
import time
from dataclasses import dataclass
from pathlib import Path

try:
    from plurichat import (
        ChatResponse,
        PLURIBUS_ROOT,
        execute_with_topology,
        get_all_provider_status,
        select_provider_for_query,
        shape_prompt,
        ProviderStatus,
    )
except ImportError:
    from nucleus.tools.plurichat import (
        ChatResponse,
        PLURIBUS_ROOT,
        execute_with_topology,
        get_all_provider_status,
        select_provider_for_query,
        shape_prompt,
        ProviderStatus,
    )


def update_providers_from_bus(current_providers: dict[str, object], bus_dir: Path) -> None:
    """Update provider status from bus events (non-blocking, tail-based)."""
    events_path = bus_dir / "events.ndjson"
    if not events_path.exists():
        return

    # Read last 200 lines (approx)
    try:
        # Optimization: seek to end minus chunk
        file_size = events_path.stat().st_size
        read_size = min(file_size, 50000) # Last 50KB
        
        with events_path.open("r", encoding="utf-8", errors="replace") as f:
            if file_size > read_size:
                f.seek(file_size - read_size)
            
            lines = f.readlines()
            
        for line in lines:
            try:
                if "provider.status" in line:
                    evt = json.loads(line)
                    topic = evt.get("topic", "")
                    if topic.startswith("provider.status."):
                        data = evt.get("data", {})
                        name = data.get("provider")
                        if name:
                            # Update or add
                            current_providers[name] = ProviderStatus(
                                name=name,
                                available=data.get("available", False),
                                model=data.get("model"),
                                error=data.get("error"),
                                blocker=data.get("note")
                            )
            except:
                pass
    except Exception:
        pass


@dataclass
class Picker:
    kind: str
    options: list[str]
    index: int = 0
    active: bool = False

    def current(self) -> str:
        return self.options[self.index] if self.options else "auto"

    def toggle(self) -> None:
        self.active = not self.active

    def move(self, delta: int) -> None:
        if not self.options:
            return
        self.index = (self.index + delta) % len(self.options)


def _render_status(stdscr, *, provider: str, persona: str, context_mode: str, art_mode: bool = False) -> None:
    h, w = stdscr.getmaxyx()
    art_status = "ON" if art_mode else "OFF"
    status = f"PluriChat TUI  provider={provider}  persona={persona}  ctx={context_mode}  art={art_status}  (TAB:providers  Ctrl-P:personas  Ctrl-A:art  Ctrl-C:quit)"
    stdscr.addstr(0, 0, status[: max(0, w - 1)], curses.A_REVERSE)


def _render_picker(stdscr, *, picker: Picker, providers: dict[str, object]) -> None:
    h, w = stdscr.getmaxyx()
    title = f"{picker.kind.upper()} picker (←/→/↑/↓, Enter select, Esc close)"
    stdscr.addstr(2, 0, title[: max(0, w - 1)], curses.A_BOLD)
    for i, opt in enumerate(picker.options[: max(0, h - 6)]):
        is_sel = i == picker.index
        avail = True
        if picker.kind == "provider":
            st = providers.get(opt)
            avail = bool(getattr(st, "available", True)) if st is not None else True
        prefix = "▶ " if is_sel else "  "
        suffix = "" if avail else " (blocked)"
        line = f"{prefix}{opt}{suffix}"
        stdscr.addstr(3 + i, 0, line[: max(0, w - 1)], curses.A_REVERSE if is_sel else 0)


def _render_transcript(stdscr, *, transcript: list[str], input_buf: str) -> None:
    h, w = stdscr.getmaxyx()
    body_top = 1
    body_bottom = h - 2
    max_lines = max(0, body_bottom - body_top)
    lines = transcript[-max_lines:]
    for i in range(max_lines):
        y = body_top + i
        stdscr.move(y, 0)
        stdscr.clrtoeol()
        if i < len(lines):
            stdscr.addstr(y, 0, lines[i][: max(0, w - 1)])

    # input line
    stdscr.move(h - 1, 0)
    stdscr.clrtoeol()
    prompt = f"> {input_buf}"
    stdscr.addstr(h - 1, 0, prompt[: max(0, w - 1)], curses.A_BOLD)


def _append_transcript(transcript: list[str], who: str, text: str) -> None:
    transcript.append(f"{who}: {text}")


def run(stdscr) -> int:
    curses.curs_set(1)
    stdscr.nodelay(False)
    stdscr.keypad(True)

    providers = get_all_provider_status()
    provider_options = ["auto", *sorted(providers.keys())]
    persona_options = ["auto", "ring0.architect", "ring0.security_auditor", "subagent.narrow_coder", "subagent.distiller", "subagent.edge_inference"]

    provider = "auto"
    persona = "auto"
    context_mode = "auto"

    provider_picker = Picker("provider", provider_options)
    persona_picker = Picker("persona", persona_options)

    transcript: list[str] = []
    input_buf = ""
    last_update = 0.0
    bus_dir = Path(os.environ.get("PLURIBUS_BUS_DIR", "/pluribus/.pluribus/bus"))
    art_mode = False

    while True:
        # Update provider status from bus (non-blocking)
        now = time.time()
        if now - last_update > 2.0:
            update_providers_from_bus(providers, bus_dir)
            last_update = now

        stdscr.erase()
        
        # Art Layer (Z-Index 0)
        if art_mode:
            h, w = stdscr.getmaxyx()
            # Simple noise pattern based on time
            for y in range(0, h, 2):
                for x in range(0, w, 4):
                    if (x + y + int(now * 2)) % 7 == 0:
                        try:
                            stdscr.addch(y, x, '.', curses.A_DIM)
                        except:
                            pass

        _render_status(stdscr, provider=provider, persona=persona, context_mode=context_mode, art_mode=art_mode, trace_id=trace_id)
        if provider_picker.active:
            _render_picker(stdscr, picker=provider_picker, providers=providers)
        elif persona_picker.active:
            _render_picker(stdscr, picker=persona_picker, providers=providers)
        else:
            _render_transcript(stdscr, transcript=transcript, input_buf=input_buf)
        stdscr.refresh()

        ch = stdscr.getch()
        if ch in (3,):  # Ctrl-C
            return 0
            
        if ch == 1: # Ctrl-A
            art_mode = not art_mode
            continue

        # Global toggles
            _render_picker(stdscr, picker=provider_picker, providers=providers)
        elif persona_picker.active:
            _render_picker(stdscr, picker=persona_picker, providers=providers)
        else:
            _render_transcript(stdscr, transcript=transcript, input_buf=input_buf)
        stdscr.refresh()

        ch = stdscr.getch()
        if ch in (3,):  # Ctrl-C
            return 0

        # Global toggles
        if ch == 9:  # TAB
            if not persona_picker.active:
                provider_picker.toggle()
                if provider_picker.active:
                    persona_picker.active = False
            continue
        if ch == 16:  # Ctrl-P
            if not provider_picker.active:
                persona_picker.toggle()
                if persona_picker.active:
                    provider_picker.active = False
            continue

        active_picker = provider_picker if provider_picker.active else (persona_picker if persona_picker.active else None)
        if active_picker:
            if ch in (curses.KEY_LEFT, curses.KEY_UP):
                active_picker.move(-1)
            elif ch in (curses.KEY_RIGHT, curses.KEY_DOWN):
                active_picker.move(1)
            elif ch in (10, 13):  # Enter
                if active_picker.kind == "provider":
                    provider = active_picker.current()
                else:
                    persona = active_picker.current()
                active_picker.active = False
            elif ch == 27:  # Esc
                active_picker.active = False
            continue

        # Input handling
        if ch in (curses.KEY_BACKSPACE, 127, 8):
            input_buf = input_buf[:-1]
            continue
        if ch in (10, 13):  # Enter submit
            msg = input_buf.strip()
            input_buf = ""
            if not msg:
                continue
            if msg in ("/quit", "/q", "/exit"):
                return 0
            if msg.startswith("/context "):
                v = msg.split(" ", 1)[1].strip().lower()
                if v in {"auto", "min", "lite", "full"}:
                    context_mode = v
                    _append_transcript(transcript, "sys", f"context={context_mode}")
                else:
                    _append_transcript(transcript, "sys", "invalid context (auto|min|lite|full)")
                continue

            _append_transcript(transcript, "you", msg)
            trace_id = str(uuid.uuid4())
            providers = get_all_provider_status()
            routing = select_provider_for_query(msg, provider, providers)
            lens = routing.lens
            if context_mode != "auto":
                lens.context_mode = context_mode  # type: ignore[misc]
            eff = shape_prompt(msg, lens=lens, persona_override=persona)
            
            # Pass trace_id via env var injection or modify execute_with_topology if it supports it.
            # Assuming execute_with_topology supports **kwargs or we set env.
            os.environ["PLURIBUS_TRACE_ID"] = trace_id
            
            resp: ChatResponse = execute_with_topology(eff, routing, Path(os.environ.get("PLURIBUS_BUS_DIR", "/pluribus/.pluribus/bus")), "plurichat-tui", "direct")
            if resp.success:
                _append_transcript(transcript, "assistant", resp.text)
                _append_transcript(transcript, "meta", f"{resp.provider} {resp.latency_ms:.0f}ms")
            else:
                _append_transcript(transcript, "error", resp.error or "unknown")
            continue

        if 32 <= ch <= 126:
            input_buf += chr(ch)


def main() -> int:
    try:
        return curses.wrapper(run)
    except KeyboardInterrupt:
        return 0


if __name__ == "__main__":
    raise SystemExit(main())

