#!/usr/bin/env python3
"""
tmux_pty_bridge.py

Spawn a tmux attach inside a PTY so WebSocket bridges can stream bytes without
"open terminal failed: not a terminal" errors.

This is intentionally small and dumb:
- stdin/stdout are pipes (from Node.js ws bridge)
- we allocate a PTY for tmux so it has a real controlling terminal

Resize:
- Web terminals (xterm.js) report cols/rows on container resize
- The tmux client size must change to render the full grid; tmux window resizing
  alone is insufficient when the controlling PTY never receives SIGWINCH
- We accept a small control protocol over stdin to apply TIOCSWINSZ to the PTY
"""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import pty
import select
import signal
import struct
import subprocess
import sys
import termios


CTRL_PREFIX = b"\x00PLURIBUS:"


class ControlDemux:
    """Demultiplex stdin bytes into (raw_input, control_frames).

    Control frames are injected by the Node bridge and are removed from the
    stream before writing to the PTY. This avoids control JSON being typed into
    tmux while still letting a plain stdin pipe carry both kinds of data.

    Frame format:
      b"\\x00PLURIBUS:" + payload + b"\\n"
    """

    def __init__(self, prefix: bytes = CTRL_PREFIX):
        self._prefix = prefix
        self._buffer = bytearray()

    def feed(self, chunk: bytes) -> tuple[bytes, list[str]]:
        if chunk:
            self._buffer.extend(chunk)

        forward = bytearray()
        frames: list[str] = []

        while True:
            idx = self._buffer.find(self._prefix)
            if idx == -1:
                keep = self._keep_suffix_len()
                if keep:
                    forward.extend(self._buffer[:-keep])
                    del self._buffer[:-keep]
                else:
                    forward.extend(self._buffer)
                    self._buffer.clear()
                break

            if idx > 0:
                forward.extend(self._buffer[:idx])
                del self._buffer[:idx]
                continue

            nl = self._buffer.find(b"\n", len(self._prefix))
            if nl == -1:
                break

            payload = bytes(self._buffer[len(self._prefix) : nl])
            del self._buffer[: nl + 1]
            frames.append(payload.decode("utf-8", errors="replace").strip())

        return bytes(forward), frames

    def _keep_suffix_len(self) -> int:
        """Keep suffix bytes that might be the start of a prefix."""
        if not self._buffer:
            return 0
        max_len = min(len(self._buffer), len(self._prefix) - 1)
        for n in range(max_len, 0, -1):
            if self._buffer[-n:] == self._prefix[:n]:
                return n
        return 0


def _set_winsize(fd: int, cols: int, rows: int) -> None:
    # TIOCSWINSZ expects rows, cols, xpixels, ypixels (unsigned short)
    winsize = struct.pack("HHHH", int(rows), int(cols), 0, 0)
    fcntl.ioctl(fd, termios.TIOCSWINSZ, winsize)


def _handle_control(payload: str, fd: int, child_pid: int) -> None:
    """Apply supported control frames (currently: resize)."""
    if not payload:
        return

    cols: int | None = None
    rows: int | None = None

    parts = payload.split()
    if len(parts) >= 3 and parts[0].lower() == "resize":
        try:
            cols = int(float(parts[1]))
            rows = int(float(parts[2]))
        except ValueError:
            cols = None
            rows = None
    else:
        # Allow JSON payloads for future-proofing.
        try:
            msg = json.loads(payload)
            if isinstance(msg, dict) and msg.get("type") == "resize":
                cols = int(float(msg.get("cols", 0)))
                rows = int(float(msg.get("rows", 0)))
        except Exception:
            return

    if cols is None or rows is None:
        return
    if cols <= 0 or rows <= 0:
        return

    try:
        _set_winsize(fd, cols, rows)
    except Exception:
        return

    try:
        os.kill(child_pid, signal.SIGWINCH)
    except Exception:
        pass


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(prog="tmux_pty_bridge.py")
    ap.add_argument("--session", required=True, help="tmux session name to attach")
    ap.add_argument("--ensure", default=None, help="Optional script to ensure session exists")
    args = ap.parse_args(argv)

    if args.ensure:
        try:
            subprocess.run(["bash", args.ensure], check=False)
        except Exception:
            pass

    cmd = ["tmux", "attach-session", "-t", args.session]

    pid, fd = pty.fork()
    if pid == 0:
        os.execvp(cmd[0], cmd)
        raise SystemExit(127)

    demux = ControlDemux()

    try:
        while True:
            r, _, _ = select.select([sys.stdin.buffer, fd], [], [])

            if sys.stdin.buffer in r:
                chunk = sys.stdin.buffer.read1(4096)
                if not chunk:
                    break
                forward, frames = demux.feed(chunk)
                if forward:
                    os.write(fd, forward)
                for payload in frames:
                    _handle_control(payload, fd, pid)

            if fd in r:
                d = os.read(fd, 4096)
                if not d:
                    break
                sys.stdout.buffer.write(d)
                sys.stdout.buffer.flush()
    except OSError:
        pass
    finally:
        try:
            os.kill(pid, 15)
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
