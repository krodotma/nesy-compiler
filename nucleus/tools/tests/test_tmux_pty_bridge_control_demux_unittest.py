#!/usr/bin/env python3
from __future__ import annotations

import pathlib
import sys
import unittest

TOOLS_DIR = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS_DIR))

import tmux_pty_bridge


class TestTmuxPtyBridgeControlDemux(unittest.TestCase):
    def test_pass_through_plain_bytes(self):
        demux = tmux_pty_bridge.ControlDemux()
        out, frames = demux.feed(b"hello world")
        self.assertEqual(out, b"hello world")
        self.assertEqual(frames, [])

    def test_extracts_control_frame_in_middle(self):
        demux = tmux_pty_bridge.ControlDemux()
        out, frames = demux.feed(b"abc\x00PLURIBUS:resize 80 24\nxyz")
        self.assertEqual(out, b"abcxyz")
        self.assertEqual(frames, ["resize 80 24"])

    def test_prefix_split_across_chunks(self):
        demux = tmux_pty_bridge.ControlDemux()

        out1, frames1 = demux.feed(b"abc\x00PLUR")
        self.assertEqual(out1, b"abc")
        self.assertEqual(frames1, [])

        out2, frames2 = demux.feed(b"IBUS:resize 120 30\nxyz")
        self.assertEqual(out2, b"xyz")
        self.assertEqual(frames2, ["resize 120 30"])

    def test_control_frame_split_across_chunks(self):
        demux = tmux_pty_bridge.ControlDemux()

        out1, frames1 = demux.feed(b"abc\x00PLURIBUS:resize 12")
        self.assertEqual(out1, b"abc")
        self.assertEqual(frames1, [])

        out2, frames2 = demux.feed(b"0 3")
        self.assertEqual(out2, b"")
        self.assertEqual(frames2, [])

        out3, frames3 = demux.feed(b"0\nxyz")
        self.assertEqual(out3, b"xyz")
        self.assertEqual(frames3, ["resize 120 30"])


if __name__ == "__main__":
    unittest.main()
