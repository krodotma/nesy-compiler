#!/usr/bin/env python3
"""
Portal Denoiser - Step 9 of PORTAL Implementation.
Cleans raw universal ingest fragments for Etymon extraction.
"""
import sys
import re
import json

def denoise_fragment(raw_text):
    # 1. Remove ANSI escape codes
    text = re.sub(r'\x1B(?:[@-Z\-_]|\[[0-?]*[ -/]*[@-~])', '', raw_text)
    # 2. Normalize whitespace
    text = " ".join(text.split())
    # 3. Strip metadata markers commonly found in YouTube/Transcript exports
    text = re.sub(r'\[\d+:\d+\]', '', text)
    return text.strip()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        print(denoise_fragment(sys.argv[1]))
    else:
        # Stream processing from stdin
        for line in sys.stdin:
            print(denoise_fragment(line))
