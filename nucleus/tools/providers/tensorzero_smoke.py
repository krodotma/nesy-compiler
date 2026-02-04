#!/usr/bin/env python3
"""
TensorZero Gateway Smoke Test
"""
import argparse
import os
import sys
import json
import urllib.request
import urllib.error

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prompt", required=True)
    p.add_argument("--model", default="gpt-4o-mini") # Default gateway model alias
    args = p.parse_args()

    base_url = os.environ.get("TENSORZERO_GATEWAY_URL", "http://localhost:3000")
    
    payload = {
        "model": args.model,
        "messages": [{"role": "user", "content": args.prompt}],
        "stream": False
    }

    try:
        req = urllib.request.Request(
            f"{base_url}/inference",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            # Adjust based on actual TensorZero response format (OpenAI compat or custom)
            # Assuming OpenAI compat for inference endpoint
            content = data.get("choices", [{}])[0].get("message", {}).get("content", "")
            if not content:
                # Try TensorZero specific format
                content = data.get("content", "")
            
            print(content)
            return 0
    except Exception as e:
        sys.stderr.write(f"TensorZero Error: {e}\n")
        return 1

if __name__ == "__main__":
    sys.exit(main())
