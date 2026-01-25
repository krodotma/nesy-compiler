#!/usr/bin/env python3
"""
LASER Token Auth Tool
=====================
Codename: LASER-OAUTH
Goal: Laptop OAuth -> Token -> VPS Push -> Webchat Interaction

Usage:
  python3 laser_auth_tool.py auth <provider>
  python3 laser_auth_tool.py token <provider> <token_string>
  python3 laser_auth_tool.py push <vps_host> [--user root]
  python3 laser_auth_tool.py status
"""
import argparse
import json
import os
import sys
import time
import webbrowser
import subprocess
from pathlib import Path
from urllib.parse import urlencode

# Constants
AUTH_FILE = Path.home() / ".pluribus" / "auth.json"
CALLBACK_PORT = 14556
CALLBACK_URL = f"http://localhost:{CALLBACK_PORT}/callback"

PROVIDERS = {
    "claude": {
        "auth_url": "https://claude.ai/oauth/authorize",  # Placeholder - actual endpoint research needed
        "token_url": "https://claude.ai/oauth/token",
        "scope": "chat",
        "client_id": "laser_cli_client" # Often public for CLIs
    },
    "gemini": {
        "auth_url": "https://accounts.google.com/o/oauth2/v2/auth",
        "scope": "https://www.googleapis.com/auth/generative-language.retriever",
        "client_id": "REPLACE_WITH_REAL_CLIENT_ID" 
    },
    "codex": {
        "auth_url": "https://github.com/login/oauth/authorize",
        "scope": "copilot",
        "client_id": "Iv1.b507a08c87ecfe98" # GitHub CLI client ID
    }
}

def ensure_auth_dir():
    if not AUTH_FILE.parent.exists():
        AUTH_FILE.parent.mkdir(parents=True, exist_ok=True)

def ensure_auth_file():
    ensure_auth_dir()
    if not AUTH_FILE.exists():
        # Initialize with empty JSON directly to avoid recursion
        with open(AUTH_FILE, 'w') as f:
            json.dump({}, f)
        os.chmod(AUTH_FILE, 0o600)

def load_auth_data():
    ensure_auth_file()
    try:
        with open(AUTH_FILE, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        return {}

def save_auth_data(data):
    ensure_auth_dir()
    with open(AUTH_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    if os.path.exists(AUTH_FILE):
        os.chmod(AUTH_FILE, 0o600)

import secrets
import hashlib
import base64

def generate_pkce_pair():
    """Generate code_verifier and code_challenge for PKCE."""
    code_verifier = secrets.token_urlsafe(64)
    code_challenge = hashlib.sha256(code_verifier.encode('ascii')).digest()
    code_challenge = base64.urlsafe_b64encode(code_challenge).decode('ascii').rstrip('=')
    return code_verifier, code_challenge

def cmd_auth_url(provider):
    if provider not in PROVIDERS:
        print(f"Error: Unknown provider '{provider}'. Available: {list(PROVIDERS.keys())}")
        return

def cmd_auth_url(provider):
    if provider not in PROVIDERS:
        print(f"Error: Unknown provider '{provider}'. Available: {list(PROVIDERS.keys())}")
        return

    print(f"\n=== LASER Auth: {provider.upper()} ===")
    
    if provider == "claude":
        # Strategy: Use the official CLI
        print("💡 Detecting official Claude CLI...")
        claude_path = "/opt/homebrew/bin/claude"
        
        if os.path.exists(claude_path):
            print(f"✅ Found official CLI at {claude_path}")
            print("🚀 Launching 'claude login' for you...")
            print("   (Follow the browser instructions, then return here)")
            print("---------------------------------------------------------")
            
            try:
                # Run interactively
                subprocess.run([claude_path, "login"])
                
                print("\n---------------------------------------------------------")
                print("✅ Login process finished.")
                print("👉 Now run this to import the token:")
                print(f"   python3 nucleus/tools/laser_auth_tool.py token claude --auto")
            except Exception as e:
                print(f"❌ Failed to run claude login: {e}")
                print("Please run 'claude login' manually.")
            return
        else:
            print("⚠️  Official Claude CLI not found. Falling back to manual method.")

    if provider == "codex":
        print("💡 Detecting official GitHub CLI (gh)...")
        gh_path = "/opt/homebrew/bin/gh"
        if os.path.exists(gh_path):
             print(f"✅ Found gh CLI at {gh_path}")
             print("🚀 Launching 'gh auth login' for you...")
             try:
                 subprocess.run([gh_path, "auth", "login", "-h", "github.com", "-p", "https", "-w"])
                 print("\n---------------------------------------------------------")
                 print("✅ Login process finished.")
                 print("👉 Now run this to import the token:")
                 print(f"   python3 nucleus/tools/laser_auth_tool.py token codex --auto")
             except Exception as e:
                 print(f"❌ Failed to run gh auth login: {e}")
             return

    if provider == "gemini":
        print("\n---------------------------------------------------------")
        print("MANUAL API KEY (Gemini)")
        print("1. Go to https://aistudio.google.com/app/apikey")
        print("2. Create or copy your API Key.")
        print("\nThen run:")
        print(f"python3 nucleus/tools/laser_auth_tool.py token gemini \"<YOUR_API_KEY>\"")
        print(f"python3 nucleus/tools/laser_auth_tool.py push fittwin")
        return

    # Fallback / Original Manual Method
    if provider == "claude":
        # ... (keep existing claude manual fallback)
        print("\n---------------------------------------------------------")
        print("MANUAL SESSION KEY (Fallback)")
        print("1. Log in to https://claude.ai in your browser.")
        print("2. Open DevTools (F12) -> Application -> Cookies.")
        print("3. Copy the value of the 'sessionKey' cookie (sk-ant-sid01-...).")
        print("\nThen run:")
        print(f"python3 nucleus/tools/laser_auth_tool.py token claude \"<SESSION_KEY>\"")
        print(f"python3 nucleus/tools/laser_auth_tool.py push fittwin")
        return

    # Generic OAuth for others...
    p = PROVIDERS[provider]
    # ... (rest defaults)
    verifier, challenge = generate_pkce_pair()
    
    # ... (save verifier logic) ...
    data = load_auth_data()
    data.setdefault("pending_auth", {})[provider] = {
        "verifier": verifier,
        "timestamp": int(time.time())
    }
    save_auth_data(data)

    params = {
        "client_id": p.get("client_id", "laser_cli_client"),
        "redirect_uri": CALLBACK_URL,
        "response_type": "code",
        "scope": p.get("scope", ""),
        "state": f"laser-{int(time.time())}",
        "code_challenge": challenge,
        "code_challenge_method": "S256"
    }
    
    url = f"{p['auth_url']}?{urlencode(params)}"
    
    print(f"\n=== LASER Auth: {provider.upper()} ===")
    print("---------------------------------------------------------")
    print("OPTION 1: AUTOMATED OAUTH (Try this first)")
    print(f"1. Opening browser to: {url}")
    print("2. If successful, you will be redirected to localhost.")
    print("3. Copy the 'code=' parameter from the URL bar.")
    print("---------------------------------------------------------")
    print("OPTION 2: MANUAL SESSION KEY (Fallback)")
    print(f"   (Use this if you see 'Invalid OAuth Request')")
    
    try:
        webbrowser.open(url)
    except Exception as e:
        print(f"   (Failed to open browser: {e})")

def cmd_token_paste(provider, token=None, auto_mode=False):
    # Update to handle --auto flag logic
    if auto_mode:
        print(f"🔍 Auto-discovering token for {provider}...")
        
        found = None
        
        if provider == "claude":
            # ... (keep existing claude logic) ...
            paths = [
                Path.home() / ".claude/settings.json",
                Path.home() / "Library/Application Support/Claude/config.json",
                Path.home() / ".config/claude/config.json",
                Path.home() / ".anthropic/config.json",
                Path.home() / "Library/Preferences/Claude/config.json"
            ]
            for p in paths:
                if p.exists():
                    print(f"   Found config at: {p}")
                    try:
                        data = json.load(open(p))
                        if "sessionKey" in data:
                             found = data["sessionKey"]
                        elif "token" in data:
                             found = data["token"]
                        elif "oauth_token" in data:
                             found = data["oauth_token"]
                    except:
                        pass
        
        elif provider == "codex":
             # Use gh auth token command
             gh_path = "/opt/homebrew/bin/gh"
             if os.path.exists(gh_path):
                 try:
                     result = subprocess.run([gh_path, "auth", "token"], capture_output=True, text=True)
                     if result.returncode == 0:
                         found = result.stdout.strip()
                         print("   Retrieved token via 'gh auth token'")
                 except Exception as e:
                     print(f"   Failed to run gh: {e}")
        
        elif provider == "gemini":
             print("   Note: Gemini (Google) typically requires a manual API key from AI Studio.")
             # Check env vars just in case
             if "GOOGLE_API_KEY" in os.environ:
                 found = os.environ["GOOGLE_API_KEY"]
                 print("   Found GOOGLE_API_KEY in environment.")
        
        if found:
            token = found
            print("✅ Auto-discovered token!")
        else:
            print("❌ Could not auto-discover token. Please paste it manually.")
            return

    if not token and not auto_mode:
        print("Error: You must provide a token string or use --auto")
        return
        
    data = load_auth_data()
    # ... rest of save logic ...
    
    if "providers" not in data:
        data["providers"] = {}
        
    data["providers"][provider] = {
        "token_type": "manual_paste",
        "access_token": token,
        "updated_at": int(time.time()),
        "manual_entry": True
    }
    
    save_auth_data(data)
    print(f"✅ Saved token for {provider} to {AUTH_FILE}")

def cmd_push(host, user="root"):
    if not AUTH_FILE.exists():
        print(f"Error: {AUTH_FILE} does not exist. Run 'auth' or 'token' first.")
        return

    # 1. Push Auth Token
    dest_auth = f"{user}@{host}:/pluribus/.pluribus/auth.json"
    print(f"🚀 Pushing Auth: {AUTH_FILE} -> {dest_auth}...")
    
    # 2. Push Updated Code (providers.py)
    # Assume tool is run from workspace root or tools dir, try to locate providers.py
    current_dir = Path.cwd()
    providers_file = None
    
    potential_paths = [
        current_dir / "nucleus/tools/providers.py",
        current_dir / "providers.py",
        Path(__file__).parent / "providers.py"
    ]
    
    for p in potential_paths:
        if p.exists():
            providers_file = p
            break
            
    if providers_file:
        dest_code = f"{user}@{host}:/pluribus/nucleus/tools/providers.py"
        print(f"📦 Pushing Code: {providers_file} -> {dest_code}...")
        try:
             subprocess.check_call(["scp", str(providers_file), dest_code])
        except subprocess.CalledProcessError as e:
             print(f"⚠️ Failed to push providers.py: {e}")
    else:
        print("⚠️ Warning: Could not find local providers.py to deploy.")

    try:
        subprocess.check_call(["scp", str(AUTH_FILE), dest_auth])
        print("✅ Push successful.")
        
        # Verify on remote
        verify_cmd = ["ssh", f"{user}@{host}", "ls -l /pluribus/.pluribus/auth.json"]
        subprocess.check_call(verify_cmd)
        
    except subprocess.CalledProcessError as e:
        print(f"❌ Push failed: {e}")

def cmd_status():
    data = load_auth_data()
    print(f"\n📂 Auth File: {AUTH_FILE}")
    if not data or "providers" not in data:
        print("   No tokens found.")
        return

    print("\n🔐 Stored Tokens:")
    for provider, info in data.get("providers", {}).items():
        updated = time.ctime(info.get("updated_at", 0))
        token_preview = info.get("access_token", "")[:10] + "..."
        print(f"   - {provider.ljust(10)}: {token_preview} (Updated: {updated})")

def main():
    parser = argparse.ArgumentParser(description="LASER Token Auth Tool")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # auth url
    p_auth = subparsers.add_parser("auth", help="Generate OAuth URL")
    p_auth.add_argument("provider", choices=PROVIDERS.keys())

    # token paste
    p_token = subparsers.add_parser("token", help="Manually save a token")
    p_token.add_argument("provider", choices=PROVIDERS.keys())
    p_token.add_argument("token_string", nargs="?", help="The actual token or code (optional if --auto used)")
    p_token.add_argument("--auto", action="store_true", help="Auto-discover token from official CLI configs")

    # push
    p_push = subparsers.add_parser("push", help="Push auth.json to VPS")
    p_push.add_argument("host", help="VPS hostname or IP")
    p_push.add_argument("--user", default="root", help="SSH user")

    # status
    subparsers.add_parser("status", help="Show stored tokens")

    args = parser.parse_args()

    if args.command == "auth":
        cmd_auth_url(args.provider)
    elif args.command == "token":
        cmd_token_paste(args.provider, args.token_string, args.auto)
    elif args.command == "push":
        cmd_push(args.host, args.user)
    elif args.command == "status":
        cmd_status()

if __name__ == "__main__":
    main()
