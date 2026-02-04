#!/usr/bin/env python3
import sys
import subprocess
import os

def main():
    """Wrapper to launch vLLM as a service."""
    # Default model if not provided
    # Users can override via args in service definition
    
    cmd = [sys.executable, "-m", "vllm.entrypoints.openai.api_server"]
    
    # Pass all arguments from the service registry to vLLM
    cmd.extend(sys.argv[1:])
    
    print(f"Starting vLLM: {' '.join(cmd)}", file=sys.stderr)
    
    env = os.environ.copy()
    # Ensure CUDA is visible if available, etc.
    
    try:
        subprocess.run(cmd, env=env, check=True)
    except KeyboardInterrupt:
        pass
    except subprocess.CalledProcessError as e:
        sys.exit(e.returncode)

if __name__ == "__main__":
    main()
