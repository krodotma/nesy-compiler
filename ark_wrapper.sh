#!/bin/bash
# ARK CLI Wrapper
# Ensures correct PYTHONPATH for nucleus modules regardless of CWD

# Get the directory where this script is located
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Set PYTHONPATH to the repo root
export PYTHONPATH="$DIR"

# Execute the CLI
python3 "$DIR/nucleus/ark/cli.py" "$@"
