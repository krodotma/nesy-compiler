#!/bin/bash
# run_ark_tests.sh - Run ARK operational tests

set -e

export PYTHONPATH=/Users/kroma/pluribus:$PYTHONPATH
cd /Users/kroma/pluribus

echo "Running ARK Operational Tests..."
python3 nucleus/ark/tests/test_ark_operational.py
