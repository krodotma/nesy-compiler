#!/bin/bash
# run_validation.sh - Run validation directly without tselector 

export PYTHONPATH=/Users/kroma/pluribus:$PYTHONPATH
cd /Users/kroma/pluribus
exec python3 nucleus/tools/distillation/validate_distillation.py
