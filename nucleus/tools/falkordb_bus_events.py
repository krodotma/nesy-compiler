#!/usr/bin/env python3
"""
##############################################################################
#                 PIA SHIM - DO NOT CHANGE OR DELETE                          #
#                                                                             #
# This shim delegates to /root/pib/pia/core/memory/falkordb_bus_events.py    #
# FalkorDB runs on port 6380 (not 6379) - this is set automatically.         #
# IRKG bus event integration for timeline queries.                            #
##############################################################################
"""
import subprocess
import sys
import os

# Set FalkorDB port (container maps 6380->6379)
if "FALKORDB_PORT" not in os.environ:
    os.environ["FALKORDB_PORT"] = "6380"

# Resolve PIA target
PIA_ROOT = os.environ.get("PIA_ROOT", "/root/pib/pia")
target_path = os.path.join(PIA_ROOT, "core/memory", os.path.basename(__file__))

if not os.path.exists(target_path):
    print(f"CRITICAL: PIA target not found: {target_path}", file=sys.stderr)
    sys.exit(1)

# Execute with full argument passthrough
sys.exit(subprocess.call([sys.executable, target_path] + sys.argv[1:], env=os.environ))
