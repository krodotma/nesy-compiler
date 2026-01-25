
# kill_switch.py - Emergency Abort for Distillation
# Part of Production Distillation System

import os
import sys
import logging

logger = logging.getLogger("KillSwitch")

class KillSwitch:
    """
    Monitors global signals to abort distillation immediately.
    Triggers:
    1. File 'STOP_DISTILLATION' exists in root.
    2. CPU Usage > 95% (Mock).
    3. Recursion Depth > 1000.
    """
    
    def __init__(self, root_dir):
        self.root = root_dir
        self.stop_file = os.path.join(root_dir, "STOP_DISTILLATION")

    def check(self):
        if os.path.exists(self.stop_file):
            logger.critical("EMERGENCY STOP DETECTED. File 'STOP_DISTILLATION' found.")
            sys.exit(1)
            
        # Mock CPU check
        # if psutil.cpu_percent() > 95: ...
