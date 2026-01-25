"""
Theia Deployment Daemon.

Runs the VLM Specialist as a persistent service.
"""

import time
import signal
import sys
from theia.vlm.specialist import VLMSpecialist, TheiaConfig

def run_daemon():
    """Main daemon loop."""
    config = TheiaConfig(vps_mode=True)
    theia = VLMSpecialist(config)
    
    # Handle signals
    def signal_handler(sig, frame):
        print("\n[Theia] Stopping...")
        theia.shutdown()
        sys.exit(0)
        
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    # Boot
    status = theia.boot()
    print(f"[Theia] Started. Status: {status}")
    
    # Main Loop
    while True:
        try:
            perception = theia.perceive()
            if perception["has_input"]:
                print(f"[Theia] Input detected. processing...")
                # Logic to determine action would go here
            
            time.sleep(1.0) # 1Hz heartbeat
            
        except Exception as e:
            print(f"[Theia] Error in loop: {e}")
            time.sleep(5.0)

if __name__ == "__main__":
    run_daemon()
