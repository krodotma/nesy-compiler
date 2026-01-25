import os
import json
import time
import hashlib
import logging
import argparse

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - DKIN - %(levelname)s - %(message)s')
logger = logging.getLogger("DKIN_Ledger")

class DKINLedger:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(DKINLedger, cls).__new__(cls)
            cls._instance._init_paths()
        return cls._instance

    def _init_paths(self):
        # Derive DKIN root from PLURIBUS_BUS_DIR parent or default
        bus_dir = os.environ.get("PLURIBUS_BUS_DIR", ".pluribus/bus")
        # Go up one level from bus to get .pluribus root, then into dkin
        self.pluribus_root = os.path.dirname(bus_dir)
        self.dkin_dir = os.path.join(self.pluribus_root, "dkin")
        self.ledger_file = os.path.join(self.dkin_dir, "ledger.jsonl")

        if not os.path.exists(self.dkin_dir):
            try:
                os.makedirs(self.dkin_dir, exist_ok=True)
                logger.info(f"Created DKIN directory: {self.dkin_dir}")
            except Exception as e:
                logger.error(f"Failed to create DKIN dir: {e}")

    def _calculate_hash(self, data: dict, previous_hash: str) -> str:
        """Calculates SHA256 hash of the entry + previous hash for chain integrity."""
        # Canonical JSON string representation
        payload = json.dumps(data, sort_keys=True)
        content = f"{previous_hash}|{payload}"
        return hashlib.sha256(content.encode('utf-8')).hexdigest()

    def _get_last_hash(self) -> str:
        """Reads the last line of the ledger to get the previous hash."""
        if not os.path.exists(self.ledger_file):
            return "0" * 64  # Genesys Block Hash
        
        try:
            with open(self.ledger_file, 'rb') as f:
                try:
                    f.seek(-2, os.SEEK_END)
                    while f.read(1) != b'\n':
                        f.seek(-2, os.SEEK_CUR)
                except OSError:
                    f.seek(0)
                last_line = f.readline().decode()
                if not last_line:
                    return "0" * 64
                
                try:
                    entry = json.loads(last_line)
                    return entry.get('hash', "0" * 64)
                except json.JSONDecodeError:
                    return "0" * 64
        except Exception:
            return "0" * 64

    def record(self, entry_type: str, data: dict):
        """Records a new entry to the immutable ledger."""
        timestamp = time.time()
        prev_hash = self._get_last_hash()
        
        entry = {
            "type": entry_type,
            "timestamp": timestamp,
            "data": data,
            "prev_hash": prev_hash
        }
        
        # Calculate integrity hash for this entry
        entry['hash'] = self._calculate_hash(entry, prev_hash)
        
        try:
            with open(self.ledger_file, 'a') as f:
                f.write(json.dumps(entry) + "\n")
            logger.info(f"Recorded {entry_type} [{entry['hash'][:8]}]")
            return entry['hash']
        except Exception as e:
            logger.error(f"Write failed: {e}")
            return None

ledger = DKINLedger()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DKIN Ledger Operator")
    parser.add_argument("--init", action="store_true", help="Initialize DKIN storage")
    parser.add_argument("--record", type=str, help="Record an entry 'type:json_data'")
    
    args = parser.parse_args()
    
    if args.init:
        print(f"DKIN Initialized at {ledger.ledger_file}")
    
    if args.record:
        try:
            etype, payload = args.record.split(":", 1)
            data = json.loads(payload)
            ledger.record(etype, data)
        except Exception as e:
            print(f"Error: {e}")
