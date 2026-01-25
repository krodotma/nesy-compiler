import logging
import json
import os
import time
from typing import Callable, Dict, List, Any
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger("AgentBus")

class AgentBus:
    """
    A file-backed event bus for asynchronous multi-agent communication.
    Canonical Location: nucleus/tools/agent_bus.py
    Standard: Veteran Protocol (Protocol v2)
    """
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(AgentBus, cls).__new__(cls)
            cls._instance.subscribers = {}
            cls._instance._init_paths()
            cls._instance._load_specs()
        return cls._instance

    def _init_paths(self):
        # Respect the PLURIBUS_BUS_DIR environment variable or fallback to canonical default
        self.bus_dir = os.environ.get("PLURIBUS_BUS_DIR", os.path.join(os.getcwd(), ".pluribus/bus"))
        self.bus_file = os.path.join(self.bus_dir, "events.ndjson")
        
        # Ensure directory exists
        if not os.path.exists(self.bus_dir):
            try:
                os.makedirs(self.bus_dir, exist_ok=True)
            except Exception as e:
                logger.error(f"Failed to create bus directory {self.bus_dir}: {e}")

        # Ensure file exists
        if not os.path.exists(self.bus_file):
            try:
                with open(self.bus_file, 'w') as f:
                    pass
            except Exception as e:
                logger.error(f"Failed to create bus file {self.bus_file}: {e}")
                
        logger.info(f"AgentBus initialized at {self.bus_file}")

    def _load_specs(self):
        """Load valid topics from semops.json"""
        self.valid_topics = set()
        spec_path = os.path.join(os.getcwd(), "nucleus/specs/semops.json")
        if os.path.exists(spec_path):
            try:
                with open(spec_path, 'r') as f:
                    data = json.load(f)
                    self.valid_topics = set(data.get("topics", {}).keys())
            except Exception as e:
                logger.warning(f"Failed to load semops.json: {e}")
        else:
            logger.warning(f"semops.json not found at {spec_path}, validation disabled.")

    def subscribe(self, topic: str, callback: Callable[[Any], None]):
        """Subscribe a callback function to a specific topic (Runtime only)."""
        if topic not in self.subscribers:
            self.subscribers[topic] = []
        self.subscribers[topic].append(callback)

    def publish(self, topic: str, data: Any):
        """Publish data to a topic, persisting it to disk and notifying local subscribers."""
        if self.valid_topics and topic not in self.valid_topics:
            logger.warning(f"Topic '{topic}' is not in semops.json registry.")

        event = {
            "timestamp": time.time(),
            "topic": topic,
            "data": data
        }
        
        # 1. Persist to disk (IPC)
        try:
            with open(self.bus_file, 'a') as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            logger.error(f"Failed to write to bus file {self.bus_file}: {e}")

        # 2. Notify local runtime subscribers
        if topic in self.subscribers:
            for callback in self.subscribers[topic]:
                try:
                    callback(data)
                except Exception as e:
                    logger.error(f"Error in subscriber callback for {topic}: {e}")
        
        logger.info(f"Published to {topic}: {str(data)[:50]}...")

    def read_recent_events(self, limit: int = 10) -> List[Dict]:
        """Read the most recent events from the bus file."""
        events = []
        if os.path.exists(self.bus_file):
            try:
                with open(self.bus_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines[-limit:]:
                        try:
                            events.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
            except Exception as e:
                logger.error(f"Failed to read bus file: {e}")
        return events

# Global instance
bus = AgentBus()
