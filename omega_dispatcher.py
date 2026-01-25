import sys
import os
import argparse
import logging
from typing import Dict, List, Optional

# Add current directory to sys.path to ensure local imports work
sys.path.append(os.getcwd())

from nucleus.tools.agent_bus import bus

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("OmegaDispatcher")

class OmegaDispatcher:
    def __init__(self, manifest_path: str = "MANIFEST.yaml"):
        self.manifest_path = manifest_path
        self.manifest = self._load_manifest()
        self.domain_map = self._build_domain_map()

    def _load_manifest(self) -> Dict:
        """Loads the MANIFEST.yaml file with a manual fallback if PyYAML is missing."""
        if not os.path.exists(self.manifest_path):
            logger.error(f"Manifest not found at {self.manifest_path}")
            return {}
            
        try:
            import yaml
            with open(self.manifest_path, 'r') as f:
                return yaml.safe_load(f)
        except ImportError:
            logger.warning("PyYAML not found, using manual fallback parser.")
            return self._manual_yaml_parse(self.manifest_path)
        except Exception as e:
            logger.error(f"Failed to load manifest: {e}")
            return {}

    def _manual_yaml_parse(self, path: str) -> Dict:
        """
        Extremely basic YAML parser for the specific App-of-Apps manifest structure.
        Does not support full YAML spec, just enough to bootstrap.
        """
        data = {'apps': {}}
        current_app = None
        
        with open(path, 'r') as f:
            for line in f:
                line = line.rstrip()
                if not line or line.startswith('#'): continue
                
                stripped = line.strip()
                
                if line.startswith('apps:'):
                    continue
                    
                # Detect app definition (indented, ends with :)
                if line.startswith('  ') and line.endswith(':') and not line.startswith('    '):
                    current_app = stripped[:-1]
                    data['apps'][current_app] = {'domains': []}
                    continue
                    
                # Detect properties
                if current_app and line.startswith('    '):
                    key, val = stripped.split(':', 1)
                    val = val.strip()
                    if key == 'domains':
                        # Parse [a, b, c]
                        val = val.strip('[]')
                        domains = [d.strip() for d in val.split(',')]
                        data['apps'][current_app]['domains'] = domains
                    else:
                        data['apps'][current_app][key] = val
                        
        return data

    def _build_domain_map(self) -> Dict[str, str]:
        """Maps domains to their parent app for quick routing."""
        mapping = {}
        if not self.manifest or 'apps' not in self.manifest:
            return mapping
        
        for app_name, config in self.manifest['apps'].items():
            for domain in config.get('domains', []):
                mapping[domain] = app_name
        return mapping

    def extract_domain(self, prompt: str) -> str:
        """
        Simple keyword heuristic to infer domain from prompt.
        In a real implementation, this would use a more sophisticated classifier.
        """
        prompt = prompt.lower()
        
        # Simple keyword mapping based on architecture doc
        keywords = {
            'kroma': ['dashboard', 'frontend', 'ui', 'button', 'css'],
            'pqc': ['security', 'ring', 'crypto', 'compartment'],
            'rag': ['knowledge', 'vector', 'embedding'],
            'cinema': ['video', 'art', 'render'],
            'pluribus': ['bus', 'agent', 'dispatcher']
        }
        
        for domain, keys in keywords.items():
            if any(k in prompt for k in keys):
                return domain
        
        return 'pluribus'  # Default to core

    def route_task(self, task_description: str):
        """Routes a task to the appropriate app/agent via the bus."""
        domain = self.extract_domain(task_description)
        target_app = self.domain_map.get(domain, "pluribus-core")
        
        logger.info(f"Routing task '{task_description}' to domain: {domain} (App: {target_app})")
        
        # In a real system, this would trigger a PAIP isolation or sub-agent spawn
        # For now, we publish to the bus
        
        payload = {
            "task": task_description,
            "target_domain": domain,
            "target_app": target_app,
            "clearance_check": "BYPASSED (Dev Mode)" 
        }
        
        bus.publish("task.dispatch", payload)
        return payload

def main():
    parser = argparse.ArgumentParser(description="Omega Dispatcher - Pluribus Task Router")
    parser.add_argument("--validate-manifest", action="store_true", help="Validate the MANIFEST.yaml structure")
    parser.add_argument("--dispatch", type=str, help="Dispatch a task description string")
    
    args = parser.parse_args()
    
    dispatcher = OmegaDispatcher()
    
    if args.validate_manifest:
        if dispatcher.manifest:
            print(f"✅ Manifest loaded successfully. {len(dispatcher.manifest.get('apps', {}))} apps configured.")
            sys.exit(0)
        else:
            print("❌ Validation failed.")
            sys.exit(1)
            
    if args.dispatch:
        dispatcher.route_task(args.dispatch)

if __name__ == "__main__":
    main()
