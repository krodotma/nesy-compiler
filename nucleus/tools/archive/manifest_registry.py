"""
manifest_registry.py - App-of-Apps MANIFEST Registry

Provides the source of truth for all available applications,
their routes, and metadata for the world_router.

Phase 6.1 - App Routing Mesh
"""

from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
import yaml


@dataclass
class AppManifest:
    """Single app manifest entry."""
    name: str
    route: str
    description: str
    version: str = "1.0.0"
    enabled: bool = True
    ring: int = 2  # Default ring level
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    

class ManifestRegistry:
    """
    Centralized registry for all app manifests.
    Source of truth for world_router routing decisions.
    """
    
    def __init__(self, manifest_path: Optional[Path] = None):
        self.manifests: Dict[str, AppManifest] = {}
        self.manifest_path = manifest_path or Path("/pluribus/MANIFEST.yaml")
        self._load_manifests()
    
    def _load_manifests(self):
        """Load manifests from MANIFEST.yaml or defaults."""
        if self.manifest_path.exists():
            try:
                with open(self.manifest_path) as f:
                    data = yaml.safe_load(f)
                    apps = data.get("apps", [])
                    for app in apps:
                        manifest = AppManifest(
                            name=app.get("name", "unknown"),
                            route=app.get("route", "/"),
                            description=app.get("description", ""),
                            version=app.get("version", "1.0.0"),
                            enabled=app.get("enabled", True),
                            ring=app.get("ring", 2),
                            tags=app.get("tags", []),
                            metadata=app.get("metadata", {}),
                        )
                        self.manifests[manifest.name] = manifest
            except Exception as e:
                print(f"Error loading manifests: {e}")
                self._load_defaults()
        else:
            self._load_defaults()
    
    def _load_defaults(self):
        """Load default app manifests."""
        defaults = [
            AppManifest(
                name="dashboard",
                route="/",
                description="Main Pluribus Dashboard",
                tags=["core", "ui"],
                ring=0,
            ),
            AppManifest(
                name="auralux",
                route="/studio/auralux",
                description="Neural Voice Synthesis Studio",
                tags=["voice", "ai", "auralux"],
                ring=1,
            ),
            AppManifest(
                name="art",
                route="/art",
                description="Art Gallery & Shader Playground",
                tags=["art", "shaders", "visual"],
                ring=1,
            ),
            AppManifest(
                name="docs",
                route="/docs",
                description="Documentation Portal",
                tags=["docs", "reference"],
                ring=2,
            ),
        ]
        for manifest in defaults:
            self.manifests[manifest.name] = manifest
    
    def get(self, name: str) -> Optional[AppManifest]:
        """Get app manifest by name."""
        return self.manifests.get(name)
    
    def get_by_route(self, route: str) -> Optional[AppManifest]:
        """Get app manifest by route prefix."""
        for manifest in self.manifests.values():
            if route.startswith(manifest.route):
                return manifest
        return None
    
    def list_apps(self, enabled_only: bool = True) -> List[AppManifest]:
        """List all registered apps."""
        apps = list(self.manifests.values())
        if enabled_only:
            apps = [a for a in apps if a.enabled]
        return sorted(apps, key=lambda a: a.route)
    
    def register(self, manifest: AppManifest) -> bool:
        """Register a new app manifest."""
        if manifest.name in self.manifests:
            return False
        self.manifests[manifest.name] = manifest
        return True
    
    def unregister(self, name: str) -> bool:
        """Unregister an app by name."""
        if name in self.manifests:
            del self.manifests[name]
            return True
        return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Export registry as dict for serialization."""
        return {
            "apps": [
                {
                    "name": m.name,
                    "route": m.route,
                    "description": m.description,
                    "version": m.version,
                    "enabled": m.enabled,
                    "ring": m.ring,
                    "tags": m.tags,
                    "metadata": m.metadata,
                }
                for m in self.manifests.values()
            ]
        }


# Singleton instance
_registry: Optional[ManifestRegistry] = None


def get_registry() -> ManifestRegistry:
    """Get or create the singleton registry instance."""
    global _registry
    if _registry is None:
        _registry = ManifestRegistry()
    return _registry


if __name__ == "__main__":
    # Self-test
    reg = get_registry()
    print(f"Loaded {len(reg.list_apps())} apps:")
    for app in reg.list_apps():
        print(f"  - {app.name}: {app.route} (ring {app.ring})")
