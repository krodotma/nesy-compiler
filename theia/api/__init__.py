"""
Theia API Layer — REST Interface.

Provides endpoints for:
    - Visual ingestion (/ingest)
    - Action execution (/act)
    - VLM inference (/infer)
    - System status (/status)
"""

from theia.api.server import create_app, run_server

__all__ = ["create_app", "run_server"]
