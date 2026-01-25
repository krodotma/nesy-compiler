"""
Theia API Server — FastAPI implementation.

Exposes Theia capabilities over HTTP for integration with:
    - Dashboard (visual feedback)
    - Other agents (collaboration)
    - External tools (CI/CD, webhooks)
"""

import time
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel

from theia.vlm import VLMSpecialist, TheiaConfig, ScreenshotExample
from theia.capture import Frame

# Global instance (lazy init)
theia: Optional[VLMSpecialist] = None

def get_theia() -> VLMSpecialist:
    """Get or create Theia instance."""
    global theia
    if theia is None:
        config = TheiaConfig(vps_mode=True)
        theia = VLMSpecialist(config)
        theia.boot()
    return theia

# Data Models
class IngestRequest(BaseModel):
    frames: List[str]  # Base64 images
    source: str = "external"
    metadata: Dict[str, Any] = {}

class ActionRequest(BaseModel):
    intention: str
    context: Dict[str, Any] = {}

class InferRequest(BaseModel):
    image: Optional[str] = None
    prompt: str

# App Factory
def create_app() -> FastAPI:
    app = FastAPI(title="Theia Vision Agent", version="0.1.0")
    
    @app.on_event("startup")
    async def startup():
        get_theia()
        print("[Theia API] Startup complete")
    
    @app.on_event("shutdown")
    async def shutdown():
        if theia and theia.active:
            theia.shutdown()
            
    @app.get("/")
    async def root():
        return {"service": "theia", "status": "active"}
        
    @app.get("/status")
    async def status():
        """Get system status report."""
        agent = get_theia()
        return agent.status_report()
        
    @app.post("/v1/theia/ingest")
    async def ingest(req: IngestRequest, background_tasks: BackgroundTasks):
        """Ingest visual frames."""
        # For now, we just log size. Real impl would push to ring buffer.
        # Background task to avoid blocking
        background_tasks.add_task(process_ingest, req)
        return {"status": "accepted", "count": len(req.frames)}
        
    @app.post("/v1/theia/act")
    async def act(req: ActionRequest):
        """Execute an action."""
        agent = get_theia()
        result = agent.act(req.intention)
        return result
        
    @app.post("/v1/theia/infer")
    async def infer(req: InferRequest):
        """VLM inference (stub/mock for now)."""
        # Connect to Ollama here ideally
        return {
            "response": "Inference stub response",
            "usage": {"tokens": 10}
        }
        
    return app

def process_ingest(req: IngestRequest):
    """Process ingested frames in background."""
    # This would simulate 'seeing' the ingested frames
    pass

def run_server(host: str = "0.0.0.0", port: int = 8000):
    """Run server programmatically."""
    import uvicorn
    uvicorn.run(create_app(), host=host, port=port)

if __name__ == "__main__":
    run_server()
