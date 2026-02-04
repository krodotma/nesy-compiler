#!/usr/bin/env python3
"""
Pluribus Studio Server (Backend)
================================

Serves the Neurosymbolic Flow Editor API and WebSockets.
Bridges the static `InferCell` state to a live, reactive XYFlow graph.

Port: 9202 (Studio)
"""
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

# Third-party imports (assumed available or mockable for scaffold)
try:
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.middleware.cors import CORSMiddleware
    import uvicorn
except ImportError:
    print("FastAPI/Uvicorn not found. Please install: pip install fastapi uvicorn")
    sys.exit(1)

sys.path.append(str(Path(__file__).resolve().parents[2]))
from nucleus.tools.infercell_manager import InferCellManager

app = FastAPI(title="Pluribus Studio")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: dict):
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except Exception:
                pass

manager = ConnectionManager()
cell_manager = InferCellManager(Path(os.environ.get("PLURIBUS_ROOT", "/pluribus")))

def cell_to_node(cell: Any) -> dict:
    """Convert an InferCell to an XYFlow Node."""
    # Heuristic layout (simple vertical spacing based on time)
    # In a real app, we'd use dagre or elkjs on the frontend for layout.
    return {
        "id": cell.cell_id,
        "type": "inferNode", # Custom node type
        "position": {"x": 0, "y": 0}, # Placeholder
        "data": {
            "trace_id": cell.trace_id,
            "state": cell.state,
            "reason": cell.fork_point.reason,
            "timestamp": cell.fork_point.timestamp
        }
    }

def cell_to_edge(cell: Any) -> dict | None:
    """Convert parent->child relationship to an XYFlow Edge."""
    if not cell.parent_trace_id:
        return None
    # We need to resolve parent_trace_id to parent_cell_id
    parent_cell = cell_manager.get_cell(cell.parent_trace_id)
    if not parent_cell:
        return None
    
    return {
        "id": f"e-{parent_cell.cell_id}-{cell.cell_id}",
        "source": parent_cell.cell_id,
        "target": cell.cell_id,
        "type": "smoothstep",
        "animated": cell.state == "active"
    }

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        # 1. Send Initial State
        cells = cell_manager.list_cells()
        nodes = [cell_to_node(c) for c in cells]
        edges = [e for e in (cell_to_edge(c) for c in cells) if e]
        
        await websocket.send_json({
            "type": "init",
            "payload": {"nodes": nodes, "edges": edges}
        })

        # 2. Listen for commands (Generative UI requests)
        while True:
            data = await websocket.receive_json()
            # Handle client-side graph edits here (e.g. "fork_cell")
            if data.get("type") == "fork":
                # Call manager.fork()
                pass
    except WebSocketDisconnect:
        manager.disconnect(websocket)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9202)
