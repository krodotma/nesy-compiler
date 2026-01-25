"""
Theia Vision WebSocket Bridge for Real-Time Frame Streaming.

Provides bidirectional communication:
- Client → Server: Frame data, commands
- Server → Client: Inference results, synthesis state

PLURIBUS v1 compliant with A2A heartbeat protocol.
"""

import asyncio
import json
import time
import uuid
from typing import Dict, Any, Set, Optional
from dataclasses import dataclass, field

# WebSocket server (using asyncio)
try:
    import websockets
    from websockets.server import WebSocketServerProtocol
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    print("[ws_bridge] websockets not installed, using fallback")

# Bus event emission
BUS_PATH = "/pluribus/.pluribus/bus/events.ndjson"
PROTO = "PLURIBUS v1"

def emit_bus_event(topic: str, kind: str, level: str, data: Dict[str, Any], actor: str = "theia_ws") -> None:
    """Emit event to PLURIBUS bus."""
    event = {
        "id": str(uuid.uuid4()),
        "ts": time.time(),
        "iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "topic": topic,
        "kind": kind,
        "level": level,
        "actor": actor,
        "proto": PROTO,
        "data": data
    }
    try:
        with open(BUS_PATH, "a") as f:
            f.write(json.dumps(event) + "\n")
    except Exception as e:
        print(f"[ws_bridge] Bus emit failed: {e}")


@dataclass
class VisionClient:
    """Connected client state."""
    client_id: str
    websocket: Any  # WebSocketServerProtocol
    connected_at: float
    last_heartbeat: float
    frame_count: int = 0
    inference_count: int = 0


class VisionWebSocketBridge:
    """
    WebSocket bridge for Theia Vision.
    
    Handles:
    - Frame ingestion from clients
    - Inference result distribution
    - A2A heartbeat protocol (5 min intervals)
    - Synthesis state broadcasting
    """
    
    def __init__(self, host: str = "0.0.0.0", port: int = 8091):
        self.host = host
        self.port = port
        self.clients: Dict[str, VisionClient] = {}
        self.frame_buffer: list = []
        self.buffer_capacity = 60
        self.running = False
        self._heartbeat_interval = 300  # 5 minutes (A2A protocol)
        self._inference_queue: asyncio.Queue = None
        
    async def start(self) -> None:
        """Start WebSocket server."""
        if not WEBSOCKETS_AVAILABLE:
            print("[ws_bridge] websockets library not available")
            return
            
        self._inference_queue = asyncio.Queue()
        self.running = True
        
        emit_bus_event(
            topic="theia.ws.start",
            kind="event",
            level="info",
            data={"host": self.host, "port": self.port}
        )
        
        async with websockets.serve(self._handle_client, self.host, self.port):
            print(f"[ws_bridge] Server started on ws://{self.host}:{self.port}")
            
            # Start heartbeat task
            heartbeat_task = asyncio.create_task(self._heartbeat_loop())
            
            try:
                await asyncio.Future()  # Run forever
            finally:
                heartbeat_task.cancel()
                self.running = False
    
    async def _handle_client(self, websocket: 'WebSocketServerProtocol') -> None:
        """Handle individual client connection."""
        client_id = str(uuid.uuid4())[:8]
        client = VisionClient(
            client_id=client_id,
            websocket=websocket,
            connected_at=time.time(),
            last_heartbeat=time.time()
        )
        self.clients[client_id] = client
        
        emit_bus_event(
            topic="vision.ws.connect",
            kind="event",
            level="info",
            data={"client_id": client_id}
        )
        
        # Send welcome message
        await websocket.send(json.dumps({
            "type": "welcome",
            "client_id": client_id,
            "proto": PROTO,
            "capabilities": ["frame_ingest", "inference", "synthesis"]
        }))
        
        try:
            async for message in websocket:
                await self._process_message(client, message)
        except Exception as e:
            print(f"[ws_bridge] Client {client_id} error: {e}")
        finally:
            del self.clients[client_id]
            emit_bus_event(
                topic="vision.ws.disconnect",
                kind="event",
                level="info",
                data={"client_id": client_id, "frames": client.frame_count}
            )
    
    async def _process_message(self, client: VisionClient, raw: str) -> None:
        """Process incoming WebSocket message."""
        try:
            msg = json.loads(raw)
            msg_type = msg.get("type", "unknown")
            
            if msg_type == "frame":
                await self._handle_frame(client, msg)
            elif msg_type == "heartbeat":
                await self._handle_heartbeat(client, msg)
            elif msg_type == "analyze":
                await self._handle_analyze(client, msg)
            elif msg_type == "ping":
                await client.websocket.send(json.dumps({"type": "pong", "ts": time.time()}))
            else:
                await client.websocket.send(json.dumps({"type": "error", "message": f"Unknown type: {msg_type}"}))
                
        except json.JSONDecodeError:
            await client.websocket.send(json.dumps({"type": "error", "message": "Invalid JSON"}))
    
    async def _handle_frame(self, client: VisionClient, msg: Dict[str, Any]) -> None:
        """Handle incoming frame data."""
        frame_data = msg.get("data")
        if not frame_data:
            return
            
        # Add to buffer
        if len(self.frame_buffer) >= self.buffer_capacity:
            self.frame_buffer.pop(0)
        self.frame_buffer.append({
            "data": frame_data,
            "ts": time.time(),
            "client_id": client.client_id,
            "meta": msg.get("meta", {})
        })
        
        client.frame_count += 1
        
        # Acknowledge
        await client.websocket.send(json.dumps({
            "type": "frame_ack",
            "count": client.frame_count,
            "buffer_size": len(self.frame_buffer)
        }))
        
        # Emit bus event (throttled - every 10 frames)
        if client.frame_count % 10 == 0:
            emit_bus_event(
                topic="vision.ws.frames",
                kind="metric",
                level="debug",
                data={"client_id": client.client_id, "count": client.frame_count}
            )
    
    async def _handle_heartbeat(self, client: VisionClient, msg: Dict[str, Any]) -> None:
        """Handle A2A heartbeat."""
        client.last_heartbeat = time.time()
        await client.websocket.send(json.dumps({
            "type": "heartbeat_ack",
            "ts": time.time(),
            "a2a": True
        }))
    
    async def _handle_analyze(self, client: VisionClient, msg: Dict[str, Any]) -> None:
        """Handle analysis request on buffered frames."""
        frames_to_analyze = msg.get("count", 10)
        frames = self.frame_buffer[-frames_to_analyze:]
        
        # Mock analysis (real impl would call VLMSpecialist)
        result = {
            "type": "inference_result",
            "frames_analyzed": len(frames),
            "synthesis": {
                "pattern": "visual_context",
                "confidence": 0.85,
                "grammar": "S -> VPrimitive | VComposite",
                "tokens": ["screen", "window", "ui_element"]
            },
            "ts": time.time()
        }
        
        client.inference_count += 1
        await client.websocket.send(json.dumps(result))
        
        emit_bus_event(
            topic="vision.ws.inference",
            kind="event",
            level="info",
            data={"client_id": client.client_id, "frames": len(frames)}
        )
    
    async def _heartbeat_loop(self) -> None:
        """Send periodic heartbeats to all clients (A2A protocol)."""
        while self.running:
            await asyncio.sleep(self._heartbeat_interval)
            
            for client in list(self.clients.values()):
                try:
                    await client.websocket.send(json.dumps({
                        "type": "a2a_heartbeat",
                        "ts": time.time(),
                        "buffer_size": len(self.frame_buffer),
                        "clients": len(self.clients)
                    }))
                except Exception:
                    pass  # Client may have disconnected
            
            emit_bus_event(
                topic="a2a.heartbeat",
                kind="metric",
                level="debug",
                data={"clients": len(self.clients), "buffer": len(self.frame_buffer)},
                actor="theia_ws"
            )
    
    async def broadcast_synthesis(self, synthesis_state: Dict[str, Any]) -> None:
        """Broadcast synthesis state to all connected clients."""
        msg = json.dumps({
            "type": "synthesis_update",
            "state": synthesis_state,
            "ts": time.time()
        })
        
        for client in list(self.clients.values()):
            try:
                await client.websocket.send(msg)
            except Exception:
                pass


def run_bridge(host: str = "0.0.0.0", port: int = 8091) -> None:
    """Run WebSocket bridge server."""
    if not WEBSOCKETS_AVAILABLE:
        print("[ws_bridge] Install websockets: pip install websockets")
        return
        
    bridge = VisionWebSocketBridge(host=host, port=port)
    asyncio.run(bridge.start())


if __name__ == "__main__":
    run_bridge()
