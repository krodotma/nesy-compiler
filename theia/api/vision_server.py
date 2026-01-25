"""
Theia Vision API Server with /v1/eyes/ingest endpoint.

Receives frames from VisionEye.tsx and forwards to VLMSpecialist.
PLURIBUS v1 compliant with bus event emission.
"""

import json
import time
import uuid
import base64
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from http.server import HTTPServer, BaseHTTPRequestHandler
from io import BytesIO
import threading

# Bus event emission
BUS_PATH = "/pluribus/.pluribus/bus/events.ndjson"
PROTO = "PLURIBUS v1"

def emit_bus_event(topic: str, kind: str, level: str, data: Dict[str, Any], actor: str = "theia_api") -> None:
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
        print(f"[Theia API] Bus emit failed: {e}")


@dataclass
class FrameBuffer:
    """Ring buffer for incoming frames."""
    capacity: int = 60
    frames: List[Dict[str, Any]] = field(default_factory=list)
    
    def add(self, frame_data: str, timestamp: float, meta: Optional[Dict] = None) -> int:
        """Add frame to buffer, return buffer size."""
        if len(self.frames) >= self.capacity:
            self.frames.pop(0)
        self.frames.append({
            "data": frame_data,
            "ts": timestamp,
            "meta": meta or {}
        })
        return len(self.frames)
    
    def get_latest(self, n: int = 1) -> List[Dict]:
        """Get latest N frames."""
        return self.frames[-n:]
    
    def clear(self) -> None:
        """Clear buffer."""
        self.frames.clear()


# Global frame buffer
FRAME_BUFFER = FrameBuffer()


class TheiaAPIHandler(BaseHTTPRequestHandler):
    """HTTP request handler for Theia Vision API."""
    
    def _send_json(self, status: int, data: Dict[str, Any]) -> None:
        """Send JSON response."""
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))
    
    def do_OPTIONS(self) -> None:
        """Handle CORS preflight."""
        self._send_json(200, {"status": "ok"})
    
    def do_GET(self) -> None:
        """Handle GET requests."""
        if self.path == "/v1/health":
            self._send_json(200, {
                "status": "healthy",
                "proto": PROTO,
                "buffer_size": len(FRAME_BUFFER.frames),
                "timestamp": time.time()
            })
        elif self.path == "/v1/eyes/buffer":
            self._send_json(200, {
                "count": len(FRAME_BUFFER.frames),
                "capacity": FRAME_BUFFER.capacity,
                "latest_ts": FRAME_BUFFER.frames[-1]["ts"] if FRAME_BUFFER.frames else None
            })
        else:
            self._send_json(404, {"error": "Not found"})
    
    def do_POST(self) -> None:
        """Handle POST requests."""
        if self.path == "/v1/eyes/ingest":
            self._handle_ingest()
        elif self.path == "/v1/eyes/analyze":
            self._handle_analyze()
        else:
            self._send_json(404, {"error": "Not found"})
    
    def _handle_ingest(self) -> None:
        """Handle /v1/eyes/ingest - receive frames from VisionEye."""
        try:
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length)
            payload = json.loads(body.decode("utf-8"))
            
            frames = payload.get("frames", [])
            timestamp = payload.get("timestamp", time.time())
            meta = payload.get("meta", {})
            
            # Add frames to buffer
            for frame_data in frames:
                FRAME_BUFFER.add(frame_data, timestamp, meta)
            
            # Emit bus event
            emit_bus_event(
                topic="vision.ingest.received",
                kind="event",
                level="info",
                data={
                    "count": len(frames),
                    "buffer_size": len(FRAME_BUFFER.frames),
                    "timestamp": timestamp
                }
            )
            
            # Process with VLMSpecialist (if available)
            analysis_result = self._process_frames(frames)
            
            self._send_json(200, {
                "status": "ingested",
                "count": len(frames),
                "buffer_size": len(FRAME_BUFFER.frames),
                "analysis": analysis_result
            })
            
        except json.JSONDecodeError:
            self._send_json(400, {"error": "Invalid JSON"})
        except Exception as e:
            emit_bus_event(
                topic="vision.ingest.error",
                kind="alert",
                level="error",
                data={"error": str(e)}
            )
            self._send_json(500, {"error": str(e)})
    
    def _handle_analyze(self) -> None:
        """Handle /v1/eyes/analyze - trigger analysis on buffered frames."""
        try:
            frames = FRAME_BUFFER.get_latest(10)
            if not frames:
                self._send_json(400, {"error": "No frames in buffer"})
                return
            
            result = self._process_frames([f["data"] for f in frames])
            
            self._send_json(200, {
                "status": "analyzed",
                "frame_count": len(frames),
                "result": result
            })
            
        except Exception as e:
            self._send_json(500, {"error": str(e)})
    
    def _process_frames(self, frames: List[str]) -> Dict[str, Any]:
        """Process frames with VLMSpecialist."""
        try:
            # Lazy import to avoid circular deps
            from theia.vlm import VLMSpecialist, TheiaConfig
            
            config = TheiaConfig(vps_mode=True, enable_swarm=True)
            specialist = VLMSpecialist(config)
            
            # Mock processing (real impl would call specialist.analyze_frames)
            return {
                "processed": len(frames),
                "quality_score": 0.85,  # Mock
                "synthesis": "Visual analysis pending full VLM integration"
            }
        except ImportError:
            return {
                "processed": len(frames),
                "quality_score": 0.5,
                "synthesis": "VLMSpecialist not available - mock response"
            }


def run_server(host: str = "0.0.0.0", port: int = 8080) -> None:
    """Run Theia API server."""
    server = HTTPServer((host, port), TheiaAPIHandler)
    print(f"[Theia API] Server starting on {host}:{port}")
    
    emit_bus_event(
        topic="theia.api.start",
        kind="event",
        level="info",
        data={"host": host, "port": port, "proto": PROTO}
    )
    
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[Theia API] Shutting down...")
        emit_bus_event(
            topic="theia.api.stop",
            kind="event",
            level="info",
            data={"reason": "keyboard_interrupt"}
        )
        server.shutdown()


if __name__ == "__main__":
    run_server()
