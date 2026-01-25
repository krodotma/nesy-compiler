"""
Theia Screen Capture — Cross-platform screen capture for VPS.

Uses mss (fast) or PIL (fallback) for screen capture.
Outputs JPEG at configurable quality and scale.
"""

import base64
import io
from typing import Optional, Tuple

# Try mss first (faster), fallback to PIL
try:
    import mss
    HAS_MSS = True
except ImportError:
    HAS_MSS = False

try:
    from PIL import Image
    HAS_PIL = True
except ImportError:
    HAS_PIL = False


class ScreenCapture:
    """
    Cross-platform screen capture.
    
    On VPS with Xvfb, captures the virtual display.
    Downscales to 50% by default (matching VisionEye.tsx).
    Outputs JPEG at 0.7 quality by default.
    """
    
    def __init__(
        self,
        scale: float = 0.5,
        quality: int = 70,
        monitor: int = 0,
    ):
        self.scale = scale
        self.quality = quality
        self.monitor = monitor
        self._sct = None
    
    def _init_mss(self):
        """Lazy init mss screenshotter."""
        if self._sct is None and HAS_MSS:
            self._sct = mss.mss()
    
    def capture(self) -> Tuple[Optional[str], int, int]:
        """
        Capture screen and return (base64_jpeg, width, height).
        
        Returns (None, 0, 0) if capture fails.
        """
        if HAS_MSS:
            return self._capture_mss()
        elif HAS_PIL:
            return self._capture_pil()
        else:
            return None, 0, 0
    
    def _capture_mss(self) -> Tuple[Optional[str], int, int]:
        """Capture using mss (fast)."""
        try:
            self._init_mss()
            
            # Get monitor
            monitors = self._sct.monitors
            if self.monitor >= len(monitors):
                mon = monitors[0]  # Fallback to primary
            else:
                mon = monitors[self.monitor]
            
            # Capture
            screenshot = self._sct.grab(mon)
            
            # Convert to PIL for scaling/encoding
            if not HAS_PIL:
                return None, 0, 0
            
            img = Image.frombytes(
                "RGB",
                (screenshot.width, screenshot.height),
                screenshot.rgb,
            )
            
            return self._encode_image(img)
            
        except Exception as e:
            return None, 0, 0
    
    def _capture_pil(self) -> Tuple[Optional[str], int, int]:
        """Capture using PIL (slower, requires ImageGrab)."""
        try:
            from PIL import ImageGrab
            img = ImageGrab.grab()
            return self._encode_image(img)
        except Exception:
            return None, 0, 0
    
    def _encode_image(self, img: "Image.Image") -> Tuple[str, int, int]:
        """Scale and encode image to base64 JPEG."""
        # Downscale
        if self.scale != 1.0:
            new_size = (
                int(img.width * self.scale),
                int(img.height * self.scale),
            )
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Encode to JPEG
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=self.quality)
        buffer.seek(0)
        
        b64 = base64.b64encode(buffer.read()).decode("ascii")
        return b64, img.width, img.height
    
    def is_available(self) -> bool:
        """Check if screen capture is available."""
        return HAS_MSS or HAS_PIL


__all__ = ["ScreenCapture"]
