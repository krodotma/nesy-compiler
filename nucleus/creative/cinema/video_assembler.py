"""
Video Assembler Module
======================

Assembles video frames into final video output.
Handles encoding, audio sync, and export.
"""

from __future__ import annotations

import json
import subprocess
import shutil
from dataclasses import dataclass, field
from enum import Enum, auto
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple
from datetime import datetime


class VideoCodec(Enum):
    """Video encoding codecs."""
    H264 = "libx264"
    H265 = "libx265"
    VP9 = "libvpx-vp9"
    AV1 = "libaom-av1"
    PRORES = "prores_ks"
    DNxHD = "dnxhd"


class AudioCodec(Enum):
    """Audio encoding codecs."""
    AAC = "aac"
    MP3 = "libmp3lame"
    OPUS = "libopus"
    FLAC = "flac"
    PCM = "pcm_s16le"


class ContainerFormat(Enum):
    """Video container formats."""
    MP4 = "mp4"
    MKV = "mkv"
    MOV = "mov"
    WEBM = "webm"
    AVI = "avi"


class QualityPreset(Enum):
    """Encoding quality presets."""
    ULTRAFAST = "ultrafast"
    SUPERFAST = "superfast"
    VERYFAST = "veryfast"
    FASTER = "faster"
    FAST = "fast"
    MEDIUM = "medium"
    SLOW = "slow"
    SLOWER = "slower"
    VERYSLOW = "veryslow"


@dataclass
class AudioTrack:
    """
    Audio track for video assembly.

    Attributes:
        path: Path to audio file
        start_time: Start time in seconds
        duration: Duration in seconds (None for full length)
        volume: Volume multiplier (1.0 = original)
        fade_in: Fade in duration in seconds
        fade_out: Fade out duration in seconds
    """
    path: Path
    start_time: float = 0.0
    duration: Optional[float] = None
    volume: float = 1.0
    fade_in: float = 0.0
    fade_out: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            'path': str(self.path),
            'start_time': self.start_time,
            'duration': self.duration,
            'volume': self.volume,
            'fade_in': self.fade_in,
            'fade_out': self.fade_out,
        }


@dataclass
class AssemblyConfig:
    """
    Configuration for video assembly.

    Attributes:
        fps: Frames per second
        width: Output video width
        height: Output video height
        video_codec: Video encoding codec
        audio_codec: Audio encoding codec
        container: Container format
        quality_preset: Encoding quality preset
        crf: Constant Rate Factor (lower = better quality)
        bitrate: Target bitrate (e.g., "10M")
        audio_bitrate: Audio bitrate (e.g., "192k")
        two_pass: Whether to use two-pass encoding
        metadata: Additional metadata to embed
    """
    fps: int = 24
    width: int = 1920
    height: int = 1080
    video_codec: VideoCodec = VideoCodec.H264
    audio_codec: AudioCodec = AudioCodec.AAC
    container: ContainerFormat = ContainerFormat.MP4
    quality_preset: QualityPreset = QualityPreset.MEDIUM
    crf: int = 23
    bitrate: Optional[str] = None
    audio_bitrate: str = "192k"
    two_pass: bool = False
    metadata: Dict[str, str] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'fps': self.fps,
            'width': self.width,
            'height': self.height,
            'video_codec': self.video_codec.name,
            'audio_codec': self.audio_codec.name,
            'container': self.container.name,
            'quality_preset': self.quality_preset.name,
            'crf': self.crf,
            'bitrate': self.bitrate,
            'audio_bitrate': self.audio_bitrate,
            'two_pass': self.two_pass,
            'metadata': self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'AssemblyConfig':
        return cls(
            fps=data.get('fps', 24),
            width=data.get('width', 1920),
            height=data.get('height', 1080),
            video_codec=VideoCodec[data.get('video_codec', 'H264')],
            audio_codec=AudioCodec[data.get('audio_codec', 'AAC')],
            container=ContainerFormat[data.get('container', 'MP4')],
            quality_preset=QualityPreset[data.get('quality_preset', 'MEDIUM')],
            crf=data.get('crf', 23),
            bitrate=data.get('bitrate'),
            audio_bitrate=data.get('audio_bitrate', '192k'),
            two_pass=data.get('two_pass', False),
            metadata=data.get('metadata', {}),
        )


@dataclass
class AssemblyResult:
    """
    Result of video assembly.

    Attributes:
        success: Whether assembly was successful
        output_path: Path to output video file
        duration: Duration of output video in seconds
        file_size: Size of output file in bytes
        encoding_time: Time taken to encode in seconds
        warnings: Any warnings during assembly
        error: Error message if assembly failed
    """
    success: bool
    output_path: Optional[Path] = None
    duration: float = 0.0
    file_size: int = 0
    encoding_time: float = 0.0
    warnings: List[str] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'success': self.success,
            'output_path': str(self.output_path) if self.output_path else None,
            'duration': self.duration,
            'file_size': self.file_size,
            'encoding_time': self.encoding_time,
            'warnings': self.warnings,
            'error': self.error,
        }


class VideoAssembler:
    """
    Assembles video frames into final video output.

    Supports:
    - Frame sequence assembly
    - Audio track integration
    - Multiple export formats
    - Quality presets
    """

    def __init__(self, config: Optional[AssemblyConfig] = None):
        """
        Initialize the video assembler.

        Args:
            config: Assembly configuration
        """
        self.config = config or AssemblyConfig()
        self._ffmpeg_available = self._check_ffmpeg()

    def _check_ffmpeg(self) -> bool:
        """Check if ffmpeg is available."""
        return shutil.which("ffmpeg") is not None

    def assemble(self,
                 frames: List[Path],
                 output_path: Path,
                 audio_tracks: Optional[List[AudioTrack]] = None) -> AssemblyResult:
        """
        Assemble frames into a video.

        Args:
            frames: List of paths to frame files (in order)
            output_path: Path for output video
            audio_tracks: Optional audio tracks to include

        Returns:
            AssemblyResult with status and details.
        """
        import time
        start_time = time.time()

        if not frames:
            return AssemblyResult(
                success=False,
                error="No frames provided"
            )

        # Ensure output directory exists
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Check for ffmpeg
        if not self._ffmpeg_available:
            return self._assemble_fallback(frames, output_path)

        try:
            result = self._assemble_with_ffmpeg(frames, output_path, audio_tracks)
            result.encoding_time = time.time() - start_time
            return result
        except Exception as e:
            return AssemblyResult(
                success=False,
                error=str(e),
                encoding_time=time.time() - start_time
            )

    def _assemble_with_ffmpeg(self,
                               frames: List[Path],
                               output_path: Path,
                               audio_tracks: Optional[List[AudioTrack]] = None) -> AssemblyResult:
        """Assemble video using ffmpeg."""
        warnings = []

        # Create a temporary file list for ffmpeg
        list_file = output_path.parent / f".frame_list_{output_path.stem}.txt"

        try:
            # Write frame list
            with open(list_file, 'w') as f:
                for frame in frames:
                    # Calculate duration per frame
                    duration = 1.0 / self.config.fps
                    f.write(f"file '{frame.absolute()}'\n")
                    f.write(f"duration {duration}\n")
                # Add last frame again (ffmpeg quirk)
                f.write(f"file '{frames[-1].absolute()}'\n")

            # Build ffmpeg command
            cmd = self._build_ffmpeg_command(list_file, output_path, audio_tracks)

            # Run ffmpeg
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True
            )

            if result.returncode != 0:
                return AssemblyResult(
                    success=False,
                    error=f"ffmpeg failed: {result.stderr[:500]}"
                )

            # Get output file info
            if output_path.exists():
                file_size = output_path.stat().st_size
                duration = len(frames) / self.config.fps

                return AssemblyResult(
                    success=True,
                    output_path=output_path,
                    duration=duration,
                    file_size=file_size,
                    warnings=warnings
                )
            else:
                return AssemblyResult(
                    success=False,
                    error="Output file was not created"
                )

        finally:
            # Cleanup
            if list_file.exists():
                list_file.unlink()

    def _build_ffmpeg_command(self,
                               list_file: Path,
                               output_path: Path,
                               audio_tracks: Optional[List[AudioTrack]] = None) -> List[str]:
        """Build ffmpeg command for video assembly."""
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-f", "concat",
            "-safe", "0",
            "-i", str(list_file),
        ]

        # Add audio if provided
        if audio_tracks:
            for track in audio_tracks:
                cmd.extend(["-i", str(track.path)])

        # Video codec settings
        cmd.extend([
            "-c:v", self.config.video_codec.value,
            "-preset", self.config.quality_preset.value,
            "-crf", str(self.config.crf),
        ])

        # Resolution
        cmd.extend([
            "-vf", f"scale={self.config.width}:{self.config.height}:force_original_aspect_ratio=decrease,pad={self.config.width}:{self.config.height}:(ow-iw)/2:(oh-ih)/2"
        ])

        # Pixel format for compatibility
        cmd.extend(["-pix_fmt", "yuv420p"])

        # Audio settings
        if audio_tracks:
            cmd.extend([
                "-c:a", self.config.audio_codec.value,
                "-b:a", self.config.audio_bitrate,
            ])
        else:
            cmd.extend(["-an"])  # No audio

        # Bitrate if specified
        if self.config.bitrate:
            cmd.extend(["-b:v", self.config.bitrate])

        # Metadata
        for key, value in self.config.metadata.items():
            cmd.extend(["-metadata", f"{key}={value}"])

        # Output
        cmd.append(str(output_path))

        return cmd

    def _assemble_fallback(self, frames: List[Path], output_path: Path) -> AssemblyResult:
        """Fallback assembly without ffmpeg (creates manifest)."""
        # Create a manifest file instead of actual video
        manifest = {
            'type': 'video_manifest',
            'frames': [str(f) for f in frames],
            'config': self.config.to_dict(),
            'created': datetime.now().isoformat(),
            'note': 'ffmpeg not available - this is a manifest for later assembly',
        }

        manifest_path = output_path.with_suffix('.manifest.json')
        manifest_path.write_text(json.dumps(manifest, indent=2))

        return AssemblyResult(
            success=True,
            output_path=manifest_path,
            duration=len(frames) / self.config.fps,
            warnings=['ffmpeg not available - created manifest instead of video']
        )

    def concatenate(self, videos: List[Path], output_path: Path) -> AssemblyResult:
        """
        Concatenate multiple videos into one.

        Args:
            videos: List of video file paths
            output_path: Path for output video

        Returns:
            AssemblyResult with status and details.
        """
        import time
        start_time = time.time()

        if not videos:
            return AssemblyResult(success=False, error="No videos provided")

        if not self._ffmpeg_available:
            return AssemblyResult(
                success=False,
                error="ffmpeg not available for video concatenation"
            )

        # Create concat file
        concat_file = output_path.parent / f".concat_{output_path.stem}.txt"

        try:
            with open(concat_file, 'w') as f:
                for video in videos:
                    f.write(f"file '{video.absolute()}'\n")

            cmd = [
                "ffmpeg",
                "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_file),
                "-c", "copy",
                str(output_path)
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                return AssemblyResult(
                    success=False,
                    error=f"Concatenation failed: {result.stderr[:500]}",
                    encoding_time=time.time() - start_time
                )

            if output_path.exists():
                return AssemblyResult(
                    success=True,
                    output_path=output_path,
                    file_size=output_path.stat().st_size,
                    encoding_time=time.time() - start_time
                )
            else:
                return AssemblyResult(
                    success=False,
                    error="Output file not created",
                    encoding_time=time.time() - start_time
                )
        finally:
            if concat_file.exists():
                concat_file.unlink()

    def add_audio(self,
                  video_path: Path,
                  audio_track: AudioTrack,
                  output_path: Path) -> AssemblyResult:
        """
        Add audio track to an existing video.

        Args:
            video_path: Path to video file
            audio_track: Audio track to add
            output_path: Path for output video

        Returns:
            AssemblyResult with status and details.
        """
        import time
        start_time = time.time()

        if not self._ffmpeg_available:
            return AssemblyResult(
                success=False,
                error="ffmpeg not available"
            )

        cmd = [
            "ffmpeg",
            "-y",
            "-i", str(video_path),
            "-i", str(audio_track.path),
            "-c:v", "copy",
            "-c:a", self.config.audio_codec.value,
            "-b:a", self.config.audio_bitrate,
            "-map", "0:v:0",
            "-map", "1:a:0",
        ]

        # Apply volume if not 1.0
        if audio_track.volume != 1.0:
            cmd.extend(["-af", f"volume={audio_track.volume}"])

        cmd.append(str(output_path))

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)

            if result.returncode != 0:
                return AssemblyResult(
                    success=False,
                    error=f"Audio merge failed: {result.stderr[:500]}",
                    encoding_time=time.time() - start_time
                )

            if output_path.exists():
                return AssemblyResult(
                    success=True,
                    output_path=output_path,
                    file_size=output_path.stat().st_size,
                    encoding_time=time.time() - start_time
                )
            else:
                return AssemblyResult(
                    success=False,
                    error="Output file not created",
                    encoding_time=time.time() - start_time
                )
        except Exception as e:
            return AssemblyResult(
                success=False,
                error=str(e),
                encoding_time=time.time() - start_time
            )

    def get_video_info(self, video_path: Path) -> Dict[str, Any]:
        """
        Get information about a video file.

        Args:
            video_path: Path to video file

        Returns:
            Dictionary with video information.
        """
        if not self._ffmpeg_available:
            return {'error': 'ffmpeg not available'}

        cmd = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            str(video_path)
        ]

        try:
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode == 0:
                return json.loads(result.stdout)
            else:
                return {'error': result.stderr}
        except Exception as e:
            return {'error': str(e)}


__all__ = [
    'VideoCodec',
    'AudioCodec',
    'ContainerFormat',
    'QualityPreset',
    'AudioTrack',
    'AssemblyConfig',
    'AssemblyResult',
    'VideoAssembler',
]
