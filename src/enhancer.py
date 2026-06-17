"""Video quality enhancement using ffmpeg.

The enhancer re-encodes a source video to improve perceived quality before
upload. It can:
  * upscale/downscale to a target resolution (preserving aspect ratio),
  * apply a light denoise + sharpen + contrast/saturation pass,
  * re-encode with a high-quality x264 profile (CRF based),
  * normalise audio loudness.

ffmpeg/ffprobe must be installed and on PATH.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


class EnhancerError(RuntimeError):
    """Raised when ffmpeg is missing or a processing step fails."""


@dataclass
class VideoInfo:
    width: int
    height: int
    duration: float
    codec: str


def _require_binary(name: str) -> str:
    path = shutil.which(name)
    if not path:
        raise EnhancerError(
            f"'{name}' was not found on PATH. Install ffmpeg first.\n"
            "  Debian/Ubuntu: sudo apt-get install -y ffmpeg\n"
            "  Fedora:        sudo dnf install -y ffmpeg\n"
            "  macOS:         brew install ffmpeg"
        )
    return path


def probe(video_path: str) -> VideoInfo:
    """Return basic stream info for a video using ffprobe."""
    ffprobe = _require_binary("ffprobe")
    cmd = [
        ffprobe,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=width,height,codec_name:format=duration",
        "-of", "json",
        video_path,
    ]
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    except subprocess.CalledProcessError as exc:  # pragma: no cover - runtime guard
        raise EnhancerError(f"ffprobe failed for {video_path}: {exc.stderr}") from exc

    data = json.loads(out)
    stream = (data.get("streams") or [{}])[0]
    fmt = data.get("format") or {}
    return VideoInfo(
        width=int(stream.get("width", 0)),
        height=int(stream.get("height", 0)),
        duration=float(fmt.get("duration", 0.0) or 0.0),
        codec=str(stream.get("codec_name", "unknown")),
    )


def _build_filters(target_height: Optional[int], apply_filters: bool) -> str:
    """Build the ffmpeg -vf filter chain string."""
    filters = []

    if target_height:
        # Scale to target height, width auto (even), keep aspect ratio.
        filters.append(f"scale=-2:{target_height}:flags=lanczos")

    if apply_filters:
        # hqdn3d: light temporal/spatial denoise
        filters.append("hqdn3d=1.5:1.5:6:6")
        # unsharp: subtle sharpening
        filters.append("unsharp=5:5:0.8:5:5:0.0")
        # eq: gentle contrast/saturation lift
        filters.append("eq=contrast=1.05:saturation=1.08")

    return ",".join(filters)


def enhance_video(
    input_path: str,
    output_path: str,
    settings: Dict[str, Any],
    overwrite: bool = True,
) -> str:
    """Enhance ``input_path`` and write the result to ``output_path``.

    ``settings`` keys: target_height, crf, preset, audio_bitrate,
    apply_filters, max_bitrate. Returns the output path.
    """
    ffmpeg = _require_binary("ffmpeg")
    src = Path(input_path)
    if not src.exists():
        raise EnhancerError(f"Input video not found: {input_path}")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    target_height = settings.get("target_height")
    crf = int(settings.get("crf", 20))
    preset = settings.get("preset", "slow")
    audio_bitrate = settings.get("audio_bitrate", "192k")
    apply_filters = bool(settings.get("apply_filters", True))
    max_bitrate = settings.get("max_bitrate")

    cmd = [ffmpeg, "-y" if overwrite else "-n", "-i", str(src)]

    vf = _build_filters(target_height, apply_filters)
    if vf:
        cmd += ["-vf", vf]

    cmd += [
        "-c:v", "libx264",
        "-preset", str(preset),
        "-crf", str(crf),
        "-pix_fmt", "yuv420p",
        "-profile:v", "high",
        "-movflags", "+faststart",  # web-optimised: moov atom at front
    ]

    if max_bitrate:
        cmd += ["-maxrate", str(max_bitrate), "-bufsize", str(max_bitrate)]

    # Audio: re-encode to AAC and normalise loudness to broadcast-ish target.
    cmd += [
        "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
        "-c:a", "aac",
        "-b:a", str(audio_bitrate),
        str(out),
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError as exc:  # pragma: no cover - runtime guard
        raise EnhancerError(
            f"ffmpeg enhancement failed:\n{exc.stderr[-2000:]}"
        ) from exc

    if not out.exists() or out.stat().st_size == 0:
        raise EnhancerError("ffmpeg reported success but output file is empty.")

    return str(out)
