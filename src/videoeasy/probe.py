"""Stage 1: ffprobe every source file into an inventory.

Records duration, fps, resolution, codec, and the embedded start timecode
(Fuji writes one) — timecode is what lets Phase 2 conform shots back to the
originals in Resolve.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

from .config import Config, list_videos


def _ffprobe(path: Path) -> dict:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-print_format", "json",
            "-show_format", "-show_streams", str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    return json.loads(result.stdout)


def _parse_fps(rate: str) -> float:
    num, _, den = rate.partition("/")
    return float(num) / float(den or 1)


def probe_file(path: Path, role: str) -> dict:
    info = _ffprobe(path)
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    tags = {**info.get("format", {}).get("tags", {}), **video.get("tags", {})}
    return {
        "id": path.stem,
        "path": str(path),
        "role": role,  # "aroll" | "broll"
        "duration_s": float(info["format"]["duration"]),
        "fps": _parse_fps(video.get("r_frame_rate", "25/1")),
        "width": video.get("width"),
        "height": video.get("height"),
        "codec": video.get("codec_name"),
        "start_timecode": tags.get("timecode"),
        "has_audio": any(s["codec_type"] == "audio" for s in info["streams"]),
    }


def run(cfg: Config) -> dict:
    inventory = {
        "aroll": [probe_file(f, "aroll") for f in list_videos(cfg.aroll_dir)],
        "broll": [probe_file(f, "broll") for f in list_videos(cfg.broll_dir)],
    }
    out = cfg.out_dir / "inventory.json"
    out.write_text(json.dumps(inventory, indent=2))
    return inventory
