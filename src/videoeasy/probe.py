"""Stage 1: ffprobe every source file into an inventory.

Records duration, fps, resolution, codec, the embedded start timecode (Fuji
writes one; it is what lets the lay-in conform shots back to the originals in
Resolve), the container's `creation_time` exactly as written, and the camera
make and model where the file says. `creation_time` is stored unparsed: what
it means depends on the camera (some write true UTC; many drones and action
cameras write local wall-clock time labelled `Z`; a camera whose clock was
never set writes a confident wrong date). Reading it is the job of the event
schedule (`deliverables.py`), per camera, with a measured offset.
"""
from __future__ import annotations

import json
import re
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


# Make/model keys seen in the wild, in preference order (keys compared lowercased). A Fuji X-H2S writes the
# com.apple.proapps pair; an X-T4 writes only the comment "FUJIFILM DIGITAL CAMERA X-T4"; a drone file that went
# through a remux carries nothing but creation_time and an encoder string.
MAKE_KEYS = ("com.apple.proapps.manufacturer", "com.apple.quicktime.make", "make")
MODEL_KEYS = ("com.apple.proapps.modelname", "com.apple.quicktime.model", "model")
COMMENT_CAMERA = re.compile(r"^(\S+) DIGITAL CAMERA (.+)$")


def camera_of(tags: dict) -> tuple[str | None, str | None]:
    """(make, model) from container/stream tags; None for whatever the file does not say."""
    low = {str(k).lower(): str(v).strip() for k, v in tags.items() if v not in (None, "")}
    make = next((low[k] for k in MAKE_KEYS if low.get(k)), None)
    model = next((low[k] for k in MODEL_KEYS if low.get(k)), None)
    m = COMMENT_CAMERA.match(low.get("comment", ""))
    if m:
        make, model = make or m.group(1), model or m.group(2)
    enc = low.get("encoder", "")
    if not make and enc.upper().startswith("DJI"):
        make, model = "DJI", model or (enc[3:].strip() or None)
    return make, model


def creation_time_of(info: dict) -> str | None:
    """The container's creation_time as written, else the first stream's; never parsed here."""
    fmt = (info.get("format") or {}).get("tags") or {}
    for tags in [fmt, *((s.get("tags") or {}) for s in info.get("streams", []))]:
        for k, v in tags.items():
            if str(k).lower() == "creation_time" and v:
                return str(v)
    return None


def probe_file(path: Path, role: str) -> dict:
    return record(path, role, _ffprobe(path))


def record(path: Path, role: str, info: dict) -> dict:
    """The inventory entry for one file from its ffprobe JSON."""
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    tags = {**info.get("format", {}).get("tags", {}), **video.get("tags", {})}
    make, model = camera_of(tags)
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
        "creation_time": creation_time_of(info),
        "camera_make": make,
        "camera_model": model,
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
