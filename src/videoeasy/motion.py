"""Stage 4: per-shot motion statistics via optical flow.

Thumbnails lose pace; this stage puts it back as numbers. Farneback flow over
frame pairs sampled at ~2fps gives:
  - motion_energy: mean flow magnitude, normalised by frame width (0..~0.1+)
  - camera_ratio: |median flow vector| / mean magnitude — near 1.0 means the
    whole frame moves together (pan/track), near 0 means static camera with
    subject motion.
These are heuristics; they feed the VLM prompt and the bible, not decisions.
"""
from __future__ import annotations

import json

import cv2
import numpy as np

from . import segments
from .config import Config


def shot_motion(path: str, in_s: float, out_s: float, sample_fps: float, width: int) -> dict:
    cap = cv2.VideoCapture(path)
    step = 1.0 / sample_fps
    times = np.arange(in_s, out_s, step)
    prev = None
    mags, cam_ratios = [], []

    for t in times:
        cap.set(cv2.CAP_PROP_POS_MSEC, t * 1000.0)
        ok, frame = cap.read()
        if not ok:
            break
        h = round(frame.shape[0] * width / frame.shape[1])
        gray = cv2.cvtColor(cv2.resize(frame, (width, h)), cv2.COLOR_BGR2GRAY)
        if prev is not None:
            flow = cv2.calcOpticalFlowFarneback(prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
            mag = np.linalg.norm(flow, axis=2)
            mean_mag = float(mag.mean())
            if mean_mag > 1e-3:
                global_vec = np.median(flow.reshape(-1, 2), axis=0)
                cam_ratios.append(float(np.linalg.norm(global_vec)) / mean_mag)
            mags.append(mean_mag / width)
        prev = gray
    cap.release()

    if not mags:
        return {"motion_energy": 0.0, "camera_ratio": 0.0, "pairs": 0}
    return {
        "motion_energy": round(float(np.mean(mags)), 5),
        "motion_peak": round(float(np.max(mags)), 5),
        "camera_ratio": round(float(np.mean(cam_ratios)), 3) if cam_ratios else 0.0,
        "pairs": len(mags),
    }


def describe(stats: dict) -> str:
    """Plain-language gloss included in the VLM prompt and the bible."""
    e = stats["motion_energy"]
    energy = "static" if e < 0.002 else "gentle" if e < 0.008 else "moderate" if e < 0.02 else "energetic"
    cam = "camera moves with the scene" if stats["camera_ratio"] > 0.6 else "camera mostly still"
    return f"{energy} movement, {cam}"


def run(cfg: Config) -> None:
    # Windows inside a long take need their own numbers: the statistics for a
    # whole 320-second drone take say nothing about which of its windows is
    # moving. A-roll stays excluded exactly as before.
    shots = segments.units(cfg)
    out_path = cfg.out_dir / "motion.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}

    for shot in shots:
        if shot["shot_id"] in results or shot["role"] != "broll":
            continue
        stats = shot_motion(
            shot["source_path"], shot["in_s"], shot["out_s"],
            cfg.motion_sample_fps, cfg.motion_downscale_width,
        )
        stats["gloss"] = describe(stats)
        results[shot["shot_id"]] = stats
        out_path.write_text(json.dumps(results, indent=2))
