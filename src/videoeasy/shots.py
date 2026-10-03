"""Stage 2: split b-roll files into individual shots.

Some b-roll files contain several distinct shots; scene detection splits them.
A-roll files are kept whole (one 'shot' each) — the transcript, not the
picture, segments those.
"""
from __future__ import annotations

import json

from scenedetect import AdaptiveDetector, detect

from .config import Config


def _detect_shots(path: str, threshold: float, min_seconds: float) -> list[tuple[float, float]]:
    scenes = detect(path, AdaptiveDetector(adaptive_threshold=threshold))
    spans = [(s.get_seconds(), e.get_seconds()) for s, e in scenes]
    if not spans:
        return []
    # Merge fragments shorter than min_seconds into their predecessor —
    # F-Log flicker and exposure shifts can fire false cuts.
    merged = [spans[0]]
    for start, end in spans[1:]:
        if end - start < min_seconds:
            merged[-1] = (merged[-1][0], end)
        else:
            merged.append((start, end))
    return merged


def run(cfg: Config) -> list[dict]:
    inventory = json.loads((cfg.out_dir / "inventory.json").read_text())
    shots: list[dict] = []

    for clip in inventory["broll"]:
        spans = _detect_shots(clip["path"], cfg.adaptive_threshold, cfg.min_shot_seconds)
        if not spans:  # single-shot file, or detection found nothing
            spans = [(0.0, clip["duration_s"])]
        for i, (start, end) in enumerate(spans):
            shots.append({
                "shot_id": f"{clip['id']}_s{i:02d}" if len(spans) > 1 else clip["id"],
                "source_id": clip["id"],
                "source_path": clip["path"],
                "role": "broll",
                "in_s": round(start, 3),
                "out_s": round(end, 3),
                "duration_s": round(end - start, 3),
                "fps": clip["fps"],
                "start_timecode": clip["start_timecode"],
            })

    for clip in inventory["aroll"]:
        shots.append({
            "shot_id": clip["id"],
            "source_id": clip["id"],
            "source_path": clip["path"],
            "role": "aroll",
            "in_s": 0.0,
            "out_s": clip["duration_s"],
            "duration_s": clip["duration_s"],
            "fps": clip["fps"],
            "start_timecode": clip["start_timecode"],
        })

    (cfg.out_dir / "shots.json").write_text(json.dumps(shots, indent=2))
    return shots
