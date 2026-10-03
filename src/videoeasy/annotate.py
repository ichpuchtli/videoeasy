"""Stage 5: VLM annotation of every b-roll shot.

Sends the LUT-graded frame set + motion gloss to the local vision model
(via LM Studio, see vlm.py) and stores structured annotations. Idempotent:
already-annotated shots are skipped unless --force.
"""
from __future__ import annotations

import json

import httpx

from . import segments
from .config import Config
from .prompts import annotation_system, annotation_user_prompt
from .vlm import chat_vision, extract_json


def _frames(cfg: Config, shot_id: str) -> list[bytes]:
    frame_dir = cfg.frames_dir / shot_id
    frames = sorted(frame_dir.glob("f*.jpg"))
    if not frames:
        raise FileNotFoundError(f"no frames for {shot_id} — run `videoeasy frames` first")
    return [f.read_bytes() for f in frames]


def annotate_shot(cfg: Config, shot: dict, motion_gloss: str | None) -> dict:
    raw = chat_vision(
        cfg.vision_url,
        cfg.vision_model,
        annotation_system(cfg.film.context),
        annotation_user_prompt(shot, motion_gloss),
        _frames(cfg, shot["shot_id"]),
    )
    return extract_json(raw)


def run(cfg: Config, force: bool = False, only: str | None = None) -> None:
    shots = segments.units(cfg)
    motion_path = cfg.out_dir / "motion.json"
    motion = json.loads(motion_path.read_text()) if motion_path.exists() else {}
    out_path = cfg.out_dir / "annotations.json"
    annotations = json.loads(out_path.read_text()) if out_path.exists() else {}
    if force:
        # --force --only X re-annotates just X; a bare --force starts over
        if only:
            annotations.pop(only, None)
        else:
            annotations = {}

    # A-roll is annotated too: walking conversations and candid takes are
    # visual coverage as much as sound, and the story room needs to know what
    # they look like. The prompt is told the role so it describes staging.
    # Whole takes and their windows are both units here. A window marked
    # visual_tag=false (a locked-off sit-down: see segments.index_only_sources)
    # is indexed for search but not sent to the VLM, because ninety near
    # identical tags would dilute the bible rather than sharpen it. Naming one
    # with --only still annotates it.
    todo = [s for s in shots if s.get("visual_tag", True)]
    if only:
        todo = [s for s in shots if s["shot_id"] == only]
        if not todo:
            raise SystemExit(f"no shot or segment with id {only!r}")

    for i, shot in enumerate(todo):
        sid = shot["shot_id"]
        if sid in annotations and "error" not in annotations[sid]:
            continue
        gloss = motion.get(sid, {}).get("gloss")
        try:
            annotations[sid] = annotate_shot(cfg, shot, gloss)
        except (json.JSONDecodeError, httpx.HTTPError, RuntimeError) as e:
            # A single stubborn shot must not kill a batch run — record the
            # error; a later `annotate` pass retries error entries.
            annotations[sid] = {"error": str(e)}
        out_path.write_text(json.dumps(annotations, indent=2))
        print(f"[{i + 1}/{len(todo)}] {sid}"
              + (f"  ERROR: {annotations[sid]['error']}" if "error" in annotations[sid] else ""))
