"""Stage 3: extract graded frames per shot, plus a contact sheet.

Grading is not cosmetic: log frames are flat and desaturated, and a VLM
captioning them reads 'washed out, muted, melancholy' on footage that is
actually warm. Every frame the model sees goes through the source's grade
profile (config `grade:`) — Fuji log through the ETERNA LUT (the intended
final look), DJI through a saturation/contrast lift.
"""
from __future__ import annotations

import subprocess
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

from . import segments
from .config import Config


def _escape_filter_path(path: Path) -> str:
    # ffmpeg filter-arg escaping: backslash, colon, comma, quote
    s = str(path)
    for ch in ("\\", ":", ",", "'"):
        s = s.replace(ch, "\\" + ch)
    return s


def extract_frame(src: Path, t: float, dest: Path, grade, max_px: int) -> None:
    filters = []
    if grade.lut is not None:
        filters.append(f"lut3d={_escape_filter_path(grade.lut)}")
    if grade.filters:
        filters.append(grade.filters)
    filters.append(f"scale='min({max_px},iw)':-2")
    subprocess.run(
        [
            "ffmpeg", "-v", "error", "-y",
            "-ss", f"{t:.3f}", "-i", str(src),
            "-frames:v", "1", "-vf", ",".join(filters),
            "-q:v", "3", str(dest),
        ],
        check=True, capture_output=True,
    )


def sample_times(in_s: float, out_s: float, n: int) -> list[float]:
    # Evenly spaced, inset from the edges so we never grab a cut frame.
    dur = out_s - in_s
    if n == 1:
        return [in_s + dur / 2]
    inset = min(0.25, dur * 0.05)
    lo, hi = in_s + inset, out_s - inset
    return [lo + (hi - lo) * i / (n - 1) for i in range(n)]


def contact_sheet(frame_paths: list[Path], dest: Path, cols: int = 4, tile_w: int = 480,
                  labels: list[str] | None = None) -> None:
    if not frame_paths or cols < 1 or tile_w < 1:
        raise ValueError("contact sheet requires frames and positive dimensions")
    if labels is not None and len(labels) != len(frame_paths):
        raise ValueError("one label is required per frame")
    tiles = []
    for path in frame_paths:
        with Image.open(path) as img:
            h = max(1, round(img.height * tile_w / img.width))
            tiles.append(img.resize((tile_w, h)))
    tile_h = max(t.height for t in tiles)
    label_h = 30 if labels is not None else 0
    cell_h = tile_h + label_h
    rows = (len(tiles) + cols - 1) // cols
    sheet = Image.new("RGB", (cols * tile_w, rows * cell_h), (16, 16, 16))
    draw = ImageDraw.Draw(sheet)
    try:
        font = ImageFont.load_default(size=18) if labels is not None else None
    except TypeError:  # Pillow 10.0 lacks the size argument.
        font = ImageFont.load_default()
    for i, tile in enumerate(tiles):
        x, y = (i % cols) * tile_w, (i // cols) * cell_h
        sheet.paste(tile, (x, y))
        if labels is not None:
            draw.text((x + 6, y + tile_h + 4), labels[i], fill="white", font=font)
    sheet.save(dest, quality=85)


def sample_count(cfg: Config, unit: dict, aroll_frames: int, rules) -> int:
    """A window inside a long take is sampled densely — that density is the
    whole point of windowing. Whole takes keep their original counts so no
    existing frame set or contact sheet is invalidated."""
    if segments.is_segment(unit):
        return rules.frames_per_segment
    return aroll_frames if unit["role"] == "aroll" else cfg.frames_per_shot


def run(cfg: Config, aroll_frames: int = 4, only: str | None = None) -> None:
    rules = segments.rules_from(cfg)
    shots = segments.units(cfg)
    if only is not None:
        shots = [s for s in shots if s["shot_id"] == only]
        if not shots:
            raise SystemExit(f"no shot or segment with id {only!r}")
    for profile in cfg.grade_profiles.values():
        if profile.lut is not None and not profile.lut.is_file():
            raise SystemExit(
                f"LUT for grade profile '{profile.name}' not found at {profile.lut} — "
                "refusing to extract ungraded log frames. Fix grade.profiles in config.yaml."
            )

    for shot in shots:
        # A-roll is framing/lighting context only; b-roll gets the full sample.
        n = sample_count(cfg, shot, aroll_frames, rules)
        grade = cfg.grade_for(shot["source_path"])
        shot_dir = cfg.frames_dir / shot["shot_id"]
        shot_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, t in enumerate(sample_times(shot["in_s"], shot["out_s"], n)):
            dest = shot_dir / f"f{i:02d}.jpg"
            if not dest.exists():
                extract_frame(Path(shot["source_path"]), t, dest, grade, cfg.frame_max_px)
            paths.append(dest)
        sheet = cfg.sheets_dir / f"{shot['shot_id']}.jpg"
        if not sheet.exists():
            times = sample_times(shot["in_s"], shot["out_s"], n)
            contact_sheet(paths, sheet, labels=[f"f{i:02d} | source {t:.3f}s" for i, t in enumerate(times)])
