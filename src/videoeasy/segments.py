"""Stage 2b: window long takes into searchable moments.

A single tag over a 358-second walk-and-talk is a summary of four frames
spread ninety seconds apart, and the September 2026 calibration caught it
inventing continuity between them. Long takes are therefore windowed into
shorter units, each with its own graded samples and its own tag.

A window is an INDEXING boundary, never a claimed cut. Where the boundary
came from is recorded per edge:

- `speech_pause` — a real silence in the local transcript, so a window starts
  and ends where the talking does.
- `even_split` — no transcript, or no pause near the wanted position: the
  take is divided evenly and the record says so. Nothing in the picture is
  being claimed as a cut.
- `source_start` / `source_end` — the take's own ends.

Short takes are left exactly as they were: no segment, no new frames, and the
whole-source shot record stays the unit of annotation. Segments are
shot-shaped (`shot_id`, `in_s`, `out_s`, `source_path`, `role`) so frames,
annotate and audit consume them without special cases; `parent_shot_id` is
what distinguishes a window from a whole take.

Transcript text is carried on the segment for SEARCH only. It is never given
to the vision model: a spoken topic is not a visible fact.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path

from .config import Config

DEFAULTS = {
    "min_source_seconds": 60.0,   # shorter takes stay whole
    "target_seconds": 45.0,       # wanted window length
    "max_seconds": 90.0,          # no window longer than this
    "min_seconds": 12.0,          # no window shorter than this
    "pause_seconds": 1.0,         # silence that may serve as a boundary
    "frames_per_segment": 6,      # graded samples per window
    "index_only_sources": [],     # windowed for search, not sent to the VLM
}


@dataclass
class SegmentRules:
    min_source_seconds: float = DEFAULTS["min_source_seconds"]
    target_seconds: float = DEFAULTS["target_seconds"]
    max_seconds: float = DEFAULTS["max_seconds"]
    min_seconds: float = DEFAULTS["min_seconds"]
    pause_seconds: float = DEFAULTS["pause_seconds"]
    frames_per_segment: int = DEFAULTS["frames_per_segment"]
    index_only_sources: tuple[str, ...] = field(default_factory=tuple)


def rules_from(cfg: Config) -> SegmentRules:
    """Read the optional `segments:` config block. Absent block = defaults,
    so a film config written before this stage existed still loads."""
    raw = getattr(cfg, "raw", {}) or {}
    options = raw.get("segments", {})
    if not isinstance(options, dict):
        raise ValueError("segments must be a mapping")
    unknown = set(options) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown segments options: {', '.join(sorted(unknown))}")
    values = {**DEFAULTS, **options}
    sources = values["index_only_sources"]
    if not isinstance(sources, list) or any(not isinstance(s, str) or not s.strip() for s in sources):
        raise ValueError("segments.index_only_sources must be a list of source ids")
    if type(values["frames_per_segment"]) is not int or not 2 <= values["frames_per_segment"] <= 16:
        raise ValueError("segments.frames_per_segment must be an integer from 2 to 16")
    numbers = {k: values[k] for k in
               ("min_source_seconds", "target_seconds", "max_seconds", "min_seconds", "pause_seconds")}
    for key, value in numbers.items():
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"segments.{key} must be a positive number")
    if not numbers["min_seconds"] < numbers["target_seconds"]:
        raise ValueError("segments.min_seconds must be below segments.target_seconds")
    # A window is placed within half a target of its ideal position, so the
    # final window can run to 1.5 targets. Anything less would let a window
    # silently exceed max_seconds.
    if numbers["max_seconds"] < 1.5 * numbers["target_seconds"]:
        raise ValueError("segments.max_seconds must be at least 1.5x segments.target_seconds")
    if numbers["min_source_seconds"] < numbers["target_seconds"]:
        raise ValueError("segments.min_source_seconds must be at least segments.target_seconds")
    return SegmentRules(**numbers, frames_per_segment=values["frames_per_segment"],
                        index_only_sources=tuple(sources))


def is_segment(unit: dict) -> bool:
    return bool(unit.get("parent_shot_id"))


def speech_boundaries(transcript_segments: list[dict], in_s: float, out_s: float,
                      pause_seconds: float) -> list[tuple[float, float]]:
    """Midpoints of silences at least `pause_seconds` long, inside the take.

    Returns (time, silence_length). A silence is evidence from the transcript,
    which is why it is preferred over an arbitrary split — but it is still a
    boundary in the SOUND, never a claim about a cut in the picture.
    """
    spans = sorted((float(s["start"]), float(s["end"])) for s in transcript_segments
                   if float(s["end"]) > in_s and float(s["start"]) < out_s)
    found: list[tuple[float, float]] = []
    spoken_until: float | None = None
    for start, end in spans:
        if spoken_until is not None and start - spoken_until >= pause_seconds:
            middle = (spoken_until + start) / 2
            if in_s < middle < out_s:
                found.append((round(middle, 3), round(start - spoken_until, 2)))
        spoken_until = end if spoken_until is None else max(spoken_until, end)
    return found


def cut_points(in_s: float, out_s: float, boundaries: list[tuple[float, float]],
               rules: SegmentRules) -> list[tuple[float, float | None]]:
    """Choose window edges: evenly spaced ideals, each snapped to the nearest
    speech pause within half a target. No pause in reach means an even split,
    recorded as such (silence: None)."""
    duration = out_s - in_s
    count = max(1, math.ceil(duration / rules.target_seconds))
    count = min(count, max(1, int(duration // rules.min_seconds)))
    if count == 1:
        return []
    cuts: list[tuple[float, float | None]] = []
    previous = in_s
    for index in range(1, count):
        ideal = in_s + duration * index / count
        low = max(previous + rules.min_seconds, ideal - rules.target_seconds / 2)
        high = min(out_s - rules.min_seconds, ideal + rules.target_seconds / 2,
                   previous + rules.max_seconds)
        if high < low:  # no lawful position left; this take gets fewer windows
            continue
        reachable = [b for b in boundaries if low <= b[0] <= high]
        if reachable:
            time, silence = min(reachable, key=lambda b: abs(b[0] - ideal))
        else:
            time, silence = min(max(ideal, low), high), None
        cuts.append((round(time, 3), silence))
        previous = time
    return cuts


def _speech(transcript_segments: list[dict], in_s: float, out_s: float) -> dict | None:
    """Transcript segments whose midpoint falls in the window — midpoints keep
    a sentence straddling an edge in exactly one window."""
    inside = [s for s in transcript_segments if in_s <= (float(s["start"]) + float(s["end"])) / 2 < out_s]
    if not inside:
        return None
    words = [w for s in inside for w in s.get("words", [])]
    return {
        "segment_count": len(inside),
        "first_word_s": round(float(words[0]["s"]) if words else float(inside[0]["start"]), 3),
        "last_word_s": round(float(words[-1]["e"]) if words else float(inside[-1]["end"]), 3),
        "text": " ".join(s["text"].strip() for s in inside if s.get("text")),
    }


def _windows(shot: dict, transcript: dict, rules: SegmentRules) -> list[dict]:
    spoken = transcript.get("segments", []) if transcript else []
    boundaries = speech_boundaries(spoken, shot["in_s"], shot["out_s"], rules.pause_seconds)
    cuts = cut_points(shot["in_s"], shot["out_s"], boundaries, rules)
    if not cuts:
        return []
    edges = [(shot["in_s"], "source_start", None)]
    edges += [(time, "speech_pause" if silence else "even_split", silence) for time, silence in cuts]
    edges.append((shot["out_s"], "source_end", None))

    segments = []
    for index, ((start, start_method, start_silence), (end, end_method, end_silence)) in enumerate(
            zip(edges, edges[1:])):
        def edge(method: str, silence: float | None) -> dict:
            return {"method": method} if silence is None else {"method": method, "silence_s": silence}

        segment = {
            "shot_id": f"{shot['shot_id']}_w{index:02d}",
            "parent_shot_id": shot["shot_id"],
            "source_id": shot["source_id"],
            "source_path": shot["source_path"],
            "role": shot["role"],
            "in_s": round(start, 3),
            "out_s": round(end, 3),
            "duration_s": round(end - start, 3),
            "fps": shot["fps"],
            "start_timecode": shot.get("start_timecode"),
            "parent_in_s": shot["in_s"],
            "parent_out_s": shot["out_s"],
            "parent_duration_s": shot["duration_s"],
            "start_boundary": edge(start_method, start_silence),
            "end_boundary": edge(end_method, end_silence),
            "visual_tag": shot["source_id"] not in rules.index_only_sources,
        }
        speech = _speech(spoken, start, end)
        if speech is not None:
            segment["speech"] = speech
        segments.append(segment)
    return segments


def build(cfg: Config, force: bool = False) -> list[dict]:
    """Window every take at or over `min_source_seconds`.

    Existing windows are kept as they are: once frames are extracted and a
    window is annotated, silently shifting its boundaries would orphan both.
    `--force` recomputes, and the caller is told which ids disappeared.
    """
    rules = rules_from(cfg)
    shots = json.loads((cfg.out_dir / "shots.json").read_text())
    transcripts = _read(cfg.out_dir / "transcripts.json", {})
    existing: dict[str, list[dict]] = {}
    for segment in load(cfg):
        existing.setdefault(segment["parent_shot_id"], []).append(segment)

    segments: list[dict] = []
    for shot in shots:
        if shot["duration_s"] < rules.min_source_seconds:
            continue
        kept = existing.get(shot["shot_id"])
        if kept and not force:
            segments.extend(kept)
            continue
        segments.extend(_windows(shot, transcripts.get(shot["source_id"], {}), rules))

    (cfg.out_dir / "segments.json").write_text(json.dumps(segments, indent=2))
    return segments


def load(cfg: Config) -> list[dict]:
    return _read(cfg.out_dir / "segments.json", [])


def units(cfg: Config) -> list[dict]:
    """Every taggable unit: whole-take shots first, then their windows. The
    whole-take record is never replaced — it stays as the take's context."""
    shots = json.loads((cfg.out_dir / "shots.json").read_text())
    return shots + load(cfg)


def _read(path: Path, default):
    return json.loads(path.read_text()) if path.exists() else default
