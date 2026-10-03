"""Stage 8: assemble every artifact into the footage bible context file.

This produces `out/bible-context.md` — the complete film in one document:
every b-roll shot's annotation + motion stats + contact-sheet path, and the
full timestamped transcript with the client's highlighted selects marked.

The relationship layer (themes across the corpus, which shots serve which
beats, motifs, gaps) is deliberately NOT generated here — that synthesis is
done in conversation with a frontier model reading this file, where the editor
can argue with it and corrections persist.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone

from .config import Config


def _fmt_ts(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


def _one_line(value) -> str:
    """Keep untrusted model prose inside its evidence-list item."""
    return " ".join(str(value).split())


def _evidence_lines(cfg: Config, shot: dict, annotation: dict) -> list[str]:
    # Older callers constructing a minimal configuration retain compatibility.
    if not hasattr(cfg, "work_dir"):
        return []
    from .evidence import read_evidence

    view = read_evidence(cfg, shot, annotation)
    sid = shot["shot_id"]
    lines = [f"- tag consistency: {view['status']} (local-model check; not footage verification)"]
    for error in view["errors"]:
        lines.append(f"  - audit unavailable: {_one_line(error)}")
    for finding in view["findings"]:
        refs = ", ".join(finding["evidence_ids"])
        lines.append(f"  - {_one_line(finding['kind'])} in {_one_line(finding['field'])}: "
                     f"{_one_line(finding['claim'])} [{refs}]")
        lines.append(f"    - reason: {_one_line(finding['reason'])}")
    ledger = view["ledger"]
    if ledger is not None:
        lines.append(f"- frame evidence ledger: `{cfg.work_dir / 'evidence' / sid / 'ledger.json'}`")
        lines.append("- sampled-frame observations (unverified local-model descriptions, not continuous playback):")
        for frame in ledger["frames"]:
            lines.append(f"  - {frame['evidence_id']} @ source {frame['source_time_s']:.3f}s "
                         f"(timestamp: {frame['timestamp_provenance']})")
            for fact in frame["observation"]["visible_facts"]:
                lines.append(f"    - observed description: {_one_line(fact)}")
            for uncertainty in frame["observation"]["uncertainties"]:
                lines.append(f"    - uncertain: {_one_line(uncertainty)}")
    return lines


def _annotation_lines(ann: dict) -> list[str]:
    if "error" in ann:
        return [f"- ANNOTATION FAILED: {ann['error']}"]
    if not ann:
        return []
    return [
        f"- subject: {ann.get('subject')}",
        f"- shot: {ann.get('shot_type')}; {ann.get('movement')}",
        f"- emotional metaphor: {ann.get('emotional_metaphor')}",
        f"- visual metaphor: {ann.get('visual_metaphor')}",
        f"- themes: {', '.join(ann.get('storytelling_themes', []))}",
        f"- mood: {ann.get('mood')}; light: {ann.get('light')}; "
        f"palette: {ann.get('color_palette')}",
        f"- holds: ~{ann.get('hold_seconds')}s; people: {ann.get('people')}",
        f"- cut notes: {ann.get('cut_notes')}",
    ]


def _edge(boundary: dict) -> str:
    label = {"source_start": "take start", "source_end": "take end",
             "speech_pause": "speech pause", "even_split": "even split"}.get(
                 boundary.get("method"), str(boundary.get("method")))
    silence = boundary.get("silence_s")
    return f"{label} {silence}s" if silence is not None else label


def _quote(text: str, words: int = 30) -> str:
    spoken = _one_line(text).split()
    return " ".join(spoken[:words]) + ("…" if len(spoken) > words else "")


def _segment_lines(cfg: Config, segment: dict, annotation: dict) -> list[str]:
    """One window, nested under its take. The window is an index into a
    continuous take, so it is rendered under that take rather than replacing
    it — the whole-take record stays the context for every window in it."""
    sid = segment["shot_id"]
    lines = [
        f"  - **{sid}** | {_fmt_ts(segment['in_s'])}–{_fmt_ts(segment['out_s'])} "
        f"({segment['duration_s']:.1f}s) | edges: {_edge(segment['start_boundary'])} → "
        f"{_edge(segment['end_boundary'])}"
    ]
    if hasattr(cfg, "sheets_dir"):
        # Windows are framed on demand, so point at a sheet only when it is
        # really there — a path to a missing file reads as a sheet to open.
        sheet = cfg.sheets_dir / (sid + ".jpg")
        lines.append(f"    - contact sheet: `{sheet}`" if sheet.exists()
                     else "    - frames not yet extracted for this window")
    speech = segment.get("speech")
    if speech:
        lines.append(f"    - speech [{_fmt_ts(speech['first_word_s'])}–{_fmt_ts(speech['last_word_s'])}]: "
                     f"\"{_quote(speech['text'])}\"")
    if not annotation and not segment.get("visual_tag", True):
        lines.append("    - not visually tagged: indexed for search only "
                     "(segments.index_only_sources)")
    lines += [f"    {line}" for line in _annotation_lines(annotation)]
    if annotation:
        # An untagged window has no claim to audit; the index line is the
        # whole record, and 170 "unchecked" lines would bury the tagged ones.
        lines += [f"    {line}" for line in _evidence_lines(cfg, segment, annotation)]
    return lines


def run(cfg: Config) -> str:
    out = cfg.out_dir

    def load(name: str, default):
        p = out / name
        return json.loads(p.read_text()) if p.exists() else default

    shots = load("shots.json", [])
    annotations = load("annotations.json", {})
    motion = load("motion.json", {})
    transcripts = load("transcripts.json", {})
    segments_by_parent: dict[str, list[dict]] = {}
    for segment in load("segments.json", []):
        segments_by_parent.setdefault(segment["parent_shot_id"], []).append(segment)
    selects = load("selects.json", {})
    marks = selects.get("marks", [])
    comments = selects.get("comments", [])

    lines = [
        "# Footage Bible — context",
        f"_Assembled {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M')} UTC. "
        "Machine-generated from ingest artifacts; the synthesis layer lives in "
        "bible-synthesis.md and is maintained in conversation._",
        "",
        "## Shot annotations (B-roll and A-roll)",
        "",
    ]

    for shot in shots:
        sid = shot["shot_id"]
        ann = annotations.get(sid, {})
        mot = motion.get(sid, {})
        lines.append(f"### {sid}")
        lines.append(f"- role: {shot['role']}")
        lines.append(
            f"- source: `{shot['source_path']}` @ {shot['in_s']:.1f}s–{shot['out_s']:.1f}s "
            f"({shot['duration_s']:.1f}s, tc {shot.get('start_timecode') or 'n/a'})"
        )
        lines.append(f"- contact sheet: `{cfg.sheets_dir / (sid + '.jpg')}`")
        if mot:
            lines.append(f"- measured motion: {mot.get('gloss', 'n/a')} "
                         f"(energy {mot.get('motion_energy')}, camera_ratio {mot.get('camera_ratio')})")
        lines.extend(_annotation_lines(ann))
        lines.extend(_evidence_lines(cfg, shot, ann))
        windows = segments_by_parent.get(sid, [])
        if windows:
            lines.append(f"- moments: {len(windows)} windows inside this take. Window edges are "
                         "indexing boundaries (a pause in the recorded speech, or an even "
                         "division), NOT cuts in the footage. Each window has its own graded "
                         "samples and its own tag; this take's record above stays the context.")
            for window in windows:
                lines.extend(_segment_lines(cfg, window, annotations.get(window["shot_id"], {})))
        lines.append("")

    lines += ["## A-roll transcript", ""]
    select_texts = [m["text"] for m in marks]
    for clip_id, tr in transcripts.items():
        lines.append(f"### {clip_id}")
        if tr.get("audio_source"):
            lines.append(f"- audio source: {tr['audio_source']}")
        for seg in tr.get("segments", []):
            marker = ""
            # crude but effective: mark segments whose text appears inside a
            # client-highlighted passage (or vice versa)
            for sel in select_texts:
                if seg["text"] and (seg["text"] in sel or sel in seg["text"]):
                    marker = " ★SELECT"
                    break
            lines.append(f"- [{_fmt_ts(seg['start'])}–{_fmt_ts(seg['end'])}]{marker} {seg['text']}")
        lines.append("")

    if marks:
        lines += ["## Marked passages (from Google Doc highlights/shading)", ""]
        for m in marks:
            lines.append(f"- ({m['color']}) {m['text']}")
        lines.append("")

    if comments:
        lines += ["## Transcript comments (from Google Doc)", ""]
        for c in comments:
            anchor = f' — on: "{c["anchor"]}"' if c["anchor"] else ""
            lines.append(f"- {c['author']}: {c['comment']}{anchor}")
        lines.append("")

    if cfg.draft_txt and cfg.draft_txt.is_file():
        lines += [
            "## The editor's draft radio cut (prior editorial intent — the arc already heard)",
            "",
            "```",
            cfg.draft_txt.read_text().strip(),
            "```",
            "",
        ]

    dest = out / "bible-context.md"
    dest.write_text("\n".join(lines))
    print(f"wrote {dest} ({len(lines)} lines)")
    return str(dest)
