"""Automatic eval of a rendered cut: picture-vs-words, exposure, rhythm, story.

Frontier model never sees pixels. Frames from the (graded) render go to the
local vision model in LM Studio; the spoken transcript goes to a local text
model in Ollama; everything else is measured with ffmpeg. Output is a JSON
scorecard plus a markdown report the story room reads.

Usage:
    uv run python -m videoeasy.cuteval \
        --cut data/films/<film>/editorial/cut-v2.json \
        --render data/films/<film>/editorial/renders/<film>-cut-v2.mp4 \
        --transcripts data/films/<film>/out/transcripts.json \
        --out data/films/<film>/editorial/eval/cut-v2-auto \
        [--film data/films/<film>] [--tier 1] [--skip-vlm] [--skip-story]

The film's profile (film.yaml, see film.py) says which clips are presenter
pieces and which are drone; it is found beside the cut unless --film names it.

Layout rule mirrors the Resolve lay-in: per beat, tier-N audio picks are
butted in order (span = sum of durations, then `gap_after_s` of room tone);
video candidates are laid from the beat start in order, BARE rows advance the
cursor without a clip, a clip is clipped to the room left and dropped when
under one second of room remains. Trailing room is bare.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from .film import Film, film_dir_for, load_film
from .sources import is_drone, label as source_label, to_camera
from .evalrun import INVALID, OK, UNCHECKED, bytes_hash, ffprobe_duration, fingerprint, refuse_overwrite, source_hash, text_hash, write_manifest
from .vlm import chat_vision, extract_json
from .textmodel import text_model, text_url

OLLAMA_URL = text_url()        # VIDEOEASY_TEXT_URL overrides (textmodel.py)
OLLAMA_MODEL = text_model()    # VIDEOEASY_TEXT_MODEL overrides
VISION_URL = "http://localhost:1234/v1"
VISION_MODEL = "google/gemma-4-26b-a4b"


# --------------------------------------------------------------------- layout
def layout(cut: dict, transcripts: dict, tier: int = 1, protected: dict[str, str] | None = None, profile: Film | None = None) -> list[dict]:
    """Return render-time stretches: the union of picture-row boundaries and
    spoken-pick boundaries, each with the words spoken during it and the one
    pick under it.

    Picture rows carrying `dest_in_s`/`dest_out_s` (cutbuild ≥ milestone two)
    are placed where they say. Legacy rows are laid consecutively from the
    beat start: BARE advances the cursor, a clip is clipped to the room left
    and dropped under one second of room, trailing room is bare. Either way
    a row that spans two picks becomes two stretches, so a judgment is
    attributed to exactly one pick and a long row is sampled per pick."""
    segs: list[dict] = []
    t = 0.0
    for bi, beat in enumerate(cut["beats"], 1):
        picks = [(j, a) for j, a in enumerate(beat["audio"], 1) if a.get("tier", 1) <= tier]
        if not picks:
            continue
        span = sum(a["duration_s"] for _, a in picks)
        end = t + span
        # words and pick spans in render time
        words: list[tuple[float, float, str]] = []
        spans: list[tuple[float, float, str, str]] = []
        pt = t
        for j, a in picks:
            clip = a["clip"].split("_w")[0]
            tr = transcripts.get(clip)
            if tr:
                for s in tr["segments"]:
                    for w in s["words"]:
                        if w["s"] >= a["in_s"] - 0.05 and w["e"] <= a["out_s"] + 0.05:
                            words.append((pt + (w["s"] - a["in_s"]), pt + (w["e"] - a["in_s"]), w["w"].strip()))
            spans.append((pt, pt + a["duration_s"], f"b{bi}p{j}", a["clip"]))
            pt += a["duration_s"]
        # picture rows in render time
        rows: list[dict] = []
        if beat["video"] and all("dest_in_s" in v for v in beat["video"]):
            for v in beat["video"]:
                kind = "bare" if v["type"] == "bare" else ("placeholder" if v["type"] == "placeholder" else "video")
                rows.append(dict(kind=kind, clip=v.get("clip"), src_in=v.get("in_s"), t0=v["dest_in_s"], t1=v["dest_out_s"],
                                 note=v.get("note", "") or v.get("text", "")))
        else:
            cursor = t
            for v in beat["video"]:
                room = end - cursor
                if room <= 0:
                    break
                if v["type"] == "bare":
                    d = min(v["duration_s"], room)
                    rows.append(dict(kind="bare", clip=None, src_in=None, t0=cursor, t1=cursor + d, note=v.get("note", "")))
                    cursor += d
                    continue
                if room < 1.0:
                    continue
                d = min(v["duration_s"], room)
                kind = "placeholder" if v["type"] == "placeholder" else "video"
                rows.append(dict(kind=kind, clip=v.get("clip"), src_in=v.get("in_s"), t0=cursor, t1=cursor + d,
                                 note=v.get("note", "") or v.get("text", "")))
                cursor += d
            if end - cursor > 0.25:
                rows.append(dict(kind="bare", clip=None, src_in=None, t0=cursor, t1=end, note="(uncovered tail of beat)"))
        # split every row at the pick boundaries it crosses
        for r in rows:
            for p0, p1, pid, pclip in spans:
                a, b = max(r["t0"], p0), min(r["t1"], p1)
                if b - a <= 0.04:
                    continue
                src_in = None if r["src_in"] is None else round(r["src_in"] + (a - r["t0"]), 3)
                segs.append(dict(beat=bi, title=beat["title"], kind=r["kind"], clip=r["clip"], src_in=src_in, t0=a, t1=b,
                                 note=r["note"], pick=pid, pick_clip=pclip,
                                 words=" ".join(w for w0, w1, w in words if w1 > a and w0 < b),
                                 to_camera=r["kind"] == "bare" and to_camera(pclip, profile), protected=(protected or {}).get(pid)))
        t = end + beat.get("gap_after_s", 0.0)
    for i, s in enumerate(segs):
        s["id"] = i
        s["dur"] = round(s["t1"] - s["t0"], 2)
    return segs


def pick_spans(cut: dict, tier: int = 1) -> list[dict]:
    """Render-time span of every tier-N audio pick, with the ids brollmatch uses (b<beat>p<n>)."""
    out = []
    t = 0.0
    for bi, beat in enumerate(cut["beats"], 1):
        picks = [(j, a) for j, a in enumerate(beat["audio"], 1) if a.get("tier", 1) <= tier]
        if not picks:
            continue
        for j, a in picks:
            out.append(dict(id=f"b{bi}p{j}", key=f"{a['clip']}@{a['in_s']:.1f}", beat=bi, t0=t, t1=t + a["duration_s"], text=a["text"],
                            clip=a["clip"], src_in=a["in_s"], src_out=a["out_s"], source_path=a.get("source_path")))
            t += a["duration_s"]
        t += beat.get("gap_after_s", 0.0)
    return out


# --------------------------------------------------------------- measurement
def frames_dir_for(base: Path, render_fp: str) -> Path:
    """Frames are cached under the render's content hash, so a re-render at the
    same path can never be judged on the old picture."""
    d = base / render_fp
    d.mkdir(parents=True, exist_ok=True)
    return d


def grab_frame(render: Path, t: float, out: Path, width: int = 960) -> bytes:
    """Extract one frame. `out` must live under `frames_dir_for(...)`: the cache
    hit below is safe only because the directory is keyed on the render's hash."""
    if not out.exists():
        p = subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t:.3f}", "-i", str(render), "-frames:v", "1",
                            "-vf", f"scale={width}:-2", "-q:v", "3", str(out)], capture_output=True, text=True)
        if p.returncode != 0 or not out.exists():
            raise RuntimeError(f"ffmpeg frame grab failed at {t:.2f}s: {p.stderr.strip()[:160]}")
    return out.read_bytes()


def signalstats(frame: Path) -> dict:
    """Luma/saturation stats of one extracted frame (0-255 luma). Reads the
    JPEG rather than seeking the render: movie=...:sp= lands on black or
    half-decoded frames after long seeks and reports YAVG 16."""
    spec = f"movie={frame},signalstats"
    keys = ["YAVG", "YMIN", "YMAX", "YLOW", "YHIGH", "SATAVG"]
    if not frame.exists():
        raise RuntimeError(f"frame missing: {frame}")
    p = subprocess.run(["ffprobe", "-v", "error", "-f", "lavfi", "-i", spec, "-read_intervals", "%+#1",
                        "-show_entries", "frame_tags=" + ",".join(f"lavfi.signalstats.{k}" for k in keys),
                        "-of", "json"], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffprobe signalstats failed: {p.stderr.strip()[:160]}")
    try:
        tags = json.loads(p.stdout)["frames"][0]["tags"]
        return {k: float(tags[f"lavfi.signalstats.{k}"]) for k in keys}
    except (ValueError, KeyError, IndexError) as e:
        raise RuntimeError(f"ffprobe signalstats returned no frame tags ({e})")


def joins(render: Path) -> dict:
    """Black frames and audio dropouts anywhere in the render: the faults a lay-in makes at the joins.

    Each detector is a separate check; when one fails its list is None and the
    status is `unchecked`, never an empty list that reads as a clean result."""
    out: dict = {"status": OK, "black": None, "silence": None, "errors": []}
    if not Path(render).exists():
        out.update(status=UNCHECKED, errors=[f"render not found: {render}"])
        return out
    p = subprocess.run(["ffmpeg", "-v", "info", "-i", str(render), "-vf", "blackdetect=d=0.04:pix_th=0.10", "-an", "-f", "null", "-"],
                       capture_output=True, text=True)
    if p.returncode == 0:
        out["black"] = [(round(float(m.group(1)), 2), round(float(m.group(2)), 2))
                        for m in re.finditer(r"black_start:([\d.]+) black_end:([\d.]+)", p.stderr)]
    else:
        out["errors"].append(f"blackdetect exit {p.returncode}: {p.stderr.strip()[-160:]}")
    p = subprocess.run(["ffmpeg", "-v", "info", "-i", str(render), "-vn", "-af", "silencedetect=n=-50dB:d=0.15", "-f", "null", "-"],
                       capture_output=True, text=True)
    if p.returncode == 0:
        starts = [float(x) for x in re.findall(r"silence_start: ([\d.]+)", p.stderr)]
        ends = [float(x) for x in re.findall(r"silence_end: ([\d.]+)", p.stderr)]
        out["silence"] = [(round(a, 2), round(b, 2)) for a, b in zip(starts, ends)]
    else:
        out["errors"].append(f"silencedetect exit {p.returncode}: {p.stderr.strip()[-160:]}")
    if out["errors"]:
        out["status"] = UNCHECKED
    return out


def exposure_flags(st: dict | None) -> list[str] | None:
    """None when there are no stats: an unmeasured frame is unchecked, not clean."""
    if not st:
        return None
    f = []
    if st["YAVG"] < 45:
        f.append("dark")
    if st["YAVG"] > 185:
        f.append("bright")
    if st["YHIGH"] >= 250 and st["YAVG"] > 150:
        f.append("clipped highlights")
    if st["YLOW"] <= 3 and st["YAVG"] < 70:
        f.append("crushed blacks")
    if st["SATAVG"] < 12:
        f.append("desaturated")
    return f


# ------------------------------------------------------------------ VLM eval
VLM_SYSTEM = (
    "You are a documentary picture editor reviewing a rough cut with a colleague. "
    "You are shown one or two frames from a stretch of the cut and the words spoken over that stretch. "
    "Judge only what is visible in the frames and what the words say. Be blunt and specific. "
    "Answer with a single JSON object and nothing else."
)

VLM_USER = """Beat: {title}
Stretch: {dur:.1f} s of the film. Picture: {kind_desc}
Words spoken over it: "{words}"

Return JSON with these keys:
- on_screen: what the frames show, at most 25 words
- relation: one of illustrates (the picture shows the thing or action the words describe), supports (relevant to the topic, mood or place of the words), generic (could sit under any words in this film), mismatch (the picture contradicts or has nothing to do with the words), distracting (the picture pulls attention from the words or is unwatchable)
- score: 0-3 where 3 = illustrates, 2 = supports, 1 = generic, 0 = mismatch or distracting
- technical: a list drawn from: underexposed, overexposed, soft focus, motion blur, shaky, colour cast, dead frame, blank card, none
- would_rather_see: at most 25 words describing the picture that would serve these words better, or an empty string if the picture already serves them
{bare_rule}"""

BARE_RULE = (
    "The picture is the interview two-shot with no cover. Score it 3 when the words are personal, emotional or a "
    "reaction and the speaker's face carries them; score it 1 when the words describe a place, plant, animal, object, "
    "document or action that the film could show instead."
)


PRESENTER_RULE = (
    "The picture is the presenter addressing the audience directly. Score it 3 when the words are a greeting, an "
    "acknowledgement, a hand-over or a sign-off that belongs on the presenter's face; score it 1 when the words "
    "describe a place, plant, animal, object or action that the film could show instead."
)


PROTECTED_RULE = (
    " The room decided this stretch stays uncovered ({rule}): score how well the speaker's face carries these words, "
    "and leave would_rather_see empty unless the picture itself is faulty."
)

RELATIONS = {"illustrates": 3, "supports": 2, "generic": 1, "mismatch": 0, "distracting": 0}
TECHNICAL = {"underexposed", "overexposed", "soft focus", "motion blur", "shaky", "colour cast", "dead frame", "blank card", "none"}
VLM_PROMPT_HASH = text_hash(VLM_SYSTEM, VLM_USER, BARE_RULE, PRESENTER_RULE, PROTECTED_RULE)


def validate_judgement(j, by_rule: bool = False) -> dict:
    """Normalise one picture-vs-words verdict. Status is ok only when the
    relation is one of the five asked for, the score is an integer 0-3 that
    agrees with the relation, and the technical list is a list. Anything else
    is invalid and carries no score: a 99 is not a 3, and a relation without
    its score is not a score.

    `by_rule` is for bare and placeholder stretches, whose score comes from
    the bare, presenter or card rule (a face may 'support' the words and
    still score 1 or 3); the relation is recorded but does not fix the score."""
    if not isinstance(j, dict):
        return {"status": INVALID, "score": None, "why": "response is not an object"}
    out = dict(j)
    problems = []
    rel = str(j.get("relation", "")).strip().lower()
    if rel not in RELATIONS:
        problems.append(f"relation {rel!r} not one of {sorted(RELATIONS)}")
    try:
        sc = int(j.get("score"))
    except (TypeError, ValueError):
        sc = None
        problems.append("score missing or not an integer")
    if sc is not None and not 0 <= sc <= 3:
        problems.append(f"score {sc} out of range 0-3")
        sc = None
    if sc is not None and rel in RELATIONS and RELATIONS[rel] != sc and not by_rule:
        problems.append(f"relation {rel} implies {RELATIONS[rel]} but score is {sc}")
        sc = None
    tech = j.get("technical", [])
    if isinstance(tech, str):
        tech = [tech]
    if not isinstance(tech, list):
        tech = []
        problems.append("technical is not a list")
    tech = [str(x).strip().lower() for x in tech]
    unknown = [x for x in tech if x not in TECHNICAL]
    known = [x for x in tech if x in TECHNICAL]
    out["technical"] = known if (known or unknown) else ["none"]
    if unknown:
        out["technical_unknown"] = unknown
    out["relation"] = rel
    out["score"] = sc
    out["status"] = INVALID if problems else OK
    if problems:
        out["why"] = "; ".join(problems)
    return out


def vlm_judge(seg: dict, frames: list[bytes], url: str, model: str) -> dict:
    if seg["kind"] == "bare" and seg.get("to_camera"):
        kind_desc = "the presenter speaking to camera, uncovered."
        bare_rule = PRESENTER_RULE
    elif seg["kind"] == "bare":
        kind_desc = "the interview two-shot, uncovered."
        bare_rule = BARE_RULE
    elif seg["kind"] == "placeholder":
        kind_desc = "a placeholder card standing in for material not yet in hand."
        bare_rule = "Score a placeholder card 1 and say in would_rather_see what real picture the card promises."
    else:
        kind_desc = "a B-roll clip laid over the interview audio."
        bare_rule = ""
    if seg["kind"] == "bare" and seg.get("protected"):
        bare_rule += PROTECTED_RULE.format(rule=seg["protected"])
    user = VLM_USER.format(title=seg["title"], dur=seg["dur"], kind_desc=kind_desc,
                           words=seg["words"] or "(no words: room tone)", bare_rule=bare_rule)
    try:
        text = chat_vision(url, model, VLM_SYSTEM, user, frames, max_tokens=1500, temperature=0.2)
        j = extract_json(text)
    except Exception as e:  # noqa: BLE001
        return {"status": UNCHECKED, "score": None, "error": str(e)[:200], "model": model, "prompt_hash": VLM_PROMPT_HASH}
    out = validate_judgement(j, by_rule=seg["kind"] in ("bare", "placeholder"))
    out.update(model=model, prompt_hash=VLM_PROMPT_HASH)
    return out


def judged(seg: dict) -> bool:
    """True when the stretch carries an accepted picture verdict."""
    v = seg.get("vlm") or {}
    return v.get("status") == OK and isinstance(v.get("score"), int)


def coverage(segs: list[dict], skipped: bool) -> dict:
    """What the picture judge actually covered, and why the rest is unchecked."""
    total_s = sum(s["dur"] for s in segs)
    ok = [s for s in segs if judged(s)]
    reasons: dict[str, int] = {}
    for s in segs:
        if judged(s):
            continue
        v = s.get("vlm") or {}
        why = "not run" if skipped or not v else (v.get("status") or UNCHECKED) + (": " + v["error"][:60] if v.get("error") else "") \
            + (": " + v["why"][:60] if v.get("why") else "")
        reasons[why] = reasons.get(why, 0) + 1
    return dict(stretches=len(segs), total_s=round(total_s, 1), judged=len(ok), judged_s=round(sum(s["dur"] for s in ok), 1),
                unchecked=len(segs) - len(ok), unchecked_s=round(total_s - sum(s["dur"] for s in ok), 1), reasons=reasons)


# ------------------------------------------------------------ dialogue level
def pick_loudness(render: Path, spans: list[dict], measure=None, channels=None, profile: Film | None = None) -> list[dict]:
    """Integrated loudness and true peak of each pick's span of the render.
    A measurement, not a judgment: the delivery target is the editor's to set."""
    if measure is None:
        from .layin import lufs_peak
        measure = lambda a, b: lufs_peak(str(render), a, b, False)  # noqa: E731
    if channels is None:
        from .layin import channel_lufs
        channels = lambda a, b: channel_lufs(str(render), a, b)  # noqa: E731
    out = []
    for sp in spans:
        row = dict(pick=sp["id"], beat=sp["beat"], clip=sp["clip"], context=source_label(sp["clip"], profile), t0=round(sp["t0"], 2),
                   t1=round(sp["t1"], 2), lufs=None, peak=None, lr_diff=None, status=UNCHECKED)
        if sp["t1"] - sp["t0"] < 0.4:
            row["error"] = "under 0.4 s: too short to integrate"
        else:
            try:
                I, P = measure(sp["t0"], sp["t1"])
                L, R = channels(sp["t0"], sp["t1"])
                row.update(lufs=I, peak=P, lr_diff=None if L is None or R is None else round(abs(L - R), 1), status=OK if I is not None else UNCHECKED)
            except Exception as e:  # noqa: BLE001
                row["error"] = str(e)[:120]
        out.append(row)
    return out


def loudness_summary(rows: list[dict], target: float = -24.0, spread: float = 3.0, mono: dict[str, bool] | None = None) -> dict:
    """Median per recording context, and the picks more than `spread` LU under the target or over -1 dBTP. The
    one-channel check applies to the picks the lay-in plan maps as mono (`mono`); a stereo sit-down pick carries
    the lav on one side and the camera on the other and differs left/right by design."""
    ok = [r for r in rows if r["status"] == OK]
    by_ctx: dict[str, list[float]] = {}
    for r in ok:
        by_ctx.setdefault(r["context"], []).append(r["lufs"])
    med = {k: round(sorted(v)[len(v) // 2], 1) for k, v in by_ctx.items()}
    quiet = [r for r in ok if r["lufs"] < target - spread]
    hot = [r for r in ok if r["peak"] is not None and r["peak"] > -1.0]
    mono = mono or {}
    one = [r for r in ok if mono.get(r["pick"]) and r.get("lr_diff") is not None and r["lr_diff"] > 3.0]   # a mono pick on one channel only
    return dict(target_lufs=target, measured=len(ok), unmeasured=len(rows) - len(ok), median_by_context=med,
                quiet=[dict(pick=r["pick"], context=r["context"], lufs=r["lufs"], peak=r["peak"]) for r in quiet],
                hot=[dict(pick=r["pick"], context=r["context"], lufs=r["lufs"], peak=r["peak"]) for r in hot],
                one_channel=[dict(pick=r["pick"], context=r["context"], lufs=r["lufs"], lr_diff=r["lr_diff"]) for r in one],
                mono_checked=sum(1 for r in ok if mono.get(r["pick"])))


# ---------------------------------------------------------------- rhythm eval
def rhythm(cut: dict, segs: list[dict], tier: int) -> dict:
    per_beat = []
    t = 0.0
    for bi, beat in enumerate(cut["beats"], 1):
        picks = [a for a in beat["audio"] if a.get("tier", 1) <= tier]
        if not picks:
            continue
        bs = [s for s in segs if s["beat"] == bi]
        spoken = sum(a["duration_s"] for a in picks)
        bare = sum(s["dur"] for s in bs if s["kind"] == "bare")
        shots = [s for s in bs if s["kind"] != "bare"]
        longest_bare = max([s["dur"] for s in bs if s["kind"] == "bare"], default=0.0)
        # jump cuts: consecutive picks from the same source clip, joined while the picture is bare
        jumps = []
        pt = t
        for a, b in zip(picks, picks[1:]):
            pt += a["duration_s"]
            same = a["clip"].split("_w")[0] == b["clip"].split("_w")[0]
            if same and any(s["kind"] == "bare" and s["t0"] - 0.05 <= pt <= s["t1"] + 0.05 for s in bs):
                jumps.append(round(pt, 1))
        per_beat.append(dict(beat=bi, title=beat["title"], spoken_s=round(spoken, 1), bare_s=round(bare, 1),
                             bare_pct=round(100 * bare / spoken) if spoken else 0, shots=len(shots),
                             mean_hold_s=round(sum(s["dur"] for s in shots) / len(shots), 1) if shots else None,
                             longest_bare_s=round(longest_bare, 1), jump_cuts_at=jumps, t0=round(t, 1)))
        t += spoken + beat.get("gap_after_s", 0.0)
    total = sum(b["spoken_s"] for b in per_beat)
    bare = sum(b["bare_s"] for b in per_beat)
    return dict(total_spoken_s=round(total, 1), bare_s=round(bare, 1), bare_pct=round(100 * bare / total),
                jump_cuts=sum(len(b["jump_cuts_at"]) for b in per_beat), beats=per_beat)


# ----------------------------------------------------------------- story eval
STORY_PROMPT = """You are watching a rough cut of a short documentary for the first time. You know nothing about it beyond what is spoken. Below is everything said in the film, in order, grouped by the editor's beats with their length in seconds. Read it as a viewer, not as an editor.

{transcript}

Answer with one JSON object:
{{
  "logline": "what this film is about, in one sentence, as you understood it",
  "who_is_who": "who the speakers seem to be and how you worked that out, at most 40 words",
  "beats": [ for every beat, in order: {{"beat": <number>, "clarity": 1-5 (did you follow it), "pull": 1-5 (did it hold you), "drag": true/false, "problem": "the one thing wrong with this beat as a viewer, or empty", "wants_to_see": "what you wished the picture showed here, at most 15 words"}} ],
  "lost_me_at": "the first beat number where attention drifted, or null",
  "missing": ["things a viewer needs and never gets, each at most 15 words"],
  "reorder": ["beats that would land better elsewhere and where, each at most 20 words"],
  "cut_first": ["stretches that should go if the film must lose two minutes, each at most 20 words"],
  "ending": "does the film end, or stop? one sentence"
}}"""


def story_eval(cut: dict, tier: int, url: str = OLLAMA_URL, model: str = OLLAMA_MODEL) -> dict:
    lines = []
    for bi, beat in enumerate(cut["beats"], 1):
        picks = [a for a in beat["audio"] if a.get("tier", 1) <= tier]
        if not picks:
            continue
        lines.append(f"\n## Beat {bi}: {beat['title']}  ({sum(a['duration_s'] for a in picks):.0f} s)")
        for a in picks:
            lines.append(f"- {a['text']}")
    prompt = STORY_PROMPT.format(transcript="\n".join(lines))
    r = httpx.post(f"{url}/api/chat", json={"model": model, "stream": False, "think": False, "format": "json",
                                            "options": {"temperature": 0.3, "num_ctx": 32768},
                                            "messages": [{"role": "user", "content": prompt}]}, timeout=900)
    r.raise_for_status()
    return json.loads(r.json()["message"]["content"])


# --------------------------------------------------------------------- report
def write_report(out: Path, cut: dict, segs: list[dict], rh: dict, story: dict | None, render: Path,
                 manifest: dict | None = None, skipped_vlm: bool = False, intent_findings: list[dict] | None = None,
                 loud: dict | None = None, rules: dict | None = None) -> None:
    L = [f"# Automatic eval of `{render.name}` against `cut-v{cut.get('version')}`", ""]
    if manifest:
        L.append(f"Provenance: render sha `{manifest['render']['sha']}`, cut sha `{manifest['cut']['sha']}`, "
                 f"tool `{manifest['tool_sha']}`; full manifest beside this report.")
    cov = coverage(segs, skipped_vlm)
    scored = [s for s in segs if judged(s)]
    if skipped_vlm:
        L.append("Picture-vs-words: **not run** (--skip-vlm). No picture score exists for this render.")
    elif not scored:
        L.append(f"Picture-vs-words: **no accepted judgment** on any of {cov['stretches']} stretches: "
                 + "; ".join(f"{k} ×{v}" for k, v in cov["reasons"].items()) + ".")
    else:
        wmean = sum(s["vlm"]["score"] * s["dur"] for s in scored) / sum(s["dur"] for s in scored)
        L.append(f"Picture-vs-words (gemma-4, duration-weighted mean of 0-3): **{wmean:.2f}** over "
                 f"{cov['judged']} of {cov['stretches']} stretches ({cov['judged_s']:.0f} of {cov['total_s']:.0f} s).")
        if cov["unchecked"]:
            L.append(f"Unchecked: **{cov['unchecked']} stretches, {cov['unchecked_s']:.0f} s** carry no accepted judgment ("
                     + "; ".join(f"{k} ×{v}" for k, v in cov["reasons"].items()) + "). They are not in the mean.")
        dist = {k: round(sum(s["dur"] for s in scored if s["vlm"]["score"] == k)) for k in range(4)}
        L.append("Seconds at each score: " + ", ".join(f"{k}: {v}s" for k, v in dist.items()))
        real = [s for s in scored if s["kind"] != "placeholder"]
        if len(real) < len(scored):
            rmean = sum(s["vlm"]["score"] * s["dur"] for s in real) / sum(s["dur"] for s in real)
            L.append(f"Excluding placeholder cards (fixed at 1 until the material arrives): **{rmean:.2f}** over "
                     f"{sum(s['dur'] for s in scored) - sum(s['dur'] for s in real):.0f}s fewer.")
    L.append(f"Bare two-shot: **{rh['bare_pct']}%** of {rh['total_spoken_s']:.0f} spoken seconds; "
             f"jump cuts on bare picture: **{rh['jump_cuts']}**.")
    j = rh.get("joins") or {}
    if not j:
        L.append("Joins: **not measured**.")
    else:
        parts = []
        for key, label in (("black", "black stretches"), ("silence", "audio dropouts over 0.15 s")):
            v = j.get(key)
            if v is None:
                parts.append(f"{label}: unchecked")
            else:
                parts.append(f"**{len(v)}** {label}" + (f" (first at {v[0][0]}s)" if v else ""))
        L.append("Joins: " + "; ".join(parts) + "." + (" Errors: " + "; ".join(j.get("errors", [])) if j.get("errors") else ""))
    unexposed = [s for s in segs if s.get("exposure_flags") is None]
    if unexposed:
        L.append(f"Exposure: **{len(unexposed)}** stretches unmeasured (frame grab or signalstats failed).")
    prot = [s for s in segs if s.get("protected")]
    if prot:
        L.append(f"Protected stretches (bare by decision): **{len(prot)}**, {sum(s['dur'] for s in prot):.0f} s; their scores say how the face "
                 f"carries the words, never that cover is wanted.")
    if intent_findings is not None:
        broken = [f for f in intent_findings if f["status"] == "broken"]
        L.append(f"Editorial intent (editorial/intent.json, checked without a model): **{'BROKEN' if broken else 'no rule broken'}**; "
                 + ", ".join(f"{f['rule']} {f['status']}" for f in intent_findings if f["status"] != "honoured") + ".")
    if loud is not None:
        L.append(f"Dialogue level (render, per pick, working target {loud['target_lufs']} LUFS; the editor has not set a delivery target): "
                 f"{loud['measured']} measured, {loud['unmeasured']} unmeasured; median by context "
                 + ", ".join(f"{k} {v}" for k, v in loud["median_by_context"].items())
                 + f"; **{len(loud['quiet'])}** picks more than 3 LU under target, **{len(loud['hot'])}** over -1 dBTP"
                 + (f"; **{len(loud['one_channel'])} picks on one channel only** (left/right differ by over 3 LU: the render ignored the channel map)."
                    if loud.get("one_channel") else (f"; all {loud.get('mono_checked', 0)} mono picks reach both channels." if loud.get("mono_checked")
                                                     else "; channel balance unchecked (no lay-in plan beside the cut names the mono picks).")))
    if rules is not None:
        L.append(f"Cover rules (measured on the render and the transcript, no model): talking cover **{len(rules['talking'])}** stretches "
                 f"({sum(s['dur'] for s in rules['talking']):.0f} s; a person on screen speaks while another pick's audio plays), "
                 f"{len(rules['talking_unknown'])} stretches from untranscribed sources unknown; camera shake over {rules['max_shake_pct']}% of frame width "
                 + (f"**{len(rules['shaky'])}** cover stretches ({sum(s['dur'] for s in rules['shaky']):.0f} s), {rules['unmeasured']} unmeasured"
                    + (f"; {len(rules['sync_shaky'])} sync-picture stretches over the limit are allowed (the speaker's own take)." if rules.get("sync_shaky") else ".")
                    if rules.get("measured") else "**not measured** (--skip-steadiness)."))
        if rules.get("measured") and (rules.get("drone_clean") or rules.get("drone_unclean") or rules.get("drone_unmeasured")):
            L.append(f"Drone moves (measured on the render, no model): **{len(rules['drone_clean'])}** drone stretches on one clean move or hold, "
                     f"**{len(rules['drone_unclean'])}** crossing a turn, a hesitation or shake "
                     f"({sum(s['dur'] for s in rules['drone_unclean']):.0f} s), {len(rules['drone_unmeasured'])} too short to measure.")
    L += ["", "## Per beat", "", "| beat | spoken | bare | shots | mean hold | longest bare | jump cuts | pic score | viewer clarity/pull | drag |",
          "|---|---|---|---|---|---|---|---|---|---|"]
    sb = {b["beat"]: b for b in (story or {}).get("beats", []) if isinstance(b, dict)}
    for b in rh["beats"]:
        bs = [s for s in scored if s["beat"] == b["beat"]]
        nb = sum(1 for s in segs if s["beat"] == b["beat"])
        ps = f"{sum(s['vlm']['score'] * s['dur'] for s in bs) / sum(s['dur'] for s in bs):.1f}" if bs else "-"
        if bs and len(bs) < nb:
            ps += f" ({len(bs)}/{nb})"
        v = sb.get(b["beat"], {})
        L.append(f"| {b['beat']} {b['title'][:32]} | {b['spoken_s']:.0f}s | {b['bare_pct']}% | {b['shots']} | "
                 f"{b['mean_hold_s'] or '-'} | {b['longest_bare_s']}s | {len(b['jump_cuts_at'])} | {ps} | "
                 f"{v.get('clarity', '-')}/{v.get('pull', '-')} | {'yes' if v.get('drag') else ''} |")
    L += ["", "## Stretches scoring 0 or 1 (fix list)", ""]
    for s in segs:
        v = s.get("vlm", {})
        if v.get("score") is not None and v["score"] <= 1:
            tag = f" [protected by {s['protected']}: not a cover request]" if s.get("protected") else ""
            L.append(f"- **{s['t0']:.0f}-{s['t1']:.0f}s** beat {s['beat']} {s['kind']} {s['clip'] or ''} "
                     f"(score {v['score']}, {v.get('relation')}){tag}: on screen: {v.get('on_screen', '')}. "
                     f"Words: \"{(s['words'] or '')[:90]}\". Rather see: {v.get('would_rather_see', '')}")
    if intent_findings:
        L += ["", "## Editorial intent (rules, not scores)", ""]
        for f in intent_findings:
            L.append(f"- {f['rule']}: **{f['status']}**. {f['evidence']}")
    if rules is not None and (rules["talking"] or rules["shaky"]):
        L += ["", "## Cover rules: stretches that are not cover (measured, no model)", "",
              "Each is a fix whatever the picture score says: the judge sees stills and cannot see lips or shake.", ""]
        for s in rules["talking"]:
            L.append(f"- **{s['t0']:.0f}-{s['t1']:.0f}s** beat {s['beat']} {s['clip']} under {s['pick']} ({s['pick_clip']}): "
                     f"{s['talking_s']} s of the covered person's speech in the source range (picture score {(s.get('vlm') or {}).get('score')})")
        for s in rules["shaky"]:
            st = s["steadiness"]
            L.append(f"- **{s['t0']:.0f}-{s['t1']:.0f}s** beat {s['beat']} {s['clip']} under {s['pick']}: camera shake {st['shake_pct']}% "
                     f"(pan {st['pan_pct_per_s']}%/s, picture score {(s.get('vlm') or {}).get('score')}, judge technical {(s.get('vlm') or {}).get('technical')})")
    if rules is not None and rules.get("drone_unclean"):
        L += ["", "## Drone stretches not on one clean move (measured, no model)", "",
              "The stills cannot show a move; the render's own frames can. Each lists the pieces the stretch crosses.", ""]
        for s in rules["drone_unclean"]:
            parts = "; ".join(f"{x['label']}" + (f" ({', '.join(x['why'])})" if x["why"] else "") for x in s["move"]["segments"])
            L.append(f"- **{s['t0']:.0f}-{s['t1']:.0f}s** beat {s['beat']} {s['clip']} under {s['pick']}: longest clean piece "
                     f"{s['move']['share']:.0%} of the stretch; {parts}")
    if loud is not None and loud.get("one_channel"):
        L += ["", "## Picks rendered on one channel only (measured)", "",
              "Left and right differ by more than 3 LU on a dialogue pick: the source channel map was not honoured by the render. "
              "Re-render after reopening the timeline (`layin` does this); the level table above is 3 LU low on these.", ""]
        for r in loud["one_channel"]:
            L.append(f"- {r['pick']} ({r['context']}): {r['lufs']} LUFS, left/right differ by {r['lr_diff']} LU")
    if loud is not None and (loud["quiet"] or loud["hot"]):
        L += ["", "## Dialogue level: picks outside the working band", "",
              "A measurement of the render. The fix is an agreed target and a measured gain or dynamics change, not more clip gain "
              "(XT4 items already sit at the API's +30 dB cap).", ""]
        for r in loud["quiet"]:
            L.append(f"- {r['pick']} ({r['context']}): {r['lufs']} LUFS, peak {r['peak']} dBTP: quiet")
        for r in loud["hot"]:
            L.append(f"- {r['pick']} ({r['context']}): {r['lufs']} LUFS, peak {r['peak']} dBTP: hot")
    unc = [s for s in segs if not judged(s)]
    if unc and not skipped_vlm:
        L += ["", "## Unchecked stretches (no accepted picture judgment)", ""]
        for s in unc:
            v = s.get("vlm") or {}
            L.append(f"- **{s['t0']:.0f}-{s['t1']:.0f}s** beat {s['beat']} {s['kind']} {s['clip'] or ''}: "
                     f"{v.get('status', UNCHECKED)} {v.get('error', v.get('why', ''))[:120]}")
    L += ["", "## Exposure flags (measured on the graded render)", ""]
    for s in segs:
        if s.get("exposure_flags"):
            st = s["stats"]
            L.append(f"- {s['t0']:.0f}s {s['kind']} {s['clip'] or ''}: {', '.join(s['exposure_flags'])} "
                     f"(YAVG {st['YAVG']:.0f}, range {st['YLOW']:.0f}-{st['YHIGH']:.0f}, sat {st['SATAVG']:.0f})")
    tech = [(s, s["vlm"]["technical"]) for s in scored if s["vlm"].get("technical") and s["vlm"]["technical"] not in (["none"], "none", [])]
    if tech:
        L += ["", "Vision-model technical notes:"]
        for s, tflags in tech:
            L.append(f"- {s['t0']:.0f}s {s['kind']} {s['clip'] or ''}: {tflags}")
    if story:
        L += ["", "## Viewer read (local text model, transcript only)", "",
              f"**Logline:** {story.get('logline', '')}", "", f"**Who is who:** {story.get('who_is_who', '')}", "",
              f"**Lost me at:** beat {story.get('lost_me_at')}", "", f"**Ending:** {story.get('ending', '')}", ""]
        for key in ("missing", "reorder", "cut_first"):
            if story.get(key):
                L.append(f"**{key}:**")
                L += [f"- {x}" for x in story[key]]
                L.append("")
        L.append("Per beat:")
        for b in story.get("beats", []):
            if isinstance(b, dict):
                L.append(f"- beat {b.get('beat')}: clarity {b.get('clarity')}, pull {b.get('pull')}"
                         f"{', drags' if b.get('drag') else ''}. {b.get('problem', '')} Wants: {b.get('wants_to_see', '')}")
    out.with_suffix(".md").write_text("\n".join(L) + "\n")


def cover_rules(segs: list[dict], transcripts: dict, render: Path, cache_dir: Path | None, max_shake: float,
                cut: dict | None = None, tier: int = 1, people: dict[str, str] | None = None, exempt: set[str] | None = None,
                profile: Film | None = None) -> dict:
    """Measured cover rules on the render's stretches (no model). Talking cover: transcribed speech of the
    covered source inside the stretch's source range. Shaky: camera path residual over `max_shake` % of frame
    width, from the render's own frames (steadiness.py); `cache_dir` None skips the measurement."""
    from . import speech as speech_mod, steadiness as steady_mod
    talking, unknown, shaky, sync_shaky = [], [], [], []
    starts = {sp["id"]: (sp["clip"], sp["src_in"], sp["t0"]) for sp in pick_spans(cut, tier)} if cut else {}
    people = people or {}
    for s in segs:
        if s["kind"] != "video" or not s.get("clip") or s.get("src_in") is None:
            continue
        if speech_mod.in_sync(s["clip"], s["src_in"], s["t0"], starts.get(s.get("pick"))):
            s["talking_s"], s["talking"], s["sync"] = None, False, True
            continue
        if speech_mod.NOBODY.match(people.get(s["clip"]) or ""):
            s["talking_s"], s["talking"], s["off_camera"] = None, False, True
            continue
        if speech_mod.is_exempt(s["clip"], exempt):
            s["talking_s"], s["talking"], s["exempt"] = None, False, True
            continue
        sp = speech_mod.spoken_seconds(transcripts, s["clip"], s["src_in"], s["src_in"] + s["dur"])
        s["talking_s"] = sp
        s["talking"] = sp is not None and sp >= speech_mod.TALKING_COVER_MIN_S
        (unknown if sp is None else talking if s["talking"] else []).append(s)
    measured = cache_dir is not None
    unmeasured = 0
    drone_clean, drone_unclean, drone_unmeasured = [], [], []
    if measured:
        # drone cover: the move the render actually shows, measured on its own frames (moves.py, 29 Sep 2026).
        # A stretch should sit on one clean move or hold; its length is the builder's business, so only the move is judged.
        from . import moves as moves_mod
        drones = [s for s in segs if s["kind"] == "video" and s.get("clip") and is_drone(s["clip"], profile) and not s.get("sync")]
        if drones:
            mrows = moves_mod.cached_motion(render, cache_dir)
            edge = 0.25   # the pair across a cut is not a camera move
            for s in drones:
                sub = moves_mod.segments(moves_mod.slice_rows(mrows, s["t0"] + edge, s["t1"] - edge), t_end=s["t1"] - edge)
                span = max(s["t1"] - s["t0"] - 2 * edge, 1e-6)
                good = [x for x in sub if moves_mod.clean_except_length(x)]
                best = max(good, key=lambda x: x["t1"] - x["t0"], default=None)
                share = (best["t1"] - best["t0"]) / span if best else 0.0
                s["move"] = dict(label=(best or (sub[0] if sub else {})).get("label"), share=round(share, 2),
                                 segments=[dict(t0=x["t0"], t1=x["t1"], label=x["label"], why=x["why"]) for x in sub])
                (drone_unmeasured if not sub else drone_clean if share >= 0.9 else drone_unclean).append(s)
        pairs = steady_mod.cached_path(render, cache_dir)
        for s in segs:
            if s["kind"] != "video":
                continue
            st = steady_mod.shake_stats(steady_mod.slice_pairs(pairs, s["t0"], s["t1"]))
            s["steadiness"] = st
            s["shaky"] = steady_mod.shaky(st, max_shake)
            if st.get("status") != OK:
                unmeasured += 1
            elif s["shaky"] and s.get("sync"):
                sync_shaky.append(s)   # the speaker's own picture: the shake is the price of the take, not a cover fault (the editor, 28 Sep 2026)
            elif s["shaky"]:
                shaky.append(s)
    return dict(talking=talking, talking_unknown=unknown, shaky=shaky, sync_shaky=sync_shaky, measured=measured, unmeasured=unmeasured,
                max_shake_pct=max_shake, drone_clean=drone_clean, drone_unclean=drone_unclean, drone_unmeasured=drone_unmeasured)


# ----------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cut", required=True)
    ap.add_argument("--render", required=True)
    ap.add_argument("--transcripts", required=True)
    ap.add_argument("--out", required=True, help="output stem; writes <stem>.json and <stem>.md")
    ap.add_argument("--frames-dir", default=None)
    ap.add_argument("--tier", type=int, default=1)
    ap.add_argument("--skip-vlm", action="store_true")
    ap.add_argument("--skip-story", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--overwrite", action="store_true", help="replace an existing report at --out")
    ap.add_argument("--intent", default=None, help="editorial/intent.json (default: intent.json beside --cut when present)")
    ap.add_argument("--skip-loudness", action="store_true")
    ap.add_argument("--resume", default=None, help="prior scorecard json of the same render, cut and prompt: its accepted judgments are reused")
    ap.add_argument("--skip-steadiness", action="store_true", help="do not measure camera shake on the render")
    ap.add_argument("--max-shake", type=float, default=None, help="camera shake limit, %% of frame width (default steadiness.MAX_SHAKE_PCT)")
    ap.add_argument("--film", default=None, help="film directory holding film.yaml (default: the nearest one above --cut)")
    args = ap.parse_args(argv)
    profile = load_film(args.film or film_dir_for(args.cut))

    render = Path(args.render)
    out = Path(args.out)
    refuse_overwrite([out.with_suffix(".json"), out.with_suffix(".md")], args.overwrite)
    if not render.exists():
        raise SystemExit(f"render not found: {render}")
    dur = ffprobe_duration(render)
    if dur is None:
        raise SystemExit(f"ffprobe cannot read {render}; nothing was scored")
    cut = json.load(open(args.cut))
    transcripts = json.load(open(args.transcripts))
    render_fp = fingerprint(render)
    frames_dir = frames_dir_for(Path(args.frames_dir or out.parent / (out.name + "-frames")), render_fp)

    from . import intent as intent_mod
    intent_path = Path(args.intent) if args.intent else Path(args.cut).parent / "intent.json"
    intent_doc = json.loads(intent_path.read_text(encoding="utf-8")) if intent_path.exists() else None
    protected = intent_mod.protected_picks(cut, intent_doc, args.tier) if intent_doc else {}
    segs = layout(cut, transcripts, args.tier, protected, profile)
    laid = segs[-1]["t1"] if segs else 0
    print(f"{len(segs)} stretches; layout ends {laid:.1f}s, render is {dur:.1f}s; render sha {render_fp}", file=sys.stderr)
    if abs(laid - dur) > 2.0:
        print("WARNING: layout and render disagree by more than 2 s; the map is wrong somewhere", file=sys.stderr)

    # frames + stats: a failed grab or measurement leaves the stretch unchecked, never clean
    for s in segs:
        pts = [s["t0"] + 0.5, s["t0"] + s["dur"] * 0.75] if s["dur"] >= 2.5 else [s["t0"] + s["dur"] / 2]
        s["frames"], s["frame_times"], s["frame_hashes"] = [], [], []
        try:
            for k, t in enumerate(pts):
                f = frames_dir / f"s{s['id']:03d}_{k}.jpg"
                s["frame_hashes"].append(bytes_hash(grab_frame(render, t, f)))
                s["frames"].append(str(f))
                s["frame_times"].append(round(t, 3))
            s["stats"] = signalstats(Path(s["frames"][0]))
            s["exposure_flags"] = exposure_flags(s["stats"])
        except Exception as e:  # noqa: BLE001
            s["stats"], s["exposure_flags"] = None, None
            s["frame_error"] = str(e)[:200]

    # cover rules, measured: the covered person's speech from the transcript, camera shake from the render's own frames
    from . import speech as speech_mod, steadiness as steady_mod
    max_shake = steady_mod.MAX_SHAKE_PCT if args.max_shake is None else args.max_shake
    ann_path = Path(args.cut).parent.parent / "out/annotations.json"
    people = {k: (a.get("people") or "") for k, a in json.loads(ann_path.read_text()).items()} if ann_path.exists() else {}
    rules = cover_rules(segs, transcripts, render, None if args.skip_steadiness else frames_dir, max_shake, cut, args.tier, people,
                        exempt=intent_mod.talking_exempt(intent_doc), profile=profile)
    print(f"cover rules: {len(rules['talking'])} talking-cover stretches, {len(rules['shaky'])} shaky over {max_shake}%"
          + ("" if rules["measured"] else " (steadiness not measured)"), file=sys.stderr)

    reused = 0
    prior = {}
    if args.resume and not args.skip_vlm:
        pj = json.loads(Path(args.resume).read_text())
        pm = pj.get("manifest", {})
        if pm.get("cut", {}).get("sha") != fingerprint(args.cut) or (pm.get("vision") or {}).get("prompt_sha") != VLM_PROMPT_HASH:
            raise SystemExit("--resume refused: the prior scorecard is of a different cut or prompt")
        if pm.get("render", {}).get("sha") != render_fp:
            # a re-render of the same timeline (audio fixed, picture unchanged): judgments are reused only where the
            # grabbed frames hash the same, which the per-stretch check below enforces
            print(f"--resume: prior scorecard is of render {pm.get('render', {}).get('sha')}; reusing judgments only on identical frames", file=sys.stderr)
        prior = {(s["t0"], s["t1"], s["kind"], s["clip"]): s["vlm"] for s in pj["stretches"] if judged(s)}
    if not args.skip_vlm:
        def job(s):
            nonlocal reused
            key = (s["t0"], s["t1"], s["kind"], s["clip"])
            if key in prior and s["frame_hashes"] == next((p["frame_hashes"] for p in pj["stretches"]
                                                         if (p["t0"], p["t1"], p["kind"], p["clip"]) == key), None):
                s["vlm"] = dict(prior[key], reused_from=args.resume)
                reused += 1
                return
            if not s["frames"]:
                s["vlm"] = {"status": UNCHECKED, "score": None, "error": "no frames: " + s.get("frame_error", "")}
                return
            frames = [Path(f).read_bytes() for f in s["frames"]]
            s["vlm"] = vlm_judge(s, frames, VISION_URL, VISION_MODEL)
            print(f"  {s['id']:3d} {s['t0']:6.1f}s {s['kind']:11} {s['clip'] or '':14} -> {s['vlm'].get('status')} "
                  f"{s['vlm'].get('score')} {s['vlm'].get('relation', s['vlm'].get('error', s['vlm'].get('why', '')))}", file=sys.stderr)
        with ThreadPoolExecutor(args.workers) as ex:
            list(ex.map(job, segs))

    rh = rhythm(cut, segs, args.tier)
    rh["joins"] = joins(render)
    jn = rh["joins"]
    print(f"joins: status {jn['status']}; black {None if jn['black'] is None else len(jn['black'])}; "
          f"silences {None if jn['silence'] is None else len(jn['silence'])}", file=sys.stderr)
    story = None
    story_status = "not run"
    if not args.skip_story:
        try:
            story = story_eval(cut, args.tier)
            story_status = OK
        except Exception as e:  # noqa: BLE001
            story_status = f"{UNCHECKED}: {str(e)[:160]}"
            print(f"viewer read failed: {e}", file=sys.stderr)

    intent_findings = intent_mod.check_cut(cut, intent_doc, args.tier, profile=profile) if intent_doc else None
    loud_rows = [] if args.skip_loudness else pick_loudness(render, pick_spans(cut, args.tier), profile=profile)
    plan_path = Path(args.cut).parent / "lay" / (Path(args.cut).stem + ".plan.json")
    mono = {a["pick"]: bool(a.get("mono")) for a in json.loads(plan_path.read_text())["audio"]} if plan_path.exists() else {}
    loud = None if args.skip_loudness else loudness_summary(loud_rows, mono=mono)
    cov = coverage(segs, args.skip_vlm)
    manifest = write_manifest(
        out, tool="cuteval", tool_sha=source_hash(__file__),
        render=dict(path=str(render), sha=render_fp, duration_s=round(dur, 3), layout_end_s=round(laid, 3)),
        cut=dict(path=args.cut, sha=fingerprint(args.cut), version=cut.get("version")),
        transcripts=dict(path=args.transcripts, sha=fingerprint(args.transcripts)), tier=args.tier,
        vision=None if args.skip_vlm else dict(model=VISION_MODEL, url=VISION_URL, prompt_sha=VLM_PROMPT_HASH, max_tokens=1500, temperature=0.2,
                                               reused_from=args.resume, reused=reused),
        viewer_read=None if args.skip_story else dict(model=OLLAMA_MODEL, url=OLLAMA_URL, prompt_sha=text_hash(STORY_PROMPT),
                                                       temperature=0.3, num_ctx=32768, status=story_status),
        frames_dir=str(frames_dir), picture_coverage=cov, joins_status=jn["status"], joins_errors=jn["errors"],
        exposure_unmeasured=sum(1 for s in segs if s.get("exposure_flags") is None),
        intent=None if not intent_doc else dict(path=str(intent_path), sha=fingerprint(intent_path), summary=intent_mod.summary(intent_findings),
                                                protected_picks=protected),
        loudness=None if loud is None else dict(measured=loud["measured"], unmeasured=loud["unmeasured"], target_lufs=loud["target_lufs"]),
        cover_rules=dict(max_shake_pct=max_shake, steadiness_measured=rules["measured"], talking=len(rules["talking"]),
                         talking_unknown=len(rules["talking_unknown"]), shaky=len(rules["shaky"]), sync_shaky=len(rules.get("sync_shaky", [])),
                         shake_unmeasured=rules["unmeasured"], drone_clean=len(rules.get("drone_clean", [])),
                         drone_unclean=len(rules.get("drone_unclean", [])), drone_unmeasured=len(rules.get("drone_unmeasured", []))),
    )
    result = dict(cut=args.cut, render=str(render), tier=args.tier, manifest=manifest, rhythm=rh, story=story, stretches=segs,
                  intent=intent_findings, loudness=dict(summary=loud, picks=loud_rows) if loud is not None else None)
    out.with_suffix(".json").write_text(json.dumps(result, indent=1))
    write_report(out, cut, segs, rh, story, render, manifest, args.skip_vlm, intent_findings, loud, rules)
    print(f"wrote {out}.json, .md and .manifest.json; picture: {cov['judged']}/{cov['stretches']} judged", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
