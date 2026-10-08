"""Facial register moments: where a face shows laughter, grief, contemplation ..., with the evidence for each.

The editor (4 Oct 2026, an event shoot): tag each clip at the timecode where a
person's face carries an emotional register, then place B-roll against the
testimony with that map. Three modalities are read apart and fused in code
(docs/sibling-project.md, Register):

  measured  faces.py: per-track blendshape cues against the track's own
            baseline. A cue run (`CUE_RULES`: an absolute floor AND a rise
            over baseline, held MIN_RUN_S) is a CANDIDATE moment, not a label.
  heard     sounds.py: laughter, crying, a sigh, a shout on the clip's audio.
            Never attributed to a face (the laugh may be off camera).
  seen      the local vision model reads graded crops of that one face across
            the moment, plus one full frame with the face boxed. It gets the
            vocabulary's definitions and nothing else: no transcript, no name,
            no session, no measured cue (a spoken topic is not a visible fact,
            and a number in the prompt would steer the reading).

Fusion (`fuse`): a label stands only when the seen register is supported by a
measured cue or a heard family listed for that register (SUPPORT, or the
term's `cues`/`heard` in film.yaml). Otherwise the moment keeps its readings
and a status saying which modality said what: `agreed`, `seen_only`,
`measured_only`, `none`, `unreadable`, or the reading's own `unchecked` /
`invalid`. Disagreement is a searchable result, not an error.

Labels describe what a face SHOWS in a clip, never what a person feels, and
are stored against footage, never collected into profiles of people. No
label is a result until the editor has labelled a blind sample (`review`,
`calibrate`); every threshold here is provisional until then.

Controls: a share of readable face stretches with no cue run (CONTROL_SHARE)
is read too, so calibration can say what the candidate step misses.

    uv run python -m videoeasy.register moments  --config C     # candidates from faces.json + sounds.json (no model)
    uv run python -m videoeasy.register read     --config C [--limit N] [--only SRC] [--workers 2]   # the vision reading
    uv run python -m videoeasy.register map      --config C     # fuse -> out/register.json + out/register-map.md
    uv run python -m videoeasy.register find     --config C [--register grief] [--min-strength 2] [--silent] [--status agreed,seen_only] [--person LABEL]
    uv run python -m videoeasy.register review   --config C [--n 120]   # local blind labelling page (never published)
    uv run python -m videoeasy.register showcase --config C [--per 6]   # local page: the run's numbers, its most confident tags, its disagreements
    uv run python -m videoeasy.register calibrate --config C --labels FILE   # editor labels vs measured / seen / fused
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from . import faces as faces_mod
from . import sounds as sounds_mod
from .evalrun import INVALID, OK, UNCHECKED, source_hash, text_hash, write_manifest

# The generic vocabulary: what each register looks like on a face. A film.yaml `face_registers` replaces it.
DEFAULT_VOCAB = {   # no "word:" gloss at the start: the model answered "effort: straining" for a definition led by one (4 Oct 2026)
    "laughter": "laughing, the mouth open in a smile, cheeks raised and eyes creased",
    "joy": "smiling warmly or beaming, without laughing",
    "sadness": "a heavy, downcast face, the inner brows raised or the mouth corners down, without tears",
    "grief": "crying or on the edge of tears, the face crumpling, eyes wet or reddened, a hand to the face",
    "contemplation": "still and inward, the eyes closed or the gaze lowered or far away, the face at rest",
    "anger": "brows drawn hard down, the jaw set or teeth bared, shouting or glaring",
    "fear": "eyes wide, brows raised and drawn together, the mouth tense or stretched",
    "effort": "teeth gritted or the face tensed with physical exertion",
}
# Which measured cues and heard families can support each register (provisional; film.yaml `cues`/`heard` override).
SUPPORT = {
    "laughter": ({"smile", "cheek", "jaw_open"}, {"laughter"}),
    "joy": ({"smile", "cheek"}, {"laughter", "cheer"}),
    "sadness": ({"brow_inner_up", "frown", "look_down", "eyes_closed", "press"}, {"crying", "breath"}),
    "grief": ({"brow_inner_up", "frown", "stretch", "squint", "eyes_closed", "press"}, {"crying", "breath"}),
    "contemplation": ({"eyes_closed", "look_down", "still"}, set()),
    "anger": ({"brow_down", "jaw_open", "sneer", "press"}, {"shout"}),
    "fear": ({"brow_inner_up", "eye_wide", "stretch"}, {"shout", "breath"}),
    "effort": ({"brow_down", "squint", "press", "jaw_open", "stretch"}, {"shout", "breath"}),
}
# Provisional (4 Oct 2026 probes): a cue is up when it is at or over the floor AND this far over the track's baseline.
CUE_RULES = {
    "smile": (0.45, 0.20), "cheek": (0.35, 0.15), "brow_inner_up": (0.35, 0.15), "brow_down": (0.40, 0.15),
    "frown": (0.30, 0.12), "eyes_closed": (0.60, 0.30), "jaw_open": (0.45, 0.20), "press": (0.35, 0.15),
    "stretch": (0.30, 0.12), "eye_wide": (0.35, 0.15), "squint": (0.50, 0.20), "look_down": (0.50, 0.20),
    "sneer": (0.30, 0.15),
}
EXPRESSIVE = ("smile", "cheek", "brow_inner_up", "brow_down", "frown", "jaw_open", "press", "stretch", "eye_wide", "sneer")
MIN_RUN_S = 0.8          # a cue held this long is a candidate
EYES_CLOSED_MIN_S = 1.5  # shorter closures are blinks
MERGE_S = 0.6            # cue runs this close on one track are one moment
PAD_S = 0.4              # the reading looks this far either side of the run
LIPS_SD = 0.05           # provisional: jaw_open spread over a moment at or over this = lips moving (speech, a laugh, a shout)
STILL_S = 2.0            # control windows, and the 'still' pseudo-cue: this long with no expressive cue up and lips still
CONTROL_SHARE = 0.15     # controls read, as a share of face moments
HEARD_PAD_S = 0.5
N_CROPS = 5
CROP_PX = 384
CONTEXT_PX = 768
HEARD_FAMILIES = ("laughter", "crying", "breath", "shout", "cheer")   # sound events that become moments of their own
MAX_TOKENS = 4000


# ------------------------------------------------------------------ vocabulary
def vocabulary(profile) -> dict[str, dict]:
    """term -> {definition, cues, heard} from film.yaml `face_registers`, else the default."""
    raw = getattr(profile, "face_registers", None) or {k: {"definition": v} for k, v in DEFAULT_VOCAB.items()}
    out = {}
    for term, spec in raw.items():
        cues, heard = SUPPORT.get(term, (set(), set()))
        out[term] = dict(definition=spec["definition"], cues=set(spec.get("cues", cues)), heard=set(spec.get("heard", heard)))
        if re.match(r"^\s*[\w'-]+\s*:", spec["definition"]):   # lessons 2.12: the answer comes back as "term: gloss"
            raise ValueError(f"face_registers.{term}: start the definition with the description, not a gloss word and a colon")
        bad = out[term]["cues"] - set(faces_mod.CUES) - {"still"}
        if bad:
            raise ValueError(f"face_registers.{term}: unknown cues {sorted(bad)} (known: {sorted(faces_mod.CUES)} and 'still')")
    return out


# --------------------------------------------------------------------- moments
def runs(mask: np.ndarray, t: np.ndarray, min_s: float, merge_s: float, step: float) -> list[tuple[float, float]]:
    """Stretches where mask holds, joined across gaps up to merge_s, kept when at least min_s long (a sample counts `step`)."""
    out: list[list[float]] = []
    for i in np.flatnonzero(mask):
        if out and t[i] - out[-1][1] <= merge_s + 1e-6:
            out[-1][1] = float(t[i])
        else:
            out.append([float(t[i]), float(t[i])])
    return [(a, b) for a, b in out if b - a + step >= min_s - 1e-6]


def population_baseline(tracks: list[dict]) -> dict[str, float]:
    """Median cue over every readable sample of every track: the baseline for tracks too short for their own."""
    vals: dict[str, list[float]] = {c: [] for c in faces_mod.CUES}
    for tr in tracks:
        for s in tr["samples"]:
            if s[5] and s[8]:
                for c in faces_mod.CUES:
                    vals[c].append(s[8][c])
    return {c: round(float(np.median(v)), 3) if v else 0.0 for c, v in vals.items()}


def lips(samples: list, t0: float, t1: float) -> float | None:
    """Spread of jaw_open over the readable samples in [t0, t1]; None with fewer than 3."""
    v = [s[8]["jaw_open"] for s in samples if s[5] and s[8] and t0 - 1e-6 <= s[0] <= t1 + 1e-6]
    return round(float(np.std(v)), 3) if len(v) >= 3 else None


def cue_runs(tr: dict, base: dict[str, float], fps: float) -> list[dict]:
    """Every cue run on one track: [{cue, t0, t1, peak, peak_t, baseline}], plus 'still' runs."""
    rs = [s for s in tr["samples"] if s[5] and s[8]]
    if len(rs) < 2:
        return []
    t = np.array([s[0] for s in rs])
    step = 1.0 / fps
    out = []
    for cue, (floor, rise) in CUE_RULES.items():
        v = np.array([s[8][cue] for s in rs])
        b = base.get(cue, 0.0)
        min_s = EYES_CLOSED_MIN_S if cue == "eyes_closed" else MIN_RUN_S
        for a, z in runs((v >= floor) & (v >= b + rise), t, min_s, step * 1.5, step):
            sel = (t >= a) & (t <= z)
            k = int(np.argmax(np.where(sel, v, -1)))
            out.append(dict(cue=cue, t0=a, t1=z, peak=round(float(v[k]), 3), peak_t=float(t[k]), baseline=b))
    up = np.zeros(len(rs), bool)
    for cue in EXPRESSIVE:
        floor, rise = CUE_RULES[cue]
        v = np.array([s[8][cue] for s in rs])
        up |= (v >= floor) & (v >= base.get(cue, 0.0) + rise)
    for a, z in runs(~up, t, STILL_S, step * 1.5, step):
        sd = lips(tr["samples"], a, z)
        if sd is not None and sd < LIPS_SD:
            out.append(dict(cue="still", t0=a, t1=z, peak=None, peak_t=(a + z) / 2, baseline=None))
    return out


def _heard_in(events: list[dict], t0: float, t1: float) -> list[dict]:
    return [dict(family=e["family"], t0=e["t0"], t1=e["t1"], peak=e["peak"]) for e in events
            if e["t0"] <= t1 + HEARD_PAD_S and e["t1"] >= t0 - HEARD_PAD_S and e["family"] in HEARD_FAMILIES]


def _pick(key: str, share: float) -> bool:
    """A deterministic draw: the same key is always in or always out."""
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < share


def source_moments(sid: str, rec: dict, snd: dict | None, pop: dict[str, float], vocab: dict) -> list[dict]:
    """Candidate moments of one source: cue runs per track joined into moments, sound events as moments of their own
    when no face moment covers them, and a deterministic share of quiet readable stretches as controls."""
    fps = rec["fps"]
    events = (snd or {}).get("events") or []
    out = []
    for tr in rec["tracks"]:
        if not tr["readable"]:
            continue
        base = tr["baseline"] or pop
        cr = cue_runs(tr, base, fps)
        expressive = sorted((r for r in cr if r["cue"] != "still"), key=lambda r: r["t0"])
        groups: list[list[dict]] = []
        for r in expressive:
            if groups and r["t0"] - max(x["t1"] for x in groups[-1]) <= MERGE_S:
                groups[-1].append(r)
            else:
                groups.append([r])
        for g in groups:
            t0, t1 = min(r["t0"] for r in g), max(r["t1"] for r in g)
            top = max(g, key=lambda r: r["peak"] - (r["baseline"] or 0))
            out.append(_moment(sid, tr, "face", t0, t1, top["peak_t"], g, events, rec, baseline_own=bool(tr["baseline"])))
        # contemplation lives in stillness: a still run on a readable face is a candidate for it, when the vocabulary has it
        if any("still" in v["cues"] for v in vocab.values()):
            for r in cr:
                if r["cue"] == "still" and not any(m["track"] == tr["id"] and m["t0"] <= r["t1"] and m["t1"] >= r["t0"] for m in out):
                    if _pick(f"{sid}:{tr['id']}:{r['t0']:.1f}:still", 0.5):
                        out.append(_moment(sid, tr, "still", r["t0"], r["t1"], r["peak_t"], [r], events, rec, baseline_own=bool(tr["baseline"])))
        # controls: readable stretches with no cue run at all
        busy = [(r["t0"], r["t1"]) for r in cr]
        rs = [s[0] for s in tr["samples"] if s[5]]
        t = rs[0] if rs else None
        while t is not None and t + STILL_S <= rs[-1] + 1e-6:
            if not any(a <= t + STILL_S and b >= t for a, b in busy) and _pick(f"{sid}:{tr['id']}:{t:.1f}:control", CONTROL_SHARE / 3):
                out.append(_moment(sid, tr, "control", t, t + STILL_S, t + STILL_S / 2, [], events, rec, baseline_own=bool(tr["baseline"])))
                t += STILL_S * 3
            else:
                t += STILL_S
    for e in events:
        if e["family"] in HEARD_FAMILIES and not any(m["t0"] <= e["t1"] and m["t1"] >= e["t0"] for m in out if m["kind"] == "face"):
            out.append(_moment(sid, None, "heard", e["t0"], e["t1"], (e["t0"] + e["t1"]) / 2, [], events, rec))
    return sorted(out, key=lambda m: (m["t0"], m["track"] or ""))


def _moment(sid, tr, kind, t0, t1, peak_t, cues, events, rec, baseline_own=False) -> dict:
    dur = rec["duration_s"]
    a, b = max(0.0, t0 - PAD_S), min(dur, t1 + PAD_S)
    tid = tr["id"] if tr else None
    m = dict(id=f"{sid}:{tid or 'audio'}:{t0:.1f}", source=sid, path=rec["path"], kind=kind, track=tid,
             t0=round(a, 2), t1=round(b, 2), peak_t=round(peak_t, 2),
             cues=[dict(cue=r["cue"], t0=r["t0"], t1=r["t1"], peak=r["peak"], baseline=r["baseline"]) for r in cues],
             heard=_heard_in(events, a, b), baseline="own" if baseline_own else ("population" if tr else None))
    if tr:
        m.update(px=tr["px"], yaw=tr["yaw"], lips_sd=lips(tr["samples"], a, b))
        m["lips_moving"] = None if m["lips_sd"] is None else m["lips_sd"] >= LIPS_SD
    return m


def build_moments(cfg) -> dict:
    idx = json.loads((cfg.out_dir / "faces.json").read_text())
    snd = sounds_mod.load(cfg.out_dir)
    vocab = vocabulary(cfg.film)
    recs = {}
    for sid, s in sorted(idx["sources"].items()):
        if s.get("status") == OK:
            r = faces_mod.load_source(cfg.out_dir, sid)
            if r:
                recs[sid] = r
    pop = population_baseline([tr for r in recs.values() for tr in r["tracks"]])
    moments = []
    for sid, r in recs.items():
        moments += source_moments(sid, r, snd.get(sid), pop, vocab)
    for sid, s in snd.items():   # sound events on sources with no face record at all
        if sid not in recs and s.get("status") == OK:
            rec = dict(fps=faces_mod.SAMPLE_FPS, duration_s=s["audio_s"], path=s.get("path"), tracks=[])
            moments += source_moments(sid, rec, s, pop, vocab)
    kinds: dict[str, int] = {}
    for m in moments:
        kinds[m["kind"]] = kinds.get(m["kind"], 0) + 1
    doc = dict(created=dt.datetime.now().astimezone().isoformat(timespec="seconds"), population_baseline=pop, kinds=kinds,
               method=dict(cue_rules=CUE_RULES, min_run_s=MIN_RUN_S, eyes_closed_min_s=EYES_CLOSED_MIN_S, merge_s=MERGE_S,
                           pad_s=PAD_S, lips_sd=LIPS_SD, still_s=STILL_S, control_share=CONTROL_SHARE, heard_families=HEARD_FAMILIES,
                           faces=idx.get("method"), tool_sha=source_hash(__file__)),
               moments=moments)
    (cfg.out_dir / "register-moments.json").write_text(json.dumps(doc, indent=1))
    return doc


# ------------------------------------------------------------------------ seen
SYSTEM = ("You read facial expression in documentary footage. You describe only what is visible. You never guess who "
          "a person is, what happened, or what anyone feels inside.")


def face_prompt(vocab: dict, n: int, dur: float) -> str:
    terms = "\n".join(f"- {k}: {v['definition']}" for k, v in vocab.items())
    return f"""Image 1 is a full frame with one face marked by a yellow box. Images 2-{n + 1} are close crops of that same face, in time order across {dur:.1f} seconds.

Read the marked face only. A register is what the face visibly shows, not an inner state. The vocabulary:
{terms}

Answer with one JSON object:
{{"visible": "what this face and head visibly do across the crops, in your own words and specific to these images, at most 30 words",
 "register": "exactly one vocabulary term, the single word as written before its definition; or \\"none\\" when the face shows no clear expression; or \\"unreadable\\" when it is too small, turned away, blurred or covered to tell",
 "strength": 0-3 (0 = nothing shown, 3 = unmistakable),
 "tears": "yes|no|cannot tell",
 "speaking": "yes|no|cannot tell (the mouth moves like speech across the crops)",
 "same_face": "yes|no (all crops show the face marked in image 1)"}}"""


def heard_prompt(vocab: dict, n: int, dur: float) -> str:
    terms = "\n".join(f"- {k}: {v['definition']}" for k, v in vocab.items())
    return f"""These {n} frames are in time order across {dur:.1f} seconds of one shot.

Read the faces and bodies that are visible. A register is what a face or body visibly shows, not an inner state. The vocabulary:
{terms}

Answer with one JSON object:
{{"visible": "who is visible and what their faces and bodies visibly do, in your own words and specific to these frames, at most 30 words",
 "register": "exactly one vocabulary term (the single word as written before its definition), the one most clearly shown by anyone visible; or \\"none\\"; or \\"unreadable\\" when no face or body can be read",
 "strength": 0-3 (0 = nothing shown, 3 = unmistakable),
 "tears": "yes|no|cannot tell",
 "speaking": "yes|no|cannot tell (anyone's mouth moves like speech across the frames)",
 "same_face": "yes"}}"""


TRISTATE = {"yes", "no", "cannot tell"}


def validate_reading(j, vocab: dict) -> dict:
    """A model answer as {status, ...}: anything outside the schema is invalid, never repaired into a label."""
    if not isinstance(j, dict):
        return dict(status=INVALID, why="not a JSON object")
    reg = str(j.get("register", "")).strip().lower()
    if reg not in vocab and reg not in ("none", "unreadable"):
        return dict(status=INVALID, why=f"register {reg!r} not in the vocabulary")
    try:
        strength = int(j.get("strength"))
    except (TypeError, ValueError):
        return dict(status=INVALID, why="strength is not a number")
    if not 0 <= strength <= 3:
        return dict(status=INVALID, why="strength out of range")
    if reg in vocab and strength == 0:
        return dict(status=INVALID, why=f"register {reg} with strength 0")
    if reg in ("none", "unreadable") and strength >= 2:
        return dict(status=INVALID, why=f"register {reg} with strength {strength}")
    tri = {}
    for k in ("tears", "speaking", "same_face"):
        v = str(j.get(k, "cannot tell")).strip().lower()
        tri[k] = v if v in TRISTATE else "cannot tell"
    visible = " ".join(str(j.get("visible", "")).split())[:300]
    return dict(status=OK, register=reg, strength=strength, visible=visible, **tri)


def _grab(src: str, t: float, size: tuple[int, int], grade):
    """One graded frame at the faces pass's decode size (boxes are in those pixels)."""
    from PIL import Image
    from .frames import _escape_filter_path
    vf = []
    if grade.lut is not None:
        vf.append(f"lut3d={_escape_filter_path(grade.lut)}")
    if grade.filters:
        vf.append(grade.filters)
    vf.append(f"scale={size[0]}:{size[1]}")
    raw = b""
    for hw in (["-hwaccel", "videotoolbox"], []):   # 4K HEVC seeks are the slow part of a reading
        raw = subprocess.run(["ffmpeg", "-v", "error", *hw, "-ss", f"{max(0.0, t):.3f}", "-i", src, "-frames:v", "1", "-vf", ",".join(vf),
                              "-f", "image2pipe", "-vcodec", "png", "-"], capture_output=True).stdout
        if raw:
            break
    if not raw:
        raise RuntimeError(f"no frame at {t:.2f}s")
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _jpeg(img, q: int = 88) -> bytes:
    b = io.BytesIO()
    img.save(b, "JPEG", quality=q)
    return b.getvalue()


def crop_times(m: dict, tr: dict | None, n: int = N_CROPS) -> list[list]:
    """The track samples the reading shows: n spread across the moment, readable ones first."""
    if tr is None:
        return [[round(float(t), 2), None] for t in np.linspace(m["t0"] + 0.1, m["t1"] - 0.1, 4)]
    inside = [s for s in tr["samples"] if m["t0"] - 1e-6 <= s[0] <= m["t1"] + 1e-6]
    pool = [s for s in inside if s[5]] or inside
    if not pool:
        return []
    pick = [pool[int(round(i))] for i in np.linspace(0, len(pool) - 1, min(n, len(pool)))]
    return [[s[0], s[1:5]] for s in pick]


def frame_size(m: dict, rec: dict) -> tuple[int, int]:
    """The faces pass's decode size (its boxes are in those pixels); for a source it never measured, the same rule."""
    if rec.get("frame_w"):
        return rec["frame_w"], rec["frame_h"]
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height", "-of", "csv=p=0",
                          m["path"]], capture_output=True, text=True, check=True).stdout.strip().split(",")
    w, h = int(out[0]), int(out[1])
    k = min(1.0, faces_mod.LONG_PX / max(w, h))
    return int(round(w * k / 2) * 2), int(round(h * k / 2) * 2)


def _crop(img, box):
    x, y, w, h = box
    cx, cy, s = x + w / 2, y + h / 2, max(w, h) * 1.6
    return img.crop((int(cx - s / 2), int(cy - s / 2), int(cx + s / 2), int(cy + s / 2))).resize((CROP_PX, CROP_PX))


def frames_for(m: dict, rec: dict, grade) -> tuple[list[bytes], list]:
    """Face moment: the middle frame with the face boxed, then the face crops in time order. Sound-led moment: 4 frames."""
    from PIL import ImageDraw
    size = frame_size(m, rec)
    tr = next((t for t in rec["tracks"] if t["id"] == m["track"]), None) if m["track"] else None
    times = crop_times(m, tr)
    if tr is None:
        out = []
        for t, _ in times:
            f = _grab(m["path"], t, size, grade)
            f.thumbnail((CONTEXT_PX, CONTEXT_PX))
            out.append(_jpeg(f))
        return out, times
    grabs = [_grab(m["path"], t, size, grade) for t, _ in times]
    k = len(times) // 2
    ctx = grabs[k].copy()
    x, y, w, h = times[k][1]
    ImageDraw.Draw(ctx).rectangle([x, y, x + w, y + h], outline=(255, 210, 0), width=max(3, size[0] // 400))
    ctx.thumbnail((CONTEXT_PX, CONTEXT_PX))
    return [_jpeg(ctx)] + [_jpeg(_crop(g, box)) for g, (_, box) in zip(grabs, times)], times


def read_one(m: dict, rec: dict, cfg, vocab: dict) -> dict:
    from .vlm import chat_vision, extract_json
    kind_prompt = heard_prompt if m["track"] is None else face_prompt
    try:
        imgs, times = frames_for(m, rec, cfg.grade_for(m["path"]))
    except Exception as e:  # noqa: BLE001
        return dict(status=UNCHECKED, why=f"frames: {str(e)[:160]}")
    if not imgs:
        return dict(status=UNCHECKED, why="no samples in the moment")
    n = len(imgs) - (0 if m["track"] is None else 1)
    prompt = kind_prompt(vocab, n, m["t1"] - m["t0"])
    try:
        text = chat_vision(cfg.vision_url, cfg.vision_model, SYSTEM, prompt, imgs, max_tokens=MAX_TOKENS, temperature=0.2)
        r = validate_reading(extract_json(text), vocab)
    except json.JSONDecodeError:
        r = dict(status=INVALID, why="no JSON in the answer")
    except Exception as e:  # noqa: BLE001
        r = dict(status=UNCHECKED, why=str(e)[:200])
    r.update(times=[t for t, _ in times], prompt_sha=text_hash(SYSTEM, prompt), model=cfg.vision_model)
    return r


def reading_key(m: dict, vocab: dict) -> str:
    return text_hash(m["id"], f"{m['t0']}:{m['t1']}", SYSTEM, face_prompt(vocab, N_CROPS, 0) + heard_prompt(vocab, 4, 0))


def read(cfg, limit: int | None = None, only: str | None = None, workers: int = 2, kinds: set[str] | None = None,
         retry: bool = False) -> dict:
    mom = json.loads((cfg.out_dir / "register-moments.json").read_text())
    vocab = vocabulary(cfg.film)
    seen_path = cfg.out_dir / "register-seen.json"
    seen = json.loads(seen_path.read_text()) if seen_path.exists() else {}
    todo = [m for m in mom["moments"] if (only is None or m["source"] == only) and (kinds is None or m["kind"] in kinds)]
    todo = [m for m in todo if reading_key(m, vocab) not in seen
            or (retry and seen[reading_key(m, vocab)].get("status") != OK)]
    # strongest candidates first, so a partial run reads what matters most
    order = {"face": 0, "heard": 1, "still": 2, "control": 3}
    todo.sort(key=lambda m: (order[m["kind"]], m.get("lips_moving") is True,   # silent faces first: they are the cover
                             -max(((c["peak"] or 0) - (c["baseline"] or 0) for c in m["cues"]), default=0)))
    if limit:
        todo = todo[:limit]
    recs: dict[str, dict] = {}
    for m in todo:
        if m["source"] not in recs:
            recs[m["source"]] = faces_mod.load_source(cfg.out_dir, m["source"]) or dict(tracks=[], frame_w=None, frame_h=None)
    done = 0
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = {ex.submit(read_one, m, recs[m["source"]], cfg, vocab): m for m in todo}
        for f in as_completed(futs):
            m = futs[f]
            r = f.result()
            seen[reading_key(m, vocab)] = dict(r, moment=m["id"])
            done += 1
            print(f"[{done}/{len(todo)}] {m['id']} {m['kind']}: {r.get('register', r['status'])} {r.get('strength', '')} "
                  f"{(r.get('visible') or r.get('why') or '')[:90]}", file=sys.stderr)
            if done % 20 == 0:
                seen_path.write_text(json.dumps(seen, indent=1))
    seen_path.write_text(json.dumps(seen, indent=1))
    return seen


# ------------------------------------------------------------------------ fuse
def fuse(m: dict, r: dict | None, vocab: dict) -> dict:
    """The fused label and status for one moment, from its measured cues, heard events and seen reading."""
    measured = {c["cue"] for c in m["cues"]}
    heard = {h["family"] for h in m["heard"]}
    speaking = bool(m.get("lips_moving")) or bool(r and r.get("speaking") == "yes")
    base = dict(measured=sorted(measured), heard=sorted(heard), speaking=speaking)
    if r is None:
        return dict(base, status=UNCHECKED, label=None, why="not read")
    if r["status"] != OK:
        return dict(base, status=r["status"], label=None, why=r.get("why"))
    reg = r["register"]
    if reg == "unreadable" or r.get("same_face") == "no":
        return dict(base, status="unreadable", label=None, seen=reg)
    if reg == "none":
        expressive = measured - {"still"}
        return dict(base, status="measured_only" if expressive or heard else "none", label=None, seen=reg)
    cues, fams = vocab[reg]["cues"], vocab[reg]["heard"]
    support = sorted(measured & cues) + [f"heard:{h}" for h in sorted(heard & fams)]
    if support:
        return dict(base, status="agreed", label=reg, strength=r["strength"], support=support, seen=reg)
    return dict(base, status="seen_only", label=None, seen=reg, strength=r["strength"])


def parroting(readings: list[dict], vocab: dict, n: int = 4) -> float | None:
    """Share of `visible` texts that repeat any n words of a definition verbatim (the model copying the prompt)."""
    grams = set()
    for v in vocab.values():
        w = re.findall(r"[a-z']+", v["definition"].lower())
        grams |= {tuple(w[i:i + n]) for i in range(len(w) - n + 1)}
    texts = [r["visible"] for r in readings if r.get("status") == OK and r.get("visible")]
    if not texts:
        return None
    hit = 0
    for t in texts:
        w = re.findall(r"[a-z']+", t.lower())
        hit += any(tuple(w[i:i + n]) in grams for i in range(len(w) - n + 1))
    return round(hit / len(texts), 2)


def fmt_t(s: float) -> str:
    return f"{int(s // 60):02d}:{s % 60:04.1f}"


def build_map(cfg) -> dict:
    mom = json.loads((cfg.out_dir / "register-moments.json").read_text())
    seen_path = cfg.out_dir / "register-seen.json"
    seen = json.loads(seen_path.read_text()) if seen_path.exists() else {}
    vocab = vocabulary(cfg.film)
    idx = json.loads((cfg.out_dir / "faces.json").read_text())
    out = []
    for m in mom["moments"]:
        r = seen.get(reading_key(m, vocab))
        out.append(dict(m, reading=r, fused=fuse(m, r, vocab)))
    status: dict[str, int] = {}
    labels: dict[str, int] = {}
    for m in out:
        status[m["fused"]["status"]] = status.get(m["fused"]["status"], 0) + 1
        if m["fused"]["label"]:
            labels[m["fused"]["label"]] = labels.get(m["fused"]["label"], 0) + 1
    ok_src = [s for s in idx["sources"].values() if s.get("status") == OK]
    coverage = dict(sources=len(idx["sources"]), measured=len(ok_src), footage_s=round(sum(s["duration_s"] for s in ok_src), 1),
                    readable_face_s=round(sum(s["readable_s"] for s in ok_src), 1),
                    sources_with_readable_face=sum(1 for s in ok_src if s["readable_tracks"]))
    par = parroting([m["reading"] for m in out if m["reading"]], vocab)
    doc = dict(created=dt.datetime.now().astimezone().isoformat(timespec="seconds"), vocabulary={k: v["definition"] for k, v in vocab.items()},
               coverage=coverage, status=status, labels=labels, parroting=par, calibrated=False, moments=out)
    (cfg.out_dir / "register.json").write_text(json.dumps(doc, indent=1, default=sorted))
    (cfg.out_dir / "register-map.md").write_text(map_md(doc))
    write_manifest(cfg.out_dir / "register", tool="register", tool_sha=source_hash(__file__), model=cfg.vision_model,
                   vocabulary=doc["vocabulary"], method=mom["method"], coverage=coverage, status=status, labels=labels, parroting=par,
                   unchecked=sorted(m["id"] for m in out if m["fused"]["status"] in (UNCHECKED, INVALID)),
                   calibrated=False, note="provisional: no label is a result until the editor's blind sample is scored (calibrate)")
    return doc


def _line(m: dict, who: str | None = None) -> str:
    f, r = m["fused"], m["reading"] or {}
    face = f"face {m['track']} {m['px']:.0f} px" if m.get("track") else "no face tracked (sound-led)"
    if who:
        face += f" ({who}, confirmed)"
    meas = ", ".join(f"{c['cue']} {c['peak']:.2f}" if c["peak"] is not None else c["cue"] for c in m["cues"]) or "no cue"
    heard = ", ".join(f"{h['family']} {h['peak']:.2f}" for h in m["heard"]) or "nothing"
    flags = []
    if f["speaking"]:
        flags.append("LIPS MOVING (sync, not silent cover)")
    if r.get("tears") == "yes":
        flags.append("tears seen")
    seen = f"seen: {r.get('register', '?')} {r.get('strength', '')} \"{r.get('visible', '')}\"" if r.get("status") == OK else f"seen: {r.get('status', 'not read')}"
    return (f"- `{m['source']}` **{fmt_t(m['t0'])}–{fmt_t(m['t1'])}** · {face} · measured: {meas} · heard: {heard} · {seen}"
            + (f" · {'; '.join(flags)}" if flags else ""))


def map_md(doc: dict) -> str:
    c = doc["coverage"]
    ms = doc["moments"]
    L = ["# Register map (provisional)", "",
         "What each face SHOWS at a timecode, never what anyone feels. Measured (blendshape cues against that face's "
         "baseline), heard (the clip's own audio, never attributed to a face) and seen (the local vision model on graded "
         "crops, with no transcript, name or session) are read apart and fused in code. "
         "**No label here is a result yet:** the editor's blind sample has not been scored (`register review`, `calibrate`).", "",
         f"Coverage: {c['measured']} of {c['sources']} sources measured ({c['footage_s'] / 60:.0f} min); "
         f"readable face time {c['readable_face_s'] / 60:.0f} min on {c['sources_with_readable_face']} sources. "
         "Faces under the size limit, side-on or without landmarks are unreadable, not neutral.", "",
         "Status: " + ", ".join(f"{k} {v}" for k, v in sorted(doc["status"].items())) + ".",
         f"Seen readings that repeat four words of a definition: {doc['parroting'] if doc['parroting'] is not None else 'n/a'}.", ""]
    agreed = [m for m in ms if m["fused"]["status"] == "agreed"]
    for term in doc["vocabulary"]:
        hits = sorted((m for m in agreed if m["fused"]["label"] == term),
                      key=lambda m: (-m["fused"].get("strength", 0), m["fused"]["speaking"], m["source"], m["t0"]))
        L += [f"## {term} ({len(hits)})", "", f"_{doc['vocabulary'][term]}_", ""]
        L += [_line(m) for m in hits] or ["(none agreed)"]
        L.append("")
    so = [m for m in ms if m["fused"]["status"] == "seen_only"]
    L += [f"## Seen, with no measured or heard support ({len(so)})", "",
          "The vision model named a register that no cue or sound backs. Worth a look; not a label.", ""]
    L += [_line(m) for m in sorted(so, key=lambda m: (m["reading"]["register"], -m["reading"]["strength"]))] or ["(none)"]
    mo = [m for m in ms if m["fused"]["status"] == "measured_only"]
    L += ["", f"## Measured or heard, the model saw nothing ({len(mo)})", ""]
    L += [_line(m) for m in mo[:200]] + ([f"… {len(mo) - 200} more in register.json"] if len(mo) > 200 else [])
    return "\n".join(L) + "\n"


# ---------------------------------------------------------------- calibration
def stratified(moments: list[dict], n: int) -> list[dict]:
    """A deterministic sample across fused status, seen register and kind, controls included (their recall matters)."""
    strata: dict[tuple, list[dict]] = {}
    for m in moments:
        key = (m["fused"]["status"], (m.get("reading") or {}).get("register"), m["kind"])
        strata.setdefault(key, []).append(m)
    for v in strata.values():
        v.sort(key=lambda m: hashlib.sha256(m["id"].encode()).hexdigest())
    out = []
    while len(out) < n and any(strata.values()):
        for k in sorted(strata, key=str):
            if strata[k] and len(out) < n:
                out.append(strata[k].pop(0))
    return sorted(out, key=lambda m: hashlib.sha256(("order" + m["id"]).encode()).hexdigest())


def review(cfg, n: int = 120) -> Path:
    """A LOCAL page for blind labelling: a short graded clip and the face crops per moment, the vocabulary as buttons,
    labels kept in the browser and downloaded as JSON. It is never published: these are people's faces."""
    doc = json.loads((cfg.out_dir / "register.json").read_text())
    sample = stratified([m for m in doc["moments"] if m["fused"]["status"] not in (UNCHECKED, INVALID)], n)
    day = dt.date.today().strftime("%Y%m%d")
    root = cfg.work_dir / "reviews" / f"register-{day}"
    (root / "media").mkdir(parents=True, exist_ok=True)
    recs: dict[str, dict] = {}
    cards = []
    for i, m in enumerate(sample, 1):
        rec = recs.setdefault(m["source"], faces_mod.load_source(cfg.out_dir, m["source"]) or dict(tracks=[]))
        slug = hashlib.sha256(m["id"].encode()).hexdigest()[:12]
        clip, strip = root / "media" / f"{slug}.mp4", root / "media" / f"{slug}.jpg"
        grade = cfg.grade_for(m["path"])
        if not clip.exists():
            from .frames import _escape_filter_path
            vf = ([f"lut3d={_escape_filter_path(grade.lut)}"] if grade.lut else []) + ([grade.filters] if grade.filters else []) + ["scale='if(gt(iw,ih),640,-2)':'if(gt(iw,ih),-2,640)'"]
            a = max(0.0, m["t0"] - 1.0)
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-hwaccel", "videotoolbox", "-ss", f"{a:.2f}", "-i", m["path"], "-t", f"{m['t1'] - m['t0'] + 2.0:.2f}",
                            "-vf", ",".join(vf), "-c:v", "libx264", "-crf", "26", "-preset", "veryfast", "-c:a", "aac", "-b:a", "96k",
                            "-movflags", "+faststart", str(clip)], capture_output=True)
        if not strip.exists() and m["track"]:
            try:
                imgs, _ = frames_for(m, rec, grade)
                from PIL import Image
                tiles = [Image.open(io.BytesIO(b)) for b in imgs[1:]]
                sheet = Image.new("RGB", (CROP_PX * len(tiles) // 2, CROP_PX // 2))
                for k, t in enumerate(tiles):
                    sheet.paste(t.resize((CROP_PX // 2, CROP_PX // 2)), (k * CROP_PX // 2, 0))
                sheet.save(strip, quality=85)
            except Exception:  # noqa: BLE001
                pass
        cards.append(dict(id=m["id"], n=i, clip=f"media/{clip.name}", strip=f"media/{strip.name}" if strip.exists() else None,
                          where=f"{m['source']} {fmt_t(m['t0'])}–{fmt_t(m['t1'])}", track=bool(m["track"])))
        print(f"[{i}/{len(sample)}] {m['id']}", file=sys.stderr)
    page = root / "index.html"
    page.write_text(review_html(cards, list(doc["vocabulary"]), doc["vocabulary"], day))
    (root / "sample.json").write_text(json.dumps([c["id"] for c in cards], indent=1))
    return page


def review_html(cards: list[dict], terms: list[str], defs: dict[str, str], day: str) -> str:
    data = json.dumps(dict(cards=cards, terms=terms, defs=defs, key=f"register-labels-{day}"))
    return """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Register labels</title><style>
:root{--bg:#fbfaf7;--fg:#1d1c1a;--mut:#6b675f;--line:#e2ded5;--acc:#2f5d50;--on:#fff}
@media (prefers-color-scheme:dark){:root{--bg:#161614;--fg:#ecebe6;--mut:#a29e94;--line:#34322d;--acc:#7fb8a6;--on:#10201b}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 -apple-system,system-ui,sans-serif}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:12px 16px;z-index:2;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
h1{font-size:17px;margin:0 auto 0 0}main{max-width:900px;margin:0 auto;padding:16px}
.card{border:1px solid var(--line);border-radius:10px;padding:12px;margin:0 0 18px}
.card.done{opacity:.6}video{width:100%;max-height:420px;background:#000;border-radius:6px}
.strip{width:100%;border-radius:6px;margin-top:6px}.meta{color:var(--mut);font-size:13px;margin:6px 0}
.btns{display:flex;flex-wrap:wrap;gap:6px;margin-top:8px}
button{font:inherit;border:1px solid var(--line);background:transparent;color:var(--fg);border-radius:999px;padding:5px 12px;cursor:pointer}
button.on{background:var(--acc);border-color:var(--acc);color:var(--on)}
input{font:inherit;width:100%;box-sizing:border-box;margin-top:8px;padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:transparent;color:var(--fg)}
details{color:var(--mut);font-size:13px}dl{margin:6px 0}dt{font-weight:600;color:var(--fg)}dd{margin:0 0 4px}
</style></head><body><header><h1>Register labels <span id="prog"></span></h1><button id="dl">Download labels</button></header><main>
<p>For each moment: what does the <b>boxed face</b> (the face in the strip of crops) visibly show? Pick up to two terms, or <i>none</i>,
or <i>unreadable</i> / <i>wrong face</i>. Label what you see, not what the person may feel. The model's reading is hidden on purpose.
Labels stay in this browser until you press <b>Download labels</b>; save the file where the calibrate command can read it.</p>
<details><summary>Definitions</summary><dl id="defs"></dl></details><div id="cards"></div></main><script>
const D=""" + data + """;let L={};try{L=JSON.parse(localStorage.getItem(D.key)||"{}")}catch(e){}
const save=()=>{try{localStorage.setItem(D.key,JSON.stringify(L))}catch(e){};prog()};
const prog=()=>{document.getElementById("prog").textContent=`${Object.keys(L).filter(k=>L[k].labels&&L[k].labels.length).length} / ${D.cards.length}`};
document.getElementById("defs").innerHTML=D.terms.map(t=>`<dt>${t}</dt><dd>${D.defs[t]}</dd>`).join("");
const extra=["none","unreadable","wrong face"];
document.getElementById("cards").innerHTML=D.cards.map(c=>`<section class="card" data-id="${c.id}"><div class="meta">#${c.n} · ${c.where}${c.track?"":" · sound-led: read anyone visible"}</div>
<video src="${c.clip}" controls preload="metadata" playsinline></video>${c.strip?`<img class="strip" src="${c.strip}" alt="face crops">`:""}
<div class="btns">${[...D.terms,...extra].map(t=>`<button data-t="${t}">${t}</button>`).join("")}</div><input placeholder="note (optional)"></section>`).join("");
document.querySelectorAll(".card").forEach(card=>{const id=card.dataset.id;const cur=()=>L[id]||(L[id]={labels:[],note:""});
const paint=()=>{const l=(L[id]||{}).labels||[];card.querySelectorAll("button").forEach(b=>b.classList.toggle("on",l.includes(b.dataset.t)));card.classList.toggle("done",l.length>0)};
card.querySelectorAll("button").forEach(b=>b.onclick=()=>{const c=cur(),t=b.dataset.t;if(extra.includes(t)){c.labels=c.labels.includes(t)?[]:[t]}else{c.labels=c.labels.filter(x=>!extra.includes(x));
if(c.labels.includes(t))c.labels=c.labels.filter(x=>x!==t);else if(c.labels.length<2)c.labels.push(t)}paint();save()});
const inp=card.querySelector("input");inp.value=(L[id]||{}).note||"";inp.oninput=()=>{cur().note=inp.value;save()};paint()});
document.getElementById("dl").onclick=()=>{const b=new Blob([JSON.stringify({labelled:new Date().toISOString(),labels:L},null,1)],{type:"application/json"});
const a=document.createElement("a");a.href=URL.createObjectURL(b);a.download=D.key+".json";a.click()};prog();
</script></body></html>"""


def calibrate(cfg, labels_path: Path) -> str:
    """Editor labels vs the fused label, the seen register and the measured candidate step, per term."""
    doc = json.loads((cfg.out_dir / "register.json").read_text())
    lab = json.loads(Path(labels_path).read_text())["labels"]
    by = {m["id"]: m for m in doc["moments"]}
    rows = [(by[k], v["labels"]) for k, v in lab.items() if k in by and v.get("labels") and "wrong face" not in v["labels"]]
    terms = list(doc["vocabulary"])
    L = ["# Register calibration", "", f"{len(rows)} labelled moments ({len(lab) - len(rows)} skipped: unlabelled, wrong face or unknown id).", "",
         "| term | editor | fused: hit / said (precision) | seen: hit / said | recall (fused) | recall (seen) |", "|---|---:|---:|---:|---:|---:|"]
    def ratio(hit: int, n: int) -> str:
        return f"{hit} / {n} ({hit / n:.0%})" if n else "0 / 0"

    for t in terms:
        truth = [t in ls for _, ls in rows]
        fused = [m["fused"]["label"] == t for m, _ in rows]
        seen = [(m.get("reading") or {}).get("register") == t for m, _ in rows]
        tp_f = sum(a and b for a, b in zip(truth, fused))
        tp_s = sum(a and b for a, b in zip(truth, seen))
        pos = sum(truth)
        rec_f, rec_s = (f"{tp_f / pos:.0%}", f"{tp_s / pos:.0%}") if pos else ("–", "–")
        L.append(f"| {t} | {pos} | {ratio(tp_f, sum(fused))} | {ratio(tp_s, sum(seen))} | {rec_f} | {rec_s} |")
    ctrl = [(m, ls) for m, ls in rows if m["kind"] == "control"]
    missed = [(m, ls) for m, ls in ctrl if not set(ls) & {"none", "unreadable"}]
    L += ["", f"Controls (readable stretches with no cue run): {len(ctrl)} labelled, {len(missed)} showed a register to the editor "
          "(what the candidate step misses).", ""]
    conf: dict[tuple[str, str], int] = {}
    for m, ls in rows:
        s = (m.get("reading") or {}).get("register") or m["fused"]["status"]
        for t in ls:
            conf[(t, s)] = conf.get((t, s), 0) + 1
    L += ["## Editor label vs seen register", ""] + [f"- {a} → {b}: {n}" for (a, b), n in sorted(conf.items(), key=lambda x: -x[1])]
    text = "\n".join(L) + "\n"
    (cfg.out_dir / "register-calibration.md").write_text(text)
    write_manifest(cfg.out_dir / "register-calibration", tool="register calibrate", labels=str(labels_path), labelled=len(rows),
                   register_json_created=doc["created"], tool_sha=source_hash(__file__))
    return text


def find(doc: dict, register: str | None = None, statuses: tuple[str, ...] = ("agreed",), min_strength: int = 1,
         silent: bool = False, sources: set[str] | None = None, tracks: set[tuple[str, str]] | None = None) -> list[dict]:
    """Moments matching a query, strongest first. A seen_only moment matches on its seen register.
    `tracks` keeps only moments on those (source, track) faces: one person's confirmed tracks (identity.py)."""
    out = []
    for m in doc["moments"]:
        f, r = m["fused"], m.get("reading") or {}
        if f["status"] not in statuses or (sources and m["source"] not in sources):
            continue
        if tracks is not None and (m["source"], m.get("track")) not in tracks:
            continue
        reg = f.get("label") or r.get("register")
        if register and reg != register:
            continue
        if (r.get("strength") or 0) < min_strength or (silent and f["speaking"]):
            continue
        out.append(m)
    return sorted(out, key=lambda m: (-(m.get("reading") or {}).get("strength", 0), m["source"], m["t0"]))


def load(cfg_out) -> dict | None:
    p = Path(cfg_out) / "register.json"
    return json.loads(p.read_text()) if p.exists() else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="facial register moments: measured, heard, seen, fused")
    ap.add_argument("cmd", choices=["moments", "read", "map", "find", "review", "showcase", "calibrate"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--only", help="one source id")
    ap.add_argument("--kinds", help="comma list of face,heard,still,control (read)")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--retry", action="store_true", help="read again where the last answer was not ok")
    ap.add_argument("--n", type=int, default=120)
    ap.add_argument("--per", type=int, default=6, help="showcase: examples per register")
    ap.add_argument("--labels")
    ap.add_argument("--register", help="find: one vocabulary term")
    ap.add_argument("--status", default="agreed", help="find: comma list of fused statuses (default agreed)")
    ap.add_argument("--min-strength", type=int, default=1)
    ap.add_argument("--silent", action="store_true", help="find: leave out moments where the lips move")
    ap.add_argument("--sources", help="find: comma list of source ids")
    ap.add_argument("--person", help="find: only faces the editor confirmed as this label (identity.py)")
    a = ap.parse_args(argv)
    from .config import load_config
    cfg = load_config(a.config)
    if a.cmd == "moments":
        d = build_moments(cfg)
        print(f"{len(d['moments'])} candidate moments: " + ", ".join(f"{k} {v}" for k, v in sorted(d["kinds"].items())), file=sys.stderr)
    elif a.cmd == "read":
        read(cfg, a.limit, a.only, a.workers, set(a.kinds.split(",")) if a.kinds else None, a.retry)
    elif a.cmd == "map":
        d = build_map(cfg)
        print("status: " + ", ".join(f"{k} {v}" for k, v in sorted(d["status"].items())) + "; agreed labels: "
              + (", ".join(f"{k} {v}" for k, v in sorted(d["labels"].items())) or "none") + f"; parroting {d['parroting']}", file=sys.stderr)
        print(cfg.out_dir / "register-map.md")
    elif a.cmd == "find":
        doc = load(cfg.out_dir)
        if doc is None:
            raise SystemExit("no register.json yet: run `register map` first")
        tracks, who = None, {}
        if a.person:
            from .identity import CONFIRMED, load_links
            links = [v for v in load_links(cfg).values() if v["status"] == CONFIRMED]
            tracks = {(v["source"], v["track"]) for v in links if v["person"].casefold() == a.person.casefold()}
            if not tracks:
                names = sorted({v["person"] for v in links})
                raise SystemExit(f"no confirmed faces for {a.person!r}: " + (", ".join(names) if names else "nobody is confirmed yet (identity review)"))
            who = {(v["source"], v["track"]): v["person"] for v in links}
        hits = find(doc, a.register, tuple(a.status.split(",")), a.min_strength, a.silent,
                    set(a.sources.split(",")) if a.sources else None, tracks)
        print("\n".join(_line(m, who.get((m["source"], m.get("track")))) for m in hits[: a.limit or len(hits)]) or "(nothing matches)")
        print(f"{len(hits)} moments (provisional: not calibrated)", file=sys.stderr)
    elif a.cmd == "review":
        print(review(cfg, a.n))
    elif a.cmd == "showcase":
        from .registerpage import build
        print(build(cfg, a.per))
    else:
        if not a.labels:
            ap.error("calibrate needs --labels")
        print(calibrate(cfg, Path(a.labels)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
