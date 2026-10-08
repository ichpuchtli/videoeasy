"""Faces as a measurement: face tracks per source, and what each face's muscles do over time.

The editor (4 Oct 2026, an event shoot): tag every clip at the timecode where a
person's face shows an emotional register (grief, joy, laughter,
contemplation and the rest), then place B-roll against the testimony with
that map. A face is measured before anything reads it:

  1. faceworker.py, in its own pinned environment (see its docstring for why),
     decodes the source at SAMPLE_FPS, long side LONG_PX; YuNet finds the faces
     and MediaPipe's FaceLandmarker reads 52 blendshapes and the head pose of
     every face at or over DETECT_MIN_PX. That raw file is cached per source
     content (`work/faces/<id>.<fingerprint>.raw.json`) and is the evidence;
  2. here, detections are linked into tracks by overlap or centre distance,
     with up to LINK_GAP_S of absence allowed (`link`);
  3. each sample is readable or not: a face under MIN_FACE_PX, turned past
     MAX_YAW, or with no landmarks is `unreadable`, never neutral;
  4. the blendshapes collapse into a dozen cue channels (CUES: the smile, the
     cheek raise, the inner brow, the lowered brow, eyes closed, jaw open ...),
     and each track with BASELINE_MIN_S of readable samples gets a baseline per
     cue (its median), so a moment is a change in THAT face, not a beard or a
     resting frown read as sorrow (register.py does the reading).

A track has no identity: `t03` of one source is unrelated to `t03` of another,
and nothing here names anyone.

Measured on four event clips (4 Oct 2026): processing ran at 5-8x real time on
one CPU worker. A to-camera interview face (about 280 px, frontal) read in
every frame. B-roll faces in a group were mostly 30-60 px or side-on (yaw
40-60 degrees); MediaPipe read few of them and a crowd shot read almost none.
The unreadable share is reported, never filled in. MIN_FACE_PX and MAX_YAW are
provisional until the editor's calibration (register.py `review`).

    uv run python -m videoeasy.faces --config config.<p>.yaml [--only SRC] [--workers 4] [--force]
    -> <out_dir>/faces/<id>.json (tracks), <out_dir>/faces.json (index) + faces.manifest.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from .evalrun import OK, UNCHECKED, fingerprint, source_hash, write_manifest

WORKER = Path(__file__).with_name("faceworker.py")

# Downloaded once into the model cache and checked against these hashes (4 Oct 2026).
MODELS = {
    "yunet": dict(file="face_detection_yunet_2023mar.onnx", licence="MIT (OpenCV zoo)",
                  url="https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
                  sha256="8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"),
    "landmarker": dict(file="face_landmarker.task", licence="Apache-2.0 (MediaPipe)",
                       url="https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
                       sha256="64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff"),
    "yamnet": dict(file="yamnet.tflite", licence="Apache-2.0 (MediaPipe / AudioSet YAMNet)",
                   url="https://storage.googleapis.com/mediapipe-models/audio_classifier/yamnet/float32/1/yamnet.tflite",
                   sha256="4d8b4a53282dc83ef04e3e7dbc4fbc98082e34e44ed798e16c3a0cdd4c584faf"),
}

SAMPLE_FPS = 5.0
LONG_PX = 1920          # decode size: faces in 4K wides need the pixels
DETECT_MIN_PX = 40      # the worker reads blendshapes at or over this face size
MIN_FACE_PX = 60        # provisional: smaller faces are unreadable (their blendshapes were noise or absent on the probes)
MAX_YAW = 40.0          # provisional: a face turned further is side-on; its blendshapes are unreliable
LINK_IOU = 0.3
LINK_CENTRE = 0.5       # or: centres within this fraction of the face size ...
LINK_SIZE = 1.6         # ... and sizes within this ratio
LINK_GAP_S = 1.0        # a track survives this long without a detection
MIN_TRACK_S = 1.0       # shorter tracks are listed in the counts only
BASELINE_MIN_S = 3.0    # readable time a track needs before its own median is its baseline

# Cue channels: the mean of their blendshapes (MediaPipe's ARKit-style names).
CUES = {
    "smile": ("mouthSmileLeft", "mouthSmileRight"),
    "cheek": ("cheekSquintLeft", "cheekSquintRight"),
    "brow_inner_up": ("browInnerUp",),
    "brow_outer_up": ("browOuterUpLeft", "browOuterUpRight"),
    "brow_down": ("browDownLeft", "browDownRight"),
    "frown": ("mouthFrownLeft", "mouthFrownRight"),
    "eyes_closed": ("eyeBlinkLeft", "eyeBlinkRight"),
    "squint": ("eyeSquintLeft", "eyeSquintRight"),
    "eye_wide": ("eyeWideLeft", "eyeWideRight"),
    "jaw_open": ("jawOpen",),
    "look_down": ("eyeLookDownLeft", "eyeLookDownRight"),
    "press": ("mouthPressLeft", "mouthPressRight"),
    "stretch": ("mouthStretchLeft", "mouthStretchRight"),
    "sneer": ("noseSneerLeft", "noseSneerRight"),
}


# ----------------------------------------------------------------------- models
def model_dir() -> Path:
    root = os.environ.get("VIDEOEASY_CACHE")
    return (Path(root) if root else Path.home() / ".cache" / "videoeasy") / "models"


def ensure_model(key: str, spec: dict | None = None) -> Path:
    """The model file, downloaded on first use and refused when its hash differs from the pinned one.
    `spec` (file, url, sha256) names a model kept outside MODELS (identity.py's)."""
    import httpx
    m = spec or MODELS[key]
    path = model_dir() / m["file"]
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(path.suffix + ".part")
        with httpx.stream("GET", m["url"], follow_redirects=True, timeout=120) as r:
            r.raise_for_status()
            with open(tmp, "wb") as f:
                for chunk in r.iter_bytes(1 << 20):
                    f.write(chunk)
        tmp.rename(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != m["sha256"]:
        raise SystemExit(f"{path}: sha256 {digest[:16]}… is not the pinned {m['sha256'][:16]}…; delete it to download again")
    return path


_PYTHON: dict[str, str] = {}


def worker_python(script: Path = WORKER, probe: str = "import mediapipe.tasks") -> str:
    """A worker script's environment interpreter, synced once per process. Parallel `uv run --script` calls race on the
    shared script environment and import a half-installed MediaPipe (66 sources failed that way, 4 Oct 2026)."""
    def healthy() -> str | None:
        p = subprocess.run(["uv", "python", "find", "--script", str(script)], capture_output=True, text=True)
        py = p.stdout.strip()
        if p.returncode == 0 and py and subprocess.run([py, "-c", probe], capture_output=True).returncode == 0:
            return py
        return None

    if str(script) not in _PYTHON:   # a sync can reinstall packages under a worker that is importing them: sync only a broken environment
        py = healthy()
        if py is None:
            subprocess.run(["uv", "sync", "--quiet", "--script", str(script)], check=True, capture_output=True)
            py = healthy()
        if py is None:
            raise SystemExit(f"the worker environment for {script.name} fails `{probe}` after `uv sync --script`")
        _PYTHON[str(script)] = py
    return _PYTHON[str(script)]


def run_worker(kind: str, src: Path, out: Path, **opts) -> None:
    """faceworker.py in its own environment. Raises with the useful tail of its stderr."""
    tmp = out.with_suffix(".part.json")
    args = [worker_python(), str(WORKER), kind, str(src), str(tmp)]
    for k, v in opts.items():
        args += [f"--{k.replace('_', '-')}", str(v)]
    p = subprocess.run(args, capture_output=True, text=True)
    if p.returncode != 0 or not tmp.exists():
        noise = ("W0000", "I0000", "INFO:", "WARNING:", "All log messages", "XNNPACK")
        tail = [ln for ln in p.stderr.splitlines() if ln.strip() and not ln.startswith(noise)][-4:]
        raise RuntimeError(" | ".join(tail)[:400] or f"worker exited {p.returncode}")
    tmp.rename(out)


# ---------------------------------------------------------------------- linking
def detections(raw: dict) -> list[dict]:
    """The worker's rows as dicts, each with its cue values (None when no landmarks) and readability."""
    names = raw.get("blendshape_names") or []
    idx = {n: i for i, n in enumerate(names)}
    out = []
    for t, x, y, w, h, det, bs, pose, sharp in raw["detections"]:
        d = dict(t=t, box=(x, y, w, h), px=min(w, h), det=det, pose=pose, sharp=sharp, cues=None)
        if bs:
            d["cues"] = {c: round(float(np.mean([bs[idx[n]] for n in ns])), 3) for c, ns in CUES.items()}
        d["why"] = unreadable_why(d)
        out.append(d)
    return out


def unreadable_why(d: dict) -> str | None:
    if d["px"] < MIN_FACE_PX:
        return "small"
    if d["cues"] is None:
        return "no landmarks"
    if abs(d["pose"][0]) > MAX_YAW:
        return "side-on"
    return None


def _iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0.0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0.0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / (aw * ah + bw * bh - inter + 1e-9)


def _near(a, b) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    sa, sb = max(aw, ah), max(bw, bh)
    if max(sa, sb) / max(1e-9, min(sa, sb)) > LINK_SIZE:
        return False
    return np.hypot(ax + aw / 2 - bx - bw / 2, ay + ah / 2 - by - bh / 2) <= LINK_CENTRE * max(sa, sb)


def link(dets: list[dict], gap_s: float = LINK_GAP_S) -> list[list[dict]]:
    """Greedy tracks in time order: each detection joins the live track whose last box overlaps it most
    (IoU >= LINK_IOU, else a near centre at a similar size), one detection per track per frame."""
    tracks: list[list[dict]] = []
    live: list[list[dict]] = []
    by_t: dict[float, list[dict]] = {}
    for d in dets:
        by_t.setdefault(d["t"], []).append(d)
    for t in sorted(by_t):
        live = [tr for tr in live if t - tr[-1]["t"] <= gap_s + 1e-6]
        pairs = []
        for i, tr in enumerate(live):
            for j, d in enumerate(by_t[t]):
                v = _iou(tr[-1]["box"], d["box"])
                if v >= LINK_IOU or _near(tr[-1]["box"], d["box"]):
                    pairs.append((v, i, j))
        used_tr, used_d = set(), set()
        for v, i, j in sorted(pairs, key=lambda p: -p[0]):
            if i in used_tr or j in used_d:
                continue
            live[i].append(by_t[t][j])
            used_tr.add(i)
            used_d.add(j)
        for j, d in enumerate(by_t[t]):
            if j not in used_d:
                tr = [d]
                tracks.append(tr)
                live.append(tr)
    return tracks


def track_record(tid: str, tr: list[dict], fps: float) -> dict:
    """One track: span, readable share, size and pose, the cue baseline, and its samples."""
    readable = [d for d in tr if d["why"] is None]
    why: dict[str, int] = {}
    for d in tr:
        if d["why"]:
            why[d["why"]] = why.get(d["why"], 0) + 1
    base = None
    if len(readable) / fps >= BASELINE_MIN_S:
        base = {c: round(float(np.median([d["cues"][c] for d in readable])), 3) for c in CUES}
    med = lambda xs: round(float(np.median(xs)), 1) if xs else None  # noqa: E731
    return dict(
        id=tid, t0=tr[0]["t"], t1=tr[-1]["t"], n=len(tr), readable=len(readable),
        readable_s=round(len(readable) / fps, 1), unreadable=why,
        px=med([d["px"] for d in tr]), yaw=med([abs(d["pose"][0]) for d in tr if d["pose"]]),
        sharp=med([d["sharp"] for d in readable if d["sharp"] is not None]), baseline=base,
        # [t, x, y, w, h, readable, yaw, pitch, cues|None]: boxes are in the decoded frame's pixels (see frame_w/h)
        samples=[[d["t"], *[round(v) for v in d["box"]], d["why"] is None,
                  d["pose"][0] if d["pose"] else None, d["pose"][1] if d["pose"] else None, d["cues"]] for d in tr],
    )


def tracks_of(raw: dict) -> dict:
    """The per-source record: every linked track of MIN_TRACK_S or more, plus counts for the rest."""
    fps = float(raw["fps"])
    dets = detections(raw)
    trs = [tr for tr in link(dets) if tr[-1]["t"] - tr[0]["t"] + 1 / fps >= MIN_TRACK_S]
    trs.sort(key=lambda tr: (tr[0]["t"], tr[0]["box"][0]))
    recs = [track_record(f"t{i:02d}", tr, fps) for i, tr in enumerate(trs)]
    return dict(
        status=OK, fps=fps, frame_w=raw["frame_w"], frame_h=raw["frame_h"], frames=raw["frames"],
        duration_s=round(raw["frames"] / fps, 1), detections=len(dets),
        readable_detections=sum(1 for d in dets if d["why"] is None),
        tracks=recs, readable_tracks=sum(1 for r in recs if r["readable"]),
        readable_s=round(sum(r["readable_s"] for r in recs), 1),
        largest_px=max((r["px"] for r in recs), default=None),
    )


# ------------------------------------------------------------------------ run
def sources(cfg) -> tuple[dict[str, Path], dict[str, list[str]]]:
    """Every video under the config's A-roll and B-roll dirs by id (the file stem, as probe.py names them).
    Two files with one stem are both left out and reported: an id must mean one file."""
    from .config import list_videos
    seen: dict[str, list[Path]] = {}
    for d in (cfg.aroll_dir, cfg.broll_dir):
        for p in list_videos(d):
            seen.setdefault(p.stem, []).append(p)
    ok = {k: v[0] for k, v in seen.items() if len(v) == 1}
    dup = {k: [str(p) for p in v] for k, v in seen.items() if len(v) > 1}
    return ok, dup


def measure_source(sid: str, src: Path, work: Path, force: bool) -> Path:
    fp = fingerprint(src)
    raw = work / f"{sid}.{fp}.raw.json"
    if force or not raw.exists():
        run_worker("faces", src, raw, yunet=ensure_model("yunet"), landmarker=ensure_model("landmarker"),
                   fps=SAMPLE_FPS, long=LONG_PX, min_px=DETECT_MIN_PX)
    return raw


def run(cfg, only: str | None = None, workers: int = 3, force: bool = False) -> dict:
    srcs, dup = sources(cfg)
    if only:
        srcs = {k: v for k, v in srcs.items() if k == only}
    work = cfg.work_dir / "faces"
    out_dir = cfg.out_dir / "faces"
    work.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)
    idx_path = cfg.out_dir / "faces.json"
    idx = json.loads(idx_path.read_text()) if idx_path.exists() else {}
    units = idx.setdefault("sources", {})
    for sid, paths in dup.items():
        units[sid] = dict(status=UNCHECKED, why="two files share this id: " + "; ".join(paths))
    ensure_model("yunet"), ensure_model("landmarker")
    worker_python()   # synced once, before the workers start

    def one(sid: str, src: Path) -> tuple[str, dict]:
        try:
            raw_path = measure_source(sid, src, work, force)
            rec = tracks_of(json.loads(raw_path.read_text()))
        except Exception as e:  # noqa: BLE001
            return sid, dict(status=UNCHECKED, why=str(e)[:300], path=str(src))
        rec.update(source=sid, path=str(src), raw=str(raw_path))
        (out_dir / f"{sid}.json").write_text(json.dumps(rec, separators=(",", ":")))
        summary = {k: v for k, v in rec.items() if k != "tracks"}
        summary["tracks"] = len(rec["tracks"])
        return sid, summary

    todo = sorted(srcs.items())
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = [ex.submit(one, sid, p) for sid, p in todo]
        for k, f in enumerate(as_completed(futs), 1):
            sid, s = f.result()
            units[sid] = s
            line = (f"{s['tracks']} tracks, {s['readable_tracks']} readable, {s['readable_s']} s readable, largest {s['largest_px']} px"
                    if s["status"] == OK else f"UNCHECKED {s['why']}")
            print(f"[{k}/{len(todo)}] {sid}: {line}", file=sys.stderr)
            if k % 10 == 0:
                idx_path.write_text(json.dumps(idx, indent=1))
    idx["method"] = method()
    idx_path.write_text(json.dumps(idx, indent=1))
    ok = [s for s in units.values() if s.get("status") == OK]
    write_manifest(cfg.out_dir / "faces", tool="faces", tool_sha=source_hash(__file__), worker_sha=source_hash(str(WORKER)),
                   method=idx["method"], sources=len(units), measured=len(ok),
                   unchecked=sorted(k for k, s in units.items() if s.get("status") != OK),
                   readable_s=round(sum(s["readable_s"] for s in ok), 1), seconds=round(sum(s["duration_s"] for s in ok), 1))
    return idx


def method() -> dict:
    return dict(sample_fps=SAMPLE_FPS, long_px=LONG_PX, detect_min_px=DETECT_MIN_PX, min_face_px=MIN_FACE_PX, max_yaw=MAX_YAW,
                link_iou=LINK_IOU, link_gap_s=LINK_GAP_S, min_track_s=MIN_TRACK_S, baseline_min_s=BASELINE_MIN_S,
                cues=CUES, models={k: dict(file=m["file"], sha256=m["sha256"], licence=m["licence"]) for k, m in MODELS.items()
                                   if k in ("yunet", "landmarker")})


def load_source(cfg_out: Path, sid: str) -> dict | None:
    p = Path(cfg_out) / "faces" / f"{sid}.json"
    return json.loads(p.read_text()) if p.exists() else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="face tracks and blendshape cues per source")
    ap.add_argument("--config", required=True)
    ap.add_argument("--only", help="one source id (file stem)")
    ap.add_argument("--workers", type=int, default=3, help="sources measured at once (each is one CPU-bound process)")
    ap.add_argument("--force", action="store_true", help="decode again even when the raw cache matches the source")
    a = ap.parse_args(argv)
    from .config import load_config
    idx = run(load_config(a.config), a.only, a.workers, a.force)
    units = idx["sources"]
    ok = [s for s in units.values() if s.get("status") == OK]
    secs = sum(s["duration_s"] for s in ok)
    rs = sum(s["readable_s"] for s in ok)
    print(f"{len(units)} sources, {len(ok)} measured ({secs / 60:.0f} min), {len(units) - len(ok)} unchecked; "
          f"readable face time {rs / 60:.0f} min across {sum(s['readable_tracks'] for s in ok)} tracks", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
