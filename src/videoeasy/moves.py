"""Drone moves as a measurement: what the camera does along a take, and how cleanly it does it.

The editor (29 Sep 2026): the drone B-roll has smooth, clean stretches (pans,
risers, orbits) and those are the bits to use. The shake measure cannot find
them: 1114 of the 1160 drone seconds already pass it, because a drone rarely
shakes; what goes wrong is the move itself (the ease into a move, a
hesitation, a change of direction, the pilot settling at the head of a
clip). The ingest's movement notes cannot find them either: four stills a
shot, and most of them say the samples are too far apart to tell.

Per frame pair (the same sampling as steadiness.py):

  1. track features (Lucas-Kanade) and fit a similarity transform with RANSAC;
     keep the shift of the frame CENTRE (the fit's own translation is about
     the top-left corner, so a zoom would read as a pan), the zoom (log
     scale) and the rotation;
  2. keep the median horizontal flow of the centre box against the edges,
     and the median flow of the top half against the bottom half: a pan or
     tilt moves every region alike, an orbit holds the centre while the
     edges travel, and a rise or slide moves the near ground (the bottom of a
     drone frame) faster than the far.

Along the take, in % of frame width per second (zoom and rotation as the
displacement they give the frame edge): the velocity is smoothed over 1 s
and cut into segments wherever the camera starts, stops or turns. Each
segment is labelled (hover, pan, tilt, orbit, slide, rise, descend, push-in,
pull-out, rotate) and scored:

  direction   how closely the velocity keeps one direction (mean cosine)
  dip         the deepest drop in speed between two faster moments, as a share
              of the top speed: a hesitation or a stick jog. A clean move
              rises, cruises and settles, so an ease in or out never counts
              (a first measure, speed against its 3 s average, called every
              short eased move a wobble, 29 Sep 2026)
  shake       the stabilisation residual of the centre path (steadiness.py)

A segment is CLEAN when it lasts MIN_CLEAN_S or more and every score is under
its limit; a clean hold (a steady hover) counts. The clean segments are a
drone unit's usable parts: `apply_clean` narrows the catalogue unit's
`steady` field to them, so the proposer, the verifier's frames and the
builder use the same parts, as they do for steady runs. The labels are
heuristics for the proposer and for the editor's review; the clean/not-clean call is
the measurement. Limits are provisional until the editor has watched the review.

    uv run python -m videoeasy.moves --config config.<film>.yaml [--only <unit>] [--force]
    -> <out_dir>/moves.json  {units: {unit: {status, segments: [...], clean: [{t0, t1, kind, label}], clean_s}}}
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

from .evalrun import OK, UNCHECKED, fingerprint, source_hash
from . import steadiness as steady_mod

SAMPLE_FPS = steady_mod.SAMPLE_FPS
WIDTH = steady_mod.WIDTH
MIN_INLIERS = steady_mod.MIN_INLIERS
MIN_REGION_PTS = 6
DRONE_PREFIXES = ("DJI_",)

SMOOTH_S = 1.0          # velocity smoothing
HOVER_PCT_S = 0.5       # under this speed (% of frame width per second) the camera is holding
TURN_DEG = 35.0         # a change of direction this large, held SPLIT_HOLD_S, starts a new segment
SPLIT_HOLD_S = 0.5
MIN_SEG_S = 1.0         # a shorter piece is a transition: its own segment, never clean
MIN_CLEAN_S = 3.0       # a clean segment must last this long (as a steady run must)
MIN_DIRECTION = 0.90    # provisional: mean cosine of the velocity to the segment's direction
MAX_DIP = 0.25          # provisional: deepest mid-move drop in speed, as a share of the move's top speed
MAX_UNRELIABLE = 0.2    # a segment with more unreliable pairs than this is not clean
ORBIT_RATIO = 0.4       # lateral move whose centre travels under this share of its edges: an orbit
PARALLAX_RATIO = 1.4    # near (bottom) flow over far (top) flow above this: a slide, rise or descend

COLS = ("t", "dx", "dy", "zoom", "rot", "inliers", "vx_c", "vx_e", "vx_t", "vx_b", "vy_t", "vy_b")


# ----------------------------------------------------------------- per pair
def _median(v: np.ndarray) -> float:
    return float(np.median(v)) if len(v) >= MIN_REGION_PTS else float("nan")


def pair_motion(prev: np.ndarray, cur: np.ndarray) -> tuple[float, ...]:
    """(dx, dy, zoom, rot, inliers, vx_c, vx_e, vx_t, vx_b, vy_t, vy_b) between two grey frames, in pixels of
    the analysis frame (dx, dy: the shift of the frame centre; zoom: log scale; rot: radians, clockwise on screen).
    A pair without enough tracked features returns zeros and inliers 0 (unreliable), as steadiness does."""
    import cv2
    h, w = prev.shape[:2]
    nan = float("nan")
    none = (0.0, 0.0, 0.0, 0.0, 0, nan, nan, nan, nan, nan, nan)
    p0 = cv2.goodFeaturesToTrack(prev, 400, 0.01, 8)
    if p0 is None or len(p0) < MIN_INLIERS:
        return none
    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0, None, winSize=(21, 21), maxLevel=3)
    ok = st.ravel() == 1
    a = p0.reshape(-1, 2)[ok]
    b = p1.reshape(-1, 2)[ok]
    if len(a) < MIN_INLIERS:
        return none
    M, inl = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=2.0)
    if M is None:
        return none
    cx, cy = w / 2.0, h / 2.0
    dx = float(M[0, 0] * cx + M[0, 1] * cy + M[0, 2] - cx)
    dy = float(M[1, 0] * cx + M[1, 1] * cy + M[1, 2] - cy)
    zoom = math.log(max(1e-6, math.hypot(M[0, 0], M[1, 0])))
    rot = math.atan2(M[1, 0], M[0, 0])
    f = b - a
    x, y = a[:, 0], a[:, 1]
    centre = (x >= w / 3) & (x < 2 * w / 3) & (y >= h / 3) & (y < 2 * h / 3)
    top, bottom = y < h / 2, y >= h / 2
    return (dx, dy, zoom, rot, int(inl.sum()), _median(f[centre, 0]), _median(f[~centre, 0]),
            _median(f[top, 0]), _median(f[bottom, 0]), _median(f[top, 1]), _median(f[bottom, 1]))


def motion_path(path: str | Path, in_s: float = 0.0, out_s: float | None = None,
                sample_fps: float = SAMPLE_FPS, width: int = WIDTH) -> np.ndarray:
    """Rows of COLS for the source between in_s and out_s, decoded sequentially (steadiness.camera_path's loop)."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    step = max(1, round(fps / sample_fps))
    if in_s > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(in_s * fps))
    k0 = k = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    prev = None
    rows = []
    while True:
        if not cap.grab():
            break
        t = k / fps
        if out_s is not None and t >= out_s:
            break
        if (k - k0) % step == 0:
            ok, fr = cap.retrieve()
            if not ok:
                break
            h = round(fr.shape[0] * width / fr.shape[1])
            g = cv2.cvtColor(cv2.resize(fr, (width, h)), cv2.COLOR_BGR2GRAY)
            if prev is not None:
                rows.append((t,) + pair_motion(prev, g))
            prev = g
        k += 1
    cap.release()
    return np.array(rows, dtype=float).reshape(-1, len(COLS))


def cached_motion(source_path: str | Path, cache_dir: Path) -> np.ndarray:
    """Whole-source motion path, decoded once and kept as .npy beside a fingerprint."""
    src = Path(source_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    npy = cache_dir / f"{src.stem}.{fingerprint(src)}.moves.npy"   # never the name steadiness caches under beside it
    if npy.exists():
        return np.load(npy)
    rows = motion_path(src)
    np.save(npy, rows)
    return rows


def slice_rows(rows: np.ndarray, t0: float, t1: float) -> np.ndarray:
    m = (rows[:, 0] >= t0) & (rows[:, 0] < t1)
    return rows[m]


# ---------------------------------------------------------------- along a take
def _smooth(v: np.ndarray, n: int) -> np.ndarray:
    n = max(1, n) | 1
    if len(v) < 2 or n == 1:
        return v.copy()
    pad = min(n // 2, len(v) - 1)
    ker = np.ones(2 * pad + 1) / (2 * pad + 1)
    return np.convolve(np.pad(v, (pad, pad), mode="reflect", reflect_type="odd"), ker, mode="valid")


def _fill(v: np.ndarray, good: np.ndarray) -> np.ndarray:
    """Unreliable samples take the value interpolated from their reliable neighbours."""
    if good.all() or not good.any():
        return v.copy()
    idx = np.arange(len(v))
    return np.interp(idx, idx[good], v[good])


def velocity(rows: np.ndarray, width: int = WIDTH) -> dict[str, np.ndarray]:
    """Per sample, % of frame width per second: lat (+ = scene moves right), vert (+ = scene moves down),
    zoom (+ = in), rot (+ = scene turns clockwise), each smoothed over SMOOTH_S; plus `good` (reliable pair)."""
    t = rows[:, 0]
    dt = np.diff(t, prepend=t[0] - 1.0 / SAMPLE_FPS)
    dt[dt <= 0] = 1.0 / SAMPLE_FPS
    good = rows[:, 5] >= MIN_INLIERS
    n = int(round(SMOOTH_S * SAMPLE_FPS))
    raw = {"lat": rows[:, 1] / width * 100 / dt, "vert": rows[:, 2] / width * 100 / dt,
           "zoom": rows[:, 3] * 50 / dt, "rot": rows[:, 4] * 50 / dt}
    out = {k: _smooth(_fill(v, good), n) for k, v in raw.items()}
    out["t"] = t
    out["good"] = good
    return out


def _vec(v: dict, i) -> np.ndarray:
    return np.stack([v["lat"][i], v["vert"][i], v["zoom"][i], v["rot"][i]], axis=-1)


def split_points(v: dict) -> list[int]:
    """Indices where a new segment starts: the camera starts or stops moving, or turns by TURN_DEG, and holds the
    change for SPLIT_HOLD_S (a single noisy sample never splits a move)."""
    V = _vec(v, slice(None))
    speed = np.linalg.norm(V, axis=1)
    moving = speed >= HOVER_PCT_S
    hold = max(1, int(round(SPLIT_HOLD_S * SAMPLE_FPS)))
    cos_turn = math.cos(math.radians(TURN_DEG))
    starts = [0]
    s = 0
    i = 1
    while i < len(V):
        seg_moving = moving[s:i].mean() >= 0.5
        if moving[i] != seg_moving and (moving[i:i + hold] != seg_moving).all():
            starts.append(i)
            s = i
            i += hold
            continue
        if seg_moving and moving[i]:
            d = V[s:i][moving[s:i]].mean(axis=0)
            nd = np.linalg.norm(d)
            if nd > 0:
                cos = (V[i:i + hold] @ d) / (np.linalg.norm(V[i:i + hold], axis=1) * nd + 1e-12)
                if len(cos) == hold and (cos < cos_turn).all():
                    starts.append(i)
                    s = i
                    i += hold
                    continue
        i += 1
    return starts


def dip(speed: np.ndarray) -> float:
    """The deepest drop between two faster moments, over the top speed. For a speed that only rises, cruises and
    falls (one peak), the lower of the running maxima from each end IS the speed, so the gap between them is zero;
    a hesitation opens it."""
    if len(speed) < 3 or speed.max() <= 0:
        return 0.0
    env = np.minimum(np.maximum.accumulate(speed), np.maximum.accumulate(speed[::-1])[::-1])
    return float((env - speed).max() / speed.max())


def split_dips(speed: np.ndarray, a: int, b: int, min_n: int) -> list[tuple[int, int]]:
    """[a, b) cut at its slowest mid-move moment while the speed dips more than MAX_DIP and both sides keep
    min_n samples: a long move with one slowdown is two moves an editor can use, not one bad one (v8's
    measure rejected a 62 s pull-out whole for a single dip, 29 Sep 2026)."""
    sp = speed[a:b]
    if len(sp) < 2 * min_n or sp.max() <= 0:
        return [(a, b)]
    env = np.minimum(np.maximum.accumulate(sp), np.maximum.accumulate(sp[::-1])[::-1])
    gap = env - sp
    if gap.max() / sp.max() <= MAX_DIP:
        return [(a, b)]
    lo, hi = min_n, len(sp) - min_n
    if hi <= lo:
        return [(a, b)]
    cut = lo + int(np.argmax(gap[lo:hi]))
    if gap[cut] / sp.max() <= MAX_DIP:
        return [(a, b)]
    return split_dips(speed, a, a + cut, min_n) + split_dips(speed, a + cut, b, min_n)


def label(mean: np.ndarray, region: dict[str, float]) -> tuple[str, str]:
    """(kind, label) from a segment's mean velocity (lat, vert, zoom, rot) and its median region flows."""
    speed = float(np.linalg.norm(mean))
    if speed < HOVER_PCT_S:
        return "hover", "hover"
    names = ("lat", "vert", "zoom", "rot")
    order = np.argsort(-np.abs(mean))
    first, second = names[order[0]], names[order[1]]

    def one(comp: str) -> tuple[str, str]:
        val = mean[names.index(comp)]
        if comp == "lat":
            c, e, t, b = (abs(region.get(k, float("nan"))) for k in ("vx_c", "vx_e", "vx_t", "vx_b"))
            side = "left" if val > 0 else "right"          # the scene moving right means the camera pans or travels left
            if not math.isnan(c) and not math.isnan(e) and e > 0 and c < ORBIT_RATIO * e:
                # an orbit yaws to hold its subject, so the background runs the way the camera travels
                return "orbit", f"orbit {'right' if val > 0 else 'left'}"
            if not math.isnan(t) and not math.isnan(b) and min(t, b) > 0 and max(t, b) / min(t, b) > PARALLAX_RATIO:
                return "slide", f"slide {side}"
            return "pan", f"pan {side}"
        if comp == "vert":
            t, b = (abs(region.get(k, float("nan"))) for k in ("vy_t", "vy_b"))
            parallax = not math.isnan(t) and not math.isnan(b) and min(t, b) > 0 and max(t, b) / min(t, b) > PARALLAX_RATIO
            if val > 0:                                     # the scene moving down: the camera rises or tilts up
                return ("rise", "rise") if parallax else ("tilt", "tilt up")
            return ("descend", "descend") if parallax else ("tilt", "tilt down")
        if comp == "zoom":
            return ("push-in", "push-in") if val > 0 else ("pull-out", "pull-out")
        return "rotate", "rotate " + ("clockwise" if val > 0 else "anticlockwise")

    kind, text = one(first)
    if abs(mean[order[1]]) >= 0.5 * abs(mean[order[0]]):
        text += " + " + one(second)[1]
    pace = "slow" if speed < 2.0 else "steady" if speed < 5.0 else "fast"
    return kind, f"{pace} {text}"


def segments(rows: np.ndarray, t_end: float | None = None, max_shake: float = steady_mod.MAX_SHAKE_PCT) -> list[dict]:
    """The take cut where the camera starts, stops or turns, each piece measured and labelled. A piece under
    MIN_SEG_S is a transition (the camera changing what it does) and is never clean; it is kept as its own
    segment so a turn cannot spoil the moves either side of it."""
    if rows is None or len(rows) < steady_mod.MIN_PAIRS:
        return []
    v = velocity(rows)
    t = v["t"]
    starts = split_points(v)
    V = _vec(v, slice(None))
    speed = np.linalg.norm(V, axis=1)
    min_n = int(round(MIN_SEG_S * SAMPLE_FPS))
    bounds = []
    for a, b in zip(starts, starts[1:] + [len(t)]):
        moving = speed[a:b].mean() >= HOVER_PCT_S
        bounds += split_dips(speed, a, b, min_n) if moving else [(a, b)]
    out = []
    for a, b in bounds:
        seg_t0 = float(t[a]) - 1.0 / SAMPLE_FPS if a == 0 else float(t[a])
        seg_t1 = float(t[b]) if b < len(t) else (t_end if t_end is not None else float(t[-1]))
        dur = seg_t1 - seg_t0
        sl = slice(a, b)
        mean = V[sl].mean(axis=0)
        sp = float(np.linalg.norm(mean))
        region = {c: float(np.nanmedian(rows[sl, COLS.index(c)])) if np.isfinite(rows[sl, COLS.index(c)]).any() else float("nan")
                  for c in ("vx_c", "vx_e", "vx_t", "vx_b", "vy_t", "vy_b")}
        kind, text = label(mean, region) if dur >= MIN_SEG_S else ("transition", "transition")
        if kind in ("hover", "transition"):
            direction, hesitation = 1.0, 0.0
        else:
            sp_t = np.linalg.norm(V[sl], axis=1)
            cos = (V[sl] @ mean) / (sp_t * sp + 1e-12)
            direction = float((cos * sp_t).sum() / (sp_t.sum() + 1e-12))
            hesitation = dip(sp_t)
        st = steady_mod.shake_stats(rows[sl][:, [0, 1, 2, 5]])
        shake = st.get("shake_pct") if st.get("status") == OK else None
        unreliable = float(1.0 - v["good"][sl].mean())
        why = []
        if kind == "transition":
            why.append("transition")
        if dur < MIN_CLEAN_S:
            why.append(f"{dur:.1f} s long")
        if direction < MIN_DIRECTION:
            why.append(f"direction {direction:.2f}")
        if hesitation > MAX_DIP:
            why.append(f"speed dips {hesitation:.0%}")
        if shake is None:
            why.append("shake unmeasured")
        elif shake > max_shake:
            why.append(f"shake {shake}%")
        if unreliable > MAX_UNRELIABLE:
            why.append(f"{unreliable:.0%} unreliable pairs")
        out.append(dict(t0=round(seg_t0, 2), t1=round(seg_t1, 2), kind=kind, label=text, speed=round(sp, 2),
                        lat=round(float(mean[0]), 2), vert=round(float(mean[1]), 2), zoom=round(float(mean[2]), 2), rot=round(float(mean[3]), 2),
                        direction=round(direction, 3), dip=round(hesitation, 3), shake_pct=shake, unreliable=round(unreliable, 3),
                        clean=not why, why=why))
    return out


def clean_parts(segs: list[dict]) -> list[dict]:
    return [dict(t0=s["t0"], t1=s["t1"], kind=s["kind"], label=s["label"]) for s in segs if s["clean"]]


def clean_except_length(seg: dict) -> bool:
    """A segment that fails only on length: a cover stretch the builder cut short is judged on the move itself."""
    return all(w.endswith(" s long") for w in seg["why"])


def fmt_clean(parts: list[dict]) -> str:
    return "; ".join(f"{steady_mod.fmt_runs([p])} {p['label']}" for p in parts)


# ------------------------------------------------------------ catalogue rule
def is_drone(sid: str, prefixes=DRONE_PREFIXES) -> bool:
    return any(sid.startswith(p) for p in prefixes)


def load(film_out: Path) -> dict[str, dict] | None:
    p = Path(film_out) / "moves.json"
    if not p.exists():
        return None
    return json.loads(p.read_text()).get("units")


def _intersect(a: list[dict], b: list[dict]) -> list[dict]:
    out = []
    for x in a:
        for y in b:
            lo, hi = max(x["t0"], y["t0"]), min(x["t1"], y["t1"])
            if hi - lo >= MIN_CLEAN_S - 1e-6:
                out.append(dict(y, t0=round(lo, 2), t1=round(hi, 2)))
    return out


def apply_clean(cat: dict[str, dict], moves: dict[str, dict] | None, prefixes=DRONE_PREFIXES) -> dict[str, str]:
    """A measured drone unit is usable only inside its clean segments (the editor, 29 Sep 2026): `steady` (the parts the
    proposer, verifier and builder use; None = the whole unit) is narrowed to them, `moves` carries their labels for
    the proposer, and a unit with none is returned as not cover. Run after steadiness.apply_runs. An unmeasured
    drone unit is left as it is; the scorer measures the render."""
    out: dict[str, str] = {}
    if not moves:
        return out
    for sid, c in cat.items():
        m = moves.get(sid)
        if not is_drone(sid, prefixes) or not m or m.get("status") != OK:
            continue
        parts = m.get("clean") or []
        base = c.get("steady") if c.get("steady") is not None else [dict(t0=c["in_s"], t1=c["out_s"])]
        use = _intersect(base, parts)
        c["steady"] = use
        c["moves"] = use
        if not use:
            out[sid] = "drone: no clean move or hold of 3 s or more"
    return out


# ----------------------------------------------------------------- source pass
def run(cfg, only: str | None = None, force: bool = False, prefixes=DRONE_PREFIXES) -> dict:
    from . import segments as segments_mod
    out_path = cfg.out_dir / "moves.json"
    doc = json.loads(out_path.read_text()) if out_path.exists() and not force else {}
    units = doc.setdefault("units", {})
    doc["method"] = dict(sample_fps=SAMPLE_FPS, width=WIDTH, smooth_s=SMOOTH_S, hover_pct_s=HOVER_PCT_S, turn_deg=TURN_DEG,
                         min_clean_s=MIN_CLEAN_S, min_direction=MIN_DIRECTION, max_dip=MAX_DIP,
                         max_shake_pct=steady_mod.MAX_SHAKE_PCT, orbit_ratio=ORBIT_RATIO, parallax_ratio=PARALLAX_RATIO,
                         prefixes=list(prefixes), tool_sha=source_hash(__file__))
    todo = [u for u in segments_mod.units(cfg) if is_drone(u["source_id"], prefixes) and (only is None or u["shot_id"] == only)
            and (force or u["shot_id"] not in units)]
    by_source: dict[str, list[dict]] = {}
    for u in todo:
        by_source.setdefault(u["source_path"], []).append(u)
    for k, (sp, us) in enumerate(by_source.items(), 1):
        try:
            rows = cached_motion(sp, cfg.work_dir / "moves")
        except Exception as e:  # noqa: BLE001
            for u in us:
                units[u["shot_id"]] = dict(status=UNCHECKED, why=f"decode failed: {str(e)[:120]}")
            continue
        for u in us:
            segs = segments(slice_rows(rows, u["in_s"], u["out_s"]), t_end=u["out_s"])
            if not segs:
                units[u["shot_id"]] = dict(status=UNCHECKED, why="too short to measure", source=u["source_id"], in_s=u["in_s"], out_s=u["out_s"])
                continue
            parts = clean_parts(segs)
            units[u["shot_id"]] = dict(status=OK, source=u["source_id"], in_s=u["in_s"], out_s=u["out_s"], segments=segs,
                                       clean=parts, clean_s=round(sum(p["t1"] - p["t0"] for p in parts), 1))
        print(f"[{k}/{len(by_source)}] {Path(sp).name}: " + ", ".join(
            f"{u['shot_id']} {units[u['shot_id']].get('clean_s', '?')}s clean" for u in us), file=sys.stderr)
        out_path.write_text(json.dumps(doc, indent=1))
    out_path.write_text(json.dumps(doc, indent=1))
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--only", default=None, help="one unit id (shot or window)")
    ap.add_argument("--force", action="store_true", help="re-measure units already in moves.json")
    ap.add_argument("--prefix", action="append", default=None, help="source-name prefix to measure (default DJI_)")
    args = ap.parse_args(argv)
    from .config import load_config
    cfg = load_config(args.config)
    doc = run(cfg, args.only, args.force, tuple(args.prefix) if args.prefix else DRONE_PREFIXES)
    units = doc["units"]
    ok = {k: u for k, u in units.items() if u.get("status") == OK}
    total = sum(u["out_s"] - u["in_s"] for k, u in ok.items() if "_w" not in k)
    clean = sum(u["clean_s"] for k, u in ok.items() if "_w" not in k)
    kinds: dict[str, float] = {}
    for k, u in ok.items():
        if "_w" in k:
            continue
        for p in u["clean"]:
            kinds[p["kind"]] = kinds.get(p["kind"], 0.0) + p["t1"] - p["t0"]
    print(f"{len(units)} units, {len(ok)} measured; takes {total:.0f} s, clean {clean:.0f} s: "
          + ", ".join(f"{k} {v:.0f}s" for k, v in sorted(kinds.items(), key=lambda x: -x[1])), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
