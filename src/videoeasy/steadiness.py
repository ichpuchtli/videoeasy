"""Camera steadiness as a measurement: how far the camera path departs from a smoothed version of itself.

The editor (26 Sep 2026): B-roll with heavy camera shake is being used as cover and
should be trimmed and never used. The vision judge sees stills, so it wrote
"none" under technical for nearly every shaky stretch of cut-v6. Shake is a
motion property, so it is measured, not judged:

  1. track features between frames sampled at ~12 fps (Lucas-Kanade), fit a
     rigid transform per pair with RANSAC, keep the translation and the inlier
     count;
  2. the camera path is the cumulative translation; a pair with too few
     inliers (a cut, a black frame, no texture) contributes nothing;
  3. smooth the path with a one-second moving average; the residual is what a
     stabiliser would have to correct;
  4. shake_pct = RMS residual as a percentage of frame width.

On one cut-v6 render (26 Sep 2026) the
tripod sit-down reads 0.06, cover stretches 0.4 median, the handheld walk
windows 2-5, and the worst cover 13. MAX_SHAKE_PCT is provisional until the editor
confirms the ranked list; it is not a calibrated number.

Source pass (one decode per source, cached; every catalogue unit sliced from it):

    uv run python -m videoeasy.steadiness --config config.<film>.yaml [--only <unit>] [--force]
    -> <out_dir>/steadiness.json  {unit_id: {shake_pct, pan_pct_per_s, unreliable_frac, inlier_med, n, status,
                                             stable_runs: [{t0, t1, shake_pct}], stable_s, profile_max}}

A unit is one number for the rule, but shake changes along a take: the walk
settles, the handheld pan lands. `profile()` reads the residual in 2 s windows
every 0.5 s and `stable_runs()` keeps the time no window over the limit
touches, in stretches (>= 3 s) a cut could still use, each re-measured whole. Those runs
ride in steadiness.json so a shaky unit is barred whole only when it has none.
The worst window rides too (`profile_max`): a unit under the limit as a whole
can still hold a shaky stretch, and v8 cut two such units straight across
theirs (a handheld insert, 1.7 % whole, 2.5 % where it was used; a drone sub-shot,
1.6 % whole, 3.9 % where it was used; 29 Sep 2026). Any unit with a window over the
limit is therefore restricted to its runs; one with none is used whole, however
short (a 3 s steady unit has no 3 s run and is still cover).

The proposer drops units over the threshold from the catalogue it shows the
model, the builder refuses them, and the scorer measures every cover stretch
of the render itself (cuteval), so a shot that slips through is still caught.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from .evalrun import OK, UNCHECKED, fingerprint, source_hash

MAX_SHAKE_PCT = 2.0       # provisional (see module docstring); over this a unit is not cover
SAMPLE_FPS = 12.0
WIDTH = 480               # analysis width in px; shake_pct is relative to it
MIN_INLIERS = 12          # a pair with fewer RANSAC inliers is unreliable and contributes no motion
SMOOTH_S = 1.0
MIN_PAIRS = 12            # a unit shorter than this many pairs (~1 s) is unchecked
PROFILE_WIN_S = 2.0       # shake is read along a unit in windows this long ...
PROFILE_HOP_S = 0.5       # ... every this often
MIN_STABLE_S = 3.0        # a steady stretch shorter than this is not worth a cover row


def camera_path(path: str | Path, in_s: float = 0.0, out_s: float | None = None,
                sample_fps: float = SAMPLE_FPS, width: int = WIDTH) -> np.ndarray:
    """Per-pair rows (t, dx, dy, inliers) for the source between in_s and out_s, decoded sequentially."""
    import cv2
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
    step = max(1, round(fps / sample_fps))
    if in_s > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(in_s * fps))
    k0 = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
    k = k0
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
                dx = dy = 0.0
                inl = 0
                p0 = cv2.goodFeaturesToTrack(prev, 300, 0.01, 8)
                if p0 is not None and len(p0) >= MIN_INLIERS:
                    p1, st, _ = cv2.calcOpticalFlowPyrLK(prev, g, p0, None, winSize=(21, 21), maxLevel=3)
                    a = p0[st.ravel() == 1]
                    b = p1[st.ravel() == 1]
                    if len(a) >= MIN_INLIERS:
                        M, inliers = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=2.0)
                        if M is not None:
                            dx, dy, inl = float(M[0, 2]), float(M[1, 2]), int(inliers.sum())
                rows.append((t, dx, dy, inl))
            prev = g
        k += 1
    cap.release()
    return np.array(rows, dtype=float).reshape(-1, 4)


def shake_stats(pairs: np.ndarray, width: int = WIDTH, sample_fps: float = SAMPLE_FPS, smooth_s: float = SMOOTH_S) -> dict:
    """Stabilisation residual of a camera path. `pairs` rows are (t, dx, dy, inliers)."""
    if pairs is None or len(pairs) < MIN_PAIRS:
        return dict(status=UNCHECKED, n=0 if pairs is None else int(len(pairs)), why="too few frame pairs")
    good = pairs[:, 3] >= MIN_INLIERS
    d = pairs[:, 1:3].copy()
    d[~good] = 0.0
    r = residual_series(pairs, width, sample_fps, smooth_s)[:, 1]
    shake = float(np.sqrt((r ** 2).mean()))
    pan = float(np.linalg.norm(d[good].mean(0)) * sample_fps / width * 100) if good.any() else 0.0
    return dict(status=OK, n=int(len(pairs)), shake_pct=round(shake, 3), pan_pct_per_s=round(pan, 2),
                unreliable_frac=round(float((~good).mean()), 3), inlier_med=int(np.median(pairs[:, 3])))


def residual_series(pairs: np.ndarray, width: int = WIDTH, sample_fps: float = SAMPLE_FPS, smooth_s: float = SMOOTH_S) -> np.ndarray:
    """Rows (t, residual in % of frame width) for every pair: the same path, smoothing and padding as shake_stats,
    kept per sample so a unit can be read along its length instead of as one number."""
    good = pairs[:, 3] >= MIN_INLIERS
    d = pairs[:, 1:3].copy()
    d[~good] = 0.0
    path = np.cumsum(d, axis=0)
    win = 2 * max(1, int(round(smooth_s * sample_fps / 2))) + 1   # odd, centred: an even window shifts the path by half a sample
    ker = np.ones(win) / win
    pad = (win // 2, win // 2)
    # odd reflection at the ends: a linear pan leaves no residual, where edge padding would charge the ends of every pan
    smooth = np.stack([np.convolve(np.pad(path[:, i], pad, mode="reflect", reflect_type="odd"), ker, mode="valid") for i in range(2)], 1)
    r = np.linalg.norm(path - smooth, axis=1) / width * 100
    return np.column_stack([pairs[:, 0], r])


def profile(pairs: np.ndarray, win_s: float = PROFILE_WIN_S, hop_s: float = PROFILE_HOP_S) -> list[tuple[float, float]]:
    """(window start, shake_pct over the window) every hop_s along the unit: shake as a function of time.
    Windows shorter than win_s at the tail are dropped; a unit under one window returns []."""
    if pairs is None or len(pairs) < MIN_PAIRS:
        return []
    rs = residual_series(pairs)
    t, r = rs[:, 0], rs[:, 1]
    t0, t_end = float(t[0]), float(t[-1])
    out = []
    a = t0
    while a + win_s <= t_end + 1e-6:
        m = (t >= a) & (t < a + win_s)
        if m.sum() >= MIN_PAIRS // 2:
            out.append((round(a, 2), round(float(np.sqrt((r[m] ** 2).mean())), 3)))
        a += hop_s
    return out


def stable_runs(pairs: np.ndarray, max_pct: float = MAX_SHAKE_PCT, min_s: float = MIN_STABLE_S,
                win_s: float = PROFILE_WIN_S, hop_s: float = PROFILE_HOP_S) -> list[dict]:
    """The stretches of a unit that hold under the limit for at least min_s, each re-measured whole:
    [{t0, t1, shake_pct}] in unit source time. A steady unit returns one run covering it; a shaky unit
    returns the parts a cut could still use. The editor (28 Sep 2026): shaky B-roll should use its stable parts,
    not be barred whole."""
    prof = profile(pairs, win_s, hop_s)
    if not prof:
        return []
    # a moment is steady when a measured window holds it and no window over the limit does. Joining overlapping
    # steady windows instead let a run bridge the window between them: 46 over-limit windows sat inside B-roll runs,
    # up to 2.58 %, where a short row would measure shaky on the render (29 Sep 2026)
    under = _union([(a, a + win_s) for a, v in prof if v <= max_pct])
    over = _union([(a, a + win_s) for a, v in prof if v > max_pct])
    runs = _minus(under, over)
    out = []
    for a, b in runs:
        if b - a + 1e-6 < min_s:
            continue
        st = shake_stats(slice_pairs(pairs, a, b))
        if st.get("status") == OK and st["shake_pct"] <= max_pct:
            out.append(dict(t0=round(a, 2), t1=round(b, 2), shake_pct=st["shake_pct"]))
    return out


def _union(iv: list[tuple[float, float]]) -> list[list[float]]:
    out: list[list[float]] = []
    for a, b in sorted(iv):
        if out and a <= out[-1][1] + 1e-6:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return out


def _minus(keep: list[list[float]], cut: list[list[float]]) -> list[list[float]]:
    out = []
    for a, b in keep:
        pieces = [[a, b]]
        for c, d in cut:
            nxt = []
            for x, y in pieces:
                if d <= x or c >= y:
                    nxt.append([x, y])
                    continue
                if c > x:
                    nxt.append([x, c])
                if d < y:
                    nxt.append([d, y])
            pieces = nxt
        out += pieces
    return out


def slice_pairs(pairs: np.ndarray, t0: float, t1: float) -> np.ndarray:
    m = (pairs[:, 0] >= t0) & (pairs[:, 0] < t1)
    return pairs[m]


def measure(path: str | Path, in_s: float, out_s: float) -> dict:
    """One unit straight from its source (no cache)."""
    return shake_stats(camera_path(path, in_s, out_s))


def shaky(st: dict | None, max_pct: float = MAX_SHAKE_PCT) -> bool:
    return bool(st) and st.get("status") == OK and st.get("shake_pct", 0.0) > max_pct


def has_shaky_stretch(st: dict | None, max_pct: float = MAX_SHAKE_PCT) -> bool:
    """A 2 s window of the unit over the limit, whatever the unit measures whole. A measured unit without
    `profile_max` predates the field: stop rather than pass it whole, which is how v8 got its two shaky rows."""
    if not st or st.get("status") != OK:
        return False
    if "profile_max" not in st:
        raise SystemExit("steadiness.json predates profile_max (the worst 2 s window per unit): re-run "
                         "videoeasy.steadiness --force before building or proposing; a unit steady as a whole can hide a shaky stretch")
    return st["profile_max"] is not None and st["profile_max"] > max_pct


def usable_runs(st: dict | None, max_pct: float = MAX_SHAKE_PCT) -> list[dict] | None:
    """None when the unit may be used whole (every window under the limit, or unmeasured: the scorer measures the
    render); otherwise the steady parts a cut may use, [] when there are none. A unit is restricted when it is over
    the limit as a whole or when any 2 s window of it is. Runs found at another limit stop the run (never a silent
    whole-unit bar)."""
    if not shaky(st, max_pct) and not has_shaky_stretch(st, max_pct):
        return None
    if st.get("stable_limit") != max_pct:
        # runs found at another limit say nothing about this one; barring the whole unit instead would discard
        # footage the editor wants used (28 Sep 2026), so the caller must re-measure rather than proceed
        raise SystemExit(f"steadiness.json holds steady parts found at {st.get('stable_limit') or 'no'}% (this unit is over {max_pct}%): "
                         f"re-run videoeasy.steadiness --force (with MAX_SHAKE_PCT set to {max_pct}) before using this limit")
    return [dict(r) for r in st.get("stable_runs") or []]


def ineligible(steady: dict[str, dict] | None, units: list[str], max_pct: float = MAX_SHAKE_PCT) -> dict[str, str]:
    """unit -> reason for every unit the measurement rules out as cover: over the limit with no steady part long
    enough to use. A shaky unit with steady parts is not ruled out; it is restricted to them (`apply_runs`).
    An unmeasured unit is not ruled out here; the scorer measures the render, so it cannot pass unseen."""
    if not steady:
        return {}
    out = {}
    for u in units:
        st = steady.get(u)
        runs = usable_runs(st, max_pct)
        if runs is not None and not runs:
            worst = max(st["shake_pct"], st.get("profile_max") or 0.0)
            out[u] = f"camera shake up to {worst}% of frame width (limit {max_pct}), no steady part of {MIN_STABLE_S:.0f} s or more"
    return out


def apply_runs(cat: dict[str, dict], steady: dict[str, dict] | None, max_pct: float = MAX_SHAKE_PCT) -> dict[str, str]:
    """Mark every catalogue unit with the steady parts it is limited to (`steady`: list of runs, or None when the
    whole unit may be used) and return the units ruled out altogether. The proposer, the verifier and the builder
    all read `steady` from the unit, so a shaky take is used only inside its steady parts (the editor, 28 Sep 2026)."""
    for sid, c in cat.items():
        c["steady"] = usable_runs((steady or {}).get(sid), max_pct)
    return ineligible(steady, list(cat), max_pct)


def clamp_to_runs(start: float, hold: float, runs: list[dict], min_s: float = 1.0) -> tuple[float, float] | None:
    """(start, hold) inside one steady part at or after `start`: the first part that holds the whole row, else the
    part that holds the most of it (at least min_s), else None. The first part with room is not enough: v8's butcher
    bird was cut to 5 s of a 5 s part when a 10 s part followed (29 Sep 2026)."""
    best = None
    for r in sorted(runs, key=lambda r: r["t0"]):
        a = max(start, r["t0"])
        room = r["t1"] - a
        if room < min_s - 1e-6:
            continue
        got = min(hold, room)
        if got >= hold - 0.01:
            return round(a, 2), round(got, 2)
        if best is None or got > best[1] + 1e-6:
            best = (round(a, 2), round(got, 2))
    return best


def fmt_runs(runs: list[dict]) -> str:
    def f(t):
        m, sec = divmod(t, 60)
        return f"{int(m)}:{sec:04.1f}"
    return ", ".join(f"{f(r['t0'])}-{f(r['t1'])}" for r in runs)


def load(film_out: Path) -> dict[str, dict] | None:
    p = Path(film_out) / "steadiness.json"
    if not p.exists():
        return None
    doc = json.loads(p.read_text())
    return doc.get("units", doc)


# ----------------------------------------------------------------- source pass
def cached_path(source_path: str | Path, cache_dir: Path) -> np.ndarray:
    """Whole-source camera path, decoded once and kept as .npy beside a fingerprint."""
    src = Path(source_path)
    cache_dir.mkdir(parents=True, exist_ok=True)
    fp = fingerprint(src)
    npy = cache_dir / f"{src.stem}.{fp}.npy"
    if npy.exists():
        return np.load(npy)
    pairs = camera_path(src)
    np.save(npy, pairs)
    return pairs


def run(cfg, only: str | None = None, force: bool = False, skip_sources: set[str] | None = None) -> dict:
    from . import segments
    out_path = cfg.out_dir / "steadiness.json"
    doc = json.loads(out_path.read_text()) if out_path.exists() and not force else {}
    units = doc.setdefault("units", {})
    doc["method"] = dict(sample_fps=SAMPLE_FPS, width=WIDTH, smooth_s=SMOOTH_S, min_inliers=MIN_INLIERS, max_shake_pct=MAX_SHAKE_PCT,
                         profile_win_s=PROFILE_WIN_S, profile_hop_s=PROFILE_HOP_S, min_stable_s=MIN_STABLE_S,
                         tool_sha=source_hash(__file__))
    cache_dir = cfg.work_dir / "steadiness"
    skip = skip_sources or set()
    todo = [u for u in segments.units(cfg) if (only is None or u["shot_id"] == only) and u["source_id"] not in skip
            and (force or u["shot_id"] not in units)]
    by_source: dict[str, list[dict]] = {}
    for u in todo:
        by_source.setdefault(u["source_path"], []).append(u)
    for k, (sp, us) in enumerate(by_source.items(), 1):
        try:
            pairs = cached_path(sp, cache_dir)
        except Exception as e:  # noqa: BLE001
            for u in us:
                units[u["shot_id"]] = dict(status=UNCHECKED, why=f"decode failed: {str(e)[:120]}")
            continue
        for u in us:
            sl = slice_pairs(pairs, u["in_s"], u["out_s"])
            st = shake_stats(sl)
            st.update(source=u["source_id"], in_s=u["in_s"], out_s=u["out_s"], role=u["role"])
            if st.get("status") == OK:
                runs = stable_runs(sl)
                prof = profile(sl)
                st.update(stable_runs=runs, stable_s=round(sum(r["t1"] - r["t0"] for r in runs), 1), stable_limit=MAX_SHAKE_PCT,
                          profile_max=max(v for _, v in prof) if prof else None)
            units[u["shot_id"]] = st
        print(f"[{k}/{len(by_source)}] {Path(sp).name}: {len(us)} units, "
              + ", ".join(f"{u['shot_id']} {units[u['shot_id']].get('shake_pct', '?')}" for u in us[:6]) + (" …" if len(us) > 6 else ""),
              file=sys.stderr)
        out_path.write_text(json.dumps(doc, indent=1))
    out_path.write_text(json.dumps(doc, indent=1))
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--only", default=None, help="one unit id (shot or window)")
    ap.add_argument("--force", action="store_true", help="re-measure units already in steadiness.json")
    ap.add_argument("--skip-source", action="append", default=[], help="source id to leave out (e.g. the index-only sit-down take)")
    args = ap.parse_args(argv)
    from .config import load_config
    cfg = load_config(args.config)
    skip = set(args.skip_source) | set((cfg.raw.get("segments") or {}).get("index_only_sources") or [])
    doc = run(cfg, args.only, args.force, skip)
    units = doc["units"]
    ok = [u for u in units.values() if u.get("status") == OK]
    over = sorted(((k, u["shake_pct"]) for k, u in units.items() if shaky(u)), key=lambda x: -x[1])
    hidden = sorted(k for k, u in units.items() if not shaky(u) and has_shaky_stretch(u))
    print(f"{len(units)} units, {len(ok)} measured, {len(units) - len(ok)} unchecked; {len(over)} over {MAX_SHAKE_PCT}%: "
          + ", ".join(f"{k} {v}" for k, v in over[:20]), file=sys.stderr)
    print(f"{len(hidden)} more under the limit as a whole with a shaky 2 s stretch (steady parts only): " + ", ".join(hidden), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
