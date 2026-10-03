"""Lay a cut into a DaVinci Resolve timeline, read it back, and conform.

Three parts, kept apart so the first and last need no Resolve:

  plan(cut, ...)         frame-exact placement from the cut's destination
                         intervals: A-roll picks on alternating tracks with
                         crossfades capped by neighbouring words, cover on V3,
                         room-tone fills, per-pick gains measured on the source
  lay(plan, ...)         the writes, each verified by reading it back
  readback(timeline)     every item on every track: record frame, duration,
                         source in-point, fades, gain, channel map
  conform(plan, items)   planned vs read back, within a frame tolerance;
                         unmatched rows and unexpected items are listed, never
                         inferred from a total duration

Frame conventions: the timeline runs at 24 fps and record frames are
relative to the timeline start (86400 in this project). `AppendToTimeline`
takes startFrame/endFrame in the SOURCE's own rate, but `GetLeftOffset`
reports the in-point in TIMELINE frames (measured 25 Sep 2026: a drone
in-point planned at source frame 3842 read back as 3845 = 3842 × 24/23.976),
so conform converts before comparing. Drone rows also take one more timeline
frame per ~40 s than their source frame count, which is why matches use a
tolerance of two frames. A row's destination is `dest_in_s` from the
cut; the lay-in never re-derives placement by laying rows consecutively.

The Resolve project, bins, grades, placeholder cards and render preset come
from the film's profile (`film.yaml` `layin`, see film.py).

Usage:
    uv run python -m videoeasy.layin --film data/films/<film> --cut editorial/cut-v6.json \\
        --name "<film> cut-v6 (AI)" [--replace] [--render <stem>] [--plan-only] [--conform-only]
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from .evalrun import OK, UNCHECKED, fingerprint, refuse_overwrite, source_hash, write_manifest
from .film import Film, load_film
from .sources import base_clip, is_drone, placeholder_file

FPS = 24
FPS_DJI = 24000 / 1001
EXT = 6                 # frames of extension each side of a join -> 12-frame crossfade
TARGET_LUFS = -24.0
PEAK_CEIL = -1.5
GAIN_CAP = 30.0         # the API accepts no more; XT4 picks sit 5-8 dB low at this cap (known)
TOL = 2                 # frames: 23.976 rows push later items one frame


def f24(s: float) -> int:
    return int(round(s * FPS))


# ------------------------------------------------------------- audio on source
def channel_rms(path: str) -> list[float]:
    p = subprocess.run(["ffmpeg", "-v", "info", "-i", path, "-vn", "-af", "astats=measure_overall=none:measure_perchannel=RMS_level",
                        "-f", "null", "-"], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"astats failed on {path}: {p.stderr[-160:]}")
    return [float(x) for x in re.findall(r"RMS level dB:\s*(-?[\d.]+|-inf)", p.stderr.replace("-inf", "-99"))][:2]


def lufs_peak(path: str, a: float, b: float, mono_left: bool) -> tuple[float | None, float | None]:
    af = ("pan=mono|c0=c0," if mono_left else "") + "ebur128=peak=true:framelog=quiet"
    p = subprocess.run(["ffmpeg", "-v", "info", "-ss", f"{a:.3f}", "-t", f"{max(0.4, b - a):.3f}", "-i", path, "-vn", "-af", af,
                        "-f", "null", "-"], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ebur128 failed on {path}: {p.stderr[-160:]}")
    I = re.search(r"I:\s+(-?[\d.]+) LUFS", p.stderr)
    P = re.search(r"Peak:\s+(-?[\d.]+) dBFS", p.stderr)
    return (float(I.group(1)) if I else None, float(P.group(1)) if P else None)


CHANNEL_TOL_DB = 3.0   # a mono pick rendered on one channel only reads ~3 LU low overall and ~12+ dB apart per channel


def channel_lufs(path: str, a: float, b: float) -> tuple[float | None, float | None]:
    """Integrated loudness of the left and the right channel separately over [a, b]."""
    out = []
    for ch in (0, 1):
        p = subprocess.run(["ffmpeg", "-v", "info", "-ss", f"{a:.3f}", "-t", f"{max(0.4, b - a):.3f}", "-i", path, "-vn",
                            "-af", f"pan=mono|c0=c{ch},ebur128=framelog=quiet", "-f", "null", "-"], capture_output=True, text=True)
        m = re.search(r"I:\s+(-?[\d.]+) LUFS", p.stderr) if p.returncode == 0 else None
        out.append(float(m.group(1)) if m else None)
    return out[0], out[1]


def verify_channels(pl: dict, path: str, measure=None, tol: float = CHANNEL_TOL_DB) -> dict:
    """After a render: every mono pick must reach both channels. Resolve accepted the source channel map on the
    items (read back as stored) and still rendered cut-v7's 17 mono picks on the left channel only until the
    timeline was reopened (26 Sep 2026); only the rendered file can say whether the map was honoured."""
    measure = measure or (lambda a, b: channel_lufs(path, a, b))
    rows = []
    for r in pl["audio"]:
        if not r.get("mono"):
            continue
        a, b = r["rec"] / FPS, (r["rec"] + r["dur"]) / FPS
        if b - a < 0.4:
            continue
        L, R = measure(a, b)
        rows.append(dict(pick=r["pick"], left=L, right=R, diff=None if L is None or R is None else round(abs(L - R), 1),
                         status=UNCHECKED if L is None or R is None else (OK if abs(L - R) <= tol else "one_channel")))
    bad = [x for x in rows if x["status"] == "one_channel"]
    unc = [x for x in rows if x["status"] == UNCHECKED]
    return dict(checked=len(rows), one_channel=bad, unchecked=len(unc), status=OK if not bad and not unc else ("broken" if bad else UNCHECKED))


def gain_for(I: float | None, P: float | None) -> float | None:
    if I is None:
        return None
    return round(min(GAIN_CAP, TARGET_LUFS - I, PEAK_CEIL - (P if P is not None else 0.0)), 1)


LEVELER_MAX = 6.0  # Resolve's Dialogue Leveler output gain range is 0..6 dB


def leveler_for(I: float | None) -> float:
    """Dialogue Leveler output gain for a pick the +30 dB clip gain cannot lift to target (the editor, 26 Sep 2026):
    the shortfall past the cap, up to the leveler's own 6 dB; 0 when the clip gain suffices. The leveler runs in
    'optimize moderate levels' mode: measured on diagnostic renders it leaves the quiet XT4 picks 3-6 LU under
    target (accepted residual), while the lifting modes clip."""
    if I is None:
        return 0.0
    short = TARGET_LUFS - I - GAIN_CAP
    return round(min(LEVELER_MAX, short), 1) if short > 0.05 else 0.0


def word_bounds(transcripts: dict, clip: str, in_s: float, out_s: float) -> tuple[float, float]:
    """Room (seconds) before in_s and after out_s in the source before a neighbouring word."""
    words = [w for sg in transcripts.get(clip, {"segments": []})["segments"] for w in sg["words"]]
    before = [w["e"] for w in words if w["e"] <= in_s + 0.02]
    after = [w["s"] for w in words if w["s"] >= out_s - 0.02]
    head = (in_s - max(before) - 0.05) if before else in_s
    tail = (min(after) - out_s - 0.05) if after else 3.0
    return max(0.0, head), max(0.0, tail)


# --------------------------------------------------------------------- plan
def inventory_paths(film_root: Path | None) -> dict[str, str]:
    """Clip id -> source path from the film's ingest inventory (out/inventory.json), {} when there is none."""
    p = Path(film_root) / "out" / "inventory.json" if film_root else None
    if not p or not p.exists():
        return {}
    inv = json.loads(p.read_text(encoding="utf-8"))
    return {c["id"]: c["path"] for role in ("aroll", "broll") for c in inv.get(role, []) if c.get("id") and c.get("path")}


def pick_source_path(pick: dict, clip: str, film_cfg: dict, inv_paths: dict[str, str]) -> str:
    """The audio pick's source: the path the cut carries, else the inventory's, else `<aroll_dir>/<clip>.MOV`."""
    return pick.get("source_path") or inv_paths.get(clip) or str(Path(film_cfg["aroll_dir"]) / f"{clip}.MOV")


def plan(cut: dict, transcripts: dict, profile: Film | None = None, tier: int = 1,
         measure=None, mono_of=None) -> dict:
    """Frame-exact placement. `measure(path, in_s, out_s, mono) -> (LUFS, peak)`
    and `mono_of(clip, path) -> bool` are injectable so the arithmetic can be
    tested without ffmpeg; the defaults measure the source. `profile` (film.py)
    gives the A-roll folder, the drone sources and the placeholder cards."""
    profile = profile or load_film(None)
    inv_paths = inventory_paths(profile.root)
    film_cfg = profile.layin
    cards = set((film_cfg.get("placeholders") or {}).values())
    mono_cache: dict[str, bool] = {}

    def default_mono(clip: str, path: str) -> bool:
        if clip not in mono_cache:
            lr = (channel_rms(path) + [-99.0, -99.0])[:2]
            mono_cache[clip] = (lr[0] - lr[1]) > 8.0
        return mono_cache[clip]

    measure = measure or lufs_peak
    mono_of = mono_of or default_mono
    aud: list[dict] = []
    vid: list[dict] = []
    marks: list[tuple[int, str, str]] = []
    t = 0.0
    for bi, b in enumerate(cut["beats"], 1):
        picks = [(j, a) for j, a in enumerate(b["audio"], 1) if a.get("tier", 1) <= tier]
        if not picks:
            continue
        marks.append((f24(t), f"B{bi:02d} {b['title']}", b.get("notes") or ""))
        cursor = t
        for j, a in picks:
            clip = base_clip(a["clip"])
            path = pick_source_path(a, clip, film_cfg, inv_paths)
            mono = mono_of(clip, path)
            I, P = measure(path, a["in_s"], a["out_s"], mono)
            head_room, tail_room = word_bounds(transcripts, clip, a["in_s"], a["out_s"])
            # a line cut inside a phrase carries its own measured room: the stored transcript can miss the word beside
            # the cut (a one-word reply just before the hook's first line was not in it, 30 Sep 2026)
            head_room = min(head_room, a.get("max_head_s", head_room))
            tail_room = min(tail_room, a.get("max_tail_s", tail_room))
            aud.append(dict(pick=f"b{bi}p{j}", name=clip + ".MOV", clip=clip, sf=f24(a["in_s"]), ef=f24(a["out_s"]),
                            rec=f24(cursor), dur=f24(a["duration_s"]), gain=gain_for(I, P), mono=mono, lufs=I, peak=P,
                            gain_status=OK if I is not None else UNCHECKED, leveler=leveler_for(I),
                            max_head=int(head_room * FPS), max_tail=int(tail_room * FPS), gap_after=0,
                            hard_in=a.get("max_head_s") == 0))
            cursor += a["duration_s"]
        aud[-1]["gap_after"] = f24(b.get("gap_after_s", 0.0))
        span_end = t + sum(a["duration_s"] for _, a in picks)
        rows_here: list[dict] = []
        for v in b["video"]:
            if v["type"] == "bare":
                continue
            if "dest_in_s" not in v:
                raise ValueError(f"beat {bi}: video row without a destination interval; rebuild the cut with cutbuild")
            d = v["dest_out_s"] - v["dest_in_s"]
            if d < 1.0:
                continue
            rec = f24(v["dest_in_s"])
            durf = f24(v["dest_out_s"]) - rec
            if v["type"] == "placeholder":
                name, sf, fps = placeholder_file(v["text"], profile), 0, float(FPS)
                marks.append((rec + 1, "PLACEHOLDER", v["text"]))
            else:
                fps = FPS_DJI if is_drone(v["clip"], profile) else float(FPS)
                name = base_clip(v["clip"]) + ".MOV"
                sf = int(round(v["in_s"] * fps))
            ef = sf + int(round(durf * fps / FPS))
            rows_here.append(dict(pick=v.get("pick"), name=name, sf=sf, ef=ef, rec=rec, dur=durf, note=(v.get("note") or "")[:60], fps=fps))
        # cover that reaches the end of the beat also spans the room-tone gap, so the picture never drops to V1;
        # it runs to the frame the next beat starts on, not end + round(gap): rounded separately the two can differ by one
        gap_f = f24(span_end + b.get("gap_after_s", 0.0)) - (rows_here[-1]["rec"] + rows_here[-1]["dur"]) if rows_here else 0
        if rows_here and b.get("gap_after_s", 0.0) and rows_here[-1]["rec"] + rows_here[-1]["dur"] >= f24(span_end) - 1 \
                and rows_here[-1]["name"] not in cards:
            r = rows_here[-1]
            r["ef"] += int(round(gap_f * r["fps"] / FPS))
            r["dur"] += gap_f
        vid += rows_here
        t = span_end + b.get("gap_after_s", 0.0)
    # crossfades: alternate tracks so neighbours may overlap; extensions capped by neighbouring words
    residual: list[tuple[str, int, int]] = []
    for i, r in enumerate(aud):
        r["track"] = 1 + (i % 2)
        r["head"] = min(EXT, r["max_head"]) if i > 0 else 0
    for i, r in enumerate(aud):
        if i + 1 < len(aud):
            nxt = aud[i + 1]
            # the gap in frames as placed: start and length are each rounded from seconds, so two picks meant to butt
            # can land a frame apart (two of v8's joins each rendered one black frame, 29 Sep 2026)
            gap = nxt["rec"] - (r["rec"] + r["dur"])
            want = gap + 2 * EXT - nxt["head"]
            r["tail"] = min(want, r["max_tail"])
            short = want - r["tail"]
            if short > 0:  # outgoing take has a word coming: let the incoming head fill instead
                nxt["head"] = min(nxt["max_head"], nxt["head"] + short)
            overlap = r["tail"] + nxt["head"] - gap
            r["fade_out"] = max(2, overlap)
            nxt["fade_in"] = max(2, overlap)
            if nxt["hard_in"]:  # the in-point sits on a word: a fade there dips it, so the outgoing tail fades out under it
                nxt["fade_in"] = 2
            if overlap < 0:
                residual.append((r["pick"], r["rec"] + r["dur"] + r["tail"], -overlap))
        else:
            r["tail"] = min(2 * EXT, r["max_tail"])
            r["fade_out"] = 2 * EXT
    aud[0]["fade_in"] = 2 * EXT
    # a hole neither side can fill with sound still gets picture: a video-only slice of the outgoing take on V3,
    # unless cover already spans it (then the slice would overlap and Resolve would drop it)
    fills = []
    for pick, hole_rec, hole in residual:
        r = next(x for x in aud if x["pick"] == pick)
        covered = any(v["rec"] <= hole_rec and v["rec"] + v["dur"] >= hole_rec + hole for v in vid)
        if covered:
            continue
        fills.append(dict(pick=pick, name=r["name"], sf=r["ef"] + r["tail"], ef=r["ef"] + r["tail"] + hole, rec=hole_rec, dur=hole,
                          note="picture across a word-capped gap", fps=float(FPS)))
    vid += fills
    end_f = max([r["rec"] + r["dur"] + r["tail"] for r in aud] + [v["rec"] + v["dur"] for v in vid])
    # gain diagnostics: a pick the cap or the peak ceiling keeps under target is a level problem the gain cannot fix
    capped = [dict(pick=r["pick"], lufs=r["lufs"], needed=round(TARGET_LUFS - r["lufs"], 1), applied=r["gain"])
              for r in aud if r["gain_status"] == OK and TARGET_LUFS - r["lufs"] > GAIN_CAP + 0.05]
    peak_bound = [dict(pick=r["pick"], lufs=r["lufs"], peak=r["peak"], applied=r["gain"], short_by=round(TARGET_LUFS - r["lufs"] - r["gain"], 1))
                  for r in aud if r["gain_status"] == OK and r["peak"] is not None and r["gain"] < TARGET_LUFS - r["lufs"] - 0.05
                  and r["gain"] < GAIN_CAP - 0.05]
    return dict(audio=aud, video=vid, markers=marks, residual_gaps=residual, fills=len(fills), end_frame=end_f,
                unmeasured_gains=[r["pick"] for r in aud if r["gain_status"] != OK], capped_gains=capped, peak_bound_gains=peak_bound)


def grade_trim_for(trims: list[dict], name: str, src_in: float | None, src_out: float | None) -> dict | None:
    """The exposure trim for one timeline item: the same clip, and when the trim names a source span, an item whose
    source span overlaps it (a walk take can be bright in one stretch and right in the next)."""
    c = base_clip(Path(name).stem)
    for t in trims:
        if t["clip"] != c:
            continue
        if "in_s" in t and (src_in is None or src_out is None or src_out <= t["in_s"] or src_in >= t["out_s"]):
            continue
        return t
    return None


def item_source_span(pl: dict, name: str, rec: int, track: int) -> tuple[float | None, float | None]:
    """Media seconds a laid item shows, from the plan row it was laid from: cover (V3) by its record frame, the A-roll
    picture (V1/V2) by its pick's record frame less the head handle."""
    if track == 3:
        v = next((v for v in pl["video"] if v["name"] == name and abs(v["rec"] - rec) <= TOL), None)
        return (v["sf"] / v["fps"], v["ef"] / v["fps"]) if v else (None, None)
    r = next((r for r in pl["audio"] if r["name"] == name and abs((r["rec"] - r["head"]) - rec) <= TOL), None)
    return ((r["sf"] - r["head"]) / FPS, (r["ef"] + r["tail"]) / FPS) if r else (None, None)


# ------------------------------------------------------------------ Resolve
def _find(folder, parts):
    if not parts:
        return folder
    for s in folder.GetSubFolderList() or []:
        if s.GetName() == parts[0]:
            return _find(s, parts[1:])
    return None


def _timeline(project, name):
    for i in range(1, project.GetTimelineCount() + 1):
        tl = project.GetTimelineByIndex(i)
        if tl.GetName() == name:
            return tl
    return None


def readback(tl) -> dict:
    """Everything on the timeline, read, not assumed."""
    start = tl.GetStartFrame()
    out = dict(timeline=tl.GetName(), start=start, end=tl.GetEndFrame(), fps=tl.GetSetting("timelineFrameRate"), items=[])
    for typ in ("video", "audio"):
        for ti in range(1, tl.GetTrackCount(typ) + 1):
            for it in tl.GetItemListInTrack(typ, ti) or []:
                row = dict(type=typ, track=ti, name=it.GetName(), rec=it.GetStart() - start, dur=it.GetDuration(),
                           src_in=it.GetLeftOffset(), src_start=it.GetSourceStartFrame(), src_end=it.GetSourceEndFrame())
                if typ == "audio":
                    row["fades"] = it.GetFades()
                    props = it.GetProperty() or {}
                    row["gain"] = props.get("AudioVolume")
                    row["leveler"] = float(props.get("AudioDialogueLevelerOutputGain") or 0.0) if props.get("AudioDialogueLevelerEnabled") else 0.0
                    try:
                        row["mono_left"] = json.loads(it.GetSourceAudioChannelMapping())["track_mapping"].get("1", {}).get("channel_idx") == [1]
                    except (TypeError, ValueError, KeyError):
                        row["mono_left"] = None
                else:
                    try:
                        row["lut"] = it.GetNodeGraph().GetLUT(1)
                    except Exception:  # noqa: BLE001
                        row["lut"] = None
                out["items"].append(row)
    out["markers"] = tl.GetMarkers()
    return out


def conform(pl: dict, rb: dict, tol: int = TOL) -> dict:
    """Planned rows vs read-back items on the tracks they were meant for.
    Every planned row is matched by name, track and record frame within
    `tol`; then its duration and source in-point are compared. Unmatched
    rows and unexpected items are listed. Nothing is inferred from totals."""
    items = rb["items"]
    used = set()

    def take(typ, track, name, rec):
        best = None
        for k, it in enumerate(items):
            if k in used or it["type"] != typ or it["track"] != track or it["name"] != name:
                continue
            if abs(it["rec"] - rec) <= tol and (best is None or abs(it["rec"] - rec) < abs(items[best]["rec"] - rec)):
                best = k
        if best is not None:
            used.add(best)
        return None if best is None else items[best]

    diffs = []
    matched = 0
    for r in pl["audio"]:
        rec, sf = r["rec"] - r["head"], r["sf"] - r["head"]
        dur = r["dur"] + r["head"] + r["tail"]
        for typ in ("audio", "video"):
            it = take(typ, r["track"], r["name"], rec)
            if not it:
                diffs.append(dict(kind="missing", type=typ, track=r["track"], pick=r["pick"], name=r["name"], rec=rec))
                continue
            matched += 1
            if abs(it["dur"] - dur) > tol:
                diffs.append(dict(kind="duration", type=typ, pick=r["pick"], name=r["name"], planned=dur, got=it["dur"]))
            if abs(it["src_in"] - sf) > tol:
                diffs.append(dict(kind="source_in", type=typ, pick=r["pick"], name=r["name"], planned=sf, got=it["src_in"]))
            if typ == "audio":
                f = it.get("fades") or {}
                if abs((f.get("FadeIn") or 0) - r["fade_in"]) > 0.5 or abs((f.get("FadeOut") or 0) - r["fade_out"]) > 0.5:
                    diffs.append(dict(kind="fades", pick=r["pick"], planned=(r["fade_in"], r["fade_out"]), got=(f.get("FadeIn"), f.get("FadeOut"))))
                if r["gain"] is not None and (it.get("gain") is None or abs(it["gain"] - r["gain"]) > 0.2):
                    diffs.append(dict(kind="gain", pick=r["pick"], planned=r["gain"], got=it.get("gain")))
                if r["mono"] and it.get("mono_left") is not True:
                    diffs.append(dict(kind="mono", pick=r["pick"], got=it.get("mono_left")))
                want_lev = r.get("leveler", 0.0) or 0.0
                if "leveler" in it and abs((it.get("leveler") or 0.0) - want_lev) > 0.2:
                    diffs.append(dict(kind="leveler", pick=r["pick"], planned=want_lev, got=it.get("leveler")))
    for v in pl["video"]:
        it = take("video", 3, v["name"], v["rec"])
        if not it:
            diffs.append(dict(kind="missing", type="video", track=3, pick=v.get("pick"), name=v["name"], rec=v["rec"], dur=v["dur"], note=v["note"]))
            continue
        matched += 1
        if abs(it["dur"] - v["dur"]) > tol:
            diffs.append(dict(kind="duration", type="video", track=3, pick=v.get("pick"), name=v["name"], planned=v["dur"], got=it["dur"]))
        # Resolve reports the in-point in timeline frames; the plan holds source frames at the clip's own rate
        sf_tl = int(v["sf"] * FPS / v.get("fps", FPS))
        if abs(it["src_in"] - sf_tl) > tol:
            diffs.append(dict(kind="source_in", type="video", track=3, pick=v.get("pick"), name=v["name"], planned=sf_tl,
                              planned_source_frames=v["sf"], got=it["src_in"]))
    unexpected = [it for k, it in enumerate(items) if k not in used]
    planned = 2 * len(pl["audio"]) + len(pl["video"])
    return dict(planned=planned, matched=matched, diffs=diffs, unexpected=unexpected,
                status=OK if not diffs and not unexpected else "mismatch")


def apply_leveler(tl, pl: dict, resolve, report: dict) -> None:
    """Dialogue Leveler on the A-roll items whose plan row asks for it (Active Timeline Only: the
    caller makes the timeline current). Every write is read back; failures go to report['audio_fail']."""
    start = tl.GetStartFrame()
    rows = [r for r in pl["audio"] if (r.get("leveler") or 0.0) > 0]
    report.setdefault("levelers", 0)
    for r in rows:
        want = start + r["rec"] - r["head"]
        hits = [it for it in (tl.GetItemListInTrack("audio", r["track"]) or []) if it.GetName() == r["name"] and abs(it.GetStart() - want) <= TOL]
        if not hits:
            report["audio_fail"].append(("leveler unmatched", r["pick"], r["name"]))
            continue
        it = hits[0]
        it.SetProperties({"AudioDialogueLevelerEnabled": True, "AudioDialogueLevelerLiftSoftDialogue": True,
                          "AudioDialogueLevelerReduceLoudDialogue": False,
                          "AudioDialogueLevelerMode": resolve.DIALOGUE_LEVELER_MODE_OPTIMIZE_MODERATE_LEVELS,  # 26 Sep 2026: the lifting modes clip
                          "AudioDialogueLevelerOutputGain": r["leveler"]})
        got = it.GetProperties()
        if got.get("AudioDialogueLevelerEnabled") and abs(float(got.get("AudioDialogueLevelerOutputGain") or 0) - r["leveler"]) < 0.2:
            report["levelers"] += 1
        else:
            report["audio_fail"].append(("leveler", r["pick"], r["leveler"], got.get("AudioDialogueLevelerEnabled"), got.get("AudioDialogueLevelerOutputGain")))


def grade_for(name: str, grades: list[dict]) -> dict | None:
    """The profile's grade for a timeline item: the first `match` the item name starts with."""
    return next((g for g in grades if name.startswith(g["match"])), None)


def lay(pl: dict, name: str, profile: Film | None = None, replace: bool = False, render_name: str | None = None) -> dict:
    """Create the timeline and place everything; every write is checked by
    reading it back. Returns the write report; the caller runs `readback` +
    `conform` for the placement proof."""
    from .resolve import get_project, get_resolve
    film_cfg = (profile or load_film(None)).layin
    if not film_cfg.get("project"):
        raise RuntimeError("film.yaml names no layin.project: the lay-in needs the Resolve project")
    grades = film_cfg.get("grades") or []
    resolve = get_resolve()
    project = get_project(film_cfg["project"])
    if resolve.GetCurrentPage() is None:
        raise RuntimeError("Resolve is on no page: a dialog or the Project Manager is open. Ask the editor to close it.")
    mp = project.GetMediaPool()
    film_bin = _find(mp.GetRootFolder(), film_cfg["bin"])
    drafts = _find(film_bin, [film_cfg["drafts_bin"]])
    pool: dict = {}
    for bname in film_cfg["clip_bins"]:
        fo = _find(film_bin, [bname])
        for c in (fo.GetClipList() if fo else []) or []:
            pool.setdefault(c.GetName(), c)
    missing = sorted(({r["name"] for r in pl["audio"]} | {v["name"] for v in pl["video"]}) - set(pool))
    if missing:
        raise RuntimeError("clips not in the media pool: " + ", ".join(missing))
    old = _timeline(project, name)
    if old:
        if not replace:
            raise RuntimeError(f"{name} already exists (pass --replace to rebuild it)")
        # DeleteTimelines returns False on the current timeline: make another one current first
        cur = project.GetCurrentTimeline()
        if cur is not None and cur.GetName() == name:
            other = next((project.GetTimelineByIndex(i) for i in range(1, project.GetTimelineCount() + 1)
                          if project.GetTimelineByIndex(i).GetName() != name), None)
            if other is not None:
                project.SetCurrentTimeline(other)
        if not mp.DeleteTimelines([old]) or _timeline(project, name) is not None:
            raise RuntimeError("could not delete the old " + name)
    mp.SetCurrentFolder(drafts)
    tl = mp.CreateEmptyTimeline(name)
    if tl is None or tl.GetName() != name:
        raise RuntimeError("CreateEmptyTimeline failed (a dialog open in Resolve?)")
    project.SetCurrentTimeline(tl)
    start = tl.GetStartFrame()
    while tl.GetTrackCount("video") < 3:
        tl.AddTrack("video")
    while tl.GetTrackCount("audio") < 2:
        tl.AddTrack("audio", "stereo")
    for track in (1, 2):
        mp.AppendToTimeline([{"mediaPoolItem": pool[r["name"]], "startFrame": r["sf"] - r["head"], "endFrame": r["ef"] + r["tail"],
                              "recordFrame": start + r["rec"] - r["head"], "trackIndex": track} for r in pl["audio"] if r["track"] == track])
    mp.AppendToTimeline([{"mediaPoolItem": pool[v["name"]], "startFrame": v["sf"], "endFrame": v["ef"], "recordFrame": start + v["rec"],
                          "trackIndex": 3, "mediaType": 1} for v in pl["video"]])
    a_items = (tl.GetItemListInTrack("audio", 1) or []) + (tl.GetItemListInTrack("audio", 2) or [])
    report = dict(timeline=name, aroll_items=len(a_items), aroll_planned=len(pl["audio"]), fades=0, mono=0, gains=0, audio_fail=[],
                  grade=dict({g.get("name", g["match"]): 0 for g in grades}, skip=0, trim=0, fail=[]), markers=0)

    def match(it):
        rec = it.GetStart() - start
        c = [r for r in pl["audio"] if r["name"] == it.GetName() and abs((r["rec"] - r["head"]) - rec) <= TOL]
        return c[0] if c else None

    for it in a_items:
        r = match(it)
        if not r:
            report["audio_fail"].append(("unmatched", it.GetName(), it.GetStart() - start))
            continue
        if it.SetFades({"FadeIn": r["fade_in"], "FadeOut": r["fade_out"]}):
            report["fades"] += 1
        if r["mono"]:
            m = json.loads(it.GetSourceAudioChannelMapping())
            m["track_mapping"] = {"1": {"channel_idx": [1], "mute": False, "type": "mono"}}
            it.SetSourceAudioChannelMapping(json.dumps(m))
            back = json.loads(it.GetSourceAudioChannelMapping())["track_mapping"].get("1", {})
            if back.get("type") == "mono" and back.get("channel_idx") == [1]:
                report["mono"] += 1
            else:
                report["audio_fail"].append(("mono", it.GetName(), it.GetStart() - start))
        if r["gain"] is not None:
            it.SetProperties({"AudioVolume": r["gain"]})
            got = it.GetProperties()["AudioVolume"]
            if abs(got - r["gain"]) < 0.2:
                report["gains"] += 1
            else:
                report["audio_fail"].append(("gain", it.GetName(), r["gain"], got))
    apply_leveler(tl, pl, resolve, report)
    for ti in (1, 2, 3):
        for it in tl.GetItemListInTrack("video", ti) or []:
            n = it.GetName()
            gr = grade_for(n, grades)
            if gr is None:
                ok = True; key = "skip"
            elif gr.get("lut"):
                # SetLUT's return is not proof: read the node back and look for the cube's own name
                g = it.GetNodeGraph(); ok = g.SetLUT(1, gr["lut"]) and Path(gr["lut"]).stem in (g.GetLUT(1) or ""); key = gr.get("name", gr["match"])
            else:
                ok = it.SetCDL(gr["cdl"]); key = gr.get("name", gr["match"])
            if ok:
                report["grade"][key] += 1
            else:
                report["grade"]["fail"].append((n, ti, it.GetStart() - start))
            # an exposure trim is a CDL on the LUT's node; there is no CDL read-back, so the render's exposure is the check.
            # The source span comes from the plan row the item was laid from: GetSourceStartTime() answers in source
            # timecode seconds (51302.7 for a clip laid from media second 0.25, 30 Sep 2026), not media time.
            t = grade_trim_for(pl.get("grade_trims", []), n, *item_source_span(pl, n, it.GetStart() - start, ti)) if gr and gr.get("lut") else None
            if t:
                if it.SetCDL(dict(t["cdl"], NodeIndex=1)):
                    report["grade"]["trim"] += 1
                else:
                    report["grade"]["fail"].append(("trim", n, ti, it.GetStart() - start))
    for fid, mname, note in pl["markers"]:
        if tl.AddMarker(fid, "Blue" if mname.startswith("B") else "Yellow", mname, note, 1, ""):
            report["markers"] += 1
    resolve.GetProjectManager().SaveProject()
    # the render is started by main() AFTER read-back and conform: while Resolve renders, item property reads
    # return None and a read-back taken then reports every gain and leveler as missing (cut-v7, 26 Sep 2026)
    report["timeline_obj"] = tl
    report["project_obj"] = project
    return report


def reopen(project, tl) -> None:
    """Make another timeline current and come back, so Resolve rebuilds this timeline's playback graph. Without it
    a source channel map written through the API is stored (it reads back) but the next render ignores it."""
    other = next((project.GetTimelineByIndex(i + 1) for i in range(project.GetTimelineCount())
                  if project.GetTimelineByIndex(i + 1).GetName() != tl.GetName()), None)
    if other is not None:
        project.SetCurrentTimeline(other)
    project.SetCurrentTimeline(tl)
    if project.GetCurrentTimeline().GetName() != tl.GetName():
        raise RuntimeError("could not make the timeline current (a dialog may be open)")


def render_timeline(project, tl, render_name: str, profile: Film | None = None, wait: bool = True, pl: dict | None = None) -> dict:
    """Render an existing timeline with the film's preset and wait for the job; the file is verified afterwards,
    including the channel balance of every mono pick when the plan is given."""
    import time
    film_cfg = (profile or load_film(None)).layin
    reopen(project, tl)
    project.LoadRenderPreset(film_cfg["render_preset"])
    project.SetCurrentRenderFormatAndCodec("mp4", "H264")
    out_dir = Path(film_cfg["render_dir"]).resolve()
    project.SetRenderSettings({"SelectAllFrames": True, "TargetDir": str(out_dir), "CustomName": render_name,
                               "ExportVideo": True, "ExportAudio": True, "FormatWidth": 1920, "FormatHeight": 1080,
                               "FrameRate": 24.0, "ReplaceExistingFilesInPlace": True})
    job = project.AddRenderJob()
    started = bool(job) and project.StartRendering([job])
    rep = dict(job=job, started=started, file=str(out_dir / f"{render_name}.mp4"))
    if not started:
        return dict(rep, status="unchecked: render did not start")
    t0 = time.time()
    while wait and project.IsRenderingInProgress():
        time.sleep(10)
    st = project.GetRenderJobStatus(job) if wait else None
    rep.update(job_status=st, elapsed_s=round(time.time() - t0))
    f = Path(rep["file"])
    rep["status"] = OK if (wait and f.exists() and (st or {}).get("JobStatus") == "Complete") else ("unchecked: " + str((st or {}).get("JobStatus")))
    if rep["status"] == OK and pl is not None:
        rep["channels"] = verify_channels(pl, str(f))
        if rep["channels"]["status"] != OK:
            rep["status"] = f"broken: {len(rep['channels']['one_channel'])} mono picks rendered on one channel" if rep["channels"]["one_channel"] \
                else "unchecked: channel balance unmeasured"
    from .resolve import get_resolve
    get_resolve().GetProjectManager().SaveProject()
    return rep


# --------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--film", required=True)
    ap.add_argument("--cut", required=True)
    ap.add_argument("--name", required=True, help="timeline name, e.g. '<film> cut-v6 (AI)'")
    ap.add_argument("--replace", action="store_true")
    ap.add_argument("--render", default=None, help="render file stem, e.g. <film>-cut-v6")
    ap.add_argument("--plan-only", action="store_true", help="write the plan, touch nothing in Resolve")
    ap.add_argument("--conform-only", action="store_true", help="read back the named timeline and conform it to the plan")
    ap.add_argument("--render-only", action="store_true", help="render the named timeline as --render and wait; no lay, no conform")
    ap.add_argument("--leveler-only", action="store_true", help="apply the plan's Dialogue Leveler gains to the named timeline, then read back and conform")
    ap.add_argument("--out", default=None, help="output stem (default <film>/editorial/lay/<cut stem>)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    film = Path(args.film)
    profile = load_film(film)
    cut_path = film / args.cut
    cut = json.load(open(cut_path, encoding="utf-8"))
    out = Path(args.out) if args.out else film / "editorial" / "lay" / cut_path.stem
    out.parent.mkdir(parents=True, exist_ok=True)
    if args.render_only:  # an existing timeline, rendered and waited for; the plan on disk is untouched
        if not args.render:
            raise SystemExit("--render-only needs --render <stem>")
        from .resolve import get_project
        project = get_project(profile.layin["project"])
        tl = _timeline(project, args.name)
        if tl is None:
            raise SystemExit(f"timeline not found: {args.name}")
        plan_path = Path(str(out) + ".plan.json")
        pl_disk = json.loads(plan_path.read_text(encoding="utf-8")) if plan_path.exists() else None
        rep = render_timeline(project, tl, args.render, profile, pl=pl_disk)
        print(json.dumps(rep, default=str), file=sys.stderr)
        return 0 if rep["status"] == OK else 1
    refuse_overwrite([Path(str(out) + ".plan.json")], args.overwrite or args.conform_only or args.leveler_only)
    from .cutbuild import check_cut
    problems = check_cut(cut)
    if problems:
        raise SystemExit("cut fails placement invariants:\n  " + "\n  ".join(problems[:20]))
    transcripts = json.load(open(film / "out/transcripts.json", encoding="utf-8"))
    pl = plan(cut, transcripts, profile)
    trims_path = Path(profile.layin["grade_trims"])
    pl["grade_trims"] = json.loads(trims_path.read_text(encoding="utf-8"))["trims"] if trims_path.exists() else []
    Path(str(out) + ".plan.json").write_text(json.dumps(pl, indent=1), encoding="utf-8")
    print(f"plan: {len(pl['audio'])} picks, {len(pl['video'])} cover rows ({pl['fills']} gap fills), {len(pl['markers'])} markers, "
          f"{pl['end_frame'] / FPS:.1f}s; residual word-capped gaps {len(pl['residual_gaps'])}; unmeasured gains {pl['unmeasured_gains']}; "
          f"gains at the +{GAIN_CAP:.0f} dB cap {[c['pick'] for c in pl['capped_gains']]}; held under target by the peak ceiling "
          f"{[c['pick'] for c in pl['peak_bound_gains']]}; dialogue leveler on {sum(1 for r in pl['audio'] if r.get('leveler'))} picks",
          file=sys.stderr)
    if args.plan_only:
        return 0
    from .resolve import get_project
    if args.conform_only or args.leveler_only:
        project = get_project(profile.layin["project"])
        tl = _timeline(project, args.name)
        if tl is None:
            raise SystemExit(f"timeline not found: {args.name}")
        report = {"timeline": args.name}
        if args.leveler_only:
            from .resolve import get_resolve
            project.SetCurrentTimeline(tl)
            if project.GetCurrentTimeline().GetName() != tl.GetName():
                raise SystemExit("could not make the timeline current (a dialog may be open)")
            report["audio_fail"] = []
            apply_leveler(tl, pl, get_resolve(), report)
            get_resolve().GetProjectManager().SaveProject()
    else:
        report = lay(pl, args.name, profile, replace=args.replace, render_name=args.render)
        tl = report.pop("timeline_obj")
        project = report.pop("project_obj")
    rb = readback(tl)
    cf = conform(pl, rb)
    Path(str(out) + ".readback.json").write_text(json.dumps(rb, indent=1), encoding="utf-8")
    Path(str(out) + ".conform.json").write_text(json.dumps(cf, indent=1), encoding="utf-8")
    # the recorded decisions, checked on what Resolve holds, independent of any score
    from . import intent as intent_mod
    intent_path = film / "editorial/intent.json"
    intent_findings = intent_mod.check_readback(rb, pl, cut, intent_mod.load(film), profile=profile) if intent_path.exists() else None
    intent_broken = [f for f in (intent_findings or []) if f["status"] == intent_mod.BROKEN]
    write_manifest(out, tool="layin", tool_sha=source_hash(__file__), cut=dict(path=args.cut, sha=fingerprint(cut_path)),
                   timeline=args.name, write_report=report, conform=dict(status=cf["status"], planned=cf["planned"], matched=cf["matched"],
                                                                           diffs=len(cf["diffs"]), unexpected=len(cf["unexpected"])),
                   intent=None if intent_findings is None else dict(sha=fingerprint(intent_path), summary=intent_mod.summary(intent_findings),
                                                                     findings=intent_findings),
                   gains=dict(capped=pl["capped_gains"], peak_bound=pl["peak_bound_gains"], unmeasured=pl["unmeasured_gains"]),
                   grade_trims=dict(path=str(trims_path), sha=fingerprint(trims_path), n=len(pl["grade_trims"])) if trims_path.exists() else None)
    if intent_findings is not None:
        print("intent on read-back: " + intent_mod.summary(intent_findings), file=sys.stderr)
        for f in intent_findings:
            if f["status"] != intent_mod.HONOURED:
                print(f"   {f['status']:12} {f['rule']}: {f['evidence']}", file=sys.stderr)
    print(json.dumps({k: v for k, v in report.items()}, default=str), file=sys.stderr)
    print(f"conform: {cf['status']}; {cf['matched']}/{cf['planned']} planned rows matched, {len(cf['diffs'])} diffs, "
          f"{len(cf['unexpected'])} unexpected items", file=sys.stderr)
    for d in cf["diffs"][:12]:
        print("  ", d, file=sys.stderr)
    if cf["status"] != OK:
        return 1
    if intent_broken:
        return 2
    if args.render and not (args.conform_only or args.leveler_only):
        rep = render_timeline(project, tl, args.render, profile, pl=pl)   # after the conform, waited for, channels verified
        print(json.dumps(rep, default=str), file=sys.stderr)
        return 0 if rep["status"] == OK else 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
