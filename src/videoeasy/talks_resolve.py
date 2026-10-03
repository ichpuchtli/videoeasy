"""A selects timeline for one analysed talk in DaVinci Resolve: plan, write, read back, verify.

  plan(analysis, fps)     no Resolve: every ranked candidate (hooks, then bites, then
                          closers, each in rank order) as audio items from the ORIGINAL
                          float chunks, ~1 s apart on A1, with one marker per candidate
                          (colour by kind; name kind, rank and score; note = the words and
                          the cover brief). A candidate across a chunk join is two items
                          butted together.
  apply(...)              dry run unless asked: import the chunks into a bin if absent
                          (verified by re-reading the bin), create the timeline, place
                          every item with AppendToTimeline + recordFrame (the only
                          placement that does not ripple), add the markers, read back.
  verify(plan, readback)  every planned item matched by name, track and record frame
                          within two frames, duration compared; every marker found within
                          two frames by name; unexpected items listed.

NOT CHECKED LIVE: the frame units AppendToTimeline uses for an audio-only WAV. For video
clips startFrame/endFrame are the source's own frames (docs/resolve-scripting.md); this
plan assumes a WAV is addressed in frames of the timeline's rate. The read-back verify
reports any disagreement (a wrong unit shows as durations off by the rate ratio), so a
first real run is its own test: run it on a scratch timeline and read the verify table.

The Resolve guards apply: a modal dialog makes every write return None silently
(GetCurrentPage() is None then), the open project must be the one named, and nothing
is assumed placed until it is read back.
"""
from __future__ import annotations

import json
from pathlib import Path

from .evalrun import OK, refuse_overwrite, source_hash, write_manifest

MARKER_COLOR = {"hook": "Red", "bite": "Blue", "closer": "Green"}   # marker colours include Red; clip colours do not
TOL = 2
GAP_S = 1.0
NOTE_MAX = 1000
KINDS = ("hook", "bite", "closer")


def plan(analysis: dict, fps: float = 24.0, gap_s: float = GAP_S, top: int | None = None) -> dict:
    by_id = {c["id"]: c for c in analysis["candidates"] if c.get("id")}
    rows: dict[str, list[dict]] = {}
    for r in analysis["selects"]:
        rows.setdefault(r["id"], []).append(r)
    order = [cid for kind in KINDS for cid in analysis["ranked"].get(kind, []) if cid in rows]
    if top:
        order = order[:top]
    items, markers, cursor, gap = [], [], 0, int(round(gap_s * fps))
    for cid in order:
        c = by_id[cid]
        first = cursor
        for r in rows[cid]:
            sf, ef = int(round(r["source_in_s"] * fps)), int(round(r["source_out_s"] * fps))
            if ef <= sf:
                continue
            items.append(dict(cand=cid, name=r["source_file"], path=r["source_path"], sf=sf, ef=ef, rec=cursor, track=1))
            cursor += ef - sf
        markers.append(dict(cand=cid, frame=first, color=MARKER_COLOR[c["kind"]], name=f"{c['kind']} #{c['rank']} {c['score']}/6",
                            note=(f"{c['text']} | cover: {c['cover_brief']}")[:NOTE_MAX]))
        cursor += gap
    return dict(fps=fps, gap_frames=gap, items=items, markers=markers, end_frame=cursor, candidates=order)


def readback(tl) -> dict:
    start = tl.GetStartFrame()
    items = []
    for ti in range(1, (tl.GetTrackCount("audio") or 0) + 1):
        for it in tl.GetItemListInTrack("audio", ti) or []:
            items.append(dict(track=ti, name=it.GetName(), rec=it.GetStart() - start, dur=it.GetDuration()))
    marks = tl.GetMarkers() or {}
    return dict(start=start, fps=tl.GetSetting("timelineFrameRate"), items=items,
                markers=[dict(frame=int(f), **{k: v for k, v in (m or {}).items() if k in ("color", "name", "note", "duration")}) for f, m in marks.items()])


def verify(pl: dict, rb: dict, tol: int = TOL) -> dict:
    used, fails = set(), []
    ok_items = 0
    for p in pl["items"]:
        hit = next((i for i, r in enumerate(rb["items"]) if i not in used and r["track"] == p["track"] and r["name"] == p["name"]
                    and abs(r["rec"] - p["rec"]) <= tol), None)
        if hit is None:
            fails.append(dict(what="item missing", cand=p["cand"], name=p["name"], rec=p["rec"]))
            continue
        used.add(hit)
        want = p["ef"] - p["sf"]
        if abs(rb["items"][hit]["dur"] - want) > tol:
            fails.append(dict(what="duration", cand=p["cand"], name=p["name"], rec=p["rec"], planned=want, read=rb["items"][hit]["dur"]))
            continue
        ok_items += 1
    ok_marks = 0
    for m in pl["markers"]:
        if any(abs(r["frame"] - m["frame"]) <= tol and r.get("name") == m["name"] for r in rb["markers"]):
            ok_marks += 1
        else:
            fails.append(dict(what="marker missing", cand=m["cand"], frame=m["frame"], name=m["name"]))
    unexpected = [r for i, r in enumerate(rb["items"]) if i not in used]
    status = OK if not fails and not unexpected else "broken"
    return dict(status=status, items_planned=len(pl["items"]), items_verified=ok_items, markers_planned=len(pl["markers"]),
                markers_verified=ok_marks, failures=fails, unexpected=unexpected)


def _sub(folder, name):
    return next((s for s in folder.GetSubFolderList() or [] if s.GetName() == name), None)


def ensure_bin(mp, path: str):
    """The media-pool folder at 'A/B/C' under the root, created where missing; each creation re-read."""
    folder = mp.GetRootFolder()
    for part in [p for p in path.split("/") if p]:
        sub = _sub(folder, part)
        if sub is None:
            mp.AddSubFolder(folder, part)
            sub = _sub(folder, part)
            if sub is None:
                raise RuntimeError(f"could not create bin {part!r} (a dialog open in Resolve?)")
        folder = sub
    return folder


def _clips_by_path(folder) -> dict:
    return {c.GetClipProperty("File Path"): c for c in folder.GetClipList() or []}


def apply(analysis: dict, project_name: str, timeline_name: str, bin_path: str, resolve=None, top: int | None = None, gap_s: float = GAP_S) -> dict:
    if resolve is None:
        from .resolve import get_resolve
        resolve = get_resolve()
    project = resolve.GetProjectManager().GetCurrentProject()
    if project is None:
        raise RuntimeError("no project open in Resolve")
    if project.GetName() != project_name:
        raise RuntimeError(f"open project is {project.GetName()!r}, expected {project_name!r}: refusing to modify the wrong project")
    if resolve.GetCurrentPage() is None:
        raise RuntimeError("Resolve is on no page: a dialog or the Project Manager is open, and every write would silently fail. Close it first.")
    for i in range(1, (project.GetTimelineCount() or 0) + 1):
        if project.GetTimelineByIndex(i).GetName() == timeline_name:
            raise RuntimeError(f"a timeline named {timeline_name!r} already exists: choose another name")
    mp = project.GetMediaPool()
    folder = ensure_bin(mp, bin_path)
    draft = plan(analysis, 24.0, gap_s, top)
    paths = sorted({it["path"] for it in draft["items"]})
    have = _clips_by_path(folder)
    missing = [p for p in paths if p not in have]
    if missing:
        mp.SetCurrentFolder(folder)
        mp.ImportMedia(missing)
        have = _clips_by_path(folder)
        still = [p for p in paths if p not in have]
        if still:
            raise RuntimeError("chunks not in the bin after import: " + ", ".join(Path(p).name for p in still))
    mp.SetCurrentFolder(folder)
    tl = mp.CreateEmptyTimeline(timeline_name)
    if tl is None or tl.GetName() != timeline_name:
        raise RuntimeError("CreateEmptyTimeline failed (a dialog open in Resolve?)")
    project.SetCurrentTimeline(tl)
    fps = float(tl.GetSetting("timelineFrameRate") or 24.0)
    pl = plan(analysis, fps, gap_s, top)
    start = tl.GetStartFrame()
    placed = mp.AppendToTimeline([{"mediaPoolItem": have[it["path"]], "startFrame": it["sf"], "endFrame": it["ef"],
                                   "recordFrame": start + it["rec"], "trackIndex": it["track"], "mediaType": 2} for it in pl["items"]]) or []
    marker_fail = [m["name"] for m in pl["markers"] if not tl.AddMarker(m["frame"], m["color"], m["name"], m["note"], 1, "")]
    rb = readback(tl)
    v = verify(pl, rb)
    resolve.GetProjectManager().SaveProject()
    return dict(project=project_name, timeline=timeline_name, bin=bin_path, fps=fps, imported=[Path(p).name for p in missing],
                append_returned=len(placed), marker_add_failures=marker_fail, plan=pl, readback=rb, verify=v)


def main_from(args, out: Path) -> int:
    a = json.loads((out / "analysis.json").read_text()) if (out / "analysis.json").exists() else None
    if a is None:
        raise SystemExit("analysis.json not found: run the report stage first")
    talk = (a.get("meta") or {}).get("talk_id") or out.name
    bin_path = args.bin or f"Talks/{talk}"
    name = args.timeline or f"{talk} selects"
    if not args.apply:
        pl = plan(a, args.fps, top=args.top)
        refuse_overwrite([out / "resolve-plan.json"], args.overwrite)
        (out / "resolve-plan.json").write_text(json.dumps(dict(timeline=name, bin=bin_path, **pl), indent=1))
        secs = pl["end_frame"] / pl["fps"]
        print(f"resolve (dry run): {len(pl['candidates'])} candidates, {len(pl['items'])} items, {len(pl['markers'])} markers, "
              f"{secs:.0f} s at {pl['fps']:g} fps -> {out / 'resolve-plan.json'}. Nothing written; --apply --resolve-project NAME writes it.")
        return 0
    if not args.resolve_project:
        raise SystemExit("--apply needs --resolve-project NAME (the open project must be that one)")
    refuse_overwrite([out / "resolve-report.json"], args.overwrite)
    rep = apply(a, args.resolve_project, name, bin_path, top=args.top)
    (out / "resolve-report.json").write_text(json.dumps(rep, indent=1, default=str))
    write_manifest(out / "resolve-report", tool="videoeasy.talks_resolve", source=source_hash(__file__), analysis=a.get("key"),
                   project=args.resolve_project, timeline=name, verify=rep["verify"]["status"],
                   unchecked=["frame units for audio-only WAV items (first live run is the test)"])
    v = rep["verify"]
    print(f"resolve: {name}: items {v['items_verified']}/{v['items_planned']} verified, markers {v['markers_verified']}/{v['markers_planned']}, "
          f"{len(v['unexpected'])} unexpected -> {v['status']}")
    for f in v["failures"][:20]:
        print("  ", f)
    return 0 if v["status"] == OK else 2
