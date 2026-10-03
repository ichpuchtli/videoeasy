"""Event projects: many short deliverables drawn from one footage pool.

A film is one story cut from its own footage. An event shoot (a retreat, a
conference, a festival) has the opposite shape: one pool of footage from
several operators over several days, and dozens of short deliverables that
each draw on a slice of it (a speaker feature, before/after interviews with
one participant, a testimonial, a b-roll package). This module holds that
shape in a project directory:

  deliverables.yaml   hand-written: days, cameras, people, the register
                      vocabulary, the schedule of sessions with what each
                      operator captures there, and the deliverable types
  items.yaml          one entry per deliverable: stubbed by `plan`, then
                      kept current by hand or with `set` as the work moves
  items/<id>/         an item's own editorial folder (`item-dir`); it can be
                      run through the cut loop as a film directory
  <out>/sessions.json which session each clip was recorded in (`assign`)

Clips are assigned to sessions by time alone, read per camera. A camera's
`creation_time` may be true UTC, local wall-clock time labelled UTC, or a
time-of-day timecode may be the better clock, and any of them can be off by
an offset only a measurement finds (`clocks` lists each camera's first and
last clip per day against the schedule). A clip with no usable time, or whose
time fits no session or two, stays unassigned with the reason: nothing is
guessed. Two sessions may run at once only when each names the cameras
filming it and no camera is in both. What a clip's session (and its
operator's line in that session) says was being filmed becomes the clip's
`feeds_candidates` and `register_candidates`: the schedule's intent, not a
judgment of the picture.

  python -m videoeasy.deliverables validate --project data/projects/<p>
  python -m videoeasy.deliverables plan     --project data/projects/<p>
  python -m videoeasy.deliverables assign   --project data/projects/<p> --config config.<p>.yaml [--overwrite]
  python -m videoeasy.deliverables clocks   --project data/projects/<p> --config config.<p>.yaml
  python -m videoeasy.deliverables status   --project data/projects/<p> [--scope editor] [--md checklist.md]
  python -m videoeasy.deliverables set      --project data/projects/<p> <item-id> status=cut notes="..."
  python -m videoeasy.deliverables item-dir --project data/projects/<p> <item-id>
  python -m videoeasy.deliverables bins     --project data/projects/<p> --config config.<p>.yaml \\
        --resolve-project "<Resolve project>" --source-bin "Master/Footage" [--apply]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml

from .evalrun import fingerprint, refuse_overwrite, source_hash

STATUSES = ("planned", "footage", "selects", "cut", "review", "graded", "mixed", "delivered")
CLOCKS = ("creation_time_utc", "creation_time_local", "timecode")
DAY_END = 24 * 60   # minutes: a session with no end and no later neighbour runs to midnight

TOP_KEYS = {"project", "client", "timezone", "days", "cameras", "people", "registers", "sessions", "deliverable_types", "notes"}
CAMERA_KEYS = {"name", "prefix", "path_contains", "operator", "clock", "clock_offset_s", "stamp", "notes"}
PERSON_KEYS = {"id", "name", "role", "notes"}
SESSION_KEYS = {"id", "day", "start", "end", "title", "capture", "registers", "feeds", "people", "cameras", "notes"}
CAPTURE_KEYS = {"operator", "text", "feeds", "registers"}
TYPE_KEYS = {"id", "title", "count", "duration_s", "template", "scope", "includes", "notes"}
ITEM_KEYS = {"id", "type", "title", "people", "sessions", "shooter", "status", "notes"}
SET_KEYS = {"status", "notes", "shooter", "title", "people", "sessions"}


# ------------------------------------------------------------------ loading
def load(project: Path) -> dict:
    p = Path(project) / "deliverables.yaml"
    if not p.exists():
        raise SystemExit(f"no deliverables.yaml in {project} (start from deliverables.example.yaml)")
    try:
        doc = yaml.safe_load(p.read_text(encoding="utf-8"))
    except yaml.YAMLError as e:
        raise SystemExit(f"deliverables.yaml does not parse: {e}")
    if not isinstance(doc, dict):
        raise SystemExit("deliverables.yaml must be a mapping")
    return doc


def load_items(project: Path) -> tuple[list[dict], str | None]:
    """(items, file text) — ([], None) when there is no items.yaml. Refuses a file it cannot read as an items list,
    because every writer here appends to or splices that file and must not guess at its shape."""
    p = Path(project) / "items.yaml"
    if not p.exists():
        return [], None
    text = p.read_text(encoding="utf-8")
    try:
        raw = yaml.safe_load(text)
    except yaml.YAMLError as e:
        raise SystemExit(f"refusing: items.yaml does not parse ({e}); fix it by hand first")
    if raw is None:
        return [], text
    if not isinstance(raw, dict) or not isinstance(raw.get("items", []) or [], list):
        raise SystemExit("refusing: items.yaml must be a mapping with an `items:` list; fix it by hand first")
    items = raw.get("items") or []
    if not all(isinstance(i, dict) for i in items):
        raise SystemExit("refusing: every entry under items: must be a mapping")
    return items, text


def _list(v) -> list:
    return [] if v is None else list(v) if isinstance(v, (list, tuple)) else [v]


def minutes(v) -> int | None:
    """'07:30' -> 450. An unquoted 07:30 reaches here as the int 450 (YAML 1.1 reads it as base 60, which is
    exactly minutes), so ints are taken as minutes. None for anything else."""
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v if 0 <= v <= DAY_END else None
    m = re.fullmatch(r"\s*(\d{1,2}):(\d{2})\s*", str(v))
    if not m or int(m.group(2)) > 59:
        return None
    total = int(m.group(1)) * 60 + int(m.group(2))
    return total if total <= DAY_END else None


def hhmm(total: float) -> str:
    total = int(total)
    return f"{total // 60:02d}:{total % 60:02d}"


def day_date(v) -> dt.date | None:
    if isinstance(v, dt.datetime):
        return v.date()
    if isinstance(v, dt.date):
        return v
    try:
        return dt.date.fromisoformat(str(v))
    except ValueError:
        return None


def capture_entries(session: dict) -> list[dict]:
    """Capture lines as dicts; a plain string is a line no operator owns."""
    return [c if isinstance(c, dict) else {"text": str(c)} for c in _list(session.get("capture"))]


def disjoint_parallel(a: dict, b: dict) -> bool:
    """Two sessions may overlap only when both name their cameras and share none."""
    ca, cb = set(_list(a.get("cameras"))), set(_list(b.get("cameras")))
    return bool(ca) and bool(cb) and not (ca & cb)


def windows(doc: dict) -> dict[str, dict]:
    """Per session id: day key, date, start and end minutes. A missing end runs to the next later start that day
    among sessions it could not run beside (a parallel session on other cameras does not end it), else midnight."""
    days = {k: day_date(v) for k, v in (doc.get("days") or {}).items()}
    sessions = [s for s in _list(doc.get("sessions")) if isinstance(s, dict)]
    out = {}
    for s in sessions:
        start = minutes(s.get("start"))
        if start is None:
            continue
        end = minutes(s["end"]) if s.get("end") is not None else None
        if end is None:
            later = [minutes(o.get("start")) for o in sessions if o is not s and o.get("day") == s.get("day")
                     and minutes(o.get("start")) is not None and minutes(o.get("start")) > start and not disjoint_parallel(s, o)]
            end = min(later) if later else DAY_END
        out[str(s.get("id"))] = dict(day=s.get("day"), date=days.get(s.get("day")), start=start, end=end, session=s)
    return out


# ------------------------------------------------------------------ validate
def validate(doc: dict, items: list[dict] | None = None) -> tuple[list[str], list[str]]:
    """(errors, conflicts). Conflicts are overlapping sessions: `validate` fails on them, `assign` runs and leaves a clip
    that falls in two sessions unassigned as ambiguous."""
    err: list[str] = []
    conflicts: list[str] = []
    for k in sorted(set(doc) - TOP_KEYS):
        err.append(f"{k}: unknown top-level key (known: {', '.join(sorted(TOP_KEYS))})")
    if not doc.get("project"):
        err.append("project: required")
    try:
        ZoneInfo(str(doc.get("timezone")))
    except (ZoneInfoNotFoundError, ValueError):
        err.append(f"timezone: {doc.get('timezone')!r} is not a zoneinfo name (e.g. Australia/Sydney)")
    days = doc.get("days") or {}
    if not isinstance(days, dict) or not days:
        err.append("days: required mapping of day key -> date")
        days = {}
    for k, v in days.items():
        if day_date(v) is None:
            err.append(f"days.{k}: {v!r} is not a date (YYYY-MM-DD)")

    cameras = _list(doc.get("cameras"))
    cam_names, operators = set(), set()
    for i, c in enumerate(cameras):
        where = f"cameras[{i}]"
        if not isinstance(c, dict):
            err.append(f"{where}: must be a mapping")
            continue
        for k in sorted(set(c) - CAMERA_KEYS):
            err.append(f"{where}.{k}: unknown key")
        name = c.get("name")
        if not name:
            err.append(f"{where}.name: required (sessions refer to cameras by name)")
        elif name in cam_names:
            err.append(f"{where}.name: duplicate camera name {name!r}")
        cam_names.add(name)
        if c.get("operator"):
            operators.add(c["operator"])
        if not c.get("prefix") and not c.get("path_contains"):
            err.append(f"{where}: needs prefix, path_contains, or both")
        if c.get("clock", "creation_time_utc") not in CLOCKS:
            err.append(f"{where}.clock: {c.get('clock')!r} not in {', '.join(CLOCKS)}")
        if c.get("stamp", "start") not in ("start", "end"):
            err.append(f"{where}.stamp: must be start or end")
        if not isinstance(c.get("clock_offset_s", 0), (int, float)) or isinstance(c.get("clock_offset_s", 0), bool):
            err.append(f"{where}.clock_offset_s: must be a number of seconds")

    people = _list(doc.get("people"))
    person_ids = set()
    for i, p in enumerate(people):
        if not isinstance(p, dict) or not p.get("id"):
            err.append(f"people[{i}]: needs an id")
            continue
        for k in sorted(set(p) - PERSON_KEYS):
            err.append(f"people[{i}].{k}: unknown key")
        if p["id"] in person_ids:
            err.append(f"people[{i}].id: duplicate {p['id']!r}")
        person_ids.add(p["id"])
    registers = set(_list(doc.get("registers")))

    types = _list(doc.get("deliverable_types"))
    type_ids = set()
    for i, t in enumerate(types):
        where = f"deliverable_types[{i}]"
        if not isinstance(t, dict) or not t.get("id"):
            err.append(f"{where}: needs an id")
            continue
        for k in sorted(set(t) - TYPE_KEYS):
            err.append(f"{where}.{k}: unknown key")
        if t["id"] in type_ids:
            err.append(f"{where}.id: duplicate {t['id']!r}")
        type_ids.add(t["id"])
        if not isinstance(t.get("count"), int) or isinstance(t.get("count"), bool) or t["count"] < 0:
            err.append(f"{where}.count: required whole number")
        d = t.get("duration_s")
        if d is not None and not (isinstance(d, list) and len(d) == 2 and all(isinstance(x, (int, float)) for x in d) and d[0] <= d[1]):
            err.append(f"{where}.duration_s: must be [min, max] seconds")

    sessions = _list(doc.get("sessions"))
    session_ids = set()
    for i, s in enumerate(sessions):
        where = f"sessions[{i}]"
        if not isinstance(s, dict) or not s.get("id"):
            err.append(f"{where}: needs an id")
            continue
        where = f"sessions[{i}] ({s['id']})"
        for k in sorted(set(s) - SESSION_KEYS):
            err.append(f"{where}.{k}: unknown key")
        if s["id"] in session_ids:
            err.append(f"{where}.id: duplicate")
        session_ids.add(s["id"])
        if s.get("day") not in days:
            err.append(f"{where}.day: {s.get('day')!r} is not a key under days")
        start = minutes(s.get("start"))
        if start is None:
            err.append(f"{where}.start: {s.get('start')!r} is not HH:MM (quote times in YAML)")
        if s.get("end") is not None:
            end = minutes(s["end"])
            if end is None:
                err.append(f"{where}.end: {s['end']!r} is not HH:MM")
            elif start is not None and end <= start:
                err.append(f"{where}.end: {s['end']} is not after start {s.get('start')} (sessions cannot cross midnight)")
        for ref, known, what in (("cameras", cam_names, "camera name"), ("registers", registers, "register"),
                                 ("feeds", type_ids, "deliverable type"), ("people", person_ids, "person id")):
            for v in _list(s.get(ref)):
                if v not in known:
                    err.append(f"{where}.{ref}: {v!r} is not a known {what}")
        for j, c in enumerate(capture_entries(s)):
            cw = f"{where}.capture[{j}]"
            for k in sorted(set(c) - CAPTURE_KEYS):
                err.append(f"{cw}.{k}: unknown key")
            if "operator" in c and c["operator"] not in operators:
                # a typo here would silently fall back to the session's feeds for that operator's clips
                err.append(f"{cw}.operator: {c['operator']!r} is not the operator of any camera")
            for ref, known, what in (("feeds", type_ids, "deliverable type"), ("registers", registers, "register")):
                for v in _list(c.get(ref)):
                    if v not in known:
                        err.append(f"{cw}.{ref}: {v!r} is not a known {what}")

    win = windows(doc)
    by_day: dict = {}
    for sid, w in win.items():
        by_day.setdefault(w["day"], []).append((sid, w))
    for day, ws in by_day.items():
        ws.sort(key=lambda x: (x[1]["start"], x[0]))
        for a in range(len(ws)):
            for b in range(a + 1, len(ws)):
                (ia, wa), (ib, wb) = ws[a], ws[b]
                if wa["start"] < wb["end"] and wb["start"] < wa["end"] and not disjoint_parallel(wa["session"], wb["session"]):
                    conflicts.append(f"sessions {ia} ({hhmm(wa['start'])}-{hhmm(wa['end'])}) and {ib} ({hhmm(wb['start'])}-"
                                     f"{hhmm(wb['end'])}) overlap on {day}: overlap is allowed only between sessions that "
                                     f"both list cameras and share none")

    if items is not None:
        seen = set()
        per_type: dict = {}
        for i, it in enumerate(items):
            where = f"items[{i}] ({it.get('id')})"
            for k in sorted(set(it) - ITEM_KEYS):
                err.append(f"{where}.{k}: unknown key")
            if not it.get("id"):
                err.append(f"items[{i}]: needs an id")
            elif it["id"] in seen:
                err.append(f"{where}.id: duplicate")
            seen.add(it.get("id"))
            if it.get("type") not in type_ids:
                err.append(f"{where}.type: {it.get('type')!r} is not a deliverable type")
            per_type[it.get("type")] = per_type.get(it.get("type"), 0) + 1
            if it.get("status", "planned") not in STATUSES:
                err.append(f"{where}.status: {it.get('status')!r} not in {', '.join(STATUSES)}")
            for ref, known, what in (("people", person_ids, "person id"), ("sessions", session_ids, "session id")):
                for v in _list(it.get(ref)):
                    if v not in known:
                        err.append(f"{where}.{ref}: {v!r} is not a known {what}")
        for t in types:
            if isinstance(t, dict) and isinstance(t.get("count"), int) and per_type.get(t.get("id"), 0) > t["count"]:
                err.append(f"items: {per_type[t['id']]} items of type {t['id']!r}, which counts {t['count']}")
    return err, conflicts


def require_valid(doc: dict, items: list[dict] | None = None, allow_conflicts: bool = False) -> list[str]:
    errors, conflicts = validate(doc, items)
    if errors or (conflicts and not allow_conflicts):
        raise SystemExit("deliverables.yaml / items.yaml do not validate:\n  " + "\n  ".join(errors + conflicts))
    return conflicts


# ------------------------------------------------------------------ clocks
def camera_for(clip: dict, cameras: list[dict]) -> dict | None:
    """First camera whose prefix (filename stem startswith, case-insensitive) and path_contains (case-insensitive
    substring of the source path) both hold, of those it defines. Two operators on the same brand both write DSCF*:
    only a card or folder name in the path tells them apart."""
    cid, path = str(clip.get("id", "")).upper(), str(clip.get("path", "")).lower()
    for c in cameras:
        pre, sub = c.get("prefix"), c.get("path_contains")
        if not pre and not sub:
            continue
        if (not pre or cid.startswith(str(pre).upper())) and (not sub or str(sub).lower() in path):
            return c
    return None


def _parse_iso(s: str) -> dt.datetime | None:
    try:
        return dt.datetime.fromisoformat(str(s).strip().replace("Z", "+00:00"))
    except ValueError:
        return None


def _tc_seconds(tc: str, fps: float | None) -> float | None:
    m = re.fullmatch(r"(\d{1,2}):(\d{2}):(\d{2})[:;.](\d{1,3})", str(tc or "").strip())
    if not m:
        return None
    h, mi, s, f = map(int, m.groups())
    return h * 3600 + mi * 60 + s + (f / fps if fps else 0.0)


def true_start(clip: dict, camera: dict, tz: ZoneInfo) -> tuple[dt.datetime | None, str]:
    """(local start, how it was read) or (None, why not). The offset is added last; a camera that stamps the end of a
    recording (stamp: end) has the clip's duration taken off."""
    rule = camera.get("clock", "creation_time_utc")
    offset = float(camera.get("clock_offset_s", 0) or 0)
    ct = clip.get("creation_time")
    if rule != "timecode" and not ct:
        return None, "no time: no creation_time in the inventory (re-run `videoeasy probe` if it predates that field)"
    if rule == "creation_time_utc":
        t = _parse_iso(ct)
        if t is None:
            return None, f"no time: creation_time {ct!r} does not parse"
        t = (t if t.tzinfo else t.replace(tzinfo=dt.timezone.utc)).astimezone(tz)
    elif rule == "creation_time_local":
        t = _parse_iso(ct)
        if t is None:
            return None, f"no time: creation_time {ct!r} does not parse"
        t = t.replace(tzinfo=None).replace(tzinfo=tz)   # the camera's wall clock, whatever zone label it wrote
    else:
        secs = _tc_seconds(clip.get("start_timecode"), clip.get("fps"))
        if secs is None:
            return None, f"no time: timecode {clip.get('start_timecode')!r} missing or unreadable"
        ref = _parse_iso(ct) if ct else None
        if ref is None:
            return None, "no time: a timecode has no date and there is no creation_time to take one from"
        ref = (ref if ref.tzinfo else ref.replace(tzinfo=dt.timezone.utc)).astimezone(tz)
        # the date is whichever puts the time-of-day timecode nearest the creation time, so a camera that labels
        # local time as UTC still lands on the right day unless its clock is half a day out
        cands = [dt.datetime.combine(ref.date() + dt.timedelta(days=d), dt.time(), tzinfo=tz) + dt.timedelta(seconds=secs)
                 for d in (-1, 0, 1)]
        t = min(cands, key=lambda c: abs((c - ref).total_seconds()))
    if camera.get("stamp", "start") == "end":
        t = t - dt.timedelta(seconds=float(clip.get("duration_s") or 0))
    return t + dt.timedelta(seconds=offset), f"{rule}{offset:+g}s" + (" stamp=end" if camera.get("stamp") == "end" else "")


def inventory_clips(inventory: dict) -> list[dict]:
    return [c for role in ("aroll", "broll") for c in inventory.get(role, [])]


def _cands(session: dict, operator: str | None, key: str) -> list:
    own = [v for c in capture_entries(session) if operator and c.get("operator") == operator for v in _list(c.get(key))]
    return list(dict.fromkeys(own)) if own else list(dict.fromkeys(_list(session.get(key))))


def assign_clips(doc: dict, inventory: dict) -> dict:
    """Per clip id: its session or None with the reason. Pure: no files."""
    tz = ZoneInfo(str(doc["timezone"]))
    win = windows(doc)
    by_date: dict = {}
    for sid, w in win.items():
        by_date.setdefault(w["date"], []).append((sid, w))
    cameras = _list(doc.get("cameras"))
    out = {}
    for clip in inventory_clips(inventory):
        cam = camera_for(clip, cameras)
        rec = dict(session=None, day=None, camera=cam.get("name") if cam else None, role=clip.get("role"),
                   start_local=None, end_local=None, time_source=None, spans_boundary=False, ends_in=None,
                   feeds_candidates=[], register_candidates=[], reason=None)
        out[clip["id"]] = rec
        if cam is None:
            rec["reason"] = "no camera matches (prefix / path_contains)"
            continue
        start, how = true_start(clip, cam, tz)
        if start is None:
            rec["reason"] = how
            continue
        end = start + dt.timedelta(seconds=float(clip.get("duration_s") or 0))
        rec.update(start_local=start.isoformat(timespec="seconds"), end_local=end.isoformat(timespec="seconds"), time_source=how)
        day_ws = by_date.get(start.date())
        if not day_ws:
            rec["reason"] = f"day not in days ({start.date().isoformat()})"
            continue
        m = start.hour * 60 + start.minute + start.second / 60
        fits = lambda w: not _list(w["session"].get("cameras")) or cam.get("name") in _list(w["session"].get("cameras"))  # noqa: E731
        hits = [(sid, w) for sid, w in day_ws if w["start"] <= m < w["end"] and fits(w)]
        if not hits:
            rec["reason"] = f"outside every session for {cam.get('name')} ({start.strftime('%H:%M')} on {day_ws[0][1]['day']})"
            continue
        if len(hits) > 1:
            rec["reason"] = "ambiguous: " + ", ".join(sorted(sid for sid, _ in hits))
            continue
        sid, w = hits[0]
        rec.update(session=sid, day=w["day"])
        end_m = m + (end - start).total_seconds() / 60
        if end_m > w["end"]:
            rec["spans_boundary"] = True
            nxt = [s for s, ow in day_ws if s != sid and ow["start"] <= min(end_m, DAY_END - 1e-9) < ow["end"] and fits(ow)]
            rec["ends_in"] = nxt[0] if len(nxt) == 1 else None
        op = cam.get("operator")
        rec["feeds_candidates"] = _cands(w["session"], op, "feeds")
        rec["register_candidates"] = _cands(w["session"], op, "registers") if clip.get("role") == "broll" else []
    return out


def clock_table(doc: dict, inventory: dict) -> list[dict]:
    """Per camera and local date: clip count, first and last start, and how many fall before the day's first session
    or after its last. A camera whose clips all sit an hour early against the schedule has an offset, not a habit."""
    tz = ZoneInfo(str(doc["timezone"]))
    win = windows(doc)
    span: dict = {}
    for w in win.values():
        a, b = span.get(w["date"], (DAY_END, 0))
        span[w["date"]] = (min(a, w["start"]), max(b, w["end"]))
    rows: dict = {}
    for clip in inventory_clips(inventory):
        cam = camera_for(clip, _list(doc.get("cameras")))
        name = cam.get("name") if cam else "(no camera)"
        start = true_start(clip, cam, tz)[0] if cam else None
        key = (name, start.date() if start else None)
        r = rows.setdefault(key, dict(camera=name, date=key[1].isoformat() if key[1] else ("(no time)" if cam else "(not read)"), clips=0, first=None,
                                      last=None, before_first_session=0, after_last_session=0, day=None))
        r["clips"] += 1
        if start is None:
            continue
        r["first"] = min(filter(None, [r["first"], start.strftime("%H:%M:%S")]))
        r["last"] = max(filter(None, [r["last"], start.strftime("%H:%M:%S")]))
        r["day"] = next((k for k, v in (doc.get("days") or {}).items() if day_date(v) == start.date()), None)
        if start.date() in span:
            m = start.hour * 60 + start.minute
            r["before_first_session"] += m < span[start.date()][0]
            r["after_last_session"] += m >= span[start.date()][1]
    return sorted(rows.values(), key=lambda r: (r["camera"], r["date"]))


# ------------------------------------------------------------------ items
def feeders(doc: dict) -> dict[str, list[str]]:
    """Type id -> sessions that feed it, through the session's feeds or any of its capture lines."""
    out: dict = {}
    for s in _list(doc.get("sessions")):
        fed = set(_list(s.get("feeds"))) | {f for c in capture_entries(s) for f in _list(c.get("feeds"))}
        for f in fed:
            out.setdefault(f, []).append(s["id"])
    return out


def stubs(doc: dict, items: list[dict]) -> list[dict]:
    """New planned items so every type lists `count`; existing items are never touched."""
    have = {it.get("id") for it in items}
    fed = feeders(doc)
    new = []
    for t in _list(doc.get("deliverable_types")):
        listed = sum(1 for it in items if it.get("type") == t["id"])
        width = max(2, len(str(t["count"])))
        n = 0
        for _ in range(max(0, t["count"] - listed)):
            n += 1
            while f"{t['id']}-{n:0{width}d}" in have:
                n += 1
            iid = f"{t['id']}-{n:0{width}d}"
            have.add(iid)
            srcs = fed.get(t["id"], [])
            new.append(dict(id=iid, type=t["id"], title=f"{t.get('title', t['id'])} {n:0{width}d}", people=[],
                            sessions=srcs if len(srcs) == 1 else [], shooter="", status="planned", notes=""))
    return new


class _NoAliasDumper(yaml.SafeDumper):
    """Stubs that share one sessions list were dumped as `&id001` / `*id001`; `set` then rewrote the entry holding
    the anchor and left every later alias dangling (found on a real 65-item plan). Every entry spells out its own
    values, so each block stands alone and can be edited alone."""
    def ignore_aliases(self, data):
        return True


def _dump(obj) -> str:
    return yaml.dump(obj, Dumper=_NoAliasDumper, sort_keys=False, allow_unicode=True, width=1000, default_flow_style=False)


def _reparse(out: str):
    try:
        return yaml.safe_load(out) or {}
    except yaml.YAMLError as e:
        raise SystemExit(f"refusing: the edited items.yaml would not parse ({e.problem or e}); an anchor or alias "
                         "probably ties entries together, so edit items.yaml by hand")


def _indent(text: str, n: int) -> str:
    return "".join((" " * n + ln) if ln.strip() else ln for ln in text.splitlines(keepends=True))


ITEMS_HEADER = ("# One entry per deliverable. `plan` appends stubs and never touches existing entries;\n"
                "# edit by hand or with `set`. Status: " + " < ".join(STATUSES) + "\n")


def append_items(text: str | None, items: list[dict], new: list[dict]) -> str:
    """The items.yaml text with `new` appended after the existing entries, every existing byte kept. Verified by
    re-parsing; raises SystemExit (nothing written) when the file's layout does not allow a safe append."""
    if text is None:
        return ITEMS_HEADER + _dump({"items": new})
    lines = text.splitlines(keepends=True)
    at = next((i for i, ln in enumerate(lines) if re.match(r"^items\s*:", ln)), None)
    if at is None:
        out = text + ("" if text.endswith("\n") or not text else "\n") + _dump({"items": new})
    else:
        later_key = next((i for i in range(at + 1, len(lines)) if re.match(r"^[A-Za-z_][\w-]*\s*:", lines[i])), None)
        if later_key is not None:
            raise SystemExit("refusing: items.yaml has keys after `items:`; move `items:` to the end of the file first")
        if re.match(r"^items\s*:\s*(\[\s*\])?\s*(#.*)?$", lines[at]) and not items:
            lines[at] = "items:\n"
            indent = 2
        else:
            first = next((ln for ln in lines[at + 1:] if re.match(r"^\s*- ", ln)), None)
            if first is None:
                raise SystemExit("refusing: cannot find the block list under `items:` (flow-style lists are not appended to)")
            indent = len(first) - len(first.lstrip(" "))
        body = "".join(lines)
        out = body + ("" if body.endswith("\n") else "\n") + _indent(_dump(new), indent)
    parsed = _reparse(out)
    if parsed.get("items") != items + new:
        raise SystemExit("refusing: appending would change existing entries; edit items.yaml by hand")
    return out


def _block(lines: list[str], iid: str) -> tuple[int, int, int] | None:
    """(start, end, indent) of the block-style entry `- id: <iid>`; end is exclusive and excludes trailing blanks."""
    pat = re.compile(r"^(\s*)- id:\s*(['\"]?)" + re.escape(iid) + r"\2\s*(#.*)?$")
    for i, ln in enumerate(lines):
        m = pat.match(ln.rstrip("\n"))
        if not m:
            continue
        ind = len(m.group(1))
        j = i + 1
        while j < len(lines) and (not lines[j].strip() or len(lines[j]) - len(lines[j].lstrip(" ")) > ind):
            j += 1
        while j > i + 1 and not lines[j - 1].strip():
            j -= 1
        return i, j, ind
    return None


def parse_set(pairs: list[str], doc: dict) -> dict:
    upd = {}
    persons = {p["id"] for p in _list(doc.get("people"))}
    sessions = {s["id"] for s in _list(doc.get("sessions"))}
    for p in pairs:
        if "=" not in p:
            raise SystemExit(f"expected key=value, got {p!r}")
        k, v = p.split("=", 1)
        if k not in SET_KEYS:
            raise SystemExit(f"cannot set {k!r}; settable: {', '.join(sorted(SET_KEYS))}")
        if k == "status" and v not in STATUSES:
            raise SystemExit(f"status {v!r} not in {', '.join(STATUSES)}")
        if k in ("people", "sessions"):
            vals = [x.strip() for x in v.split(",") if x.strip()]
            known = persons if k == "people" else sessions
            bad = [x for x in vals if x not in known]
            if bad:
                raise SystemExit(f"unknown {k}: {', '.join(bad)}")
            upd[k] = vals
        else:
            upd[k] = v
    return upd


def set_item(text: str, items: list[dict], iid: str, upd: dict) -> str:
    """items.yaml text with one entry updated and every other line untouched; verified by re-parsing."""
    target = next((it for it in items if it.get("id") == iid), None)
    if target is None:
        raise SystemExit(f"no item {iid!r} in items.yaml")
    lines = text.splitlines(keepends=True)
    blk = _block(lines, iid)
    if blk is None:
        raise SystemExit(f"refusing: cannot find a block-style `- id: {iid}` entry to edit; edit items.yaml by hand")
    i, j, ind = blk
    new_item = {**target, **upd}
    lines[i:j] = [_indent(_dump([new_item]), ind)]
    out = "".join(lines)
    expect = [new_item if it.get("id") == iid else it for it in items]
    if _reparse(out).get("items") != expect:
        raise SystemExit("refusing: the edit would change other entries; edit items.yaml by hand")
    return out


# ------------------------------------------------------------------ status
def evidence(item_dir: Path, status: str) -> list[str]:
    """What the item folder fails to show for the status it claims. Empty when the claim is backed (or needs nothing)."""
    rank = STATUSES.index(status) if status in STATUSES else 0
    problems = []
    if rank >= STATUSES.index("cut") and not list(item_dir.glob("editorial/cut*.json")):
        problems.append(f"{status}: no editorial/cut*.json")
    renders = [p for p in item_dir.glob("editorial/renders/*.mp4") if not p.name.endswith(".master.mp4")]
    if rank >= STATUSES.index("review") and not renders:
        problems.append(f"{status}: no render in editorial/renders/")
    if status == "delivered":
        masters = [*item_dir.glob("master/*.mp4"), *item_dir.glob("editorial/renders/*.master.mp4")]
        verdicts = []
        for m in masters:
            man = m.parent / (m.stem + ".manifest.json")
            if man.exists():
                try:
                    verdicts.append(((json.loads(man.read_text()).get("rule") or {}).get("status")) or "unchecked")
                except ValueError:
                    verdicts.append("unreadable manifest")
        if "honoured" not in verdicts:
            problems.append("delivered: no master with a deliver manifest" if not verdicts else
                            f"delivered: delivery loudness rule {', '.join(sorted(set(verdicts)))} on every master")
    return problems


def status_rows(doc: dict, items: list[dict], project: Path, scope: str | None = None) -> tuple[list[dict], list[dict]]:
    types = [t for t in _list(doc.get("deliverable_types")) if scope is None or scope in _list(t.get("scope"))]
    rows, claims = [], []
    for t in types:
        its = [it for it in items if it.get("type") == t["id"]]
        counts = {s: sum(1 for it in its if it.get("status", "planned") == s) for s in STATUSES}
        rows.append(dict(type=t["id"], title=t.get("title", t["id"]), target=t["count"], listed=len(its), **counts))
        for it in its:
            for prob in evidence(Path(project) / "items" / it["id"], it.get("status", "planned")):
                claims.append(dict(item=it["id"], problem=prob))
    return rows, claims


def status_text(rows: list[dict], claims: list[dict]) -> str:
    head = f"{'type':<28} {'target':>6} {'listed':>6}  " + " ".join(f"{s[:7]:>7}" for s in STATUSES)
    out = [head, "-" * len(head)]
    for r in rows:
        out.append(f"{r['type']:<28} {r['target']:>6} {r['listed']:>6}  " + " ".join(f"{r[s]:>7}" for s in STATUSES))
    tot = {k: sum(r[k] for r in rows) for k in ("target", "listed", *STATUSES)}
    out.append(f"{'total':<28} {tot['target']:>6} {tot['listed']:>6}  " + " ".join(f"{tot[s]:>7}" for s in STATUSES))
    short = [r for r in rows if r["listed"] < r["target"]]
    if short:
        out += ["", "fewer items than the type counts (run `plan`): " + ", ".join(f"{r['type']} {r['listed']}/{r['target']}" for r in short)]
    if claims:
        out += ["", "claimed, no evidence:"] + [f"  {c['item']}: {c['problem']}" for c in claims]
    return "\n".join(out)


def status_md(doc: dict, items: list[dict], rows: list[dict], claims: list[dict], scope: str | None) -> str:
    """A shareable checklist: titles, statuses and gaps, never paths to footage."""
    bad = {}
    for c in claims:
        bad.setdefault(c["item"], []).append(c["problem"])
    title = f"{doc.get('project')} deliverables" + (f" ({scope})" if scope else "")
    out = [f"# {title}", "", f"_Status from items.yaml on {dt.date.today().isoformat()}; claims checked against the item folders._", ""]
    for r in rows:
        out += [f"## {r['title']}: {r['delivered']}/{r['target']} delivered", ""]
        for it in (it for it in items if it.get("type") == r["type"]):
            st = it.get("status", "planned")
            line = f"- [{'x' if st == 'delivered' and it['id'] not in bad else ' '}] {it.get('title') or it['id']} · {st}"
            if it["id"] in bad:
                line += " · claimed, no evidence: " + "; ".join(bad[it["id"]])
            out.append(line)
        if r["listed"] < r["target"]:
            out.append(f"- {r['target'] - r['listed']} more not yet listed")
        out.append("")
    return "\n".join(out)


# ------------------------------------------------------------------ item dirs
def item_dir(project: Path, iid: str) -> list[str]:
    """Make items/<id>/ runnable as a film directory for the cut loop: its own editorial/, and links to the project's
    shared out/ and work/. The project's film.yaml needs no link: load_film falls back to it for an item directory
    (an item can still hold its own). Never replaces anything that exists."""
    project = Path(project)
    d = project / "items" / iid
    made = []
    (d / "editorial").mkdir(parents=True, exist_ok=True)
    for name in ("out", "work"):
        src = project / name
        link = d / name
        if src.exists() and not link.exists() and not link.is_symlink():
            link.symlink_to(os.path.relpath(src, d))
            made.append(f"{link} -> {os.path.relpath(src, d)}")
    return made


# ------------------------------------------------------------------ Resolve bins
def bin_name(s: str) -> str:
    return str(s).replace("/", "-").strip()


def plan_bins(doc: dict, assignments: dict, clip_names: list[str], root: str | None = None) -> dict:
    """Bins to create and clips to move. Clips are matched to the inventory by file stem; register bins of the b-roll
    package are created empty (a register candidate is the schedule's intent, not a verdict on the picture)."""
    root = bin_name(root or doc.get("project") or "Project")
    win = windows(doc)
    folders = []
    for sid, w in sorted(win.items(), key=lambda kv: (str(kv[1]["day"]), kv[1]["start"], kv[0])):
        folders.append((root, bin_name(w["day"]), bin_name(f"{sid} {w['session'].get('title', '')}")))
    for r in _list(doc.get("registers")):
        folders.append((root, "B-roll package", bin_name(r)))
    target = {sid: (root, bin_name(w["day"]), bin_name(f"{sid} {w['session'].get('title', '')}")) for sid, w in win.items()}
    moves: dict = {}
    unassigned, unknown = [], []
    for name in clip_names:
        a = assignments.get(Path(name).stem)
        if a is None:
            unknown.append(name)
        elif not a.get("session"):
            unassigned.append(name)
        else:
            moves.setdefault("/".join(target[a["session"]]), []).append(name)
    return dict(root=root, folders=["/".join(f) for f in folders], moves=moves, unassigned=unassigned, not_in_inventory=unknown)


def _child(folder, name):
    return next((s for s in folder.GetSubFolderList() or [] if s.GetName() == name), None)


def _walk(media_pool, path: str, create: bool):
    folder = media_pool.GetRootFolder()
    for part in path.split("/"):
        nxt = _child(folder, part)
        if nxt is None and create:
            media_pool.AddSubFolder(folder, part)
            nxt = _child(folder, part)   # re-read: AddSubFolder's return is not trusted either
        if nxt is None:
            return None
        folder = nxt
    return folder


def apply_bins(media_pool, source, plan: dict) -> None:
    """Create the bins and move the clips. Return values are ignored on purpose: verify_bins re-reads every bin."""
    for f in plan["folders"]:
        _walk(media_pool, f, create=True)
    by_name = {c.GetName(): c for c in source.GetClipList() or []}
    for path, names in plan["moves"].items():
        dest = _walk(media_pool, path, create=True)
        clips = [by_name[n] for n in names if n in by_name]
        if dest is not None and clips:
            media_pool.MoveClips(clips, dest)


def verify_bins(media_pool, plan: dict) -> dict:
    """Planned vs found, per bin, from a fresh read of the media pool."""
    report = dict(folders_missing=[], bins={}, ok=True)
    for f in plan["folders"]:
        if _walk(media_pool, f, create=False) is None:
            report["folders_missing"].append(f)
    for path, names in plan["moves"].items():
        folder = _walk(media_pool, path, create=False)
        have = {c.GetName() for c in (folder.GetClipList() or [])} if folder else set()
        missing = [n for n in names if n not in have]
        report["bins"][path] = dict(planned=len(names), verified=len(names) - len(missing), missing=missing)
    report["ok"] = not report["folders_missing"] and all(not b["missing"] for b in report["bins"].values())
    return report


# ------------------------------------------------------------------ CLI
def _cfg(path: str):
    from .config import load_config
    return load_config(path)


def _inventory(cfg) -> tuple[Path, dict]:
    p = cfg.out_dir / "inventory.json"
    if not p.exists():
        raise SystemExit(f"no {p}: run `videoeasy probe --config ...` first")
    return p, json.loads(p.read_text())


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m videoeasy.deliverables", description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("validate", "plan", "assign", "clocks", "status", "set", "item-dir", "bins"):
        p = sub.add_parser(name)
        p.add_argument("--project", required=True, help="project directory holding deliverables.yaml")
        if name in ("assign", "clocks", "bins"):
            p.add_argument("--config", required=True, help="the ingest config whose out_dir holds inventory.json")
        if name == "assign":
            p.add_argument("--overwrite", action="store_true")
        if name == "status":
            p.add_argument("--scope", default=None, help="only types whose scope lists this name")
            p.add_argument("--md", default=None, help="also write a shareable markdown checklist here")
        if name in ("set", "item-dir"):
            p.add_argument("item")
        if name == "set":
            p.add_argument("pairs", nargs="+", help="key=value: status, notes, shooter, title, people (a,b), sessions (a,b)")
        if name == "bins":
            p.add_argument("--resolve-project", required=True, help="refuse unless this is the project open in Resolve")
            p.add_argument("--source-bin", required=True, help="media-pool path holding the clips, e.g. Master/Footage")
            p.add_argument("--root-bin", default=None, help="top bin to build under Master (default: the project name)")
            p.add_argument("--apply", action="store_true", help="make the changes (default: print the plan only)")
    args = ap.parse_args(argv)
    project = Path(args.project)
    doc = load(project)

    if args.cmd == "validate":
        items, _ = load_items(project)
        errors, conflicts = validate(doc, items if (project / "items.yaml").exists() else None)
        for e in errors + conflicts:
            print(f"error  {e}")
        if not errors and not conflicts:
            print(f"ok  {len(_list(doc.get('sessions')))} sessions, {len(_list(doc.get('deliverable_types')))} types, "
                  f"{len(items)} items")
        return 2 if errors or conflicts else 0

    if args.cmd == "plan":
        items, text = load_items(project)
        require_valid(doc, items)
        new = stubs(doc, items)
        if not new:
            print(f"every type already lists its count ({len(items)} items); nothing appended")
            return 0
        (project / "items.yaml").write_text(append_items(text, items, new), encoding="utf-8")
        print(f"appended {len(new)} planned items to {project / 'items.yaml'}")
        return 0

    if args.cmd == "assign":
        conflicts = require_valid(doc, allow_conflicts=True)
        cfg = _cfg(args.config)
        inv_path, inv = _inventory(cfg)
        out = cfg.out_dir / "sessions.json"
        refuse_overwrite([out], args.overwrite)
        clips = assign_clips(doc, inv)
        by_session: dict = {}
        for cid, r in clips.items():
            if r["session"]:
                by_session.setdefault(r["session"], []).append(cid)
        unassigned = {cid: r["reason"] for cid, r in clips.items() if not r["session"]}
        doc_out = dict(manifest=dict(tool="deliverables.assign", tool_sha=source_hash(__file__),
                                     created=dt.datetime.now().astimezone().isoformat(timespec="seconds"),
                                     inventory=dict(path=str(inv_path), sha=fingerprint(inv_path)),
                                     deliverables=dict(path=str(project / "deliverables.yaml"), sha=fingerprint(project / "deliverables.yaml")),
                                     timezone=doc["timezone"], warnings=conflicts),
                       clips=clips, sessions=by_session, unassigned=unassigned)
        out.write_text(json.dumps(doc_out, indent=1))
        win = windows(doc)
        print(f"{'session':<30} {'day':<6} {'time':<11} {'clips':>5} {'minutes':>8}")
        inv_by = {c["id"]: c for c in inventory_clips(inv)}
        for sid, w in sorted(win.items(), key=lambda kv: (str(kv[1]["day"]), kv[1]["start"], kv[0])):
            ids = by_session.get(sid, [])
            mins = sum(float(inv_by[c].get("duration_s") or 0) for c in ids) / 60
            print(f"{sid:<30} {str(w['day']):<6} {hhmm(w['start'])}-{hhmm(w['end']):<5} {len(ids):>5} {mins:>8.1f}")
        if unassigned:
            reasons: dict = {}
            for cid, why in unassigned.items():
                reasons.setdefault(why.split(" (")[0] if why.startswith(("outside", "day not")) else why, []).append(cid)
            print(f"\n{len(unassigned)} unassigned:")
            for why, ids in sorted(reasons.items(), key=lambda kv: -len(kv[1])):
                print(f"  {len(ids):>4}  {why}: {', '.join(sorted(ids)[:12])}{' ...' if len(ids) > 12 else ''}")
        for c in conflicts:
            print(f"warning  {c}")
        print(f"\nwrote {out}")
        return 0

    if args.cmd == "clocks":
        require_valid(doc, allow_conflicts=True)
        _, inv = _inventory(_cfg(args.config))
        print(f"{'camera':<16} {'date':<11} {'day':<6} {'clips':>5} {'first':>9} {'last':>9} {'early':>6} {'late':>5}")
        for r in clock_table(doc, inv):
            print(f"{r['camera']:<16} {r['date']:<11} {str(r['day'] or '-'):<6} {r['clips']:>5} {r['first'] or '-':>9} "
                  f"{r['last'] or '-':>9} {r['before_first_session']:>6} {r['after_last_session']:>5}")
        print("\nearly/late: clips before the day's first session or after its last. A camera that is consistently early or "
              "late against the schedule wants clock_offset_s; measure it on one event you can see in two cameras.")
        return 0

    if args.cmd == "status":
        items, _ = load_items(project)
        rows, claims = status_rows(doc, items, project, args.scope)
        print(status_text(rows, claims))
        if args.md:
            Path(args.md).write_text(status_md(doc, items, rows, claims, args.scope), encoding="utf-8")
            print(f"\nwrote {args.md}")
        return 0

    if args.cmd == "set":
        items, text = load_items(project)
        if text is None:
            raise SystemExit("no items.yaml: run `plan` first")
        (project / "items.yaml").write_text(set_item(text, items, args.item, parse_set(args.pairs, doc)), encoding="utf-8")
        print(f"updated {args.item}")
        return 0

    if args.cmd == "item-dir":
        items, _ = load_items(project)
        if args.item not in {it.get("id") for it in items}:
            raise SystemExit(f"no item {args.item!r} in items.yaml")
        for m in item_dir(project, args.item):
            print(f"linked {m}")
        print(f"{project / 'items' / args.item} is ready as --film for the cut loop")
        return 0

    if args.cmd == "bins":
        cfg = _cfg(args.config)
        sj = cfg.out_dir / "sessions.json"
        if not sj.exists():
            raise SystemExit(f"no {sj}: run `assign` first")
        assignments = json.loads(sj.read_text())["clips"]
        from .resolve import find_folder, get_project, get_resolve
        resolve = get_resolve()
        if resolve.GetCurrentPage() is None:
            raise SystemExit("Resolve reports no current page: a modal dialog or the Project Manager is open, and every write "
                             "would silently do nothing. Close it and re-run.")
        project_obj = get_project(args.resolve_project)
        pool = project_obj.GetMediaPool()
        source = find_folder(pool, args.source_bin)
        plan = plan_bins(doc, assignments, [c.GetName() for c in source.GetClipList() or []], args.root_bin)
        print(f"bins under {plan['root']}: {len(plan['folders'])}")
        for path, names in plan["moves"].items():
            print(f"  {len(names):>4} -> {path}")
        print(f"  {len(plan['unassigned'])} unassigned stay in {args.source_bin}; {len(plan['not_in_inventory'])} not in the inventory")
        if not args.apply:
            print("\ndry run: pass --apply to create the bins and move the clips")
            return 0
        apply_bins(pool, source, plan)
        rep = verify_bins(pool, plan)
        for path, b in rep["bins"].items():
            print(f"  {b['verified']:>4}/{b['planned']:<4} {path}" + (f"  missing: {', '.join(b['missing'][:8])}" if b["missing"] else ""))
        for f in rep["folders_missing"]:
            print(f"  bin not created: {f}")
        print("verified by re-reading every bin" if rep["ok"] else "NOT all moves verified: MoveClips can report success "
              "without moving; check the bins above")
        return 0 if rep["ok"] else 2
    return 1


if __name__ == "__main__":
    sys.exit(main())
