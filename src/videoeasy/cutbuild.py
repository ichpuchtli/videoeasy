"""Build the next cut from a previous cut's audio and a verified B-roll plan.

Audio picks (the radio cut) are copied unchanged. Video rows are rebuilt
per pick from `brollmatch`'s plan: a chosen shot held for the pick's length
(in-point moved to the first usable frame), long picks split across the
chosen shot and its verified alternatives, placeholder cards where the
plan names them, and BARE rows where the plan says the speaker's face is the
picture. Adjacent picks that chose the same shot merge into one hold when
the source has the length.

Usage:
    uv run python -m videoeasy.cutbuild --film data/films/<film> \
        --cut editorial/cut-v2.json --plan editorial/broll-plan-v3.json \
        --version 3 [--max-hold 12]
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import math
import sys
from pathlib import Path

from . import moves as moves_mod
from . import steadiness as steady_mod
from .brollmatch import load_catalogue, verdict_ok
from .evalrun import OK, UNCHECKED, fingerprint, refuse_overwrite, source_hash, text_hash

FRAME_FRACS = [0.0, 2 / 7, 5 / 7, 1.0]  # sample positions of the four verified frames


def fmt(s: float) -> str:
    m, r = divmod(max(0.0, s), 60)
    return f"{int(m):02d}:{r:05.2f}"


def shot_row(cat: dict, sid: str, hold: float, usable: list | None, note: str, used_in: dict | None = None,
             frames_used: list[dict] | None = None) -> dict | None:
    """A cover row of up to `hold` seconds starting at the first frame the
    verifier called usable. The row never backs up before that frame: when the
    source has less than `hold` left from there the row is shorter, and when
    it has under a second the row is None and the caller keeps the pick's
    timing with something else. `frames_used` (from the verdict) gives the
    real source time of each frame sent; without it the legacy quarter
    positions are assumed."""
    c = cat[sid]
    src_len = c["out_s"] - c["in_s"]
    idx = sorted({int(u) for u in (usable or []) if str(u).isdigit() and int(u) >= 1})
    if frames_used:
        idx = [i for i in idx if i <= len(frames_used)]
        start = frames_used[idx[0] - 1]["t"] if idx else c["in_s"]
    else:
        first_ok = min([i for i in idx if i <= 4], default=1)
        start = c["in_s"] + FRAME_FRACS[first_ok - 1] * src_len
    certified_from = start  # the first frame the verifier saw and accepted; never earlier
    if c.get("steady"):
        # a shaky unit: the row lives inside one steady part. A reused shot first tries the footage after its last
        # use, then the certified start again (a repeat, said so in the note); the longer of the two wins.
        tries = []
        if used_in is not None and sid in used_in:
            tries.append((max(certified_from, used_in[sid]), False))
        tries.append((certified_from, used_in is not None and sid in used_in))
        best = None
        for s0, repeat in tries:
            r = steady_mod.clamp_to_runs(s0, min(hold, c["out_s"] - s0), c["steady"])
            if r and (best is None or r[1] > best[0][1] + 0.01):
                best = (r, repeat)
            if r and r[1] >= hold - 0.01:
                break
            # fresh footage that holds a usable stretch beats a longer repeat: v9's first build played one 5.7 s
            # drone pull-out twice, four minutes apart, rather than 3.8 s of the next clean move (29 Sep 2026)
            if r and not repeat and r[1] >= min(hold, steady_mod.MIN_STABLE_S) - 0.01:
                break
        if best is None:
            return None
        (start, hold), repeat = best
        part = next((r for r in c["steady"] if r["t0"] - 0.01 <= start <= r["t1"] + 0.01), {})
        what = f"clean drone move: {part['label']}" if c.get("moves") and part.get("label") else "steady part of a shaky take"
        note = (note + "; " if note else "") + what + ("; repeats footage used earlier in the film" if repeat else "")
        if hold < 1.0:
            return None
        if used_in is not None:
            used_in[sid] = round(start + hold, 2)
        return dict(type="video", clip=sid, source_path=None, in_s=round(start, 2), out_s=round(start + hold, 2),
                    duration_s=round(hold, 2), note=note)
    # a shot reused later in the film continues from where its last use ended; when that leaves
    # less than the hold, it restarts at the certified start and repeats footage (said so in the note)
    if used_in is not None and sid in used_in:
        cont = max(start, used_in[sid])
        if c["out_s"] - cont >= hold - 0.01 or c["out_s"] - cont >= c["out_s"] - certified_from:
            start = cont
        else:
            note = (note + "; " if note else "") + "repeats footage used earlier in the film"
    hold = min(hold, c["out_s"] - start)
    if hold < 1.0:
        return None
    if used_in is not None:
        used_in[sid] = round(start + hold, 2)
    return dict(type="video", clip=sid, source_path=None, in_s=round(start, 2), out_s=round(start + hold, 2),
                duration_s=round(hold, 2), note=note)


def room_after(cat: dict, row: dict) -> float:
    """Seconds a cover row can run on past its out-point: to the end of the steady part (or clean drone move) it sits
    in, else to the end of the unit."""
    c = cat[row["clip"]]
    end = c["out_s"]
    if c.get("steady"):
        end = next((r["t1"] for r in c["steady"] if r["t0"] - 0.01 <= row["out_s"] <= r["t1"] + 0.01), row["out_s"])
    return max(0.0, end - row["out_s"])


def stretch(cat: dict, rows: list[dict], need: float, used_in: dict | None = None) -> float:
    """A pick's cover came up `need` seconds short: its rows run on where their footage allows, last row first, before
    the speaker fills the rest. Without this an evenly split pick whose last piece ran 0.15 s short flashed the speaker
    for four frames (cut-v10's hook, 29 Sep 2026) while its first piece had the room. Returns what is still missing."""
    for row in reversed(rows):
        if need <= 0.02:
            break
        if row["type"] != "video" or row.get("clip") not in cat:
            continue
        extra = round(min(need, room_after(cat, row)), 2)
        if extra > 0.01:
            row["out_s"] = round(row["out_s"] + extra, 2)
            row["duration_s"] = round(row["duration_s"] + extra, 2)
            need = round(need - extra, 2)
            if used_in is not None:
                used_in[row["clip"]] = max(used_in.get(row["clip"], 0.0), row["out_s"])
    return need


def segment_rows(r: dict, cat: dict, used_in: dict, ineligible: dict, never_prefix: str | None) -> list[dict]:
    """A hand-set plan row's picture in word-timed segments (the editor, 30 Sep 2026: the opening hook's drone only under
    its first words; a welcome on the presenter's face, then the land while the words name it, then the face again).
    `segments` = [{until_s, shot or None, note?}], seconds from the pick's start; no shot is the speaker. A segment's
    shot must be catalogued, measured eligible, allowed by intent and watchable with a verdict of 2 or more for this
    pick, or that segment goes to the speaker with the reason. A shot that runs short runs on where its footage allows,
    then the speaker fills the rest of its segment, so every later segment still starts on its word."""
    vds = r.get("verdicts", {})
    out: list[dict] = []
    t = 0.0
    for seg in r["segments"]:
        length = round(min(float(seg["until_s"]), r["duration_s"]) - t, 2)
        if length <= 0.02:
            continue
        sid, why, vd = seg.get("shot"), None, {}
        if sid:
            vd = vds.get(sid) or {}
            if sid not in cat:
                why = f"{sid} is not in the catalogue"
            elif sid in ineligible:
                why = f"{sid} is not cover: {ineligible[sid]}"
            elif never_prefix and sid.startswith(never_prefix):
                why = f"{sid} is another walk take; sync walk stays on its own picture (intent)"
            elif (vd.get("score") or 0) < 2 or not watchable(vd):
                why = f"{sid} has no accepted verdict of 2 or more for these words"
        if not sid or why:
            out.append(bare_row(length, why or seg.get("note") or "speaker (plan segment)"))
        else:
            row = shot_row(cat, sid, length, vd.get("usable"), vd.get("reason", r.get("reason", "")), used_in, vd.get("frames_used"))
            rows = [row] if row else []
            short = round(length - (row["duration_s"] if row else 0.0), 2)
            if short > 0.02 and rows:
                short = stretch(cat, rows, short, used_in)
            out += rows
            if short > 0.02:
                out.append(bare_row(short, f"{sid} ran short; speaker for the rest of the segment"))
        t = round(t + length, 2)
    return out


def bare_row(dur: float, note: str) -> dict:
    return dict(type="bare", clip=None, source_path=None, in_s=0.0, out_s=round(dur, 2), duration_s=round(dur, 2), note=note)


BAD_WORDS = ("blurry", "blur", "out of focus", "unusable", "shaky", "unwatchable")


def watchable(v: dict | None) -> bool:
    """Only a verdict that ran and was accepted can certify cover; a missing,
    errored or invalid verdict certifies nothing. Among accepted verdicts,
    one whose reason complains of blur with fewer than two usable frames is
    not cover either."""
    if not verdict_ok(v):
        return False
    reason = (v.get("reason") or "").lower()
    usable = [u for u in (v.get("usable") or []) if str(u).isdigit()]
    return not (any(w in reason for w in BAD_WORDS) and len(usable) < 2)


def build_beat_video(beat: dict, rows: list[dict], cat: dict, max_hold: float, used_in: dict[str, float],
                     t0: float = 0.0, protected: dict[str, str] | None = None, never_from: dict[str, str] | None = None,
                     ineligible: dict[str, str] | None = None) -> list[dict]:
    """Video rows for one beat from the plan rows of its picks.

    Invariant: the rows emitted for a pick sum to that pick's duration (to
    0.02 s), so every later pick's picture starts where its words start. A
    source that runs short is filled in place with a bare row, never by
    letting the next picture slide earlier. A plan row with status
    `unchecked` gets a bare row marked provisional: no cover was certified.

    Every row carries its destination: `pick` (the plan row id it serves)
    and `dest_in_s` / `dest_out_s` in film time, `t0` being the beat's start.
    A merged row spans two picks and lists both in `picks`.

    `protected` (pick -> intent rule id) forces a bare row whatever the plan
    says; `never_from` (pick -> source prefix) drops a chosen shot or
    alternative from that prefix. Both come from editorial/intent.json, so a
    plan edited by hand cannot bypass a recorded decision. `ineligible`
    (unit -> reason) drops units the measurements rule out as cover (camera
    shake over the limit, someone on screen talking): the row goes bare with
    the reason, never to a middling alternative that was not measured."""
    protected = protected or {}
    never_from = never_from or {}
    ineligible = ineligible or {}
    v2_ph = [v for v in beat["video"] if v["type"] == "placeholder"]
    # Placeholders are the editor's decision (material promised for later); if the plan did not place one,
    # attach it to the pick whose words share the most vocabulary with the card, else the first pick.
    planned_ph = {r["shot"] for r in rows if r.get("shot", "") and r["shot"].startswith("PH:")}
    forced: dict[str, list[dict]] = {}
    for k, ph in enumerate(v2_ph, 1):
        if f"PH:ph{k}" in planned_ph:
            continue
        words = {w.strip("',.()").lower() for w in ph["text"].split() if len(w) > 3}
        best = max(rows, key=lambda r: len(words & {w.strip("',.?").lower() for w in r["text"].split()}), default=None)
        if best is not None:
            forced.setdefault(best["pick"], []).append(ph)
    out: list[dict] = []
    for r in rows:
        target = r["duration_s"]
        dur = target
        shot = r.get("shot")
        vds = r.get("verdicts", {})
        pr: list[dict] = []      # rows for this pick
        merged = 0.0             # seconds added to the previous pick's last row instead
        if r["pick"] in protected:
            r = dict(r, shot=None, status=OK, note=f"held bare by intent rule {protected[r['pick']]}")
            shot = None
            forced.pop(r["pick"], None)
        if r["pick"] in never_from:
            pre = never_from[r["pick"]]
            if shot and not shot.startswith("PH:") and shot.startswith(pre):
                r = dict(r, shot=None, note=f"{shot} is another walk take; sync walk stays on its own picture (intent)")
                shot = None
            r = dict(r, alternatives=[a for a in r.get("alternatives", []) if not a.startswith(pre)])
        if ineligible:
            if shot and not shot.startswith("PH:") and shot in ineligible:
                r = dict(r, shot=None, note=f"{shot} is not cover: {ineligible[shot]}")
                shot = None
            r = dict(r, alternatives=[a for a in r.get("alternatives", []) if a not in ineligible])
        for ph in forced.get(r["pick"], []):
            card = copy.deepcopy(ph)
            card["duration_s"] = card["out_s"] = round(min(8.0, dur), 2)
            pr.append(card)
            dur = round(dur - card["duration_s"], 2)
        if r.get("segments") and shot and not shot.startswith("PH:") and not pr and r.get("status") != UNCHECKED:
            pr += segment_rows(r, cat, used_in, ineligible, never_from.get(r["pick"]))
        elif r.get("status") == UNCHECKED:
            row = bare_row(dur, "PROVISIONAL: no candidate had an accepted verdict; " + (r.get("note") or ""))
            row["provisional"] = True
            pr.append(row)
        elif dur <= 0.02:
            pass
        elif not shot:
            pr.append(bare_row(dur, r.get("note") or "speaker carries it"))
        elif shot.startswith("PH:"):
            k = int(shot[5:]) - 1 if shot[3:5] == "ph" and shot[5:].isdigit() else 0
            ph = copy.deepcopy(v2_ph[k] if k < len(v2_ph) else v2_ph[0])
            ph["duration_s"] = ph["out_s"] = round(min(dur, 12.0), 2)
            pr.append(ph)
        elif dur < 1.0:
            pr.append(bare_row(dur, "under a second left after the card; speaker"))
        else:
            # merge with the previous row when it is the same shot, its verdict for THIS pick is accepted, and the source has room
            if out and out[-1]["type"] == "video" and out[-1]["clip"] == shot and watchable(vds.get(shot)):
                # the merge may not run past the end of the steady part the row sits in
                if dur <= room_after(cat, out[-1]) + 0.01:
                    out[-1]["out_s"] = round(out[-1]["out_s"] + dur, 2)
                    out[-1]["duration_s"] = round(out[-1]["duration_s"] + dur, 2)
                    out[-1]["dest_out_s"] = round(out[-1]["dest_out_s"] + dur, 2)
                    out[-1].setdefault("picks", [out[-1]["pick"]]).append(r["pick"])
                    used_in[shot] = out[-1]["out_s"]
                    merged = dur
            if not merged:
                alts = [a for a in r.get("alternatives", []) if a in cat and (vds.get(a, {}).get("score") or 0) >= 2 and watchable(vds.get(a))]
                if not watchable(vds.get(shot)):
                    if alts:
                        shot, alts = alts[0], alts[1:]
                    else:
                        pr.append(bare_row(dur, "chosen cover was unwatchable; back to the speaker"))
                        shot = None
                if shot:
                    n = max(1, math.ceil(dur / max_hold))
                    chain = [shot] + alts
                    if n > 1 and len(chain) > 1:
                        n = min(n, len(chain))
                        piece = dur / n
                        carry = 0.0  # a short piece hands its deficit to the next piece, not to the speaker
                        for k, sid in enumerate(chain[:n]):
                            want = piece + carry
                            vd = vds.get(sid, {})
                            row = shot_row(cat, sid, want, vd.get("usable"), vd.get("reason", r.get("reason", "")), used_in, vd.get("frames_used"))
                            if row:
                                pr.append(row)
                            got = row["duration_s"] if row else 0.0
                            carry = want - got
                            if carry > 0.02 and k == n - 1:  # nothing left in the chain: earlier pieces run on, then fill in place
                                carry = stretch(cat, pr, carry, used_in)
                                if carry > 0.02:
                                    pr.append(bare_row(carry, f"{sid} ran short; speaker for the rest of the pick"))
                    else:
                        vd = vds.get(shot, {})
                        row = shot_row(cat, shot, dur, r.get("usable"), r.get("reason", ""), used_in, vd.get("frames_used"))
                        if row:
                            pr.append(row)
                        else:
                            pr.append(bare_row(dur, "no certified interval of a second or more; back to the speaker"))
        emitted = merged + sum(x["duration_s"] for x in pr)
        deficit = round(target - emitted, 2)
        if deficit > 0.02:
            pr.append(bare_row(deficit, "keeps the pick's timing"))
        elif deficit < -0.02 and pr:
            last = pr[-1]
            last["duration_s"] = round(last["duration_s"] + deficit, 2)
            last["out_s"] = round(last["out_s"] + deficit, 2)
        cursor = round(t0 + merged, 4)
        for x in pr:
            x["pick"] = r["pick"]
            x["dest_in_s"] = round(cursor, 2)
            x["dest_out_s"] = round(cursor + x["duration_s"], 2)
            if x["type"] == "placeholder":
                x["asset"] = "ph-" + text_hash(x["text"])[:8]
            cursor = round(cursor + x["duration_s"], 4)
        out.extend(pr)
        t0 = round(t0 + target, 4)
    return out


def check_cut(cut: dict, tier: int = 1, tol: float = 0.05) -> list[str]:
    """Placement invariants a cut must satisfy before it is laid or scored:
    per beat, video rows are contiguous from the beat start, never overlap,
    and end where the spoken picks end. Returns the violations (empty = ok)."""
    problems = []
    t = 0.0
    for bi, beat in enumerate(cut["beats"], 1):
        picks = [a for a in beat["audio"] if a.get("tier", 1) <= tier]
        if not picks:
            continue
        span = sum(a["duration_s"] for a in picks)
        cursor = t
        for k, v in enumerate(beat["video"]):
            if "dest_in_s" not in v:
                problems.append(f"beat {bi} row {k}: no destination interval")
                cursor += v["duration_s"]
                continue
            if abs(v["dest_in_s"] - cursor) > tol:
                problems.append(f"beat {bi} row {k} ({v['type']} {v.get('clip')}): starts {v['dest_in_s']} expected {cursor:.2f}")
            if abs((v["dest_out_s"] - v["dest_in_s"]) - v["duration_s"]) > tol:
                problems.append(f"beat {bi} row {k}: destination {v['dest_in_s']}-{v['dest_out_s']} vs duration {v['duration_s']}")
            cursor = v["dest_out_s"]
        if abs(cursor - (t + span)) > tol:
            problems.append(f"beat {bi}: video ends {cursor:.2f}, picks end {t + span:.2f}")
        t += span + beat.get("gap_after_s", 0.0)
    return problems


def write_md(cut: dict, out_md: Path, version: int) -> None:
    core = sum(a["duration_s"] for b in cut["beats"] for a in b["audio"] if a.get("tier", 1) == 1)
    prov = sum(1 for b in cut["beats"] for v in b["video"] if v.get("provisional"))
    L = [f"# {cut['film']} — radio cut v{version} (word-timed)", "",
         f"_Generated {dt.date.today():%d %b %Y} by `cutbuild` from cut-v{version - 1} audio and the B-roll plan "
         f"(`broll-plan-v{version}.json`): each tier-1 pick's cover was proposed by the local text model from the bible and "
         f"checked by the local vision model on up to four sampled frames of the shot (a sample, not a frame-by-frame check); "
         f"BARE rows are picks the plan kept on the speaker._",
         "", f"Core tier spoken time: **{fmt(core)}**.", ""]
    if prov:
        L += [f"**PROVISIONAL: {prov} picks have no certified cover** (their verdicts errored or were never run); they play bare until verified.", ""]
    t = 0.0
    for bi, b in enumerate(cut["beats"], 1):
        picks = [a for a in b["audio"] if a.get("tier", 1) == 1]
        span = sum(a["duration_s"] for a in picks)
        L += [f"## {b['title']}  ·  core {fmt(t)} → {fmt(t + span)}  ({span:.0f}s core)", ""]
        if b.get("notes"):
            L += [f"_{b['notes']}_", ""]
        L += ["| # | src | in | out | dur | text |", "|---|---|---|---|---|---|"]
        for j, a in enumerate(b["audio"], 1):
            tier = "" if a.get("tier", 1) == 1 else " *ext*"
            L.append(f"| A{j}{tier} | `{a['clip']}` | {fmt(a['in_s'])} | {fmt(a['out_s'])} | {a['duration_s']:.1f}s | {a['text'][:150]} |")
        L.append("")
        for v in b["video"]:
            if v["type"] == "bare":
                L.append(f"- BARE ~{v['duration_s']:.0f}s — {v.get('note', '')}")
            elif v["type"] == "placeholder":
                L.append(f"- **[{v['text']}]** ~{v['duration_s']:.0f}s — {v.get('note', '')}")
            else:
                L.append(f"- `{v['clip']}` {fmt(v['in_s'])}–{fmt(v['out_s'])} (~{v['duration_s']:.0f}s hold) — {v.get('note', '')}")
        L.append("")
        t += span + b.get("gap_after_s", 0.0)
    out_md.write_text("\n".join(L))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--film", required=True)
    ap.add_argument("--cut", required=True)
    ap.add_argument("--plan", required=True)
    ap.add_argument("--version", type=int, required=True)
    ap.add_argument("--max-hold", type=float, default=12.0)
    ap.add_argument("--out-dir", default=None, help="directory for cut-vN.json/.md (default <film>/editorial)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--max-shake", type=float, default=None, help="camera shake limit, %% of frame width (default steadiness.MAX_SHAKE_PCT)")
    args = ap.parse_args(argv)
    film = Path(args.film)
    out_dir = Path(args.out_dir) if args.out_dir else film / "editorial"
    out_json = out_dir / f"cut-v{args.version}.json"
    out_md = out_dir / f"cut-v{args.version}.md"
    refuse_overwrite([out_json, out_md], args.overwrite)
    cut = json.load(open(film / args.cut))
    plan = json.load(open(film / args.plan))
    cat = load_catalogue(film)
    from . import intent as intent_mod
    intent_path = film / "editorial/intent.json"
    intent_doc = intent_mod.load(film) if intent_path.exists() else None
    protected = intent_mod.protected_picks(cut, intent_doc) if intent_doc else {}
    never_from = intent_mod.walk_rules(cut, intent_doc) if intent_doc else {}
    from . import speech
    max_shake = steady_mod.MAX_SHAKE_PCT if args.max_shake is None else args.max_shake
    transcripts = json.load(open(film / "out/transcripts.json")) if (film / "out/transcripts.json").exists() else {}
    steady = steady_mod.load(film / "out")
    ineligible = steady_mod.apply_runs(cat, steady, max_shake)   # shaky units keep only their steady parts (cat[..]["steady"])
    drone_moves = moves_mod.load(film / "out")
    ineligible.update(moves_mod.apply_clean(cat, drone_moves))   # drone units keep only their clean moves and holds
    exempt = intent_mod.talking_exempt(intent_doc)
    ineligible.update(speech.talking_units(transcripts, [dict(id=k, in_s=c["in_s"], out_s=c["out_s"], people=c.get("people")) for k, c in cat.items()],
                                           exempt=exempt))
    new = copy.deepcopy(cut)
    new["version"] = args.version
    new["generated"] = dt.date.today().isoformat()
    new["broll_plan"] = args.plan
    used_in: dict[str, float] = {}
    stats = dict(picks=0, covered_s=0.0, bare_s=0.0, rows=0, provisional=0)
    t0 = 0.0
    for bi, beat in enumerate(new["beats"], 1):
        pb = plan["beats"].get(str(bi))
        picks = [a for a in beat["audio"] if a.get("tier", 1) == 1]
        if not pb:
            continue
        rows = pb["plan"]
        beat["video"] = build_beat_video(beat, rows, cat, args.max_hold, used_in, t0, protected, never_from, ineligible)
        t0 += sum(a["duration_s"] for a in picks) + beat.get("gap_after_s", 0.0)
        for v in beat["video"]:
            if v["type"] == "video":
                v["source_path"] = str(film / "inputs" / ("broll" if cat[v["clip"]]["role"] == "broll" else "aroll") / (cat[v["clip"]]["source"] + ".MOV"))
        stats["picks"] += len(rows)
        stats["rows"] += len(beat["video"])
        stats["covered_s"] += sum(v["duration_s"] for v in beat["video"] if v["type"] != "bare")
        stats["bare_s"] += sum(v["duration_s"] for v in beat["video"] if v["type"] == "bare")
        stats["provisional"] += sum(1 for v in beat["video"] if v.get("provisional"))
    new["manifest"] = dict(tool="cutbuild", tool_sha=source_hash(__file__),
                           cut=dict(path=args.cut, sha=fingerprint(film / args.cut)),
                           plan=dict(path=args.plan, sha=fingerprint(film / args.plan), manifest=plan.get("manifest")),
                           catalogue_sha=text_hash(json.dumps(cat, sort_keys=True)), max_hold=args.max_hold, stats=stats)
    problems = check_cut(new)
    if problems:
        raise SystemExit("cut violates placement invariants; nothing written:\n  " + "\n  ".join(problems[:20]))
    new["manifest"]["placement_check"] = "ok"
    # the built cut is measured again as a whole: no cover row may hold the covered person's speech
    tc = speech.talking_cover(new, transcripts, people={k: c.get("people") or "" for k, c in cat.items()}, exempt=exempt)
    talking = [t for t in tc if t["status"] == "talking"]
    new["manifest"]["cover_rules"] = dict(max_shake_pct=max_shake, steadiness_measured=steady is not None, drone_moves_measured=drone_moves is not None,
                                          ineligible_units=len(ineligible),
                                          talking_cover=len(talking), cover_from_untranscribed_sources=sum(1 for t in tc if t["status"] == "unknown"))
    if talking:
        raise SystemExit("cut puts a talking take under another pick's audio; nothing written:\n  "
                         + "\n  ".join(f"beat {t['beat']} {t['pick']} {t['clip']} {t['in_s']}-{t['out_s']}: {t['speech_s']} s of speech" for t in talking))
    if intent_doc:
        findings = intent_mod.check_cut(new, intent_doc)
        broken = [f for f in findings if f["status"] == intent_mod.BROKEN]
        new["manifest"]["intent"] = dict(path=str(intent_path), sha=fingerprint(intent_path), summary=intent_mod.summary(findings), findings=findings)
        if broken:
            raise SystemExit("cut breaks recorded editorial intent; nothing written:\n  " + "\n  ".join(f"{f['rule']}: {f['evidence']}" for f in broken))
        print(f"intent: {intent_mod.summary(findings)}", file=sys.stderr)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(new, indent=1))
    write_md(new, out_md, args.version)
    print(f"wrote {out_json} and .md: {stats}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
