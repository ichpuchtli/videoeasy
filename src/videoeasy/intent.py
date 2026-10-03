"""The resolved editorial intent, checked on the cut and on the Resolve read-back.

`editorial/intent.json` is the one place a film's decisions live: what was
decided, by whom, where it is recorded, and which earlier decision it
replaces. `resolve()` applies precedence (an explicit `supersedes` retires
the named rules; on one `topic` the latest `decided` date wins) and returns
the active brief plus what it retired. `check_cut()` and `check_readback()`
test a cut and a timeline read-back against the active rules with no model
in the loop: a quality score can never override a decision, and a plan
edited by hand cannot bypass one.

Each finding: rule id, status (honoured | broken | review | open |
not_testable), evidence. `review` lists things the rule allows only
conditionally so a person can look; it is not a pass.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from .cuteval import pick_spans
from .film import load_film
from .sources import base_clip, placeholder_file

HONOURED, BROKEN, REVIEW, OPEN, NOT_TESTABLE = "honoured", "broken", "review", "open", "not_testable"
EDGE_TOL_S = 0.1   # an excluded span and a pick that meet at an edge (adjacent lines of one take) do not overlap


def load(film: Path | str) -> dict:
    return json.loads((Path(film) / "editorial/intent.json").read_text(encoding="utf-8"))


def resolve(doc: dict) -> dict:
    """Active rules after precedence, plus the retired ones with the reason."""
    rules = {r["id"]: dict(r) for r in doc["rules"]}
    retired: dict[str, str] = {}
    for r in rules.values():
        if r.get("status") == "superseded":
            retired.setdefault(r["id"], "marked superseded")
        for old in r.get("supersedes", []):
            if old in rules:
                retired[old] = f"superseded by {r['id']} ({r.get('decided')})"
    by_topic: dict[str, list[dict]] = {}
    for r in rules.values():
        if r["id"] not in retired:
            by_topic.setdefault(r.get("topic", r["id"]), []).append(r)
    for topic, rs in by_topic.items():
        if len(rs) > 1:
            rs.sort(key=lambda r: r.get("decided", ""))
            for r in rs[:-1]:
                retired[r["id"]] = f"same topic '{topic}': later decision {rs[-1]['id']} ({rs[-1].get('decided')}) wins"
    active = [r for r in doc["rules"] if r["id"] not in retired]
    return dict(film=doc.get("film"), active=active, retired=[dict(id=k, why=v) for k, v in retired.items()])


def brief(doc: dict) -> str:
    """The current brief as text for a prompt: active decisions with their source, then what no longer applies."""
    res = resolve(doc)
    L = []
    for r in res["active"]:
        who = r.get("by", "")
        L.append(f"- [{r['id']}] {r['decision']} (decided {r.get('decided')} by {who}; source: {r.get('source')})")
    if res["retired"]:
        L.append("")
        L.append("No longer applies:")
        rules = {r["id"]: r for r in doc["rules"]}
        for x in res["retired"]:
            L.append(f"- [{x['id']}] {rules[x['id']]['decision']} ({x['why']})")
    return "\n".join(L)


def talking_exempt(doc: dict | None) -> set[str]:
    """Clips the editor has allowed as cover despite speech on their own audio (kind talking_cover_exempt). The talking-cover
    rule stays measured everywhere else; this is a named exception, never a pattern."""
    if not doc:
        return set()
    return {c for r in resolve(doc)["active"] if r["kind"] == "talking_cover_exempt" for c in r.get("clips", [])}


def story_additions(doc: dict) -> str:
    """Text for the spine section: beats that later decisions require, so a story judge does not call them orphans."""
    out = [r["story"] for r in resolve(doc)["active"] if r.get("story")]
    return "\n".join(f"- {s}" for s in out)


# ------------------------------------------------------------------- scope
def _beats_in_scope(cut: dict, scope: dict) -> list[int]:
    out = []
    for bi, b in enumerate(cut["beats"], 1):
        t, n = b.get("title", ""), b.get("notes") or ""
        if "beat_title_contains" in scope and scope["beat_title_contains"].lower() not in t.lower():
            continue
        if "beat_notes_contains" in scope and scope["beat_notes_contains"].lower() not in n.lower():
            continue
        out.append(bi)
    return out


def _picks_in_scope(cut: dict, rule: dict, tier: int = 1) -> list[dict]:
    """Pick spans (render time) a rule protects."""
    scope = rule.get("scope", {})
    spans = pick_spans(cut, tier)
    if "clip" in scope:
        return [sp for sp in spans if base_clip(sp["key"].split("@")[0]) == scope["clip"]
                and scope.get("text_contains", "").lower() in sp["text"].lower()]
    beats = _beats_in_scope(cut, scope)
    if not beats:
        return []
    picks = [sp for sp in spans if sp["beat"] in beats]
    if scope.get("pick") == "last":
        return [p for b in beats for p in [x for x in picks if x["beat"] == b][-1:]]
    return picks


def _video_rows(cut: dict, beat_no: int) -> list[dict]:
    rows = []
    for v in cut["beats"][beat_no - 1].get("video", []):
        if "dest_in_s" in v:
            rows.append(v)
    return rows


def _rows_over(cut: dict, sp: dict) -> list[dict]:
    """Video rows (with destinations) that overlap a pick span by more than 0.1 s."""
    return [v for v in _video_rows(cut, sp["beat"]) if v["dest_out_s"] > sp["t0"] + 0.1 and v["dest_in_s"] < sp["t1"] - 0.1]


def _cover_rows(cut: dict, sp: dict) -> list[dict]:
    return [v for v in _rows_over(cut, sp) if v["type"] != "bare"]


def _legacy(cut: dict, picks: list[dict]) -> list[str]:
    """Picks whose beat has video rows without destinations: nothing can be said about their picture."""
    return [sp["id"] for sp in picks if not _video_rows(cut, sp["beat"]) and cut["beats"][sp["beat"] - 1].get("video")]


def walk_rules(cut: dict, doc: dict, tier: int = 1) -> dict[str, str]:
    """pick id -> source prefix that may never cover it (sync walk covered by another walk take)."""
    out = {}
    for r in resolve(doc)["active"]:
        if r["kind"] == "cover_only_named_detail" and r.get("never_from_prefix"):
            for sp in _picks_in_scope(cut, r, tier):
                out[sp["id"]] = r["never_from_prefix"]
    return out


# --------------------------------------------------------------- check cut
def check_cut(cut: dict, doc: dict, tier: int = 1, heard_text: str | None = None, profile=None) -> list[dict]:
    """Findings per active rule. `heard_text` (the render's transcript, lower
    case) lets word rules judge what plays; without it they judge the cut's
    intended text and say so. `profile` (film.py) names the placeholder cards."""
    res = resolve(doc)
    spans = pick_spans(cut, tier)
    used_clips = {base_clip(sp["key"].split("@")[0]) for sp in spans}
    intended = " ".join(sp["text"].lower() for sp in spans)
    text, basis = (heard_text, "render transcript") if heard_text is not None else (intended, "cut text")
    out = []
    for r in res["active"]:
        k = r["kind"]
        f = dict(rule=r["id"], kind=k, status=NOT_TESTABLE, evidence="")
        if k == "words_forbidden":
            hits = [w for w in r["words"] if re.search(rf"\b{re.escape(w)}\b", text)]
            f.update(status=BROKEN if hits else HONOURED, evidence=(f"heard in {basis}: {', '.join(hits)}" if hits else f"none of {r['words']} in {basis}"))
        elif k == "words_allowed_history":
            hits = [p for p in r.get("forbidden_phrases", []) if p in text]
            mentions = [sp["id"] for sp in spans if any(re.search(rf"\b{w}\b", sp["text"].lower()) for w in r["words"])]
            if hits:
                f.update(status=BROKEN, evidence=f"forbidden framing in {basis}: {', '.join(hits)}")
            elif mentions:
                f.update(status=REVIEW, evidence=f"mentions allowed as history in picks {', '.join(mentions)}; framing check on {basis}: clean")
            else:
                f.update(status=HONOURED, evidence="no mention")
        elif k == "clip_span_excluded":
            # one span (clip, in_s, out_s), or `spans`: a list of them (lines dropped by name); a pick that only shares an edge
            # with a span (the next line of the same take) does not touch it
            excl = r.get("spans") or [dict(clip=r["clip"], in_s=r["in_s"], out_s=r["out_s"])]
            bad = [sp["id"] for sp in spans for x in excl if base_clip(sp["key"].split("@")[0]) == x["clip"]
                   and sp["src_out"] > x["in_s"] + EDGE_TOL_S and sp["src_in"] < x["out_s"] - EDGE_TOL_S]
            where = f"{r['clip']} {r['in_s']}-{r['out_s']}s" if "spans" not in r else f"{len(excl)} excluded spans"
            f.update(status=BROKEN if bad else HONOURED, evidence=(f"picks inside {where}: {bad}" if bad else f"no pick touches {where}"))
        elif k in ("no_cover", "no_cover_pick"):
            picks = _picks_in_scope(cut, r, tier)
            if not picks:
                f.update(status=NOT_TESTABLE, evidence="scope matches no pick in this cut")
            else:
                cov = [(sp["id"], v.get("clip") or v.get("text", "")[:30], round(v["dest_in_s"], 1)) for sp in picks for v in _cover_rows(cut, sp)]
                legacy = _legacy(cut, picks)
                if legacy:
                    f.update(status=NOT_TESTABLE, evidence=f"rows without destinations under {legacy}; rebuild the cut")
                else:
                    f.update(status=BROKEN if cov else HONOURED,
                             evidence=(f"cover over protected picks: {cov}" if cov else f"bare under {[sp['id'] for sp in picks]}"))
        elif k == "cover_only_named_detail":
            picks = _picks_in_scope(cut, r, tier)
            rows = [(sp["id"], v.get("clip") or "", v.get("note", "")[:60]) for sp in picks for v in _cover_rows(cut, sp)]
            bad = [x for x in rows if r.get("never_from_prefix") and x[1].startswith(r["never_from_prefix"])]
            if not picks:
                f.update(status=NOT_TESTABLE, evidence="scope matches no pick")
            elif _legacy(cut, picks):
                f.update(status=NOT_TESTABLE, evidence=f"rows without destinations under {_legacy(cut, picks)}; rebuild the cut")
            elif bad:
                f.update(status=BROKEN, evidence=f"walk sync covered with another walk take: {bad}")
            elif rows:
                f.update(status=REVIEW, evidence=f"cover on walk beats, each must show a detail the words name: {rows}")
            else:
                f.update(status=HONOURED, evidence=f"walk beats bare: {[sp['id'] for sp in picks]}")
        elif k == "must_be_present":
            missing = [c for c in r["clips"] if c not in used_clips]
            f.update(status=BROKEN if missing else HONOURED, evidence=(f"no tier-{tier} pick from {missing}" if missing else f"picks present from all of {r['clips']}"))
        elif k == "clips_excluded":
            used = [c for c in r["clips"] if c in used_clips]
            f.update(status=BROKEN if used else HONOURED, evidence=(f"used: {used}" if used else f"none of {r['clips']} used"))
        elif k == "cover_required":
            picks = _picks_in_scope(cut, r, tier)
            ok = [sp["id"] for sp in picks if any(base_clip(v.get("clip") or "") in r["clips_any"] for v in _cover_rows(cut, sp))]
            bad = [sp["id"] for sp in picks if sp["id"] not in ok]
            if not picks:
                f.update(status=NOT_TESTABLE, evidence="scope matches no pick")
            elif _legacy(cut, picks):
                f.update(status=NOT_TESTABLE, evidence=f"rows without destinations under {_legacy(cut, picks)}; rebuild the cut")
            else:
                f.update(status=BROKEN if bad else HONOURED, evidence=(f"{bad} not covered by any of {r['clips_any']}" if bad else f"{ok} covered by {r['clips_any']}"))
        elif k == "placeholder_required":
            have = set()
            for b in cut["beats"]:
                for v in b.get("video", []):
                    if v["type"] == "placeholder":
                        try:
                            have.add(next(kk for kk in r["keys"] if placeholder_file(v.get("text", ""), profile).endswith(placeholder_file(kk, profile))))
                        except (KeyError, StopIteration):
                            pass
            missing = [kk for kk in r["keys"] if kk not in have]
            f.update(status=BROKEN if missing else REVIEW,
                     evidence=(f"placeholder cards missing for {missing}" if missing else f"all cards present ({sorted(have)}); still missing material for delivery"))
        elif k == "open":
            f.update(status=OPEN, evidence=r["decision"])
        elif k == "talking_cover_exempt":
            f.update(status=NOT_TESTABLE, evidence=f"named exception to the talking-cover rule, applied by the proposer, builder and scorer: {r.get('clips')}")
        elif k == "note":
            f.update(status=NOT_TESTABLE, evidence="a recorded decision about method, not a testable property of the cut: " + r["decision"][:80])
        elif k == "delivery_loudness":
            f.update(status=NOT_TESTABLE, evidence=f"measured on the master by videoeasy.deliver (target {r['target_lufs']} LUFS, {r['max_true_peak_dbtp']} dBTP)")
        out.append(f)
    return out


# ---------------------------------------------------------- check read-back
def check_readback(rb: dict, plan: dict, cut: dict, doc: dict, tier: int = 1, profile=None) -> list[dict]:
    """The same rules against what Resolve actually holds. Protected spans come
    from the plan's audio rows (pick -> record frames); the read-back's video
    items on tracks above V1 are the cover that exists."""
    res = resolve(doc)
    fps = float(rb.get("fps") or 24.0)
    rec_of = {a["pick"]: (a["rec"], a["rec"] + a["dur"]) for a in plan["audio"]}
    # the A-roll's own picture sits on V1/V2 beside its sound; everything else with a picture is cover
    aroll = [(a["name"], a.get("track", 1), a["rec"] - a.get("head", 0)) for a in plan["audio"]]
    cover = [i for i in rb["items"] if i["type"] == "video"
             and not any(i["name"] == n and i["track"] == t and abs(i["rec"] - r) <= 3 for n, t, r in aroll)]
    audio_names = {i["name"] for i in rb["items"] if i["type"] == "audio"}
    out = []
    for r in res["active"]:
        k = r["kind"]
        f = dict(rule=r["id"], kind=k, status=NOT_TESTABLE, evidence="")
        if k in ("no_cover", "no_cover_pick", "cover_required", "cover_only_named_detail"):
            picks = [sp for sp in _picks_in_scope(cut, r, tier) if sp["id"] in rec_of]
            if not picks:
                f.update(evidence="scope matches no laid pick")
                out.append(f)
                continue
            hits = []
            for sp in picks:
                a, b = rec_of[sp["id"]]
                for i in cover:
                    if i["rec"] < b - 2 and i["rec"] + i["dur"] > a + 2:
                        hits.append((sp["id"], i["name"], i["track"], round(i["rec"] / fps, 1)))
            if k == "cover_required":
                ok = [h for h in hits if base_clip(h[1].split(".")[0]) in r["clips_any"]]
                f.update(status=HONOURED if ok else BROKEN, evidence=(f"cover found: {ok}" if ok else f"no item from {r['clips_any']} over {[p['id'] for p in picks]}"))
            elif k == "cover_only_named_detail":
                bad = [h for h in hits if r.get("never_from_prefix") and h[1].startswith(r["never_from_prefix"])]
                f.update(status=BROKEN if bad else (REVIEW if hits else HONOURED),
                         evidence=(f"other walk take over sync: {bad}" if bad else (f"cover items on walk beats: {hits}" if hits else "walk beats bare")))
            else:
                f.update(status=BROKEN if hits else HONOURED, evidence=(f"cover items over protected picks: {hits}" if hits else f"no cover item over {[p['id'] for p in picks]}"))
        elif k == "must_be_present":
            missing = [c for c in r["clips"] if f"{c}.MOV" not in audio_names]
            f.update(status=BROKEN if missing else HONOURED, evidence=(f"no audio item from {missing}" if missing else "audio items present"))
        elif k == "clips_excluded":
            used = [c for c in r["clips"] if f"{c}.MOV" in audio_names]
            f.update(status=BROKEN if used else HONOURED, evidence=(f"audio items from {used}" if used else "none used"))
        elif k == "placeholder_required":
            names = {i["name"] for i in cover}
            missing = [kk for kk in r["keys"] if placeholder_file(kk, profile) not in names]
            f.update(status=BROKEN if missing else REVIEW, evidence=(f"cards missing on the timeline: {missing}" if missing else "all cards on the timeline; still missing material"))
        elif k == "open":
            f.update(status=OPEN, evidence=r["decision"])
        else:
            f.update(evidence="a word rule: judged on the transcript, not the timeline")
        out.append(f)
    return out


def summary(findings: list[dict]) -> str:
    c: dict[str, int] = {}
    for f in findings:
        c[f["status"]] = c.get(f["status"], 0) + 1
    return ", ".join(f"{k} {v}" for k, v in sorted(c.items()))


def protected_picks(cut: dict, doc: dict, tier: int = 1) -> dict[str, str]:
    """pick id -> rule id for every pick a no-cover rule protects (for the scorer and the proposer)."""
    out = {}
    for r in resolve(doc)["active"]:
        if r["kind"] in ("no_cover", "no_cover_pick"):
            for sp in _picks_in_scope(cut, r, tier):
                out[sp["id"]] = r["id"]
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse
    import sys
    ap = argparse.ArgumentParser(description="check a cut (and optionally a lay-in read-back) against editorial/intent.json")
    ap.add_argument("--film", required=True)
    ap.add_argument("--cut", required=True)
    ap.add_argument("--readback", default=None, help="layin read-back json")
    ap.add_argument("--plan", default=None, help="layin plan json (needed with --readback)")
    ap.add_argument("--tier", type=int, default=1)
    args = ap.parse_args(argv)
    film = Path(args.film)
    doc = load(film)
    profile = load_film(film)
    cut = json.loads(Path(args.cut if Path(args.cut).is_absolute() else film / args.cut).read_text(encoding="utf-8"))
    res = resolve(doc)
    print(f"active rules {len(res['active'])}, retired {len(res['retired'])}: " + "; ".join(f"{x['id']} ({x['why']})" for x in res["retired"]))
    for f in check_cut(cut, doc, args.tier, profile=profile):
        print(f"  cut      {f['status']:12} {f['rule']:32} {f['evidence']}")
    if args.readback:
        rb = json.loads(Path(args.readback).read_text(encoding="utf-8"))
        pl = json.loads(Path(args.plan).read_text(encoding="utf-8"))
        for f in check_readback(rb, pl, cut, doc, args.tier, profile=profile):
            print(f"  readback {f['status']:12} {f['rule']:32} {f['evidence']}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
