"""Propose and verify B-roll per spoken pick, with local models only.

Two stages:
  propose  - a local text model (Ollama) reads one beat's picks plus the
             compact B-roll catalogue from the bible and names candidate
             shots per pick, or says the pick should stay on the speaker.
  verify   - the local vision model (LM Studio) sees four graded frames of
             each candidate next to the pick's words and scores the match
             0-3, naming the usable frames.
The result is a plan: one chosen shot (or bare) per pick with a verified
score, which `cutbuild` turns into the video rows of the next cut.

Usage:
    uv run python -m videoeasy.brollmatch \
        --film data/films/<film> --cut editorial/cut-v2.json \
        --out editorial/broll-plan-v3.json [--beats 1,2] [--skip-verify]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from . import moves as moves_mod
from . import steadiness as steady_mod
from .cuteval import pick_spans
from .evalrun import INVALID, OK, UNCHECKED, fingerprint, source_hash, text_hash
from .film import Film, load_film
from .frames import sample_times
from .sources import drone_prefixes, never_cover, sync_walk
from .vlm import chat_vision, extract_json
from .textmodel import text_model, text_url

OLLAMA_URL = text_url()        # VIDEOEASY_TEXT_URL overrides (textmodel.py)
OLLAMA_MODEL = text_model()    # VIDEOEASY_TEXT_MODEL overrides
VISION_URL = "http://localhost:1234/v1"
VISION_MODEL = "google/gemma-4-26b-a4b"

# Never proposed as cover: units of a role marked `never_cover` in the film's profile (the sit-down itself and the
# presenter pieces), so the proposer cannot cover a pick with its own A-roll.


# ------------------------------------------------------------------ catalogue
NOTES_CHARS = 400
HEARD_WORDS = 40


DEFAULT_SYNTHESIS = "out/bible-synthesis.md"


def synthesis_path(film: Path, synthesis: str | None = None) -> Path:
    """The hand-curated synthesis a run reads: `--synthesis` relative to the film dir, else the first build.
    Later builds sit beside the first (out/bible-synthesis-v2.md, 28 Sep 2026); none replaces another."""
    return Path(film) / (synthesis or DEFAULT_SYNTHESIS)


SOURCE_ID = r"\b[A-Z][A-Z0-9]*_?\d{3,}(?!\d)"   # camera-style clip names: CAM0031, C0012, DJI_1001 (a window suffix is dropped)


def synthesis_links(film: Path, synthesis: str | None = None, ids: set[str] | None = None) -> dict[str, list[str]]:
    """source id -> the synthesis sections (heading text) that cite it. Read-only over the hand-curated
    synthesis; the relationship is 'this section talks about this source', nothing more. With `ids` (the
    film's source ids) only those are looked for; without, any camera-style clip name."""
    import re
    p = synthesis_path(film, synthesis)
    if not p.exists():
        return {}
    pat = re.compile("|".join(rf"\b{re.escape(i)}(?!\d)" for i in sorted(ids, key=len, reverse=True)) if ids else SOURCE_ID)
    links: dict[str, list[str]] = {}
    heading = "(top)"
    for line in p.read_text(encoding="utf-8").splitlines():
        if line.startswith("#"):
            heading = line.lstrip("#").strip()[:60]
            continue
        if ids is not None and not ids:
            continue
        for sid in set(pat.findall(line)):
            if heading not in links.setdefault(sid, []):
                links[sid].append(heading)
    return links


def heard_in(transcripts: dict, source: str, in_s: float, out_s: float, max_words: int = HEARD_WORDS) -> dict:
    """What the unit's own audio recorded: words inside the range, or the reason none are known. `status` is
    'speech' | 'silent' (transcribed, nothing said) | 'not_transcribed' | 'no_audio'."""
    tr = transcripts.get(source)
    if not tr:
        return dict(status="not_transcribed", words=[], seconds=None)
    if (tr.get("quality") or {}).get("status") == "silent":
        return dict(status="no_audio", words=[], seconds=0.0)
    words, secs = [], 0.0
    for seg in tr.get("segments", []):
        for w in seg.get("words", []):
            a, b = max(float(w["s"]), in_s), min(float(w["e"]), out_s)
            if b > a:
                words.append(w["w"].strip())
                secs += b - a
    return dict(status="speech" if words else "silent", words=words[:max_words], seconds=round(secs, 1), truncated=len(words) > max_words)


def load_catalogue(film: Path, synthesis: str | None = None, profile: Film | None = None) -> dict[str, dict]:
    profile = profile or load_film(film)
    ann = json.load(open(film / "out/annotations.json"))
    shots = {s["shot_id"]: s for s in json.load(open(film / "out/shots.json"))}
    segs = {s["shot_id"]: s for s in json.load(open(film / "out/segments.json"))}
    parents_with_windows = {s["parent_shot_id"] for s in segs.values()}
    tr_path = film / "out/transcripts.json"
    transcripts = json.load(open(tr_path)) if tr_path.exists() else {}
    st_path = film / "out/steadiness.json"
    steady = (json.load(open(st_path)).get("units") or {}) if st_path.exists() else {}
    links = synthesis_links(film, synthesis, {m["source_id"] for m in [*shots.values(), *segs.values()]})
    cat: dict[str, dict] = {}
    for sid, a in ann.items():
        if never_cover(sid, profile):
            continue
        if "error" in a or not a.get("subject"):
            continue
        if sid in parents_with_windows:
            continue  # the windows carry the tags for long takes
        meta = segs.get(sid) or shots.get(sid)
        if not meta:
            continue
        cat[sid] = dict(
            id=sid, source=meta["source_id"], source_path=meta.get("source_path"), role=meta["role"], in_s=meta["in_s"], out_s=meta["out_s"],
            duration_s=round(meta["duration_s"], 1), subject=a["subject"], movement=a.get("movement", ""),
            shot_type=a.get("shot_type", ""), mood=a.get("mood", ""), themes=a.get("storytelling_themes", []),
            hold=a.get("hold_seconds"), notes=(a.get("cut_notes") or "")[:NOTES_CHARS], people=a.get("people", ""),
            heard=heard_in(transcripts, meta["source_id"], meta["in_s"], meta["out_s"]),
            shake_pct=(steady.get(sid) or {}).get("shake_pct"), synthesis=links.get(meta["source_id"], []),
            steady=None,  # set by steadiness.apply_runs at the caller's limit: None = whole unit, [runs] = steady parts only
        )
    return cat


def catalogue_text(cat: dict[str, dict], exclude: set[str]) -> str:
    """One line per unit for the proposer. What the tags SAW and what the clip's audio HEARD are separate fields:
    a spoken topic is not a visible fact (Codex clip-context review, 26 Sep 2026)."""
    lines = []
    for sid, c in cat.items():
        used = " [ALREADY USED]" if sid in exclude else ""
        shake = f", shake {c['shake_pct']}%" if c.get("shake_pct") is not None else ""
        if c.get("moves"):
            shake += f"; CLEAN DRONE MOVES ONLY {moves_mod.fmt_clean(c['moves'])} (the rest of the flight turns, hesitates or shakes)"
        elif c.get("steady"):
            shake += f"; STEADY PARTS ONLY {steady_mod.fmt_runs(c['steady'])} (the rest shakes)"
        h = c.get("heard") or {}
        if h.get("status") == "speech":
            heard = f" Heard on its audio (said, not necessarily visible): \"{' '.join(h['words'])}{'…' if h.get('truncated') else ''}\" ({h['seconds']} s of speech)."
        elif h.get("status") == "silent":
            heard = " Audio: no speech."
        elif h.get("status") == "no_audio":
            heard = " Audio: silent track."
        else:
            heard = " Audio: not checked."
        people = f" People: {c['people']}." if c.get("people") else ""
        syn = f" Synthesis cites it under: {'; '.join(c['synthesis'][:3])}." if c.get("synthesis") else ""
        lines.append(f"{sid} ({c['duration_s']:.0f}s, {c['shot_type']}, hold {c['hold']}s{shake}){used}: {c['subject']}.{people} "
                     f"Movement: {c['movement']}. Mood: {c['mood']}. Themes: {', '.join(c['themes'])}. Notes: {c['notes']}{heard}{syn}")
    return "\n".join(lines)


# -------------------------------------------------------------------- propose
PROPOSE_PROMPT = """You are the picture editor on a short documentary in which the picture shows what the speaker names.{film} You are assigning cover for one beat of the interview. Below are the spoken picks of this beat, in order, each with an id, then the catalogue of every available B-roll shot with the tags a vision model wrote for it.

Beat: {title}
Editor's note for this beat: {note}
Placeholder cards available (material promised but not shot; use only where the words name exactly that material): {placeholders}

PICKS
{picks}
{review}

CATALOGUE
{catalogue}

For every pick decide what the audience should be looking at while these words are spoken:
- "illustrate": a shot that shows the very thing, place, action or person named in the words.
- "support": no shot shows the named thing, but a shot of the right place, mood or activity keeps the picture alive.
- "bare": the words are a personal admission, a joke, a reaction or a moment between the two speakers, and the speaker's face is the picture. Also use bare where the editor's note says to.
Rules: never name a shot that is not in the catalogue; avoid shots marked ALREADY USED unless nothing else fits; prefer shots whose notes do not warn of focus or blur problems; a catalogue line's "Heard on its audio" is what the camera recorded, not what is visible: it tells you the activity and the moment, never that a named plant or person is in frame; "People" says who is visible, so match the speaker's subject (a speaker's own hands for that speaker's method, never another's) and never show a face while another voice speaks; a line marked STEADY PARTS ONLY is usable only inside the listed times (the rest of it shakes), so weigh the length of those parts against the pick; a line marked CLEAN DRONE MOVES ONLY is usable only inside the listed times, each named by what the camera does there, so choose the move whose direction and pace suit the words; one shot can cover a pick of up to about 12 seconds, so give longer picks two or three candidates in the order they should play; a pick under 3 seconds may share its neighbour's shot (say "same as previous").

Answer with one JSON object: {{"picks": [ {{"pick": "<id>", "need": "illustrate|support|bare", "candidates": [ {{"shot": "<catalogue id or PH:<placeholder key>>", "why": "at most 15 words"}} ] }} ]}}
Candidates ordered best first, at most 4 per pick, empty list when need is bare."""


def review_text(picks: list[dict], feedback: dict[str, dict]) -> str:
    lines = []
    for p in picks:
        fb = feedback.get(p.get("key"))
        if not fb or not fb.get("bad"):
            continue
        for clip, score, rather in fb["bad"]:
            what = f"shot {clip}" if clip else "the bare two-shot"
            lines.append(f"- {p['id']}: last cut used {what}, scored {score}/3. Reviewer wanted: {rather or 'something that shows the words'}")
    if not lines:
        return ""
    return "REVIEW OF THE LAST CUT (a vision model scored the last cut's picture against these same words; do not propose a shot named here as scoring 0 or 1 again for that pick)\n" + "\n".join(lines)


def propose_template(context: str = "") -> str:
    """The proposer's prompt with the film's register line (film.yaml `context`) in place."""
    film = f" The film: {context.strip()}" if context and context.strip() else ""
    return PROPOSE_PROMPT.replace("{film}", film.replace("{", "{{").replace("}", "}}"))


def propose_beat(beat: dict, picks: list[dict], cat: dict, used: set[str], placeholders: dict[str, str],
                 feedback: dict[str, dict] | None = None, url: str = OLLAMA_URL, model: str = OLLAMA_MODEL, context: str = "") -> dict:
    ptxt = "\n".join(f"{p['id']} ({p['duration_s']:.1f}s): \"{p['text']}\"" for p in picks)
    ph = "; ".join(f"PH:{k} = {v}" for k, v in placeholders.items()) or "none"
    prompt = propose_template(context).format(title=beat["title"], note=beat.get("notes") or "none", placeholders=ph,
                                   picks=ptxt, review=review_text(picks, feedback or {}), catalogue=catalogue_text(cat, used))
    r = httpx.post(f"{url}/api/chat", json={"model": model, "stream": False, "think": False, "format": "json",
                                            "options": {"temperature": 0.2, "num_ctx": 40000},
                                            "messages": [{"role": "user", "content": prompt}]}, timeout=1200)
    r.raise_for_status()
    out = json.loads(r.json()["message"]["content"])
    # drop hallucinated ids
    for p in out.get("picks", []):
        p["candidates"] = [c for c in p.get("candidates", [])
                           if isinstance(c, dict) and (c.get("shot") in cat or str(c.get("shot", "")).startswith("PH:"))]
    return out


# --------------------------------------------------------------------- verify
VERIFY_SYSTEM = (
    "You are a documentary picture editor. You see four frames sampled evenly across one B-roll shot, in order, "
    "and the words that would be spoken over it. Judge only what is visible. Answer with a single JSON object."
)
VERIFY_USER = """Words to be spoken over this shot: "{words}"
The shot is {dur:.0f} s long; you see {n} frames from it, in order, taken at {positions} of the way through. Shot tags say: {subject}.

Return JSON:
- shows: what the frames show, at most 20 words
- score: 3 if the shot shows the thing, place, action or person the words name; 2 if it is the right place, activity or mood without showing the named thing; 1 if it is generic coverage; 0 if it contradicts the words, shows a different person doing something unrelated, or is unwatchable (blur, shake, blown out, black)
- usable: list of frame numbers (1-{n}) that are sharp, steady and well exposed
- reason: at most 20 words"""
VERIFY_PROMPT_HASH = text_hash(VERIFY_SYSTEM, VERIFY_USER)


def candidate_frames(c: dict, fdir: Path) -> list[dict]:
    """Up to four of the cached ingest frames of a shot, spread over whatever
    the cache holds (4, 6 or 8 files), each with the source time it was
    sampled at. The verdict records these so nothing downstream has to guess
    a frame's position from its ordinal."""
    files = sorted(fdir.glob("f0[0-9].jpg"))
    n = len(files)
    if n == 0:
        return []
    times = sample_times(c["in_s"], c["out_s"], n)
    span = max(c["out_s"] - c["in_s"], 1e-6)
    runs = c.get("steady")
    if runs:  # a shaky unit: only frames inside its steady parts can certify anything
        ok = [i for i in range(n) if any(r["t0"] - 1e-6 <= times[i] < r["t1"] for r in runs)]
        idx = sorted({ok[round(i * (len(ok) - 1) / 3)] for i in range(4)}) if len(ok) > 4 else ok
    else:
        idx = sorted({round(i * (n - 1) / 3) for i in range(4)}) if n > 4 else list(range(n))
    return [dict(file=str(files[i]), t=round(times[i], 2), pos=round((times[i] - c["in_s"]) / span, 2)) for i in idx]


def validate_verdict(j, n_frames: int) -> dict:
    """Normalise one cover verdict: integer score 0-3 and usable frame numbers
    within the frames actually sent, or invalid with no score."""
    if not isinstance(j, dict):
        return {"status": INVALID, "score": None, "why": "response is not an object"}
    out = dict(j)
    problems = []
    try:
        sc = int(j.get("score"))
    except (TypeError, ValueError):
        sc = None
        problems.append("score missing or not an integer")
    if sc is not None and not 0 <= sc <= 3:
        problems.append(f"score {sc} out of range 0-3")
        sc = None
    raw = j.get("usable", [])
    if not isinstance(raw, list):
        raw = [raw]
    usable = []
    for u in raw:
        try:
            k = int(u)
        except (TypeError, ValueError):
            continue
        if 1 <= k <= n_frames:
            usable.append(k)
    if raw and not usable and n_frames:
        problems.append(f"usable {raw!r} names no frame in 1-{n_frames}")
    out["usable"] = sorted(set(usable))
    out["score"] = sc
    out["status"] = INVALID if problems else OK
    if problems:
        out["why"] = "; ".join(problems)
    return out


def verify(cand_id: str, words: str, cat: dict, film: Path, url: str = VISION_URL, model: str = VISION_MODEL) -> dict:
    c = cat[cand_id]
    fu = candidate_frames(c, film / "work/frames" / cand_id)
    if len(fu) < 2:
        where = " inside its steady parts" if c.get("steady") else ""
        return {"status": UNCHECKED, "score": None, "error": f"fewer than two cached frames for {cand_id}{where}", "frames_used": fu}
    frames = [Path(f["file"]).read_bytes() for f in fu]
    positions = ", ".join(f"{int(round(f['pos'] * 100))}%" for f in fu)
    user = VERIFY_USER.format(words=words, dur=c["duration_s"], n=len(fu), positions=positions, subject=c["subject"])
    base = dict(frames_used=fu, model=model, prompt_sha=VERIFY_PROMPT_HASH)
    try:
        j = extract_json(chat_vision(url, model, VERIFY_SYSTEM, user, frames, max_tokens=1200, temperature=0.2))
    except Exception as e:  # noqa: BLE001
        return dict(base, status=UNCHECKED, score=None, error=str(e)[:200])
    out = validate_verdict(j, len(fu))
    out.update(base)
    return out


def verdict_ok(v: dict | None) -> bool:
    """An accepted verdict: status ok, or (legacy plans) an integer score with no status field."""
    if not v:
        return False
    st = v.get("status")
    if st is None:
        return isinstance(v.get("score"), int)
    return st == OK and isinstance(v.get("score"), int)


# ----------------------------------------------------------------------- plan
def choose(pick: dict, proposal: dict, verdicts: dict[str, dict], uses: dict[str, int] | None = None,
           barred: set[str] | None = None) -> dict:
    """Pick the best candidate among those with an accepted verdict. A shot
    already used twice loses 0.75 per extra use; a shot the last review scored
    0-1 for this pick is barred. A candidate whose verdict is missing, errored
    or invalid is not eligible: when nothing eligible remains the row is
    `unchecked`, with no shot and no score, never a middling guess."""
    need = proposal.get("need", "support")
    all_c = proposal.get("candidates", [])
    cands = [c for c in all_c if c["shot"] not in (barred or set())]
    if need == "bare":
        return dict(pick=pick["id"], need="bare", shot=None, score=None, status=OK)
    if not cands:
        return dict(pick=pick["id"], need="bare", shot=None, score=None, status=OK,
                    note="every candidate was barred by the last review" if all_c else "proposer named no candidate")
    ranked = []
    pending = []
    for i, c in enumerate(cands):
        sid = c["shot"]
        if sid.startswith("PH:"):
            ranked.append((2.5, -i, sid, {"reason": "placeholder", "usable": [], "status": OK}))
            continue
        v = verdicts.get(sid)
        if not verdict_ok(v):
            pending.append((sid, (v or {}).get("status") or "unverified"))
            continue
        penalty = 0.75 * max(0, (uses or {}).get(sid, 0) - 1)
        ranked.append((v["score"] - penalty, -i, sid, v))
    if not ranked:
        return dict(pick=pick["id"], need=need, shot=None, score=None, status=UNCHECKED,
                    note="no candidate has an accepted verdict: " + "; ".join(f"{s} {w}" for s, w in pending))
    ranked.sort(reverse=True)
    best = ranked[0]
    if best[0] < 1:
        return dict(pick=pick["id"], need="bare", shot=None, score=best[0], status=OK, note="all verified candidates scored 0")
    return dict(pick=pick["id"], need=need, shot=best[2], score=best[0], usable=best[3].get("usable"),
                reason=best[3].get("reason", ""), alternatives=[r[2] for r in ranked[1:]], status=OK,
                unverified=[s for s, _ in pending])


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--film", required=True)
    ap.add_argument("--cut", required=True, help="cut json, relative to --film")
    ap.add_argument("--out", required=True, help="plan json, relative to --film")
    ap.add_argument("--beats", default=None, help="comma list of beat numbers (1-based)")
    ap.add_argument("--tier", type=int, default=1)
    ap.add_argument("--skip-verify", action="store_true")
    ap.add_argument("--verify-only", action="store_true", help="reuse the proposals already in --out; run only the vision check")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--eval", default=None, help="cuteval json of the previous render of --cut: keeps 3s, bars 0-1s")
    ap.add_argument("--max-shake", type=float, default=None, help="camera shake limit, %% of frame width (default steadiness.MAX_SHAKE_PCT)")
    ap.add_argument("--synthesis", default=None, help=f"synthesis file relative to --film (default {DEFAULT_SYNTHESIS})")
    args = ap.parse_args(argv)
    film = Path(args.film)
    profile = load_film(film)
    cut = json.load(open(film / args.cut))
    cat = load_catalogue(film, args.synthesis, profile)
    # cover rules measured without a model: a unit with camera shake over the limit, or with transcribed speech
    # inside it (someone on screen talking under another pick's audio), is not cover and the proposer never sees it
    from . import speech
    max_shake = steady_mod.MAX_SHAKE_PCT if args.max_shake is None else args.max_shake
    transcripts = json.load(open(film / "out/transcripts.json")) if (film / "out/transcripts.json").exists() else {}
    steady = steady_mod.load(film / "out")
    from . import intent as intent_mod
    intent_doc = intent_mod.load(film) if (film / "editorial/intent.json").exists() else None
    ineligible = steady_mod.apply_runs(cat, steady, max_shake)
    drone_moves = moves_mod.load(film / "out")
    ineligible.update(moves_mod.apply_clean(cat, drone_moves, drone_prefixes(profile)))   # drone units: only their clean moves and holds (29 Sep 2026)
    ineligible.update(speech.talking_units(transcripts, [dict(id=k, in_s=c["in_s"], out_s=c["out_s"], people=c.get("people")) for k, c in cat.items()],
                                           exempt=intent_mod.talking_exempt(intent_doc)))
    cat_visible = {k: v for k, v in cat.items() if k not in ineligible}
    print(f"cover rules: {len(ineligible)} units are not cover ({sum(1 for r in ineligible.values() if 'shake' in r)} shake over {max_shake}%, "
          f"{sum(1 for r in ineligible.values() if 'speech' in r)} talking, {sum(1 for r in ineligible.values() if r.startswith('drone'))} drone without a clean move); "
          f"steadiness {'measured' if steady else 'NOT MEASURED (run videoeasy.steadiness)'}, drone moves {'measured' if drone_moves else 'NOT MEASURED (run videoeasy.moves)'}",
          file=sys.stderr)
    catalogue_sha = text_hash(json.dumps(cat, sort_keys=True))
    protected = intent_mod.protected_picks(cut, intent_doc) if intent_doc else {}
    never_from = intent_mod.walk_rules(cut, intent_doc) if intent_doc else {}
    print(f"catalogue: {len(cat)} shots/windows", file=sys.stderr)
    want = {int(x) for x in args.beats.split(",")} if args.beats else None
    out_path = film / args.out
    plan = json.load(open(out_path)) if out_path.exists() else {"beats": {}}
    used: set[str] = set()
    uses: dict[str, int] = {}
    feedback: dict[str, dict] = {}
    if args.eval:
        ev = json.load(open(film / args.eval))
        ev_cut_path = Path(ev["cut"]) if ev.get("cut") else None
        if ev_cut_path is not None and not ev_cut_path.exists():
            ev_cut_path = film / ev_cut_path   # scorecards record the path as it was given to cuteval
        ev_cut = json.load(open(ev_cut_path)) if ev_cut_path is not None else cut
        for sp in pick_spans(ev_cut, args.tier):
            # only accepted judgments feed back; an errored or invalid stretch is silence, not a verdict.
            # Stretches carry the pick they sit under (cuteval ≥ milestone two); older scorecards fall back to time overlap.
            keyed = any("pick" in st for st in ev["stretches"])
            over = [st for st in ev["stretches"] if verdict_ok(st.get("vlm")) and
                    (st.get("pick") == sp["id"] if keyed else st["t1"] > sp["t0"] + 0.2 and st["t0"] < sp["t1"] - 0.2)]
            if not over:
                continue
            bad = [(st["clip"], st["vlm"]["score"], st["vlm"].get("would_rather_see", "")) for st in over if st["vlm"]["score"] <= 1]
            good = [st for st in over if st["vlm"]["score"] == 3 and st["kind"] == "video"]
            keep = None
            if not bad and good and len(over) == 1:
                keep = good[0]["clip"]
            feedback[sp["key"]] = {"bad": bad, "keep": keep, "barred": {c for c, _, _ in bad if c}}
        # a shot scored 0 for blur or shake is unwatchable under any words: bar it everywhere
        faulty = {st["clip"] for st in ev["stretches"] if st.get("clip") and verdict_ok(st.get("vlm")) and st["vlm"]["score"] == 0
                  and any(f in ("motion blur", "shaky", "soft focus", "dead frame") for f in (st["vlm"].get("technical") or []))}
        for fb in feedback.values():
            fb["barred"] |= faulty
        for sp in pick_spans(ev_cut, args.tier):
            feedback.setdefault(sp["key"], {"bad": [], "keep": None, "barred": set()})["barred"] |= faulty
        if faulty:
            print(f"barred film-wide for technical faults: {sorted(faulty)}", file=sys.stderr)
        print(f"feedback: {sum(1 for f in feedback.values() if f['keep'])} picks kept, "
              f"{sum(1 for f in feedback.values() if f['bad'])} picks with a 0-1 stretch", file=sys.stderr)
    for bi, beat in enumerate(cut["beats"], 1):
        picks = [dict(a, id=f"b{bi}p{j}", key=f"{a['clip']}@{a['in_s']:.1f}") for j, a in enumerate(beat["audio"], 1) if a.get("tier", 1) <= args.tier]
        if not picks:
            continue
        if want and bi not in want:
            for row in plan["beats"].get(str(bi), {}).get("plan", []):
                if row.get("shot"):
                    used.add(row["shot"]); uses[row["shot"]] = uses.get(row["shot"], 0) + 1
            continue
        # numbered among the placeholder rows only, which is how cutbuild resolves PH:phN
        placeholders = {f"ph{k}": v["text"] for k, v in enumerate([v for v in beat["video"] if v["type"] == "placeholder"], 1)}
        if args.verify_only and str(bi) in plan["beats"]:
            by_pick = {r["pick"]: {"need": r.get("proposer_need") or "support", "candidates": r.get("candidates", []), "kept": r.get("kept", False)}
                       for r in plan["beats"][str(bi)]["plan"]}
        else:
            prop = propose_beat(beat, picks, cat_visible, used, placeholders, feedback, context=profile.context)
            by_pick = {p.get("pick"): p for p in prop.get("picks", []) if isinstance(p, dict)}
        for p in picks:  # a pick whose only stretch scored 3 last time keeps its shot
            kp = feedback.get(p["key"], {}).get("keep")
            if kp and kp in cat and kp not in ineligible:
                by_pick[p["id"]] = {"need": "illustrate", "candidates": [{"shot": kp, "why": "kept: scored 3 in the last review"}], "kept": True}
        # verify every distinct (candidate, pick) pair
        jobs = []
        verdicts: dict[str, dict[str, dict]] = {p["id"]: {} for p in picks}
        for p in picks:
            pr = by_pick.get(p["id"], {})
            if pr.get("kept"):
                verdicts[p["id"]][pr["candidates"][0]["shot"]] = {"status": OK, "score": 3, "usable": [], "reason": "kept from the last review",
                                                                  "evidence": f"cuteval {args.eval} pick {p['key']}"}
                continue
            for c in pr.get("candidates", []):
                if c["shot"] in cat:
                    jobs.append((p, c["shot"]))
        if not args.skip_verify and jobs:
            def run(job):
                p, sid = job
                verdicts[p["id"]][sid] = verify(sid, p["text"], cat, film)
                v = verdicts[p["id"]][sid]
                print(f"  beat {bi} {p['id']} {sid:14} -> {v.get('status')} {v.get('score')} {v.get('reason', v.get('error', v.get('why', '')))[:60]}", file=sys.stderr)
            with ThreadPoolExecutor(args.workers) as ex:
                list(ex.map(run, jobs))
        rows = []
        no_cover = "no cover" in (beat["title"] + " " + (beat.get("notes") or "")).lower()
        for p in picks:
            pr = by_pick.get(p["id"], {"need": "support", "candidates": []})
            if no_cover or p["id"] in protected:  # the recorded decision is a rule, not a hint
                pr = {"need": "bare", "candidates": [], "rule": protected.get(p["id"], "beat note: no cover")}
            pre = never_from.get(p["id"])
            walk_beat = "walking interlude" in (beat.get("notes") or "").lower()

            def barred_here(shot: str) -> bool:
                # sync sound of people on the move: never cover it with a different take of the same kind
                return bool(shot) and not shot.startswith("PH:") and (bool(pre and shot.startswith(pre)) or (walk_beat and sync_walk(shot, profile)))
            if pre or walk_beat:
                pr = dict(pr, candidates=[c for c in pr.get("candidates", []) if not barred_here(str(c.get("shot", "")))])
            if pr.get("need") != "bare" and not pr.get("candidates") and rows and rows[-1].get("shot") \
                    and not barred_here(rows[-1]["shot"]):
                # "same as previous": inherit the neighbour's shot and verdicts
                prev = rows[-1]
                pr = {"need": prev["need"], "candidates": [{"shot": prev["shot"], "why": "same as previous"}]}
                verdicts[p["id"]] = {prev["shot"]: dict(prev.get("verdicts", {}).get(prev["shot"], {}), score=prev.get("score"))}
            row = choose(p, pr, verdicts[p["id"]], uses, (feedback.get(p["key"], {}).get("barred") or set()) | set(ineligible))
            dropped = [c["shot"] for c in pr.get("candidates", []) if c.get("shot") in ineligible]
            if dropped:
                row["not_cover"] = {s: ineligible[s] for s in dropped}
            row.update(text=p["text"], duration_s=p["duration_s"], proposer_need=pr.get("need"),
                       candidates=pr.get("candidates", []), verdicts=verdicts[p["id"]], kept=bool(pr.get("kept")))
            if row.get("shot") and not row["shot"].startswith("PH:"):
                used.add(row["shot"]); uses[row["shot"]] = uses.get(row["shot"], 0) + 1
            rows.append(row)
        plan["beats"][str(bi)] = {"title": beat["title"], "plan": rows}
        plan["cover_rules"] = dict(max_shake_pct=max_shake, steadiness_measured=steady is not None, drone_moves_measured=drone_moves is not None,
                                   transcribed_sources=len(transcripts), ineligible=ineligible)
        syn_p = synthesis_path(film, args.synthesis)
        plan["synthesis"] = dict(path=str(syn_p), sha=fingerprint(syn_p) if syn_p.exists() else None)
        plan["manifest"] = plan_manifest(args, cut, catalogue_sha, film, plan, profile)
        out_path.write_text(json.dumps(plan, indent=1))
        covered = sum(r["duration_s"] for r in rows if r["shot"])
        total = sum(r["duration_s"] for r in rows)
        unchecked = sum(1 for r in rows if r.get("status") == UNCHECKED)
        print(f"beat {bi} {beat['title'][:40]}: {len(rows)} picks, cover {covered:.0f}/{total:.0f}s, "
              f"scores {[r['score'] for r in rows]}" + (f", {unchecked} unchecked" if unchecked else ""), file=sys.stderr)
    m = plan.get("manifest", {})
    print(f"wrote {out_path}: {m.get('rows_ok', 0)} rows chosen, {m.get('rows_unchecked', 0)} unchecked", file=sys.stderr)
    return 0


def plan_manifest(args, cut: dict, catalogue_sha: str, film: Path, plan: dict, profile: Film | None = None) -> dict:
    rows = [r for b in plan["beats"].values() for r in b["plan"]]
    return dict(tool="brollmatch", tool_sha=source_hash(__file__),
                cut=dict(path=args.cut, sha=fingerprint(film / args.cut), version=cut.get("version")),
                catalogue_sha=catalogue_sha,
                intent=dict(sha=fingerprint(film / "editorial/intent.json")) if (film / "editorial/intent.json").exists() else None,
                eval=None if not args.eval else dict(path=args.eval, sha=fingerprint(film / args.eval)),
                propose=dict(model=OLLAMA_MODEL, url=OLLAMA_URL, prompt_sha=text_hash(propose_template(profile.context if profile else "")), temperature=0.2, num_ctx=40000),
                verify=None if args.skip_verify else dict(model=VISION_MODEL, url=VISION_URL, prompt_sha=VERIFY_PROMPT_HASH, max_tokens=1200, temperature=0.2),
                rows=len(rows), rows_ok=sum(1 for r in rows if r.get("status", OK) == OK),
                rows_unchecked=sum(1 for r in rows if r.get("status") == UNCHECKED))


if __name__ == "__main__":
    sys.exit(main())
