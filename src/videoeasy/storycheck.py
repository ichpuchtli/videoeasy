"""Story-consistency check of a rendered cut against the synthesis.

Three layers, none of which needs the frontier model to watch or listen:

1. **What was actually said.** The render is transcribed beat by beat with
   mlx-whisper (each beat as its own chunk, so a long render cannot loop) and
   every pick's intended words are aligned against the words heard in its
   span as token edits (`speech.py`): omissions, substitutions, insertions
   inside the run, leaks at the boundaries. An edit on a polarity, quantity,
   modality, name or pronoun token is a semantic risk whatever the coverage.
   Every flagged pick gets a padded recheck (render excerpt with 2 s either
   side, and the source clip) before it is called anything; both attempts are
   kept and nothing is trimmed on one pass.
2. **Deterministic intent rules.** `editorial/intent.json` resolved by
   `intent.py`: forbidden words on the heard transcript, presence and cover
   rules on the cut, no model in the loop.
3. **The story against the spine.** The heard transcript, by beat and with
   its recording context labelled from the film's profile (sit-down,
   presenter to camera, walk-and-talk...), goes to the local text model with
   the synthesis's proposed spine and the recorded editorial intent. It
   reports which spine beats are present, partial, absent or out of order;
   which intent rules are honoured or broken; repeated ideas; orphan beats;
   whether the midpoint lands; and, when the profile names them
   (`story.voices`, `story.questions`), the balance of voices and the film's
   own structural questions.

Forbidden and watched words are rules in `editorial/intent.json`
(`words_forbidden`, `words_allowed_history`); there are none in the code.

Usage:
    uv run python -m videoeasy.storycheck --film data/films/<film> \
        --cut editorial/cut-v5.json --render editorial/renders/<film>-cut-v5.mp4 \
        --out editorial/eval/cut-v5-story
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import httpx

from . import intent as intent_mod
from . import speech
from .cuteval import pick_spans
from .film import Film, load_film
from .sources import label as sources_label
from .evalrun import OK, UNCHECKED, ffprobe_duration, fingerprint, refuse_overwrite, source_hash, text_hash, write_manifest
from .textmodel import text_model, text_url

OLLAMA_URL = text_url()        # VIDEOEASY_TEXT_URL overrides (textmodel.py)
OLLAMA_MODEL = text_model()    # VIDEOEASY_TEXT_MODEL overrides
WHISPER = "mlx-community/whisper-large-v3-mlx"


def source_label(clip: str, profile: Film | None = None) -> str:
    """Recording context of a pick, from the film's source table (`sources.py`), never a filename guess here."""
    return sources_label(clip, profile)


# ------------------------------------------------------------- transcription
def beat_spans(cut: dict, tier: int) -> list[tuple[int, float, float]]:
    out = []
    t = 0.0
    for bi, beat in enumerate(cut["beats"], 1):
        picks = [a for a in beat["audio"] if a.get("tier", 1) <= tier]
        if not picks:
            continue
        span = sum(a["duration_s"] for a in picks)
        out.append((bi, t, t + span))
        t += span + beat.get("gap_after_s", 0.0)
    return out


def transcript_cache_key(render_fp: str, spans: list[tuple[int, float, float]]) -> str:
    """A transcript is valid for one render content, one set of beat spans and one ASR model."""
    return text_hash(render_fp, json.dumps([[b, round(t0, 3), round(t1, 3)] for b, t0, t1 in spans]), WHISPER, "chunked-per-beat-v1")


def cache_valid(cache: Path, key: str) -> bool:
    if not cache.exists():
        return False
    try:
        return json.loads(cache.read_text()).get("key") == key
    except (OSError, ValueError):
        return False


def transcribe_by_beat(render: Path, spans: list[tuple[int, float, float]], cache: Path, key: str | None = None) -> dict:
    """Transcribe each beat's span of the render. The cache is reused only when
    its key (render hash, spans, model) matches; a re-render at the same path
    or a changed cut invalidates it."""
    if key and cache_valid(cache, key):
        return json.load(open(cache))
    import mlx_whisper  # slow import
    words = []
    with tempfile.TemporaryDirectory() as td:
        for bi, t0, t1 in spans:
            wav = Path(td) / f"b{bi}.wav"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{t0:.3f}", "-t", f"{t1 - t0 + 0.4:.3f}", "-i", str(render),
                            "-vn", "-ac", "1", "-ar", "16000", str(wav)], check=True)
            res = mlx_whisper.transcribe(str(wav), path_or_hf_repo=WHISPER, word_timestamps=True,
                                         condition_on_previous_text=False, language="en")
            for seg in res["segments"]:
                for w in seg.get("words", []):
                    words.append({"w": w["word"].strip(), "s": round(t0 + w["start"], 2), "e": round(t0 + w["end"], 2), "beat": bi})
            print(f"  beat {bi}: {len(res['segments'])} segments", file=sys.stderr)
    out = {"render": str(render), "key": key, "whisper": WHISPER, "words": words}
    cache.write_text(json.dumps(out))
    return out


# ------------------------------------------------------------- word alignment
def align(cut: dict, heard: list[dict], tier: int, lex: speech.Lexicon = speech.DEFAULT_LEXICON) -> list[dict]:
    """One row per pick: the token-edit analysis of its intended words against the words heard in its span."""
    rows = []
    spans = pick_spans(cut, tier)
    for n, sp in enumerate(spans):
        # the window excludes the crossfade zones, where the neighbouring pick's own words are audible by design
        got_raw = [w["w"] for w in heard if speech.norm(w["w"]) and sp["t0"] + 0.2 <= (w["s"] + w["e"]) / 2 <= sp["t1"] - 0.2]
        prev_words = set(speech.tokens(spans[n - 1]["text"].split()[-6:])) if n else set()
        next_words = set(speech.tokens(spans[n + 1]["text"].split()[:6])) if n + 1 < len(spans) else set()
        a = speech.analyse(sp["text"], got_raw, prev_words, next_words, lex)
        rows.append(dict(id=sp["id"], beat=sp["beat"], t0=round(sp["t0"], 1), t1=round(sp["t1"], 1), coverage=a["coverage"],
                         leak_head=a["leak_head"], leak_tail=a["leak_tail"], missing=a["omissions"][:8], substitutions=a["substitutions"][:6],
                         insertions=a["insertions"][:6], risks=a["risks"], reasons=speech.reasons(a), heard=a["heard_raw"][:160],
                         intended=sp["text"][:160], text=sp["text"], flag=a["flag"], clip=sp["clip"], src_in=sp["src_in"],
                         src_out=sp["src_out"], source_path=sp.get("source_path"), analysis=a, prev_words=sorted(prev_words),
                         next_words=sorted(next_words)))
    return rows


def _wave(w: dict) -> str:
    if w.get("status") != "ok":
        return f"not measured ({w.get('error', '')[:80]})"
    lost = w.get("lost") or []
    return (f"{w['speech_windows']} speech windows against the source, correlation min {w['min_corr']} median {w['median_corr']}"
            + (f"; stops matching at {', '.join(f'{t:.2f}s' for t, _ in lost[:6])}" if lost else "; all match"))


def recheck_flagged(render: Path, render_fp: str, rows: list[dict], cache: Path, pad: float = 2.0,
                    lex: speech.Lexicon = speech.DEFAULT_LEXICON) -> dict[str, dict]:
    """Padded second look at every flagged pick; results cached by render hash, span and pad."""
    store = {}
    if cache.exists():
        try:
            store = json.loads(cache.read_text())
        except ValueError:
            store = {}
    out = {}
    for r in rows:
        if not r["flag"]:
            continue
        pick = dict(id=r["id"], t0=r["t0"], t1=r["t1"], intended=r["text"])
        key = speech.recheck_key(render_fp, pick, pad)
        if key in store:
            out[r["id"]] = store[key]
            continue
        src = Path(r["source_path"]) if r.get("source_path") else None
        res = speech.recheck(render, pick, r["analysis"], src, r.get("src_in"), r.get("src_out"), pad,
                             prev_words=set(r["prev_words"]), next_words=set(r["next_words"]), lex=lex)
        res["key"] = key
        store[key] = res
        out[r["id"]] = res
        print(f"  recheck {r['id']}: {res.get('state')} ({'; '.join(res['first']['reasons'])} -> "
              f"{'; '.join(res.get('second', {}).get('reasons', [])) or 'clean'})", file=sys.stderr)
    cache.write_text(json.dumps(store, indent=1))
    return out


# ------------------------------------------------------------- synthesis text
def section(text: str, heading_prefix: str, level: str = "## ") -> str:
    lines = text.splitlines()
    out, on = [], False
    for ln in lines:
        if ln.startswith(level) and ln[len(level):].startswith(heading_prefix):
            on = True
            out.append(ln)
            continue
        if on and ln.startswith("#") and (ln.startswith(level) or (level == "### " and ln.startswith("## "))):
            break
        if on:
            out.append(ln)
    return "\n".join(out).strip()


STORY_PROMPT = """You are the story editor on a short documentary. Below is (A) the film as it actually plays, transcribed from the render and grouped by the cut's beats, each pick labelled with where it was recorded; (B) the proposed spine from the story room; (C) editorial intent recorded from the subjects and the producer. Judge the film against the spine and the intent. Quote the film's words as evidence; do not invent lines.

(A) THE FILM AS IT PLAYS
{film}

(B) THE PROPOSED SPINE
{spine}

(C) RECORDED INTENT
{intent}

A preference for one word over another is broken only if the film uses the disfavoured word; it is not broken by the absence of the preferred one. You only have the words. A rule about pictures, B-roll, montages, inserts, time-lapses or sound design cannot be judged here: mark it not_testable_from_words rather than broken. For (B), walk the spine's own numbered beats, not the cut's beat titles, and say which cut beats carry each. A cut beat that carries one of the additions listed under the spine is a required beat, not an orphan; list as orphans only beats that serve neither the numbered spine nor an addition.

Answer with one JSON object:
{{
  "spine": [ for every numbered beat of the spine, in spine order: {{"spine_beat": "<its name>", "status": "present|partial|absent|out_of_order", "carried_by": [<cut beat numbers>], "note": "at most 25 words"}} ],
  "intent": [ for every rule in (C) that the spoken film can honour or break: {{"rule": "<short name>", "status": "honoured|broken|not_testable_from_words", "evidence": "quote or empty"}} ],
  "repetition": [ {{"idea": "<what is said twice>", "beats": [<cut beat numbers>], "keep_in": <beat number>, "why": "at most 20 words"}} ],
  "orphans": [ {{"beat": <cut beat number>, "why": "why it serves no spine beat, at most 20 words"}} ],
{film_fields}  "midpoint_lands": {{"yes": true/false, "why": "at most 25 words"}},
  "transitions": [ {{"from_beat": <n>, "to_beat": <n>, "problem": "a join where the words do not lead into the next beat, at most 20 words"}} ],
  "top_fixes": [ "up to five, each at most 25 words, ordered by how much they would improve the story" ]
}}"""


def _js(text: str) -> str:
    """Text placed inside the prompt's JSON template: quotes escaped, braces doubled for str.format."""
    return str(text).replace('"', "'").replace("{", "{{").replace("}", "}}")


def story_template(profile: Film | None = None) -> str:
    """STORY_PROMPT with the film's own fields: `story.voices` (group -> who speaks in it) with
    `story.voices_question`, and `story.questions` (key -> question, asked as written, word limit included)."""
    st = (profile.story if profile else {}) or {}
    lines = []
    voices = st.get("voices") or {}
    if voices:
        parts = [f'"{_js(k)}_voice_seconds_est": <{_js(who)}, number>' for k, who in voices.items()]
        q = st.get("voices_question") or "is the balance of voices what the spine asks for"
        lines.append('  "voices": {{' + ", ".join(parts) + f', "comment": "{_js(q)}, at most 30 words"' + "}},\n")
    qs = st.get("questions") or {}
    if isinstance(qs, list):
        qs = {f"question_{i}": q for i, q in enumerate(qs, 1)}
    for k, q in qs.items():
        lines.append(f'  "{_js(k)}": "{_js(q)}",\n')
    return STORY_PROMPT.replace("{film_fields}", "".join(lines))


def story_model(film_text: str, spine: str, intent: str, url: str = OLLAMA_URL, model: str = OLLAMA_MODEL,
                profile: Film | None = None) -> dict:
    prompt = story_template(profile).format(film=film_text, spine=spine, intent=intent)
    r = httpx.post(f"{url}/api/chat", json={"model": model, "stream": False, "think": False, "format": "json",
                                            "options": {"temperature": 0.2, "num_ctx": 40000},
                                            "messages": [{"role": "user", "content": prompt}]}, timeout=1500)
    r.raise_for_status()
    return json.loads(r.json()["message"]["content"])


# --------------------------------------------------------------------- report
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--film", required=True)
    ap.add_argument("--cut", required=True)
    ap.add_argument("--render", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--tier", type=int, default=1)
    ap.add_argument("--skip-model", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-recheck", action="store_true", help="skip the padded second pass on flagged picks")
    ap.add_argument("--synthesis", default=None, help="synthesis file relative to --film (default out/bible-synthesis.md); its '6. Proposed spine' is the spine")
    args = ap.parse_args(argv)
    film = Path(args.film)
    profile = load_film(film)
    lex = speech.lexicon_for(profile)
    cut = json.load(open(film / args.cut))
    render = film / args.render
    out = film / args.out
    refuse_overwrite([out.with_suffix(".json"), out.with_suffix(".md")], args.overwrite)
    if not render.exists() or ffprobe_duration(render) is None:
        raise SystemExit(f"render missing or unreadable: {render}; nothing was checked")
    render_fp = fingerprint(render)

    spans = beat_spans(cut, args.tier)
    key = transcript_cache_key(render_fp, spans)
    cache = render.with_name(render.stem + f".transcript.{key[:12]}.json")
    heard = transcribe_by_beat(render, spans, cache, key)["words"]
    rows = align(cut, heard, args.tier, lex)
    flagged = [r for r in rows if r["flag"]]
    rechecks: dict[str, dict] = {}
    if flagged and not args.no_recheck:
        rechecks = recheck_flagged(render, render_fp, rows, render.with_name(render.stem + ".recheck.json"), lex=lex)
    for r in rows:
        r["recheck"] = rechecks.get(r["id"])
        r["state"] = (r["recheck"] or {}).get("state", "unresolved" if r["flag"] else "clean") if r["flag"] else "clean"
        for k in ("analysis", "text", "prev_words", "next_words"):
            r.pop(k, None)
    heard_text = " ".join(w["w"] for w in heard).lower()
    intent_path = film / "editorial/intent.json"
    intent_doc = intent_mod.load(film) if intent_path.exists() else None
    nogo, watch = [], []
    if intent_doc:
        for rule in intent_mod.resolve(intent_doc)["active"]:
            if rule["kind"] == "words_forbidden":
                nogo += [(w, rule["decision"]) for w in rule["words"] if re.search(rf"\b{re.escape(w)}", heard_text)]
            if rule["kind"] == "words_allowed_history":
                watch += [(w, rule["decision"]) for w in rule["words"] if re.search(rf"\b{w}\b", heard_text)]
        intent_findings = intent_mod.check_cut(cut, intent_doc, args.tier, heard_text=heard_text, profile=profile)
    else:
        intent_findings = None   # no recorded decisions: no word rules either

    # film as it plays, from the heard words grouped by pick
    by_src: dict[str, float] = {}
    film_lines = []
    cur_beat = None
    for sp, r in zip(pick_spans(cut, args.tier), rows):
        if sp["beat"] != cur_beat:
            cur_beat = sp["beat"]
            title = cut["beats"][cur_beat - 1]["title"]
            film_lines.append(f"\n## Cut beat {cur_beat}: {title}")
        clip = sp["key"].split("@")[0]
        label = source_label(clip, profile)
        by_src[label] = by_src.get(label, 0.0) + (sp["t1"] - sp["t0"])
        words = " ".join(w["w"] for w in heard if sp["t0"] - 0.3 <= (w["s"] + w["e"]) / 2 <= sp["t1"] + 0.3)
        film_lines.append(f"- [{label}, {sp['t1'] - sp['t0']:.0f}s] {words}")
    film_text = "\n".join(film_lines)

    from .brollmatch import synthesis_path
    syn_p = synthesis_path(film, args.synthesis)
    syn = syn_p.read_text(encoding="utf-8")
    spine = section(syn, "6. Proposed spine")
    if intent_doc:
        # one resolved brief, latest decision first in precedence, with sources; the spine carries the later additions
        adds = intent_mod.story_additions(intent_doc)
        if adds:
            spine += "\n\n### Additions decided later (required beats, not orphans)\n" + adds
        intent = intent_mod.brief(intent_doc)
    else:
        # no intent.json: the synthesis's own record of intent, by its conventional heading
        intent = section(syn, "5. Recorded editorial intent")
    story = None
    story_status = "not run"
    if not args.skip_model:
        try:
            story = story_model(film_text, spine, intent, profile=profile)
            story_status = OK
        except Exception as e:  # noqa: BLE001
            story_status = f"{UNCHECKED}: {str(e)[:160]}"
            print(f"story model failed: {e}", file=sys.stderr)

    partial = [r for r in rows if not r["flag"] and r["coverage"] < 1.0]
    states: dict[str, int] = {}
    for r in flagged:
        states[r["state"]] = states.get(r["state"], 0) + 1
    manifest = write_manifest(
        out, tool="storycheck", tool_sha=source_hash(__file__),
        render=dict(path=str(render), sha=render_fp), cut=dict(path=args.cut, sha=fingerprint(film / args.cut), version=cut.get("version")),
        synthesis=dict(path=str(syn_p), sha=fingerprint(syn_p)),
        transcript=dict(cache=str(cache), key=key, whisper=WHISPER),
        story_model=None if args.skip_model else dict(model=OLLAMA_MODEL, url=OLLAMA_URL, prompt_sha=text_hash(story_template(profile)),
                                                       temperature=0.2, num_ctx=40000, status=story_status),
        alignment=dict(rule="token edits (speech.py); flag when coverage < 0.8, >= 2 leaked words, or any edit on a polarity/quantity/"
                            "modality/name/pronoun token; window t0+0.2..t1-0.2; neighbours' words discounted",
                       recheck="padded 2 s render excerpt and source clip per flagged pick" if not args.no_recheck else "skipped",
                       states=states),
        intent=None if not intent_doc else dict(path=str(intent_path), sha=fingerprint(intent_path), summary=intent_mod.summary(intent_findings)),
        picks=len(rows), flagged=len(flagged), unflagged_below_full_coverage=len(partial), tier=args.tier,
    )
    result = dict(render=str(render), cut=args.cut, manifest=manifest, seconds_by_source={k: round(v) for k, v in by_src.items()},
                  picks=rows, flagged=len(flagged), states=states, no_go=nogo, watch=watch, intent=intent_findings, story=story)
    out.with_suffix(".json").write_text(json.dumps(result, indent=1))

    L = [f"# Story check of `{render.name}` against `{Path(args.cut).name}` and the synthesis", ""]
    L.append(f"Provenance: render sha `{render_fp}`, cut sha `{manifest['cut']['sha']}`; manifest beside this report.")
    L.append(f"Words heard vs intended: **{len(rows) - len(flagged)} of {len(rows)} picks** did not trip the alignment flags "
             f"(coverage ≥ 0.8, fewer than 2 words leaked in from beyond the pick, no edit on a polarity, quantity, modality, name or "
             f"pronoun token); {len(partial)} of those are below full coverage, so this is not proof of complete speech.")
    if flagged:
        L.append(f"Flagged picks after the padded recheck: " + ", ".join(f"**{v}** {k}" for k, v in sorted(states.items()))
                 + ". 'cleared_on_recheck' was an ASR miss; 'confirmed' means two render passes agree, the source carries the words and the render's "
                 "waveform stops matching the source's; 'waveform_intact' means the same ASR disagreement with the source's speech present "
                 "in the render throughout (nothing lost); "
                 "'intent_mismatch' means the source itself says something else; 'unresolved' needs a listen. Nothing is trimmed on this evidence.")
    if intent_findings is not None:
        broken = [f for f in intent_findings if f["status"] == intent_mod.BROKEN]
        L.append(f"Editorial intent (editorial/intent.json, resolved, checked without a model): **{'BROKEN: ' + ', '.join(f['rule'] for f in broken) if broken else 'no rule broken'}**; "
                 + "; ".join(f"{f['rule']} {f['status']}" for f in intent_findings if f["status"] not in (intent_mod.HONOURED, intent_mod.BROKEN)) + ".")
    L.append("Seconds by recording context (not speaker balance; a two-shot carries more than one voice): " + ", ".join(f"{k}: {round(v)}s" for k, v in by_src.items()) + ".")
    L.append("No-go words heard: " + (", ".join(f"'{k}' ({why})" for k, why in nogo) if nogo else "none") + ".")
    if watch:
        L.append("Words to watch: " + ", ".join(f"'{k}' ({why})" for k, why in watch) + ".")
    if story_status != OK and not args.skip_model:
        L.append(f"Story model: **{story_status}**.")
    if flagged:
        L += ["", "## Picks that tripped the alignment flags, with the padded recheck", ""]
        for r in flagged:
            rc = r.get("recheck") or {}
            L.append(f"- **{r['t0']}-{r['t1']}s** beat {r['beat']} {r['id']} ({source_label(r['clip'], profile)}): **{r['state']}**. "
                     f"First pass: {'; '.join(r['reasons'])}. Heard: \"{r['heard'][:100]}\"."
                     + (f" Recheck: {'; '.join(rc.get('second', {}).get('reasons', [])) or 'clean'}; heard \"{rc.get('second', {}).get('heard', '')[:100]}\"."
                        f" Source: {'; '.join(rc.get('source', {}).get('reasons', [])) or ('clean' if rc.get('source') else 'not checked')}."
                        + (f" Waveform: {_wave(rc['waveform'])}." if rc.get("waveform") else "")
                        + f" Action: {rc.get('action', '')}" if rc else " Recheck: not run."))
    if intent_findings:
        L += ["", "## Editorial intent (rules, not scores)", ""]
        for f in intent_findings:
            L.append(f"- {f['rule']}: **{f['status']}**. {f['evidence']}")
    if story:
        L += ["", "## The story against the spine (local text model)", ""]
        for s in story.get("spine", []):
            if isinstance(s, dict):
                L.append(f"- **{s.get('spine_beat')}**: {s.get('status')} (cut beats {s.get('carried_by')}). {s.get('note', '')}")
        L += ["", "**Intent:**"]
        for s in story.get("intent", []):
            if isinstance(s, dict):
                L.append(f"- {s.get('rule')}: {s.get('status')}. {s.get('evidence', '')}")
        for key, title in (("repetition", "Repetition"), ("orphans", "Orphan beats"), ("transitions", "Joins where the words do not lead on")):
            items = story.get(key) or []
            if items:
                L += ["", f"**{title}:**"]
                for s in items:
                    L.append(f"- {json.dumps(s) if not isinstance(s, dict) else '; '.join(f'{k}: {v}' for k, v in s.items())}")
        v = story.get("voices") or {}
        if v:
            est = ", ".join(f"{k[:-len('_voice_seconds_est')]} ~{x}s" for k, x in v.items() if k.endswith("_voice_seconds_est"))
            L += ["", f"**Voices:** {est}. {v.get('comment', '')}"]
        qs = (profile.story or {}).get("questions") or {}
        for k in (qs if isinstance(qs, dict) else [f"question_{i}" for i in range(1, len(qs) + 1)]):
            L += ["", f"**{k.replace('_', ' ').capitalize()}:** {story.get(k, '')}"]
        L += ["", f"**Midpoint lands:** {(story.get('midpoint_lands') or {}).get('yes')}. {(story.get('midpoint_lands') or {}).get('why', '')}",
              "", "**Top fixes:**"]
        L += [f"1. {x}" for x in story.get("top_fixes", [])]
    out.with_suffix(".md").write_text("\n".join(L) + "\n")
    print(f"wrote {out}.md and .manifest.json ({len(flagged)} flagged picks {states}, {len(partial)} unflagged below full coverage, "
          f"{len(nogo)} no-go hits, intent {intent_mod.summary(intent_findings) if intent_findings else 'no file'}, story model {story_status})", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
