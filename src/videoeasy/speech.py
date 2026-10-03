"""Speech alignment as token edits, with a padded recheck before any flag becomes an action.

A pick's intended words (from the cut) are aligned against the words heard
in its span of the render. The result is the complete list of edit
operations, not a coverage number: what was dropped, what was substituted,
what was inserted inside the run, and what leaked in at either boundary.
Any operation that touches a critical token (polarity, quantity, modality,
a name, a pronoun) is a semantic risk whatever the coverage.

A flag on one ASR pass is a question, not a finding. `recheck` transcribes
the pick again from a padded excerpt of the render, and the same words from
the source clip, and reports how the passes agree:

  cleared_on_recheck   the padded render pass hears the pick as intended
  confirmed            two render passes agree on the same defect, the
                       source clip does carry the words, and the render's
                       waveform stops matching the source's somewhere in the
                       pick: the render lost them
  waveform_intact      the same two ASR passes and a clean source, but every
                       half-second of the source's speech is in the render
                       (`waveform_match`): the transcriber hears the levelled
                       render differently, nothing was lost. A pick once called
                       confirmed on ASR alone matched the source throughout
                       (min 0.75, median 0.92; 29 Sep 2026)
  intent_mismatch      the source clip itself does not carry the intended
                       words: the cut's text is wrong, not the render
  unresolved           the passes disagree in any other way

Nothing here moves a trim or drops a pick. Both attempts are kept.
"""
from __future__ import annotations

import difflib
import json
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .evalrun import OK, UNCHECKED, text_hash

WHISPER = "mlx-community/whisper-large-v3-mlx"
ANALYSIS_VERSION = "token-edits-v5"  # bump when the analysis rules change; recheck caches are keyed on it

# Spellings ASR gets wrong are film vocabulary (names, species, jargon) and come from the film's profile
# (`film.yaml` `aliases`, `names`); only spelling conventions live here. A heard token on the right is the
# intended token on the left.
BASE_ALIASES = {
    "metres": {"meters"},
}


@dataclass(frozen=True)
class Lexicon:
    """The film's vocabulary for alignment: explicit aliases (intended -> heard variants) and the name tokens whose
    edit changes who is meant."""
    aliases: dict = field(default_factory=dict)
    names: frozenset = frozenset()

    @property
    def alias_of(self) -> dict[str, str]:
        return {h: k for k, hs in self.aliases.items() for h in hs}


def lexicon(aliases: dict | None = None, names=()) -> Lexicon:
    merged = {k: set(v) for k, v in BASE_ALIASES.items()}
    for k, vs in (aliases or {}).items():
        merged.setdefault(k.lower(), set()).update(v.lower() for v in vs)
    return Lexicon(aliases={k: frozenset(v) for k, v in merged.items()}, names=frozenset(n.lower() for n in names))


def lexicon_for(profile) -> Lexicon:
    """The lexicon of a film profile (film.py), or the generic one for None."""
    return lexicon(profile.aliases, profile.names) if profile is not None else DEFAULT_LEXICON


DEFAULT_LEXICON = lexicon()

POLARITY = {"not", "no", "never", "nothing", "none", "nobody", "without", "neither", "nor", "cannot"}
MODALITY = {"must", "should", "shouldn't", "could", "couldn't", "can", "can't", "may", "might", "will", "won't",
            "would", "wouldn't", "always", "only", "still", "yet", "already"}
PRONOUNS = {"i", "we", "you", "he", "she", "they", "it", "me", "us", "them", "him", "her", "my", "our", "your", "their", "his"}
INSERT_ESCALATES = {"polarity", "quantity"}  # an inserted "not" or number changes meaning; an inserted "you" or "would" rarely does
NUMBER_WORDS = {"one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "twenty", "thirty", "hundred",
                "thousand", "half", "double", "few", "many", "all", "every", "most", "several", "first", "last", "twice", "once"}


CONTRACTIONS = {"can't": "can not", "won't": "will not", "shan't": "shall not", "n't": " not",
                "'re": " are", "'ve": " have", "'ll": " will", "'m": " am", "let's": "let us"}
NO_APOSTROPHE = {k: k[:-2] + "n't" for k in ("dont", "cant", "wont", "isnt", "wasnt", "doesnt", "didnt", "couldnt", "wouldnt",
                                            "shouldnt", "hasnt", "havent", "arent", "werent")}
NUMBERS = {"0": "zero", "1": "one", "2": "two", "3": "three", "4": "four", "5": "five", "6": "six", "7": "seven", "8": "eight",
           "9": "nine", "10": "ten", "11": "eleven", "12": "twelve", "15": "fifteen", "20": "twenty", "30": "thirty", "40": "forty",
           "50": "fifty", "100": "hundred", "1000": "thousand"}


def norm(tok: str) -> str:
    return re.sub(r"[^a-z0-9']", "", tok.lower())


def tokens(words) -> list[str]:
    """Normalised tokens with contractions expanded and small numbers spelled out, so
    "we're" and "we are", "5" and "five" agree and "can't" carries an explicit "not"."""
    out = []
    for w in (words.split() if isinstance(words, str) else words):
        t = norm(w)
        if not t:
            continue
        if t in NO_APOSTROPHE:
            t = NO_APOSTROPHE[t]
        for k in ("can't", "won't", "shan't", "let's"):
            if t == k:
                t = CONTRACTIONS[k]
                break
        else:
            for suf in ("n't", "'re", "'ve", "'ll", "'m"):
                if t.endswith(suf) and len(t) > len(suf) + 1:
                    t = t[: -len(suf)] + CONTRACTIONS[suf]
                    break
        for part in t.split():
            out.append(NUMBERS.get(part, part))
    return out


def is_critical(tok: str, lex: Lexicon = DEFAULT_LEXICON) -> str | None:
    """The class that makes an edit on this token a semantic risk, or None."""
    if tok in POLARITY or tok.endswith("n't"):
        return "polarity"
    if tok in NUMBER_WORDS or re.fullmatch(r"\d+(st|nd|rd|th)?", tok):
        return "quantity"
    if tok in MODALITY:
        return "modality"
    if tok in lex.names:
        return "name"
    if tok in PRONOUNS:
        return "actor"
    return None


def canon(tok: str, vocab: set[str], lex: Lexicon = DEFAULT_LEXICON) -> str:
    """Map a heard token onto the intended token it stands for. The alias
    table is explicit; the fuzzy fallback never touches a critical token and
    never maps onto one, so 'not' cannot be normalised away."""
    if tok in vocab:
        return tok
    alias_of = lex.alias_of
    if tok in alias_of and alias_of[tok] in vocab:
        return alias_of[tok]
    if tok in lex.aliases:  # the cut text may itself carry the transcript's spelling
        for v in sorted(lex.aliases[tok]):
            if v in vocab:
                return v
    if is_critical(tok, lex) or len(tok) < 4:
        return tok
    best = difflib.get_close_matches(tok, [v for v in vocab if not is_critical(v, lex) and len(v) >= 4], n=1, cutoff=0.8)
    return best[0] if best else tok


def edits(expected: list[str], heard: list[str]) -> list[dict]:
    """Complete token-edit operations from intended to heard (both normalised).
    Each op: kind (equal|delete|insert|replace), exp, got, at (index in expected)."""
    sm = difflib.SequenceMatcher(a=expected, b=heard, autojunk=False)
    ops = []
    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == "equal":
            ops.append(dict(kind="equal", exp=expected[i1:i2], got=heard[j1:j2], at=i1))
        elif tag == "delete":
            ops.append(dict(kind="delete", exp=expected[i1:i2], got=[], at=i1))
        elif tag == "insert":
            ops.append(dict(kind="insert", exp=[], got=heard[j1:j2], at=i1))
        else:
            ops.append(dict(kind="replace", exp=expected[i1:i2], got=heard[j1:j2], at=i1))
    return ops


def analyse(intended: str, heard_tokens: list[str], prev_words: set[str] = frozenset(), next_words: set[str] = frozenset(),
            lex: Lexicon = DEFAULT_LEXICON) -> dict:
    """One pick: edit operations split into omissions, substitutions, internal
    insertions and boundary leaks, the coverage, and the semantic risks."""
    exp = tokens(intended)
    raw = tokens(heard_tokens)
    vocab = set(exp)
    got = [canon(g, vocab, lex) for g in raw]
    ops = edits(exp, got)
    n_exp = len(exp)
    matched = sum(len(o["exp"]) for o in ops if o["kind"] == "equal")
    first_eq = next((k for k, o in enumerate(ops) if o["kind"] == "equal"), None)
    last_eq = next((k for k in range(len(ops) - 1, -1, -1) if ops[k]["kind"] == "equal"), None)
    omissions, substitutions, insertions, leak_head, leak_tail = [], [], [], [], []
    for k, o in enumerate(ops):
        if o["kind"] == "delete":
            omissions += o["exp"]
        elif o["kind"] == "replace":
            substitutions.append((" ".join(o["exp"]), " ".join(o["got"])))
        elif o["kind"] == "insert":
            if first_eq is None or k < first_eq:
                leak_head += [g for g in o["got"] if g not in prev_words and g not in vocab]
            elif k > last_eq:
                leak_tail += [g for g in o["got"] if g not in next_words and g not in vocab]
            else:
                insertions += o["got"]
    risks = []
    for k, o in enumerate(ops):
        if o["kind"] == "equal":
            continue
        internal = first_eq is not None and last_eq is not None and first_eq < k < last_eq
        heard = " ".join(o["got"][:6]) + (" …" if len(o["got"]) > 6 else "")
        expected = " ".join(o["exp"][:6]) + (" …" if len(o["exp"]) > 6 else "")
        # a big replaced block is garbled ASR, not a word swap: only polarity is escalated from inside it
        small = len(o["exp"]) <= 4 and len(o["got"]) <= 4
        for tok in o["exp"]:
            c = is_critical(tok, lex)
            if not c or (c != "polarity" and not small):
                continue
            if o["kind"] == "replace" and tok in o["got"]:
                continue  # the token itself was heard; the block differs elsewhere
            risks.append(dict(cls=c, kind="dropped" if o["kind"] == "delete" else "substituted", token=tok, heard=heard,
                              expected=expected, at=o["at"]))
        seen = {r["cls"] for r in risks if r["at"] == o["at"] and r["kind"] == "substituted"}
        for tok in o["got"]:
            c = is_critical(tok, lex)
            if not c or (c != "polarity" and not small):
                continue
            if o["kind"] == "replace" and (tok in o["exp"] or c in seen):
                continue  # already reported as the substitution of the intended token of that class
            if o["kind"] == "insert" and c not in INSERT_ESCALATES:
                continue  # an interjected pronoun or modal from the other voice on a two-shot is noise, not a changed meaning
            if o["kind"] == "replace" or internal:
                risks.append(dict(cls=c, kind="inserted" if o["kind"] == "insert" else "substituted", token=tok, heard=heard,
                                  expected=expected, at=o["at"]))
    cov = matched / n_exp if n_exp else 1.0
    return dict(coverage=round(cov, 2), matched=matched, expected=n_exp, omissions=omissions, substitutions=substitutions,
                insertions=insertions, leak_head=leak_head, leak_tail=leak_tail, risks=risks, heard_raw=" ".join(raw),
                ops=[o for o in ops if o["kind"] != "equal"],
                flag=(cov < 0.8) or (len(leak_head) + len(leak_tail) >= 2) or bool(risks))


def reasons(a: dict) -> list[str]:
    r = []
    if a["coverage"] < 0.8:
        r.append(f"coverage {a['coverage']}")
    if len(a["leak_head"]) + len(a["leak_tail"]) >= 2:
        r.append(f"{len(a['leak_head']) + len(a['leak_tail'])} leaked words")
    for k in a["risks"]:
        if k["kind"] == "substituted":
            r.append(f"{k['cls']} substituted: '{k.get('expected') or k['token']}' -> '{k['heard']}'")
        else:
            r.append(f"{k['cls']} {k['kind']}: '{k['token']}'")
    return r


# ---------------------------------------------------------------- recheck
def transcribe_excerpt(media: Path, t0: float, t1: float, pad: float, model: str = WHISPER) -> list[dict]:
    """Words heard in media[t0-pad, t1+pad], timed in media seconds."""
    import mlx_whisper  # slow import
    a = max(0.0, t0 - pad)
    with tempfile.TemporaryDirectory() as td:
        wav = Path(td) / "x.wav"
        p = subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{a:.3f}", "-t", f"{t1 + pad - a:.3f}", "-i", str(media),
                            "-vn", "-ac", "1", "-ar", "16000", str(wav)], capture_output=True, text=True)
        if p.returncode != 0:
            raise RuntimeError(f"ffmpeg failed on {media}: {p.stderr[-160:]}")
        res = mlx_whisper.transcribe(str(wav), path_or_hf_repo=model, word_timestamps=True,
                                     condition_on_previous_text=False, language="en")
    return [{"w": w["word"].strip(), "s": round(a + w["start"], 2), "e": round(a + w["end"], 2)}
            for seg in res["segments"] for w in seg.get("words", [])]


WAVE_SR = 8000
WAVE_WIN_S = 0.25   # a quarter second: the 0.6 s hole test reads 0.0 here, 0.3 at half a second
WAVE_MIN_CORR = 0.6     # measured on a falsely confirmed pick: every speech window 0.75 or better, median 0.92
WAVE_SPEECH_DB = 20.0   # a window within this many dB of the pick's loudest carries speech


def _pcm(media: Path, t0: float, dur: float, pan: str):
    import numpy as np
    p = subprocess.run(["ffmpeg", "-v", "error", "-ss", f"{max(0.0, t0):.3f}", "-t", f"{dur:.3f}", "-i", str(media), "-vn",
                        "-af", f"pan=mono|c0={pan},highpass=f=150,lowpass=f=3500", "-ar", str(WAVE_SR), "-f", "f32le", "-"],
                       capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed on {media}: {p.stderr.decode(errors='replace')[-160:]}")
    return np.frombuffer(p.stdout, np.float32)


def waveform_match(render: Path, t0: float, t1: float, source: Path, src_in: float, src_out: float,
                   lead: float = 0.5, search: float = 0.08) -> dict:
    """Does the render carry the source's audio through the pick? Every half-second of the source is correlated
    against the render (quarter-second windows) at a lag searched locally (+/- `search` s, following the drift: Resolve's rate conversion
    moved v8's pick 26 ms over 39 s). Gain, the Dialogue Leveler and the mono-to-both-channels map leave a speech
    window's correlation high; a lost word leaves silence or other audio there. Returns the speech windows under
    WAVE_MIN_CORR as `lost` (render seconds)."""
    import numpy as np
    dur = min(t1 - t0, src_out - src_in)
    R = _pcm(render, t0 - lead, dur + 2 * lead, "c0")
    S = _pcm(source, src_in, dur, "c0+c1")
    w, span = int(WAVE_WIN_S * WAVE_SR), int(search * WAVE_SR)
    lag = int(lead * WAVE_SR)
    rows = []
    for i in range(0, len(S) - w + 1, w):
        b = S[i:i + w]
        best = (-2.0, lag)
        for L in range(max(0, lag - span), lag + span + 1, 2):
            a = R[L + i:L + i + w]
            if len(a) < w:
                break
            c = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
            if c > best[0]:
                best = (c, L)
        rows.append((i / WAVE_SR, best[0], 20 * np.log10(np.sqrt(float((b ** 2).mean())) + 1e-12)))
        if best[0] >= WAVE_MIN_CORR:
            lag = best[1]
    if not rows:
        return dict(status=UNCHECKED, error="pick shorter than one window")
    top = max(r[2] for r in rows)
    speech_rows = [r for r in rows if r[2] >= top - WAVE_SPEECH_DB]
    corr = sorted(r[1] for r in speech_rows)
    return dict(status=OK, windows=len(rows), speech_windows=len(speech_rows), min_corr=round(corr[0], 2),
                median_corr=round(corr[len(corr) // 2], 2),
                lost=[(round(t0 + r[0], 2), round(r[1], 2)) for r in speech_rows if r[1] < WAVE_MIN_CORR])


def in_window(words: list[dict], t0: float, t1: float, slack: float = 0.2) -> list[str]:
    """Words whose midpoint lies inside [t0+slack, t1-slack]; a negative slack widens the window."""
    return [w["w"] for w in words if t0 + slack <= (w["s"] + w["e"]) / 2 <= t1 - slack]


def same_defect(a: dict, b: dict) -> bool:
    """Two analyses report the same problem: same dropped/substituted critical tokens, or both under coverage on the same words."""
    ra = {(k["cls"], k["kind"], k["token"]) for k in a["risks"]}
    rb = {(k["cls"], k["kind"], k["token"]) for k in b["risks"]}
    same_words = set(a["omissions"]) == set(b["omissions"]) and set(a["insertions"]) == set(b["insertions"]) \
        and set(a["substitutions"]) == set(b["substitutions"])
    if ra or rb:
        return ra == rb and same_words
    return same_words and a["coverage"] < 0.8 and b["coverage"] < 0.8


def recheck_key(render_fp: str, pick: dict, pad: float, model: str = WHISPER) -> str:
    return text_hash(render_fp, pick["id"], f"{pick['t0']:.2f}-{pick['t1']:.2f}", f"{pad:.1f}", model, ANALYSIS_VERSION)


RECHECK_SLACK = -0.3  # the second look listens 0.3 s past each end of the pick, so an edge word is not lost to the window


def recheck(render: Path, pick: dict, first: dict, source: Path | None, src_in: float | None, src_out: float | None,
            pad: float = 2.0, transcribe=transcribe_excerpt, prev_words=frozenset(), next_words=frozenset(),
            waveform=waveform_match, lex: Lexicon = DEFAULT_LEXICON) -> dict:
    """Second look at a flagged pick. `pick` has id, t0, t1 (render seconds) and
    `intended` (the full text); `first` is the first-pass analysis. `transcribe`
    is injectable for tests. The window is 0.3 s wider than the pick at each
    end and the neighbouring picks' words are discounted, so a word at the
    edge is heard and a crossfaded neighbour is not a leak. Returns the state,
    both analyses and the source's own reading."""
    out: dict = dict(pick=pick["id"], pad=pad, first=dict(coverage=first["coverage"], reasons=reasons(first)), status=OK)
    try:
        words = transcribe(render, pick["t0"], pick["t1"], pad)
    except Exception as e:  # noqa: BLE001
        return dict(out, status=UNCHECKED, state="unresolved", error=f"render recheck failed: {str(e)[:160]}")
    second = analyse(pick["intended"], in_window(words, pick["t0"], pick["t1"], RECHECK_SLACK), prev_words, next_words, lex)
    out["second"] = dict(coverage=second["coverage"], reasons=reasons(second), heard=second["heard_raw"][:200])
    src = None
    if source is not None and src_in is not None and src_out is not None:
        try:
            sw = transcribe(Path(source), src_in, src_out, pad)
            src = analyse(pick["intended"], in_window(sw, src_in, src_out, RECHECK_SLACK), lex=lex)
            out["source"] = dict(coverage=src["coverage"], reasons=reasons(src), heard=src["heard_raw"][:200])
        except Exception as e:  # noqa: BLE001
            out["source"] = dict(status=UNCHECKED, error=str(e)[:160])
    if not second["flag"]:
        out["state"] = "cleared_on_recheck"
    elif src is not None and not src["flag"] and same_defect(first, second):
        # ASR on the render twice and on the source once cannot tell lost audio from audio heard differently:
        # the waveform can. A failed or partial measurement leaves the flag standing.
        try:
            wm = waveform(render, pick["t0"], pick["t1"], Path(source), src_in, src_out)
        except Exception as e:  # noqa: BLE001
            wm = dict(status=UNCHECKED, error=str(e)[:160])
        out["waveform"] = wm
        out["state"] = "waveform_intact" if wm.get("status") == OK and not wm.get("lost") else "confirmed"
    elif src is not None and src["flag"] and same_defect(second, src):
        out["state"] = "intent_mismatch"
    else:
        out["state"] = "unresolved"
    out["action"] = {"cleared_on_recheck": "none: the first pass was an ASR miss",
                     "confirmed": "the render lost these words; fix the lay-in or the pick, then re-measure",
                     "waveform_intact": "none: the render carries the source's speech throughout; the transcriber hears the processed audio differently",
                     "intent_mismatch": "the cut's text does not match what the source says; correct the pick text",
                     "unresolved": "listen to the padded excerpt; no trim moves on this evidence"}[out["state"]]
    return out


if __name__ == "__main__":  # a quick look at one pick: python -m videoeasy.speech "intended words" "heard words"
    print(json.dumps(analyse(sys.argv[1], sys.argv[2].split()), indent=1))


# ------------------------------------------------------------- cover rules
# The editor's v6 review (26 Sep 2026): one person visibly talking to camera under
# another speaker's interview audio. The picture judge sees stills and cannot see lips; the transcript knows
# exactly when a person speaks in a source, so "cover from a range where the
# covered person is speaking" is measured here without a model.
TALKING_COVER_MIN_S = 1.0   # at least this much transcribed speech inside the covered source range


def source_of(unit: str) -> str:
    """Catalogue unit id -> source id: windows are `<source>_wN`, drone sub-shots `<source>_sN`."""
    return re.sub(r"_[ws]\d+$", "", unit)


def spoken_seconds(transcripts: dict, clip: str, in_s: float, out_s: float) -> float | None:
    """Seconds of transcribed words inside [in_s, out_s] of a source; None when the source has no transcript
    (an untranscribed B-roll take is unknown, not silent)."""
    base = source_of(clip)
    tr = transcripts.get(base)
    if not tr:
        return None
    total = 0.0
    for seg in tr.get("segments", []):
        for w in seg.get("words", []):
            a, b = max(float(w["s"]), in_s), min(float(w["e"]), out_s)
            if b > a:
                total += b - a
    return round(total, 2)


SYNC_TOLERANCE_S = 0.5


def pick_starts(cut: dict, tier: int = 1) -> dict[str, tuple[str, float, float]]:
    """pick id -> (clip, source in_s, film-time start), laid as cuteval.layout lays them."""
    out = {}
    t = 0.0
    for bi, beat in enumerate(cut["beats"], 1):
        pt = t
        for j, a in enumerate(beat["audio"], 1):
            if a.get("tier", 1) > tier:
                continue
            out[f"b{bi}p{j}"] = (a["clip"], float(a["in_s"]), pt)
            pt += a["duration_s"]
        t = pt + beat.get("gap_after_s", 0.0)
    return out


def in_sync(row_clip: str, row_in_s: float, row_t0: float, pick: tuple[str, float, float] | None, tol: float = SYNC_TOLERANCE_S) -> bool:
    """The picture is the pick's own take at the pick's own time: sync, so the speech on screen is the speech heard."""
    if not pick or row_t0 is None:
        return False
    pclip, pin, pt0 = pick
    if source_of(row_clip) != source_of(pclip):
        return False
    return abs((row_in_s - pin) - (row_t0 - pt0)) <= tol


def talking_cover(cut: dict, transcripts: dict, min_s: float = TALKING_COVER_MIN_S, tier: int = 1,
                  people: dict[str, str] | None = None, exempt: set[str] | None = None) -> list[dict]:
    """Every cover row of a cut whose source range holds transcribed speech (someone on screen is talking
    while another pick's audio plays). Rows from untranscribed sources are listed as unknown, not clean. A
    row that is the pick's own take at the pick's own time (sync picture) is exempt; the same take at another
    time is not (cut-v6's walk beats showed the two speakers saying other words under their own audio)."""
    out = []
    starts = pick_starts(cut, tier)
    people = people or {}
    for bi, beat in enumerate(cut["beats"], 1):
        if not any(a.get("tier", 1) <= tier for a in beat.get("audio", [])):
            continue   # not in this tier's film: its legacy rows are not built or rendered
        for v in beat.get("video", []):
            if v.get("type") != "video" or not v.get("clip"):
                continue
            if in_sync(v["clip"], v["in_s"], v.get("dest_in_s"), starts.get(v.get("pick"))):
                continue
            if NOBODY.match(people.get(v["clip"]) or ""):
                continue   # speech off camera: the same exemption the catalogue rule gives
            if is_exempt(v["clip"], exempt):
                continue   # the editor allowed this clip by name (intent.json, kind talking_cover_exempt)
            s = spoken_seconds(transcripts, v["clip"], v["in_s"], v["out_s"])
            if s is None:
                out.append(dict(beat=bi, pick=v.get("pick"), clip=v["clip"], in_s=v["in_s"], out_s=v["out_s"],
                                dest_in_s=v.get("dest_in_s"), speech_s=None, status="unknown"))
            elif s >= min_s:
                out.append(dict(beat=bi, pick=v.get("pick"), clip=v["clip"], in_s=v["in_s"], out_s=v["out_s"],
                                dest_in_s=v.get("dest_in_s"), speech_s=s, status="talking"))
    return out


NOBODY = re.compile(r"^\s*(nobody|no people|no person|none|no one|no humans?)\b", re.I)


def is_exempt(clip: str, exempt: set[str] | None) -> bool:
    """A clip the editor has allowed by name despite speech on its audio: the unit itself or its source take."""
    return bool(exempt) and (clip in exempt or source_of(clip) in exempt)


def talking_units(transcripts: dict, units: list[dict], min_s: float = TALKING_COVER_MIN_S,
                  exempt: set[str] | None = None) -> dict[str, str]:
    """unit id -> reason for every catalogue unit whose range holds transcribed speech; such a unit is not cover.
    A unit whose tag says nobody is visible (`people`) is exempt: the speech is off camera, there are no lips to
    mismatch. A person seen from behind or hands-only is not exempt; the tag comes from sampled stills."""
    out = {}
    for u in units:
        if NOBODY.match(u.get("people") or "") or is_exempt(u["id"], exempt):
            continue
        s = spoken_seconds(transcripts, u["id"], u["in_s"], u["out_s"])
        if s is not None and s >= min_s:
            out[u["id"]] = f"{s:.0f} s of transcribed speech in the unit: someone on screen is talking"
    return out
