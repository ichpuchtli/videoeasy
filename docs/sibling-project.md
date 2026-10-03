# Sibling project: register, sound bites and meaning search

Status: design proposal, 3 Oct 2026. Nothing here is built yet. It reuses
what videoeasy learned and as much of its code as is film-agnostic.

## Decision summary

- **Build the sound-bite finder first.** It is the most measurable job: in/out
  points, edge room, overlap and loudness are numbers. It also reuses the most
  videoeasy code, and the editor can rate its output in Resolve on the first
  day. Register comes second, because it needs the editor's labels before any
  score means anything. Corpus search comes third, because it needs more than
  one talk.
- **Unit of analysis: the utterance.** That is one speaker's continuous speech,
  split at real pauses, 3–20 s long. It is an index, not a cut. Bites, register
  readings and search hits all point at utterance ids plus word times.
- **Measure before judging.** Measure who speaks, whose face is on screen,
  lips moving, pitch, energy, rate, pauses, overlap, loudness and edge room.
  Leave to local models only what cannot be measured: the register reading per
  modality, self-containedness, quotability and the relation between two
  passages.
- **Judge each modality separately, then fuse them in code.** Face, voice and
  words each get their own reading, and no prompt sees another modality's
  answer. Disagreement between them is recorded as a finding, not averaged
  away. Someone smiling while describing a loss is exactly what an editor
  looks for.
- **Use embeddings for search only.** Understanding stays in legible per-talk
  documents. Every relation an embedding proposes is checked by a local LLM
  reading both passages, then accepted or rejected by the editor.
- **Code layout: a separate repo next to videoeasy.** It depends on videoeasy
  through an editable path source while developing and a pinned git source for
  release, and it imports only the film-agnostic modules listed in §7. A shared
  `core` package gets extracted once both projects have stabilised on it.

## 1. The three jobs, as queries and outputs

| Job | The editor asks | The tool returns |
|---|---|---|
| Register | "Every moment someone is wry / animated / tender / flat, in these 40 clips." "Where does the voice say one thing and the face another?" | Ranked utterances with per-modality evidence (face, voice, words), the fused register label, the disagreement flag and a status per modality |
| Sound bites | "Top 20 self-contained 8–25 s bites from this 90-minute talk, with clean in/out room." "Same, but only where the speaker is on camera and animated." | Bites with word-exact in/out, measured head/tail room, flags (dangling start, unresolved reference, overlap, disfluency, low ASR confidence), rubric scores, pairwise rank, and a Resolve selects timeline |
| Meaning search | "Where else does anyone say something like this, and how does it relate?" "Everything about trust across these six talks." | Passages grouped by verified relation (restates / elaborates / contradicts / answers / example_of), each with quoted evidence from both sides and the channel that matched |

## 2. Principles carried over, and the lesson each one comes from

| Principle | Lesson in videoeasy |
|---|---|
| Only local models see pixels and audio. The frontier model reasons over text. | Footage privacy, and the split worked: the story room never needed raw media. |
| Measure what can be measured. Judge only the rest. | A still-frame judge passed talking cover (someone on screen speaking under someone else's audio) and camera shake. Both became measurements (`speech.talking_cover`, `steadiness.py`), and both defects went to zero. |
| Judge modalities apart. The transcript never enters a vision prompt. | A spoken topic is not a visible fact. A fusion pass that merged speech into the visual fields put a spoken subject into a "what is seen" field. |
| A window is an index, not a cut. | Four frames across a 358 s take invented a transition. Window edges now record whether they sit on a speech pause or an even split (`segments.py`). |
| Honest status: `ok` / `unchecked` / `invalid`, no fallback scores, a manifest per output, content-hash caches. | Error verdicts had been converted into mid scores and accepted. Caches keyed on file existence served stale frames (`evalrun.py`). |
| Never act on a single ASR pass. A flag gets a padded recheck, then a waveform check. | Picks flagged as "lost" on one Whisper pass turned out to be intact. `speech.recheck` has states for ASR miss, real loss, waveform intact, text wrong and unresolved. |
| Propose with one model, verify with another, and choose deterministically. | `brollmatch`: a text model proposes, a vision model verifies each candidate, and code picks. Invented ids are dropped, and a missing verdict is never eligible. |
| Editorial decisions are rules, not prompt hints. | `intent.py` resolves decisions with supersession, and the builder refuses to write a cut that breaks one. A prompt note alone was ignored more than once. |
| Calibrate against the editor before trusting any number. | The cut scorer's 0–3 scale is self-consistent but uncalibrated. Run-to-run variance was never measured, so small deltas between versions mean nothing yet. |
| Validate the fixture before grading the model. | A synthetic speech control returned success but wrote silent audio, which invalidated a whole model run. |
| Test the actual route the media takes, end to end. | Ollama's MLX backend silently dropped images, and the model described shots it never saw. One omni model followed instructions on images or audio alone but failed on the two combined. |
| Never let a prompt stay silent about a missing measurement. | Given no motion data, the model reported "no significant optical flow detected". |
| Never put example phrases in open-ended prompt fields. | The model parroted a theme example verbatim across dozens of shots. |
| Run one GPU job at a time. | LM Studio and Ollama running concurrently were about 4x slower, and the OS killed a process for low memory. |

## 3. Where "it all fits in one context" breaks

videoeasy holds a whole short film in one context, about 13k transcript tokens
plus shot records. That still holds for **one** talk: 90 minutes is about 13–15k
words. It breaks in two places:

1. **Local models.** A 32–40k-token local context cannot hold a whole talk plus
   instructions plus output. Local passes work on chapter-sized chunks with
   overlap.
2. **A corpus.** Twenty 90-minute talks are about 400k tokens.

Replacement, a hierarchy that keeps the legible-document principle:

- **Talk bible** (one markdown file per talk, fits in a frontier context):
  chapters with time spans, a numbered claims/ideas list (each with span and
  short quote), candidate bites with flags, register highlights, open
  questions. Built locally in chunks, then corrected in conversation. Curated
  edits go in a separate file that ingest never overwrites, as with
  `bible-synthesis.md`.
- **Corpus map** (one file): every talk's chapter titles and claim lines.
  20 talks × ~30 claims ≈ 600 lines, which still fits in a frontier context.
  It is the table of contents the story conversation reasons over.
- **Embedding index** (search aid, not understanding): one vector per
  utterance and per claim, kept as separate channels (`spoken`, `claim`, and
  later `seen`). A hit reports which channel matched. This rule comes from the
  clip-context review: a search for visible roots must not succeed only
  because someone said "roots".
- **Verification**: every proposed relation between two passages is read by a
  local LLM with both passages and their neighbours, labelled with a relation
  type and quoted evidence, or rejected. Only verified relations reach the
  bible. The editor's accept/reject decisions are stored as rules
  (`relation_rejected`, `relation_confirmed`) and are never re-proposed.

## 4. Pipelines

Shared front end, run once per source:

```
probe ─► transcribe (word times) ─► diarize ─► utterances ─► [face tracks + ASD]  (video sources only)
```

| Stage | Reuses | New |
|---|---|---|
| probe | `probe.py` (ffprobe, timecode) | — |
| transcribe | `transcribe.transcribe_file` (mlx-whisper, word timestamps, `condition_on_previous_text=False`). Confidence fields and the silence gate from `transcribe_broll_file` / `peak_db`. Lav-over-camera audio from `audiosync` | Verbatim mode for disfluencies (Whisper tends to drop fillers; test an initial prompt containing fillers and compare against a hand count before trusting a disfluency rate) |
| diarize | — | pyannote `speaker-diarization-community-1`. Overlapped-speech regions are kept as their own spans. Diarization labels are `spk_0…`, never names. Names come from an editor-curated registry |
| utterances | `segments.speech_boundaries` / `cut_points` logic: pause-snapped edges, each edge recording `speech_pause` or `even_split` | Split by speaker turn first, then pauses. 3–20 s bounds |
| face tracks + ASD | `frames.extract_frame` (graded crops; grading before vision is non-negotiable), `speech.talking_units` idea | Face detection and tracking at 25 fps on speech spans only. Active-speaker detection (Light-ASD) links a face track to a diarized voice: "on screen and speaking", "off screen", "on screen, someone else speaking" |

```jsonc
// out/utterances.json: one entry per utterance (an index, not a cut)
{"id": "talk03_u0142", "source": "<file>", "in_s": 812.40, "out_s": 826.95,
 "speaker": "spk_1", "edges": ["speech_pause", "speech_pause"],
 "words": [{"w": "So", "s": 812.40, "e": 812.61}, ...],
 "asr": {"avg_logprob": -0.21, "no_speech_prob": 0.02},
 "overlap_s": 0.0,
 "on_screen": {"status": "ok", "track": "f07", "speaking_face": true, "face_px": 220},
 "status": "ok"}
```

### 4.1 Register (job 1)

Per utterance, four independent readings, then fusion in code:

| Reading | How | Status rules |
|---|---|---|
| **Voice, measured** | Prosody: f0 range and variability, energy variance, speech rate (words/s from word times), pause ratio, loudness (`deliver.measure` / `layin.lufs_peak` ebur128 helpers) | `unchecked` when overlap > 0.3 s or ASR confidence is below a threshold |
| **Voice, model** | Dimensional SER: audeering `wav2vec2-large-robust-12-ft-emotion-msp-dim` (arousal, dominance, valence, ~0–1). Optionally emotion2vec+ categorical as a second opinion | Stored as raw dimensions. Never turned into a category in this stage |
| **Face, measured + seen** | Facial action-unit intensities per frame on the speaking face track (OpenFace 3.0), summarised over the utterance. Then the VLM (gemma-4 via LM Studio) gets 6–8 graded face crops, in order, and is asked only for **visible** expression cues (smile, brow raise, gaze aversion, head shake). No transcript, no name, no story | `unchecked` when the face is off screen, < ~80 px, or side-on. Absence is reported, never filled in |
| **Words** | Local text LLM (Ollama, `think: false`, `format: json`) reads the words plus one utterance either side and returns a register reading from the words alone | `invalid` on a label outside the vocabulary |

Fusion is deterministic. Each reading maps onto the editor's vocabulary
(§5) with evidence. The fused label requires at least two modalities in
agreement. Otherwise the record carries `disagreement` naming which
modalities said what. Disagreement is a searchable result, not an error.

```jsonc
// out/register.json
{"utterance": "talk03_u0142",
 "voice":  {"status": "ok", "arousal": 0.71, "valence": 0.38, "dominance": 0.55, "rate_wps": 3.4, "f0_range_st": 9.1},
 "face":   {"status": "ok", "au": {"AU06": 0.8, "AU12": 1.4}, "seen": "broad smile, eyes narrowed, head tilts back"},
 "words":  {"status": "ok", "reading": "rueful", "evidence": "\"we lost the lot that year\""},
 "fused":  {"label": null, "disagreement": {"face": "amused", "words": "rueful"}, "candidates": ["wry"]},
 "manifest": "register.manifest.json"}
```

Optional discovery pass: a native audiovisual model (Nemotron Omni 30B ran
locally through MLX and passed basic audiovisual controls) can be asked for
**checkable events**: laughter, hesitation, a restart, a long pause after a
question. Its output is an attributed observation, confirmed by measurement
or Whisper before use. In the pilot, the same model omitted audible speech and
called a 2 s audio delay "keep", so it never becomes the speech reference or
the sync judge.

### 4.2 Sound bites (job 2)

```
utterances ─► candidates ─► gates (measured) ─► rubric (LLM) ─► pairwise rank ─► recheck top-N ─► selects
```

1. **Candidates.** Sentence spans from word times and punctuation, joined
   into runs of 1–4 consecutive sentences in one speaker turn, 5–30 s long.
   Starts and ends sit on sentence boundaries. Pause lengths are kept as
   evidence.
2. **Measured gates.** These produce flags, and only a flagged
   `fatal` bars a candidate.
   - *Edge room*: silence before the first word and after the last, measured
     from the source's own energy per channel, as layin's
     `max_head_s`/`max_tail_s` already do. The transcript's word times are not
     enough, because they miss the word beside the cut. Under 0.15 s is a flag.
     A word bleeding into the room is `fatal`.
   - *Overlap*: diarized overlap inside the span over 0.3 s is a flag.
   - *Dangling start*: the first token is in {and, but, so, because, which,
     that's, as I said…}. This is a flag; the editor may trim.
   - *Disfluency*: fillers and restarts per 10 s, only after the verbatim
     check in §4 has been validated.
   - *Level*: loudness and true peak of the span, and clipping.
   - *ASR confidence*: segment `avg_logprob` / `no_speech_prob`.
   - *On camera*: from ASD, carried for filtering.
3. **Rubric (local LLM, one call per candidate, ±1 utterance of context).**
   It returns `self_contained` (can a first-time viewer follow this without
   what came before; unresolved "it/this/they" listed), `specific`
   (concrete vs generic), `charge` (with the register record attached as
   measured context, not as an instruction), `quotable`, and a one-line
   reason. Each is 0–3 with `status`.
4. **Pairwise rank.** The rubric scores shortlist; the final order comes from
   pairwise comparisons ("which bite would you rather open a chapter with")
   with order swapped on a repeat. Absolute scores drift; pairwise is more
   stable and is what the editor will be calibrated against.
5. **Recheck.** Every top-N bite gets a padded re-transcription
   (`speech.transcribe_excerpt`, `speech.analyse`, `speech.recheck` states).
   An edit on a polarity, quantity, modality, name or pronoun token flags the
   bite for listening. Nothing is trimmed automatically.

```jsonc
// out/bites.json
{"id": "talk03_b017", "utterances": ["talk03_u0142", "talk03_u0143"],
 "in_s": 812.40, "out_s": 831.10, "words": "So the year we lost the lot ...",
 "room": {"head_s": 0.42, "tail_s": 0.65, "status": "ok"},
 "flags": ["dangling_start:so"], "overlap_s": 0.0, "lufs": -23.8, "on_camera": true,
 "rubric": {"status": "ok", "self_contained": 3, "specific": 2, "charge": 3, "quotable": 2,
            "unresolved_refs": [], "reason": "complete story beat, ends on a turn"},
 "rank": {"wins": 11, "comparisons": 14},
 "recheck": {"status": "ok", "state": "cleared_on_recheck"}}
```

### 4.3 Meaning search and related dialogue (job 3)

1. **Chunked local reading** of each talk into chapters and claims. Each
   claim carries a span and a quote that must be found verbatim in the words
   (checked in code; an unfound quote is `invalid`).
2. **Index.** Local embeddings per utterance (`spoken` channel) and per claim
   (`claim` channel), stored in SQLite with the vectors. Names and terms go
   through an editor-maintained alias table (the `speech.ALIASES` idea moved
   into per-project config) and a plain text index, so a name search never
   depends on an embedding.
3. **Query.** Hybrid (text + vector) recall of ~50, then a local reranker or
   the LLM verifier on the top ~15.
4. **Related dialogue.** For a passage, take its nearest neighbours in other
   talks, then run the verifier on each pair: relation in {restates,
   elaborates, contradicts, answers, example_of, unrelated}, evidence quoted
   from both sides, `status`. Only verified, non-`unrelated` edges are stored.
   The editor's accept/reject decisions are rules.

```jsonc
// out/relations.json
{"a": "talk03_c12", "b": "talk07_u0331", "relation": "contradicts",
 "evidence": {"a": "\"you can't rush it\"", "b": "\"we pushed it through in a season\""},
 "proposed_by": {"channel": "claim", "similarity": 0.81},
 "verified_by": {"model": "<text model>", "prompt_hash": "...", "status": "ok"},
 "editor": null}
```

## 5. Register vocabulary

Use a **small editor-defined vocabulary plus dimensional scores**, not generic
emotion categories.

- Categories like "happy/sad/angry" come from acted-speech datasets and say
  little about how a moment cuts. An editor wants "wry", "earnest",
  "animated", "tender", "deadpan", "searching", "flat/tired" and "playful".
  Each depends on context and footage, and each implies a use in a cut.
- Keep it to 6–10 terms, each with a one-sentence **definition**. Do not put
  example phrases in the prompt (see §2). Reference clips the editor picked
  live in the calibration set, not in the prompt.
- The dimensional scores (arousal, valence, dominance, rate, f0 range) are
  stored raw and stay searchable: "high arousal, low valence" is a query even
  when no label fits.
- Labels describe **displayed register in a clip**, never a person's inner
  state. "Voice high-arousal, smile visible, words about loss" is
  observation. "She was hiding her grief" is not something this tool says.

Calibration, before any label is shown as a result:

1. **Stratified sample.** 80–120 utterances spread across speakers, sources
   (interview, field, stage), measured arousal bins, face visible / not, and
   overlap. The editor labels each with up to two terms or "neutral".
2. **Held out.** Split into dev (prompt and threshold work) and test (touched
   once per prompt version). Report per-term precision/recall and confusion
   on test only.
3. **Variance.** The same request three times at the working temperature.
   Report agreement. A term the model cannot repeat itself on is not yet a
   usable search facet.
4. **Pairwise.** For bites and for the register "strength" within one term,
   ask the editor to choose between pairs (30–50 pairs) and measure the
   model's agreement with those choices, which is more reliable than absolute
   ratings.
5. **Ablations.** Words-only vs voice-only vs face-only vs fused, on the same
   test set, so the project knows which modality earns its compute.

## 6. Output surfaces

- **Talk bible** (`out/<talk>/talk-bible.md`): chapters, claims with
  timecodes, top bites with flags and rank, register highlights and
  disagreements, unchecked coverage ("face not visible 38 % of speech").
  Regenerable. Curated corrections live beside it in `talk-notes.md`, which
  no script writes.
- **Corpus map** (`out/corpus-map.md`): every talk's chapters and claims, plus
  verified cross-talk relations.
- **JSON indexes**: `utterances.json`, `register.json`, `bites.json`,
  `relations.json`, `index.sqlite`, each with a `.manifest.json`
  (`evalrun.write_manifest`: input hashes, model, prompt hash, what was left
  unchecked) and `refuse_overwrite` semantics.
- **Resolve selects timeline.** Bites laid in rank order with a gap between
  them. Each item gets a marker whose name is the rank and register and whose
  note holds the flags. Layout follows layin's `plan` → `lay` → `readback` →
  `conform` (only `AppendToTimeline` with `recordFrame` places without
  rippling; every write is verified by re-reading). The editor marks
  keep/maybe/reject with flags. A read-back turns those marks into calibration
  labels, which closes the loop without a new UI.

## 7. Code layout

**Recommendation: a separate repo next to videoeasy.**

```toml
# sibling/pyproject.toml
[project]
dependencies = ["videoeasy"]

[tool.uv.sources]
videoeasy = { path = "../videoeasy-public", editable = true }   # while developing
# videoeasy = { git = "https://github.com/<owner>/videoeasy", rev = "<sha>" }   # for release
```

Why not a uv workspace now: videoeasy is being published as it stands, and
restructuring it into `packages/core` before a second consumer exists would
mean guessing the boundary. The editable path source gives atomic local edits
to shared code anyway. Rule: when the sibling needs a shared module to change,
the change is made **in videoeasy**, generalised and tested there, never copied.

**Extract a `core` package** (a uv workspace inside the videoeasy repo,
consumed through a git `subdirectory=` source) once milestone 1 is done and
both projects import the same set of modules unchanged for a few weeks.
Candidates, and what in each is still documentary-specific:

| Module | Reusable as is | Documentary-specific, stays out of core |
|---|---|---|
| `evalrun.py` | statuses, `fingerprint`, `text_hash`, `write_manifest`, `refuse_overwrite` | — |
| `transcribe.py` | `transcribe_file`, `transcribe_broll_file` confidence filters, `peak_db` | `run()` assumes the a-roll/b-roll directory split |
| `segments.py` | `speech_boundaries`, `cut_points`, edge provenance | `build()` assumes shot-shaped units and the film config |
| `speech.py` | `tokens`, `analyse`, `edits`, `transcribe_excerpt`, `waveform_match`, `recheck`, `spoken_seconds` | cut/pick shapes in `pick_starts`, `talking_cover`; the alias table belongs in per-project config |
| `vlm.py` + the `check` trap-test | `chat_vision`, `extract_json`, `loaded_context_length` | — |
| `frames.py` | `extract_frame` (graded), `sample_times`, `contact_sheet` | `run()` and sample counts tied to shots/windows |
| `config.py` | grade profiles, `list_videos` | path layout of a film |
| `audiosync.py` | `envelope`, `xcorr_match` | DJI Mic session naming and chunking |
| `deliver.py` | `measure`, two-pass `loudnorm` master | — |
| `resolve.py`, `layin.py` | Resolve bridge, `readback`, `conform`, loudness helpers | `plan()` is cut-specific (beats, cover on V3, crossfades between picks). A selects layout needs its own small `plan` |
| `intent.py` | rule resolution (supersedes, latest on topic) | the rule kinds are cut-specific; the sibling adds its own kinds |

Never imported by the sibling: `brollmatch`, `cutbuild`, `cuteval`,
`storycheck`, `bible`, `selects`, `moves`, `steadiness`, `birds`, `species`.

### Milestone 1: sound bites on one long talk

> **Status (Oct 2026).** A first cut of this milestone now lives in videoeasy
> itself as [`videoeasy.talks`](talks.md), built for an event shoot's lav
> recordings: chunk joining, float-safe gain, word-timed transcript, sentences
> with measured room and off-mic flags, LLM beats and candidates with cover
> briefs, an independent standalone check, selects CSV and a Resolve selects
> timeline. Not yet built: diarization, pairwise top-20, the editor-rated
> calibration run, and the exit criteria below. Extract it into the sibling
> repo when the register work starts.

Input: one 60–90 min talk, single speaker plus an interviewer or audience,
camera audio or lav. Stages: transcribe → diarize → utterances → candidates →
gates → rubric → pairwise top 20 → recheck → Resolve selects timeline → the
editor marks keep/maybe/reject → read-back.

Exit criteria:

- Every output has a manifest. No `unchecked` item carries a score. A re-run
  with unchanged inputs does no model work.
- 0 bites whose in/out lands inside a word (measured head/tail room). The
  selects timeline conforms 100 % on read-back.
- At least 10 of the top 20 rated keep or maybe by the editor, against a
  baseline of the 20 longest pause-bounded spans. The same comparison is
  repeated on a second talk the prompts were not developed on.
- Rubric repeatability: the same top-20 request three times keeps at least
  15 of 20 bites in common.
- The editor's time from finished ingest to a marked selects timeline is
  recorded. That is the number this project exists to shrink.

Milestone 2: register on the same talk, plus one set of field clips, with the
calibration in §5. Milestone 3: corpus map and verified relations across at
least three talks, evaluated on 20 editor-written queries (5 with no true
answer), reporting precision@5 and the false-relation rate.

## 8. Risks and open questions

- **Diarization on overlapping or similar voices.** Interviews with
  backchannel ("mm", "yeah") fragment turns. Treat overlap spans as their own
  class, and never let a bite or register reading span a speaker change
  silently.
- **SER domain shift.** The dimensional model was trained on podcast speech.
  Wind, distance, handheld field audio and lav rustle will move it.
  Calibration must include field material, and synced lav audio is preferred
  wherever it exists, as videoeasy's `transcribe` already does.
- **Faces.** Small, side-on or backlit faces make AU and VLM readings
  `unchecked`, not neutral. Expect large unchecked fractions on
  observational footage, and report them.
- **Rubric bias.** An LLM rubric will tend to favour pithy, generic lines,
  just as the cut scorer favoured literal illustration. Pairwise calibration
  against the editor is the check. Expect prompt patches; each one gets a
  regression fixture, not just a prompt note.
- **Licences.** The audeering dimensional model and OpenFace 3.0 are released
  for research use. That is fine for the editor's own work, but it blocks
  commercial distribution of a pipeline built on them. Check pyannote and
  Light-ASD licences before shipping anything.
- **Consent and privacy.** Register labels describe what a clip displays.
  They are stored against footage, never aggregated into per-person profiles
  ("speaker X is anxious"), and never leave the machine. The public repo
  carries code and method only, never labels or transcripts of real people.
- **Compute.** Face tracking at 25 fps and ASD are run on speech spans only.
  Run one GPU job at a time, and measure throughput on milestone 1 before
  estimating a corpus run.

## Appendix: models and tools named

Verified (installed locally, or the model page confirmed on 3 Oct 2026):

- Installed and used by videoeasy: `google/gemma-4-26b-a4b` (LM Studio,
  vision; images and text only, it does not take audio),
  `qwen3.8:27b-mtp-q8_0` (Ollama, text judge), `mlx-community/whisper-large-v3-mlx`.
- Installed: `text-embedding-nomic-embed-text-v1.5` (LM Studio embedding),
  `BAAI/bge-small-en-v1.5` (Hugging Face cache).
- Run locally in videoeasy's model pilot: Nemotron 3 Nano Omni 30B-A3B
  (community 8-bit MLX conversion, native audio + ~1 fps video).
- Confirmed to exist: `audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim`
  (arousal/dominance/valence, research use only);
  `emotion2vec/emotion2vec_plus_large` (9-class SER, FunASR); pyannote
  `speaker-diarization-community-1` (runs on Apple Silicon via MPS); OpenFace
  3.0 (AUs, landmarks, gaze, expression; research use); Light-ASD (CVPR 2023
  active-speaker detection, code and weights on GitHub); Ollama
  `qwen3-embedding` (0.6b/4b/8b) and `bge-m3`.

Not verified here (plausible, check before relying on them): openSMILE /
Parselmouth for prosody features, a faster CoreML diarization port, MediaPipe
or InsightFace for face detection and tracking, a verbatim Whisper variant for
disfluencies, and whether the Light-ASD weights run acceptably on MPS.
