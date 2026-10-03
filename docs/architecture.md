# Architecture and decisions

## The core insight

A 10–15 minute documentary's raw material is small: about an hour of
transcript (around 13k tokens) plus a few dozen to a few hundred richly
described shots. **The entire film fits in one frontier-model context.** So
the system deliberately avoids a retrieval/RAG architecture. Understanding
lives in a single legible document, the **footage bible**, which a strong
model holds in full and a person can read, argue with and correct.
Embeddings are deferred: they are not the understanding layer.

### Where that stops holding

The estimate grows faster than expected. One short film's bible reached
about 67,000 words once long takes were windowed and audit evidence was
attached, plus about 5,400 words of synthesis, and the B-roll catalogue alone
ran to about 16,000 words. A frontier model still holds that, but the local
text models that propose cover do not: removing unusable units brought one
proposer prompt from about 26k tokens to about 14k of a 40k context.

For multi-hour corpora (long-form talks, a series, a whole archive), the
one-document assumption fails. The recommended route keeps the principle (a
legible record is the source of truth) while adding navigation:

- a compact overview (brief, people and aliases, active decisions, arc,
  motifs, gaps) and a complete short index of moments, with detailed records
  expanded on demand;
- exact ids, text search, aliases, filters and explicit links in JSON or
  SQLite first;
- vector search only when that proves insufficient, and then indexing
  **visual content, spoken content and interpretation separately** and
  returning which channel matched. A search for visible roots should not
  succeed just because someone mentioned roots.

## Split of labour

- **Local models** (LM Studio, Ollama, mlx-whisper on an Apple Silicon Mac)
  are the only things that ever see pixels or hear audio. That is good for
  privacy, and it runs at overnight-batch pace.
- **The frontier model** (Claude, in a Claude Code session) does narrative
  reasoning over text only: the bible, the transcript, scores, metadata and
  the conversation. It orchestrates local judges and never grades footage by
  looking at it ([lessons 8.1](lessons.md#81-the-frontier-model-does-not-judge-by-eye)).
- **Deterministic code** measures everything that can be measured: shake,
  drone moves, loudness, joins, exposure, speech alignment and rule
  compliance. A model's judgment never stands in for a measurement.

## Pipeline: ingest

```
probe       ffprobe inventory: duration, fps, embedded start timecode (for the
            Resolve conform), audio presence
shots       PySceneDetect splits multi-shot B-roll files; A-roll files stay whole
transcribe  mlx-whisper large-v3, word-level timestamps (a radio cut selects
            words, not segments); prefers waveform-synced lav audio when verified;
            --broll pass with confidence filters and explicit silent/no-track states
segment     long takes cut into windows at transcript pauses: an index, not a cut
frames      N frames per shot or window, GRADED per source, plus a contact sheet each
motion      Farneback optical flow → motion_energy, camera_ratio, with a gloss in words
annotate    vision model reads the graded frames + motion gloss → the editor's schema:
            subject, movement, shot_type, emotional_metaphor, visual_metaphor,
            storytelling_themes, mood, light, palette, hold_seconds, cut_notes, people
audit       (opt-in) per-frame observations + a consistency check of each tag
selects     transcript .docx → marked passages (highlight AND w:shd shading) and
            anchored comments, parsed from the XML
assemble    everything → bible-context.md
```

Measurement passes for the eval loop: `steadiness` (camera shake and steady
runs), `moves` (drone moves), `birds` (BirdNET), `species` (BioCLIP).

Every stage is idempotent and resumable, and writes its JSON incrementally.

## Pipeline: story and assembly

```
story room (frontier model, conversation)
   reads bible-context.md + bible-synthesis.md + the film brief
   writes bible-synthesis.md (human-corrected judgment) and cut-vN.json (radio cut)
   records decisions in editorial/intent.json

eval loop (local models + code), per iteration; see eval-design.md
   cuteval → storycheck → brollmatch (propose, verify) → cutbuild → layin → render

delivery
   deliver: loudness-normalised master, measured against the recorded target
```

## Decisions and their reasons

**Grade before vision.** Log frames are flat, and a vision model reads them
as "washed out, melancholy" whatever the actual mood. Grading is per source:
log footage goes through the LUT that approximates the final grade, so
annotations describe how the *finished* film feels, and non-log footage gets
a saturation and contrast lift. The first matching filename rule wins.

**LM Studio, not Ollama, for vision.** Ollama 0.31's MLX backend silently
dropped images. `videoeasy check` traps that failure with a synthetic image
whose colours the model must name exactly.

**The bible is a document, not a database.** The editor needs deep relational
understanding: themes, feelings, A-roll ↔ B-roll relationships. Embeddings
encode relations silently. A document makes them legible and correctable, and
corrections persist. Machine-generated context (`bible-context.md`) is kept
separate from conversational synthesis (`bible-synthesis.md`), so an ingest
re-run never clobbers curated judgment.

**Prior editorial intent is data.** Transcript markup (all colours mean keep;
a designated colour means client-added), anchored comments (reorder moves,
readiness gates, coverage notes, cut flags) and the editor's own draft radio
cut are extracted into the bible. The story room starts from human judgment,
not a blank page.

**Decisions are rules.** Every decision lives in `intent.json` with who made
it, its source and what it supersedes. Code checks the cut and the timeline
read-back against the active rules, with no model in the loop. A quality
score can never override a decision.

**Film specifics are data, not code.** Recording roles, ASR aliases, the
register paragraph, Resolve project and bins, LUTs and placeholder cards all
live in `data/films/<film>/film.yaml`. The code reads them, and the repository
holds no film.

**A window is an index.** Long takes are windowed for honest per-moment tags
and for search. The edges record whether they came from a speech pause or an
even split, and no edge is ever presented as an edit point.

**Evidence is status-bearing and hash-keyed.** Every eval result is `ok`,
`unchecked` or `invalid`, and only `ok` carries a score. Caches are keyed on
content hashes. Every report has a manifest.

**Timecode from day one.** `probe` records embedded start timecode per clip,
and shot records carry source-relative seconds and fps. The Resolve conform
needs both.

**Round trip, not export.** The lay-in builds a timeline *and reads it back*,
so conversation and checks continue from the actual edit, not from the last
thing a model proposed.

## Phases

1. **Story room.** Conversational depth over the bible: radio-cut work,
   B-roll matching, the synthesis layer.
2. **Assembly.** The cut JSON as the model's working map → DaVinci Resolve via
   its Python API, built and read back. The local-model eval loop judges each
   render.
3. **Pipeline and taste** (future). Generalise beyond one film: moment records
   that join seen, heard and context; true video embeddings where they earn
   their place; a small ranking model trained on the editor's accept/reject
   history and films they admire. Explicitly not fine-tuning an LLM.

## Known approximations

- An older log profile graded through a newer profile's LUT renders slightly
  dark and contrasty. Accepted; final colour happens in Resolve.
- Whisper's wording differs slightly from a human-edited transcript, so marked
  passages are fuzzy-matched onto timecodes.
- The scene-detection threshold depends on content, and low-contrast log
  footage misses cuts. Contact sheets are the verification surface.
- Shake and move thresholds are provisional working numbers, not calibrated
  values.
