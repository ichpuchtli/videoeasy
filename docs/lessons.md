# Lessons

What building videoeasy taught, as short entries: the failure, how it was
found, the evidence, the rule now in the code, and the check that guards it.
These came from editing real documentary footage on one Mac with local
models, between July and October 2026. Numbers are from that work; the films
themselves are not part of this repository.

Read this before changing a prompt, a model, a threshold or a cache. Most
entries record a failure that looked like success at the time it happened.

Contents:

1. [Local model plumbing](#1-local-model-plumbing)
2. [Prompting vision models](#2-prompting-vision-models)
3. [Sampling, windows and grading](#3-sampling-windows-and-grading)
4. [Measure, don't judge](#4-measure-dont-judge)
5. [ASR and speech alignment](#5-asr-and-speech-alignment)
6. [Honest evaluation](#6-honest-evaluation)
7. [Editorial intent as rules](#7-editorial-intent-as-rules)
8. [Working with a frontier model in the loop](#8-working-with-a-frontier-model-in-the-loop)
9. [Open items](#9-open-items)

DaVinci Resolve API lessons have their own document:
[resolve-scripting.md](resolve-scripting.md). Model-by-model findings are in
[local-models.md](local-models.md).

---

## 1. Local model plumbing

### 1.1 A vision backend can drop images silently

- **Failure.** Ollama 0.31's MLX backend accepted image requests, returned
  confident and plausible shot descriptions, and never saw the images.
- **Found by.** `prompt_eval_count` stayed at the text-only token count
  whatever images were attached.
- **Rule.** Vision goes through LM Studio's OpenAI-compatible API
  (`localhost:1234/v1`). Never point `models.vision_url` at Ollama without
  re-running the trap.
- **Guard.** `videoeasy check` sends a synthetic 1024 px image (a red-orange
  ground, a yellow circle, a blue square) and asks for shapes and colours as
  JSON. The model must name the real colours. `SUSPECT` means images are not
  arriving: stop, because every annotation after this would be invented.

### 1.2 Thinking models spend the budget before they answer

- **Failure.** GLM-4.6V returned an empty answer with
  `finish_reason=length`: its reasoning used the whole token budget.
- **Rule.** `vlm.chat_vision` defaults to `max_tokens=6000` and retries once
  at double. A length-cut answer counts as no answer, because truncated JSON
  is useless. The client raises a clear error instead of returning a fragment.
- **Related.** A 27B text model served through LM Studio spent 5,999 of 6,000
  generated tokens on reasoning, returned nothing and took 336 s. LM Studio's
  documented reasoning-off option rejected that model's configuration, and
  native MLX with thinking disabled fixed it. Text judges served by Ollama
  run with `think: false` and `format: json`.

### 1.3 The default context window truncates the busiest shots

- **Failure.** 9 of the first 130 tags in one batch failed with
  `finish_reason=length`. Every failure was a dense, busy set of frames, and
  none was related to take length. Raising `max_tokens` changed nothing.
- **Cause.** LM Studio loads models with a 4096-token context. Eight graded
  frames plus the annotation prompt fill most of it, leaving no room for the
  answer.
- **Rule.** Load the vision model with `--context-length 16384`.
- **Guard.** `check` reads the loaded window from LM Studio's
  `/api/v0/models` and fails below `vlm.MIN_CONTEXT_TOKENS` (16384). A plain
  `annotate` re-run retries only the error entries.

### 1.4 Run one local model job at a time

- Running Ollama and LM Studio jobs together made each about 4x slower. With
  both models resident, the OS killed a background process for low memory.
- Three concurrent audit jobs on one LM Studio instance produced context
  errors. Retrying the same requests one at a time succeeded with the same
  saved observations.
- **Rule.** Schedule GPU jobs one at a time. The proposer (Ollama) and the
  scorer (LM Studio) never run together.

### 1.5 An image trap does not test an audiovisual route

- **Failure.** A 12B omni model's default MLX video-first chat template
  produced wrong colours and lost a spoken negation. An explicit audio-first
  template fixed the colour tests. Image delivery alone had looked fine.
- **Rule.** Before trusting an audiovisual model, run paired controls that
  each change one input: the same pictures with a changed spoken instruction
  (a negation), the same audio with the picture order reversed, exact
  silence, and a spoken code. Record media-token counts and tensor shapes. A
  processor that runs without error has not shown that the model perceives
  anything.

### 1.6 Validate the test fixture before grading the model

- **Failure.** macOS `say` returned success inside a sandbox and wrote
  zero-sample AIFF files. Padding turned them into silent WAVs, which made
  every spoken control a silence control. That whole run was invalidated.
- **Rule.** Check each fixture's decoded duration, non-zero samples, intended
  transformation and hash. Transcribe synthetic speech independently before
  using it as a control.

### 1.7 A model's name is not its modality

- The 26B-A4B Gemma in production takes text and images. It does not take
  audio. Sending it a video container, or calling the pipeline "video",
  gives it no hearing. Check the model card for the exact variant, then prove
  each modality with a control (1.5).

### 1.8 No silent fallback between backends

- The evidence audit talks to LM Studio's native `/api/v1/chat` with reasoning
  off and conversation storage off on every request, and with bounded output
  budgets (1,536 tokens by default, configurable from 128 to 32,768). When
  that route fails, the result is `unchecked`. It never quietly retries on a
  different backend that might drop images.

---

## 2. Prompting vision models

### 2.1 The model copies example phrases word for word

- **Failure.** The annotation prompt once gave example wording for its
  metaphor and theme fields. The exact example phrase ("patience, renewal
  after neglect") then appeared verbatim on unrelated shots across the
  corpus. A bible whose shots all read alike cannot tell shots apart, and
  telling them apart is its only job.
- **Rule.** Never put concrete example phrases in interpretive fields.
  `prompts.py` demands wording specific to the shot and allows an honest
  "neutral coverage" answer. The film's register goes into the prompt as a
  description (`film.yaml` `context:`), never as sample phrases.
- **Guard.** In calibration, look for the same metaphor wording on unrelated
  shots. That repetition means the prompt is leaking.

### 2.2 A prompt that is silent about a missing measurement invites invention

- **Failure.** On a film that had never had `motion` run, tags reported "no
  significant optical flow is detected".
- **Rule.** When the statistics are absent, `prompts.py` says so explicitly.
  Otherwise the model reports having measured what it never saw.

### 2.3 Spell out the limits of the evidence

- In sampled checks against contact sheets, the tags asserted laughter
  (from stills), inferred time of day and crew roles from clothing, and
  overclaimed camera technique. Guardrails in the system prompt removed the
  first two patterns from sampled reruns. Motion claims stayed weak: an
  optical-flow figure does not show intentional tracking or continuous
  steadiness.
- **Rule.** The prompt states that the input is sparse stills with no audio;
  that sounds, dialogue, identities, relationships and exact edit points
  must not be asserted; that optical flow measures image displacement, not
  technique; and that metaphor fields are interpretations, not facts.

### 2.4 Broad themes do not help choose between shots

- One generic theme tag appeared in 38 of 228 annotations and another in 24.
  Neither is false, and neither helps choose between moments. Prefer
  observable events (someone notices something, finishes an action, restarts
  a sentence) to shot-level virtues.

### 2.5 Keep what was heard apart from what was seen

- A window's transcript rides along for search only. It never enters the
  vision prompt. Once it does, the tags start describing what the model
  "heard": a spoken topic is not a visible fact.
- The proposer's catalogue prints a unit's own audio as "said, not
  necessarily visible", in a field separate from the visual tags.
- In a text-only fusion test, the model put a topic the speaker only *talked
  about* into a visual search field.
- **Rule.** Visual observations, audio and interpretation live in separate
  fields. If vector search is ever added, index each channel separately and
  report which one matched.

### 2.6 Structured fusion still adds unsupported claims

- Fusing visual observations, nearby speech and production context into
  "moment records" made the records searchable for what was actually
  happening. The first version still copied generic metaphors from old tags
  and suggested cover that broke a rule it had been given.
- A revision separated the visual and audio fields, required evidence IDs
  appropriate to each modality, and attached active rules deterministically
  outside the model's response. That fixed most of it. It still introduced a
  factual error, naming a place as one participant's property, and inferred
  a speaker's motive.
- **Rule.** A valid evidence ID does not make a claim supported. Keep
  generated records provisional and review the important claims.

### 2.7 The rubric decides the score as much as the picture does

- Identical frames of a deliberately uncovered moment scored 1 under the
  baseline rubric, with the judge asking for illustrative footage. With the
  no-cover decision appended, the same frames scored 3. The pictures did not
  change.
- A picture-versus-words rubric rewards literal illustration over
  supporting mood, contrast, a motif callback or a breathing hold.
- **Rule.** Keep at least three judgments apart: visible relation, editorial
  purpose, and compliance with intent. Never average them into one finish
  score.

### 2.8 Judges score protected moments low even when told

- Told that a stretch stays uncovered by decision and asked how the face
  carries the words, the vision judge still scored 2 of 5 protected stretches
  0 ("mismatch").
- **Rule.** For a protected stretch, the rule finding counts. The score is
  kept only as calibration data.

### 2.9 A shortlist always has a winner

- A softmax over a short species list read 0.99998 for a single frame. The
  open-ended tree-of-life ranking for the same frames was the real check: on
  one cut, 17 of 23 species claims read under 0.1, because a wide shot of
  mixed vegetation has no single answer.
- **Rule.** Use the open-ended answer to agree, disagree or say "cannot
  tell". The shortlist only orders candidates for a person to check.
  Nothing goes on screen without human confirmation.

### 2.10 Don't infer identity from clothing or filenames

- People in similar work clothes are an identity risk. A filename-prefix
  table once assigned the presenter role by camera prefix and mislabelled one
  presenter's introduction, which was then judged against the wrong rubric.
- **Rule.** Recording roles come from an explicit per-film table
  (`film.yaml` `roles`/`clips`/`prefixes`). That table records where the
  sound was recorded. It is not a person-recognition database. Names come
  from a curated registry. When a match is uncertain, leave it unresolved.

### 2.11 Some requests never fit the budget

- Two placeholder-card stretches hit a 3,000-token budget on every attempt.
  They stay `unchecked`. They are never scored by fiat.

### 2.12 A definition's shape becomes the answer's shape

- A closed vocabulary was given as `term: definition`, and one definition
  itself began with a gloss word and a colon ("effort: straining: teeth
  gritted …"). Each time the model chose that term it answered
  `"effort: straining"`. The validator rightly marked it `invalid`, so the
  register was unreachable: 4 of the first 300 readings, all on the
  footage where that register mattered most. The other terms were fine.
- **Rule.** Definitions start with the description itself, never with a
  gloss word and a colon, and the prompt asks for the single term as written.
  Read the `invalid` reasons as a list: one repeated reason is a prompt bug,
  not model noise. Never repair such an answer into a label. Fix the prompt
  and re-read.

---

## 3. Sampling, windows and grading

### 3.1 Grade every frame before a vision model sees it

- A vision model reads flat log frames as "washed out, melancholy" whatever
  the scene's actual mood.
- **Rule.** Grading is per source (`grade:` in config, first matching filename
  substring wins). Log footage goes through the LUT that approximates the
  intended final look, so the annotations describe the finished film. Non-log
  drone footage gets a saturation and contrast lift. A known approximation is
  accepted: one camera's older log profile through the newer profile's LUT
  renders slightly dark.

### 3.2 A few frames across a long take invent continuity

- **Failure.** Four frames sampled ninety seconds apart across a 358 s
  walk-and-talk produced a tag describing a transition that was not in
  evidence.
- **Rule.** Any take at or over `segments.min_source_seconds` is cut into
  windows. Each window gets its own graded frames and its own tag, nested
  under the whole-take record, which is preserved.
- **A window is an index, not a cut.** An edge lands on a real transcript
  silence where one is within reach (`speech_pause`, with the silence length
  recorded) and on an even division where none is (`even_split`). Nothing may
  describe a window edge as an edit point.
- Locked-off takes are listed in `segments.index_only_sources`. They are
  windowed for transcript search but not sent to the vision model. A
  58-minute locked-off interview would otherwise have sent about 79
  near-identical frame sets and flooded the bible with interchangeable tags.

### 3.3 Denser sampling does not reveal camera technique

- Windowed tags still asserted "the camera is stationary" from stills, and
  shot scale still came back wrong (a wide read as medium). Stills can screen
  subject, relevance and obvious exposure problems. They cannot establish
  continuous focus, stability, whether an action completes, performance,
  pacing or an audio transition. Those either get measured (section 4) or
  stay unknown.
- A technical finding in one sampled interval must not ban a whole source
  everywhere.

### 3.4 Record which frames were actually sent

- **Failure.** The verifier's prompt described "four positions" whatever it
  was given. Frame caches held 4, 6 or 8 frames per take, inset from the
  edges, while the builder assumed fixed fractions of the take.
- **Failure.** When only the last sample of a 10 s source was usable and a
  6 s hold was requested, the builder constructed 4–10 s, reaching back into
  footage no sample had certified.
- **Rule.** Every verdict records the file and source time of each frame sent.
  The builder starts a row at the recorded time of the first accepted frame
  and never earlier.

### 3.5 Read the frame, then measure it

- `ffmpeg -f lavfi movie=...:sp=` lands on black frames after long seeks.
  Exposure is measured with `signalstats` on the extracted JPEG.
- A low mean luma in a dark forest is an observation, not automatically a
  grade defect.

### 3.6 Caches must track their inputs

- Frame extraction that skips any frame whose filename exists will not notice
  a LUT, sampling, source or boundary change. Annotation that globs `f*.jpg`
  can send stale extra frames after the sample count drops. The eval tools
  now key every cache on content hashes (6.2). The ingest stages still have
  existence-keyed caches; see [Open items](#9-open-items).

---

## 4. Measure, don't judge

The vision judge sees stills, so it cannot see lips moving or a camera
shaking. Every defect that is a property of motion or sound became a
measurement and a rule. None of them is a score.

### 4.1 Talking cover

- **Failure.** A person visibly talking to camera was used as cover under
  another speaker's audio for 32 s. The judge scored that stretch 3.
- **Evidence.** The first measure found 10 such rows (102 s) in one cut.
  Transcribing the B-roll audio (5.2) raised it to 27 rows (269 s). The next
  iteration under the rule had 0.
- **Rule.** A cover row whose source range holds at least 1 s of transcribed
  words is not cover (`speech.talking_cover`). A unit from an untranscribed
  source counts as *unknown*, not clean. A unit whose tag says nobody is
  visible (off-camera speech) is exempt. A clip the editor allows by name
  despite speech is an intent rule (`talking_cover_exempt`), never a pattern.
- **Guard.** The proposer never sees such units, the builder refuses to write
  a cut containing one, and `cuteval` measures every cover stretch of the
  render.

### 4.2 Camera shake as a stabiliser residual

- **Method** (`steadiness.py`). Track features between frames at 12 fps
  (Lucas–Kanade) and fit a rigid transform per frame pair with RANSAC. The
  cumulative translation is the camera path. Smooth it with a 1 s centred
  moving average; the RMS residual as a percentage of frame width is what a
  stabiliser would have to correct. A pair with under 12 inliers (a cut,
  black, no texture) contributes nothing, so a cut inside a stretch cannot
  raise the figure, and a linear pan leaves no residual.
- **Evidence.** Tripod interview 0.06 %, cover median 0.4 %, handheld walk
  windows 2–5 %, worst cover 13 % (a 40 %/s swing). The vision judge wrote
  `none` under technical for 7 of the 8 worst stretches.
- **Limit.** `MAX_SHAKE_PCT = 2.0`, provisional: a working number until
  the editor confirms a ranked list. It has not been calibrated.

### 4.3 Use the steady parts, not whole units

- **Failure 1.** A whole-unit figure hides a shaky stretch. Units measuring
  1.7 % and 1.6 % as a whole measured 2.5 % and 3.9 % over the seconds a
  cut actually used. 29 units under the limit as a whole held such a stretch.
- **Failure 2.** Steady runs were first built by joining overlapping steady
  windows, which let a run bridge the over-limit window between them. 46
  over-limit windows, up to 2.58 %, sat inside "steady" runs. Fixing it cut
  steady B-roll from 4,593 s to 4,135 s.
- **Failure 3.** Barring whole units at 2 % removed 29 of 105 clips (23 min)
  that held 17.5 min of steady footage, and every barred clip had at least
  one usable run.
- **Rule.** Read the residual in 2 s windows every 0.5 s. A run is time that
  no over-limit window touches, lasts 3 s or more, and is re-measured whole
  (`stable_runs`). A unit with any over-limit window is usable only inside
  its runs, and barred only when it has none. A unit with no over-limit
  window is used whole however short. The speaker's own sync picture is
  exempt.
- **Guard.** The proposer is told `STEADY PARTS ONLY <times>`. The verifier's
  frames come from inside the runs, and fewer than two there is `unchecked`.
  The builder clamps every row and merge into one run. The scorer measures
  the render. Runs computed at one limit say nothing about another, so a
  changed `--max-shake` stops and asks for a re-measure.

### 4.4 Drone moves

- **Failure.** 1,114 of 1,160 drone seconds passed the shake measure, because
  drones rarely shake. What goes wrong is the move itself: the ease in, a
  hesitation, a change of direction, the pilot settling at the head of a
  clip. Four stills per shot could not see it either. In one cut, 13 of 18
  drone rows crossed a turn, a slowdown or a settle.
- **Method** (`moves.py`). Per frame pair: the shift of the frame *centre*
  (a similarity fit's translation is about the corner, so a zoom would read
  as a pan), zoom, rotation, the flow of the centre against the edges (an
  orbit holds its centre), and the flow of the top against the bottom (a
  rise or slide moves the near ground faster). The velocity is smoothed over
  1 s and cut where the camera starts, stops, turns by more than 35° or slows
  mid-move. The cut goes at the slowest moment, so a 62 s pull-out with one
  slowdown becomes two usable moves instead of one rejected one.
- **Clean move.** It keeps one direction (mean cosine ≥ 0.90), its speed
  never dips mid-move by more than 25 % of top speed, its shake is under the
  limit, and it lasts 3 s or more. A steady hover counts. A first version
  measured speed against a 3 s average and called every short eased move a
  wobble: an ease in or out is not a dip.
- **Evidence.** 1,155 of 1,395 drone seconds were clean. The move labels
  (pan, slide, orbit, tilt, rise, descend, push-in, pull-out, rotate, hover)
  are heuristics. The measurement is what decides clean or not.
- **Guard.** `CLEAN DRONE MOVES ONLY`, with each move named, for the proposer.
  `cuteval` lists any drone stretch of the render that is less than 90 % one
  clean move.

### 4.5 Joins

- **Failure.** Butting picks on one track with a 5-frame fade on each was a
  dip to zero at every cut. Room tone "laid as nothing" was half a second of
  black and silence per beat.
- **Rule.** Alternating tracks, handles and crossfades (see
  [resolve-scripting.md](resolve-scripting.md#joins)). One render went from
  30 black stretches and 24 dropouts to 1 and 3, all at joins whose handles
  were capped at the neighbouring word.
- **Guard.** `blackdetect d=0.04:pix_th=0.10` and
  `silencedetect n=-50dB:d=0.15` on every render. Any black is a lay-in
  fault. A one-frame black came from rounding start and length to frames
  separately, so `layin.plan` measures every join gap in placed frames.

### 4.6 Loudness

- Camera scratch audio sat at −50 LUFS on one camera and −59 LUFS on another,
  with crest factors over 20 dB. At −18 LUFS the render clipped in 113 places.
  The peak-safe working level is about −24 LUFS.
- The API caps clip gain at +30 dB, and the quiet camera's picks needed 31 to
  38 dB. Asking for more gain is not a fix. The fix is a decision about
  target and dynamics, then a measured diagnostic render.
- Dialogue Leveler modes were measured on diagnostic renders (details in
  [resolve-scripting.md](resolve-scripting.md#audio)). The lifting modes
  pushed already-loud picks to +3 to +5 dBTP and barely moved the quiet ones.
- **Delivery** (`deliver.py`): two-pass linear `loudnorm` to the target in
  `intent.json` (for example −16 LUFS / −1 dBTP for web), with a limiter
  1.5 dB under the ceiling because the sample-peak limiter overshoots true
  peak by about 1 dB. The video stream is copied untouched, so the picture
  that was scored is byte-identical to the picture delivered. The result is
  measured and its manifest says `honoured` or `broken`.

### 4.7 Mono picks on one channel

- **Failure.** Every mono pick in a render played on the left channel only,
  although the channel map read back correctly. The signature: integrated
  level 3 LU low with true peak unchanged. A level table alone showed the
  symptom and invited the wrong fix, more gain.
- **Cause.** Resolve does not apply a source channel map written through the
  API until the timeline is reopened.
- **Guard.** Reopen the timeline before every render, then measure left
  against right on every mono pick. A difference over 3 LU marks the render
  `broken`.

### 4.8 What is still not measured

- **Repeated footage.** When a reused shot has no fresh room, the builder
  repeats certified frames, and nothing scores the repeat. One cut showed the
  same 9 s in five beats.
- **Jump cuts** are a heuristic on source names and bare stretches. Of four
  flagged in one cut, one was a real face-to-face jump. The rest were a
  9-frame flash of the speaker and picture cuts on the join.

### 4.9 Heard is not seen

- BirdNET over camera audio, with species limited by location and week,
  gives detections with confidences. A bird heard on a clip is not a bird in
  the frame. Lifting quiet camera audio by 30 dB changed no detection, so the
  audio is measured as recorded. A person confirms every label before it
  goes on screen.

---

## 5. ASR and speech alignment

### 5.1 Whisper invents speech in silence

- `condition_on_previous_text=False` on every pass. It is the guard against
  invented speech during silence.

### 5.2 B-roll audio: invented speech over natural sound

- **Failure.** Whisper invents speech over wind and birds and misreads the
  language: a 20 s wind-and-birds probe came back as "nn".
- **Rule** (`transcribe --broll`). Force English. Keep Whisper's per-segment
  `no_speech_prob`, `avg_logprob` and `compression_ratio`, drop segments over
  0.6 (no-speech) or 2.4 (compression), and record what was dropped. A track
  under −50 dB peak is written as silent, and a source with no audio track is
  recorded as such, so no source is ever "not checked". A-roll entries are
  never touched.
- **Evidence.** 105 B-roll sources: 13 with no audio track, 14 silent, 78
  transcribed, 24 with speech (people talking through B-roll for up to 106 s
  a take). Three one-word results near the threshold ("Thank you.", "Yeah.")
  are probably inventions. They cost their units under the talking rule,
  which is the safe side.

### 5.3 Coverage misses meaning

- Deleting "not" from a sentence gave 90 % coverage and no flag. Inserting
  "not" inside the matched run gave 100 % coverage and no flag. Only words
  outside the run had counted as leaks.
- **Rule** (`speech.py`). Align intended against heard as complete token-edit
  operations: omissions, substitutions, insertions inside the run, and leaks
  at either boundary that are not a neighbouring pick's words. Any edit on a
  polarity, quantity, modality, name or pronoun token is a semantic risk and
  flags the pick whatever the coverage. Coverage under 0.8, or 2 or more
  leaks, also flags.
- **Guard.** Both negation fixtures are in `tests/test_speech.py`.

### 5.4 One ASR pass is a question, not a finding

- **Failure.** A three-word pick reported at zero coverage was recovered in
  full from an 11 s padded excerpt of the same render. Fed back through the
  unchanged aligner, it scored coverage 1.0.
- **Rule.** Every flagged pick is transcribed again from a 2 s padded excerpt
  of the render, and from the source clip at the pick's own in and out. The
  states are `cleared_on_recheck`, `confirmed`, `waveform_intact`,
  `intent_mismatch` (the source itself says something else, so the cut text
  is wrong) and `unresolved` (listen). Nothing is trimmed on one pass, and no
  state moves a trim or drops a pick.

### 5.5 Check the waveform before calling speech lost

- **Failure.** A pick called `confirmed` lost on two ASR passes matched the
  source waveform throughout (minimum correlation 0.75, median 0.92). The
  transcriber had heard the levelled render differently.
- **Rule** (`speech.waveform_match`). Correlate quarter-second speech windows
  at a locally searched lag. A window under 0.6 counts as lost. `confirmed`
  needs two agreeing render passes, a source that carries the words, *and* a
  waveform that stops matching. A test copy with 0.6 s silenced read 0.0 at
  the hole.
- **Limit.** A single word muted in place, shorter than a window, can pass. A
  lay-in places whole items and cannot do that, and a wrong in- or out-point
  shifts everything after it, which fails every later window.

### 5.6 Say what the check established

- A report said "68/70 picks play their words in full", yet 35 unflagged
  picks had less than full coverage. A mismatch below the flag threshold is
  not evidence of complete speech. The wording is now "did not trigger the
  configured alignment flags".

### 5.7 Aliases are explicit and narrow

- Spelling variants (proper names, botanical words, place names) go through
  an explicit per-film alias table (`film.yaml` `aliases`). The fuzzy fallback
  never touches a token under four letters or any critical token. Spelling
  aliases are kept separate from semantic corrections: a source's spelling is
  not permission to normalise arbitrary words into what the editor intended.
- The alignment thresholds (0.8 coverage, 2 leaks, a ±0.2 s window) were
  tuned on one render after three rounds of false positives and have not been
  validated on another.

### 5.8 Make cut points from measured room, not stored transcripts

- A stored transcript can miss the word beside a cut. A pick cut inside a
  phrase carries its own measured room (`max_head_s` / `max_tail_s`, from the
  source's energy per channel). A pick with zero head room starts on its word
  with a 2-frame fade-in instead of a symmetric crossfade that dips the word.

### 5.9 The human transcript and Whisper disagree

- Whisper's wording differs slightly from a human-edited transcript, so marked
  passages are fuzzy-matched onto timecodes, and those timecodes are
  approximate.
- `python-docx` sees neither `w:shd` shading nor comments, and both carry
  editorial intent. `selects` parses the docx XML directly.

### 5.10 Sync by waveform, not by clock

- Camera and lav-mic clocks disagreed by hours. Only the calendar day is used
  as a prior. `audiosync` cross-correlates 100 Hz RMS envelopes via FFT across
  a whole mic session (chunks concatenated), then verifies with a word-overlap
  check on a transcript excerpt. `transcribe` prefers verified synced lav
  audio over camera scratch audio.

### 5.11 Recording context is not speaker balance

- Seconds by recording role (interview, presenter, walk) are computed
  deterministically from the pick spans. They are not seconds per speaker: a
  two-shot holds two people. Without diarisation or editorial speaker labels,
  never report them as speaker balance.

### 5.12 32-bit float recordings clip on the way into the transcriber

- Lav recorders writing 32-bit float keep samples above full scale. Whisper's
  loader converts audio to 16-bit integers, which clips every one of them.
  The first real chunks measured peaks of +6.0 and +6.7 dBFS (30 and 24
  over-full-scale samples) on handling bumps.
- **Rule.** Build one working file per talk with a single linear gain that
  puts the talk's peak at −1 dBFS (a cut of 7–8 dB on those chunks, a lift
  of 8 dB on a quiet excerpt), never compressed, and keep a map from talk
  time back to the original float chunks so every in and out is laid from
  the originals. **Guard:** `talks prepare` records peak, overs and gain per
  talk; a test scales a sine above 0 dBFS and checks the working file is
  unclipped.

### 5.13 Measure room against the noise floor, not the speech level

- The first edge-room rule called "quiet" anything 20 dB under the median
  speech level. On a real outdoor lav excerpt that threshold sat at
  −67 dBFS, below a −58 dBFS floor of wind and birds, so all 33 sentence
  edges measured zero room.
- **Rule.** Quiet is within 6 dB of the measured noise floor (10th
  percentile of the envelope). When speech is less than 12 dB above the
  floor, room is reported as unknown, not as zero. A sentence 10 dB or more
  under the wearer's median level is flagged off-mic (probably another voice
  heard by the lav): on that excerpt, 10 of 33 sentences at −43 to −47 dB
  against −32.6 dB. All provisional until the editor listens.

---

## 6. Honest evaluation

### 6.1 Failed checks became acceptances

An independent review reproduced every one of these in the eval loop:

- a verifier result of only `{"error": "timeout"}` became a selected
  candidate scoring 1.5;
- `watchable({})` returned true, and a merge ran *before* the verdict check,
  extending a shot across a pick whose verdict said score 0, unwatchable blur;
- a score of 99 was accepted;
- the black and silence helpers returned empty findings for a missing render,
  which reads as "no fault";
- the report crashed on a result with silences but no black, which was
  exactly the shape of the real render's result.

**Rule** (`evalrun.py`). Every result carries a status: `ok`; `unchecked`
(did not run or failed to: missing file, ffmpeg error, timeout, truncation);
or `invalid` (ran, but the response is unusable: bad JSON, out-of-range score,
relation and score disagreeing). Anything not `ok` carries no score, cannot
make a candidate eligible or a stretch accepted, and cannot support a claim
of improvement. A model reporting no issue means `no_issue_detected`, never
"verified".

**Guard.** `tests/test_eval_status.py` holds the review's fixture list.

### 6.2 Caches keyed on file existence returned stale evidence

- A red test render was rendered and a frame cached. The render was replaced
  with a blue one at the same path, and the same frame request returned the
  old red frame byte for byte. A transcript cache accepted a different,
  nonexistent render with changed beat spans.
- **Rule.** Each stage's cache key depends on that stage's inputs. Frames
  live under `<frames-dir>/<render sha>/`, and transcripts are keyed on the
  render hash, the spans and the model. Changing a prompt need not re-extract
  identical frames, but it does invalidate judgments. A large file's hash is
  cached in a `.sha256` sidecar that is trusted only while size and mtime
  match.

### 6.3 A score without provenance is unattributable

- A scorecard quoted for a render was written 31 minutes *before* that render.
  Modification times are a warning, not proof.
- **Rule.** Every report writes `<out>.manifest.json`: content hashes of its
  inputs, prompt text or hash, model and settings, tool source version, sample
  times, and what was checked versus `unchecked`. Tools refuse to overwrite a
  previous run without `--overwrite`. Quote the render sha with any score.

### 6.4 Conserve timing in every pick

- **Failure.** A 20 s pick split between a 3 s source and a long one,
  followed by a 5 s pick, built 3 + 10 + 5 = 18 s, so the second pick's
  picture started at 13 s instead of 20 s. In a real cut this put two cover
  shots 6.2 s and 3.7 s early, confirmed on the timeline read-back.
- **Rule.** The rows of a pick sum to the pick. A short piece hands its
  deficit to the next piece, or fills it bare. Every row carries `pick`,
  `dest_in_s` and `dest_out_s`, and `cutbuild.check_cut` refuses a cut whose
  rows are not contiguous per beat or do not end where the words end. A
  reused shot with no room left repeats certified footage and says so: one
  cut carried 131 s of unlabelled repeats before this.

### 6.5 Identity by position breaks

- The proposer numbered placeholder cards by position among *all* rows, and
  the builder among placeholder rows only. With a video row before two cards,
  the same id meant different cards to each. Placeholders now have stable ids
  by content hash.
- Scoring stretches must split at the union of picture-row and spoken-pick
  boundaries. Before that, 8 of 109 stretches spanned two picks, and their
  feedback went to both.

### 6.6 The model's layout and the real timeline can disagree

- `cuteval` scores a layout it computes, then grabs frames from the render at
  those times. Requiring an exact frame match to Resolve's read-back once
  silently skipped 11 items. Items are now matched within ±2 frames, because
  23.976 fps drone rows push later items by a frame.
- A lay manifest listed 92 cover rows and the read-back held 89. Resolve had
  dropped three short fillers as overlaps. Reconcile the two and never treat
  a manifest as proof of placement.

### 6.7 There is no ground truth yet

- Neither judge has been calibrated against a human rating. The 0–3 scale is
  self-consistent (the same prompt and model score the render and the
  candidates), but its absolute level means nothing. Deltas between versions
  are not trustworthy either until provenance, run-to-run variance (the
  temperature is 0.2, not zero) and comparability (length, placeholders,
  scoring coverage) are established.
- There is no held-out material: the scorecard that drives the proposer is the
  same one that reports improvement, so a fault the judge cannot see is
  invisible end to end.
- **Plan.** An editor-rated sample stratified by kind of moment, *with
  audio*, held out from rubric development. Repeated identical requests for a
  variance estimate. Blind pairwise comparison on matched units with ties
  allowed.

### 6.8 Keep the acceptance layers separate

| Layer | Evidence | When unresolved |
|---|---|---|
| Evidence validity | current fingerprints, correct interval map, required stages complete, valid responses | no improvement claim; show unchecked coverage |
| Technical integrity | interval invariants, joins, speech integrity, levels, peaks, channels | block on deterministic violations; recheck ambiguous ASR |
| Editorial intent | active rules from `intent.json` | a score cannot override a decision |
| Quality comparison | rubric judgments, repeatability, editor preference | guidance with uncertainty |

A higher quality score cannot cancel a failure in the first three layers.
Draft acceptance and delivery acceptance are different profiles: placeholders
are allowed in a draft, and they are still missing material at delivery.

### 6.9 "Keep" is not evidence that a check happened

- Given an excerpt with the audio deliberately delayed 2 s, every local judge
  recommended "keep" and marked a protected-dialogue rule as respected, even
  though their own reported timings placed about 1.7 s of the protected
  greeting under the next cover shot. One judge's transcript of the delayed
  version omitted a whole preceding passage that Whisper still found.
- **Rule.** Intersect protected spoken intervals with actual picture coverage
  in code. Never accept a model's `intent_respected: true` as that check.

### 6.10 Fix the evidence before tuning the judge

- An independent review put the author's own roadmap in a different order:
  calibration cannot rescue a scorer that grades stale frames under the
  wrong pick. Make failure honest and evidence frozen, then conserve timing
  and reconcile the actual edit, then fix speech and intent checks, and only
  then calibrate.

### 6.11 Read a review as dated evidence

- A review records what was true of the code on its date. After fixes, the
  current design and code are the authority. Keep the old evidence
  directories as an inspectable record, and repeat an experiment in a new run
  directory instead of overwriting one.

### 6.12 Compare prompt variants the way you would compare treatments

- A 2×2 comparison of annotation prompts (motion statistics in or out; the
  model describing frames directly, or a literal per-frame observer feeding a
  second synthesis call) was run as a small trial, not a vibe check:
  - **Frozen inputs.** One manifest fixed the prompts, the frames (by
    SHA-256), the request order, temperature 0.3, 6000 max tokens and two
    repeats per case and variant. Each response was written exclusive-create
    and checked against its request hash when a run resumed.
  - **Split cases.** A development split for choosing, and a confirmation
    split held back and run only if a candidate qualified.
  - **Label-hidden scoring.** The responses were shuffled into a packet with
    review ids, and the key mapping ids to variants sat in a separate file,
    not read until scoring was finished. This hides the labels, not wording
    clues or the reviewer's prior knowledge of the shots.
  - **A decision rule written before scoring.** The best candidate had to
    beat the current prompt by at least one point of 8 on the mean, with
    fewer motion and grounding errors, no more schema failures and no lower
    editorial points. Costs (median end-to-end seconds per call) were
    reported beside the scores.
- **Outcome.** No candidate cleared the gate, so the current prompt stayed
  and the confirmation cases were never spent. Without the gate, the
  highest-scoring variant would have shipped on noise from two repeats.
- **Rule.** Any prompt or model change that claims to improve a judgment
  goes through the same shape. Freeze, split, hide the labels, write the
  decision rule first, report cost. This is also the template for
  calibrating a new label vocabulary (see `docs/sibling-project.md`).

### 6.13 A cache beside the file writes into the originals

- Large-file digests were cached in a `<name>.sha256` beside each file. The
  shake and drone-move measurements fingerprint source footage, so one film
  collected 123 sidecars. They landed beside symlinks only because its
  inputs were links; a config pointing straight at the footage, or at a
  cloud-synced folder, would have written into the originals' folder and
  uploaded with it.
- **Rule.** Digests are cached under `$VIDEOEASY_CACHE/fingerprints` (default
  `~/.cache/videoeasy/fingerprints`), keyed on the resolved path and trusted
  only while size and mtime match. **Guard:** a test fingerprints a large
  file and asserts its folder is unchanged.

---

## 7. Editorial intent as rules

### 7.1 Decisions scattered across prompts get lost

- The story model received the original spine and a later addendum requiring
  new beats in different prompt sections. It reported the required beats as
  "orphans".
- **Rule.** `editorial/intent.json` is the one place a decision lives: id,
  topic, kind, the decision in words, origin (a subject on camera, the editor,
  or a story-room proposal the editor accepted), who decided, source,
  date, and what it supersedes. `intent.resolve()` retires anything a later
  rule supersedes and keeps the latest decision on each topic. The story model
  gets one resolved brief, with what no longer applies listed separately.

### 7.2 Check rules in code, on the cut and on the timeline

- `check_cut` judges rows by their destination intervals: rows without
  destinations are `not_testable`, never `honoured`. `check_readback` judges
  what Resolve actually holds. The builder refuses to write a cut with a
  broken rule, and `layin` exits 2 when the read-back breaks one even though
  the conform passed.
- The first run caught two beats covered with footage a rule forbade. The
  proposer's own note-based version of the same rule had missed them.

### 7.3 The models must not be able to undo a decision

- A beat whose title or note says "no cover" stays bare whatever the proposer
  says. Protected picks are marked bare for the proposer and held bare by the
  builder, and the scorer tells the judge the stretch stays uncovered.
  Placeholder cards the proposer dropped are re-attached to the pick whose
  words match them.
- Exceptions are granted by name (a clip id in a rule), never by pattern.
- Never encode a decision only in a beat note or a prompt.

### 7.4 Each story-model patch is a prompt note, not a test

- The story model needed patching twice. Rules about pictures cannot be
  judged from words and must be marked `not_testable_from_words`, not
  `broken`. A word preference is broken only by use of the disfavoured word,
  not by absence of the preferred one. Expect more categories of rule it
  misreads.

### 7.5 Curated judgment is never regenerated

- `bible-context.md` is regenerated by `assemble`. `bible-synthesis.md` is
  argument built in conversation and corrected by the editor, and no script
  writes it. A rebuild goes in a new file beside the old one
  (`bible-synthesis-v2.md`) and stays a proposal until the editor reads it.

---

## 8. Working with a frontier model in the loop

### 8.1 The frontier model does not judge by eye

- A first draft built by the frontier model looking at frames was called
  "awful". The rule since: to validate a render, a grade or a cut, the
  frontier model writes the prompt, sends graded frames to the local vision
  model and the transcript to the local text model, reads the scores, and
  fixes the generator. Contact sheets are for discussing a specific shot,
  not for grading a cut.

### 8.2 A document, not a database

- A 10–15 minute documentary's material fits in one frontier context (see
  [architecture.md](architecture.md)). Understanding lives in a legible
  document that a person can read, argue with and correct, and corrections
  persist. Embeddings encode relations silently.

### 8.3 Prior human intent is data

- Transcript markup (highlight colours, cell shading, anchored comments) and
  the editor's own draft radio cut are extracted into the bible. The story
  room starts from them. When the footage argues against a recorded decision,
  say so and make the case; never quietly overrule it.

### 8.4 The loop changes pictures; people change the story

- Audio picks (the radio cut) change only by hand in the story room. The eval
  loop rebuilds picture rows and measures pictures and words.

### 8.5 Calibrate before committing a night

- `videoeasy all` is an overnight job and is never launched to test a change.
  Run `check`, then annotate 2–3 shots with `--only`, review their depth with
  the editor, then commit the night. After a prompt or model change,
  `annotate --force`, or completed shots are skipped and you read stale
  records.

### 8.6 A second agent finds what the author misses

- Independent reviews by a different agent found the fallback scores,
  existence caches, timing loss, missed negations and the backwards roadmap.
  The author's own design document had not. Commission adversarial reviews
  with reproductions, and keep their evidence.

### 8.7 Environment failures are not code failures

- In a sandbox, `check` and `annotate` fail with connection errors because
  the sandbox blocked localhost, not because the code broke. Check `lms ps`
  first.
- Footage on external drives: source paths are baked into artifacts, so mount
  names must stay stable. Originals are never written.

### 8.8 Footage stays local

- Hosted audiovisual models were researched as capability context and
  rejected. Only local models see pixels or audio. The frontier model
  receives text: the bible, metadata, scores and transcripts.

---

## 9. Open items

Known defects and gaps not yet fixed, from reviews whose findings still
apply to the ingest stages:

- **Ingest caches are existence-keyed.** `frames` skips existing filenames and
  `annotate` skips any id with a record. The eval tools' content-hash pattern
  (`evalrun.py`) has not been carried back to ingest.
- **An unavailable source directory yields an empty inventory**, and `probe`
  writes it, after which later stages can overwrite good artifacts. Planned
  fix: fail before replacing artifacts when configured sources are missing,
  and write JSON atomically.
- **Annotation responses lack schema validation.** Any JSON object is
  accepted, and an empty object counts as complete.
- **`audiosync` can return an impossible lag** when a mic session is shorter
  than the clip: the valid-lag slice end goes negative. A 1,000-sample clip
  against a 100-sample session returned an 11.2 s offset into a 1 s session.
  Planned fix: reject sessions shorter than the clip before the FFT.
- **Resolve helpers** (`resolve.py`) still return `MoveClips`' unreliable
  value and lack a page-state gate. `layin.py` verifies by read-back, but the
  older bin helpers do not.
- **Verify frames are ingest frames.** `brollmatch` verifies a candidate on
  the whole shot's graded frames. `cutbuild` may then use a sub-range of it,
  and only the next `cuteval` sees what was actually used.
- **The proposer is text-only over vision tags.** A wrong ingest tag becomes
  a wrong proposal, caught only by the verifier.
- **A pick whose every candidate is barred falls to bare** instead of being
  re-proposed with the bar stated in the prompt.
- The reuse penalty (0.75), the unverified weight (1.5) and the placeholder
  weight (2.5) produced sensible plans, but they are guesses that nothing
  tests.
