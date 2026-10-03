# Local models: which did what, and how they failed

videoeasy keeps footage local: only models running on the editing machine
(an Apple Silicon Mac) see pixels or audio. This page records which models
held which job and what each one got wrong, as findings with their method
limits. It is not a leaderboard. Precision was kept at 8-bit for every
comparison, because a lower-precision build is a different model.

## Production assignment

| Job | Model | Served by | Settings that matter |
|---|---|---|---|
| Shot and window annotation; picture-vs-words judge; cover verifier | `google/gemma-4-26b-a4b` (GGUF Q8_0) | LM Studio, OpenAI-compatible `:1234/v1` | `--context-length 16384`; `max_tokens` 6000 with one retry at double; temperature 0.3 |
| Text judge, cover proposer, story model, talk analysis | `qwen3.8:27b-mtp-q8_0` (Q8_0, digest `8a1582877303`) | Ollama `/api/chat` | `think: false`, `format: json`, `num_ctx` 32768–40000, temperature 0.2–0.3 |
| Transcription (sources and renders) | `mlx-community/whisper-large-v3-mlx` | in-process (mlx-whisper) | word timestamps, `condition_on_previous_text=False`, renders chunked per beat |
| Bird calls heard | BirdNET 2.4 (location + acoustic) | in-process (TFLite) | optional extra `birds` |
| Plant species on screen | BioCLIP 2 | in-process | optional extra `species`; advisory only |
| Everything measurable | ffmpeg / ffprobe / OpenCV | | `signalstats`, `blackdetect`, `silencedetect`, `ebur128`, `loudnorm`, Lucas–Kanade + RANSAC |

**Never run Ollama and LM Studio jobs at the same time.** GPU contention
slowed both by about 4x, and with both models resident the OS killed a
background process for low memory.

## Vision

### Ollama MLX drops images

Ollama 0.31's MLX backend silently dropped images. Its `prompt_eval_count`
stayed text-only while the model wrote confident, plausible descriptions.
`videoeasy check` now sends a synthetic image with known colours and requires
the model to name them. Run it after any model or backend change.

### GLM-4.6V-flash (9B, thinking)

It passed the colour trap and a multi-image ordering test. Its reasoning
consumes the token budget before any content appears, so with a small
`max_tokens` it returns an empty answer with `finish_reason=length`. In
July 2026 calibration it was replaced by Gemma 4 26B-A4B, which wrote deeper
cut notes and ran about 1.7x faster. Both pass `check`.

### Gemma 4 26B-A4B

- The production vision model. It takes text and images only; it is **not**
  the audio-capable Gemma variant. A video container gives it no audio.
- At LM Studio's default 4096-token context, 9 of the first 130 tags in a
  batch failed with `finish_reason=length`, all on busy frames. 16384 tokens
  fixed it.
- As a judge it is useful for subject and relevance screening and obvious
  exposure. It cannot see motion: it wrote `none` under technical for 7 of
  the 8 shakiest cover stretches. It names plants confidently and cannot be
  trusted to species level. It scored 2 of 5 protected, deliberately
  uncovered stretches 0 even when told they were protected.
- In a pilot with six timestamped stills plus Whisper (the "modular
  baseline"), it passed the four basic controls, but only because a
  deterministic exact-silence check ran before ASR: that 4/4 belongs to the
  pipeline, not the model. It did not notice audio deliberately delayed by
  2 s. Its first run truncated two answers at a 2,000-token budget, and 6,000
  fixed that.
- Placeholder-card stretches exhausted a 3,000-token budget on every attempt
  and stay `unchecked`.

## Text

### Qwen 27B (text)

- Through Ollama with `think: false` and `format: json`, it is the cover
  proposer (about 85 s per beat over a catalogue of about 14k tokens), the
  first-time-viewer read and the story model.
- Through LM Studio, a request spent 5,999 generated tokens on reasoning,
  returned no answer and took 336 s. LM Studio's documented reasoning-off
  route then rejected the model's reasoning configuration with a 400. Native
  MLX with thinking disabled worked.
- In the audiovisual pilot (as the text half of a Whisper + stills pipeline)
  it passed all four controls and kept all three film excerpts. It did not
  expose an injected 2 s audio delay. Its own response reported a protected
  greeting at 10.60–12.78 s and its visual sequence changing to other footage
  at 11.67 s, yet it marked the protected rule satisfied. It sometimes gave
  one person's identity to another without grounds.
- The story model needed two prompt patches: picture rules cannot be judged
  from words (`not_testable_from_words`), and a word preference is broken
  only by use of the disfavoured word.

## Audio

### Whisper large-v3 (mlx)

- It invents speech during silence without `condition_on_previous_text=False`.
- Over natural sound it invents speech and misreads the language (a 20 s
  wind-and-birds probe returned "nn"). For B-roll audio: force English, keep
  `no_speech_prob`, `avg_logprob` and `compression_ratio`, drop segments over
  0.6 no-speech or 2.4 compression, record drops, and write a track under
  −50 dB peak as silent. One-word results near the threshold ("Thank you.",
  "Yeah.") are probably still inventions.
- A single pass over a render can miss words the render contains. A
  three-word pick reported at zero coverage was recovered in full from an
  11 s padded excerpt. A levelled render can also be transcribed differently
  from its source while the waveform matches throughout (minimum correlation
  0.75, median 0.92). Hence the padded recheck and the waveform test before
  anything is called lost ([lessons 5.4–5.5](lessons.md#54-one-asr-pass-is-a-question-not-a-finding)).
- Its wording differs slightly from a human-edited transcript, so marked
  passages are fuzzy-matched.

## Audiovisual ("omni") contenders, September 2026

A pilot compared three downloadable audiovisual models at 8-bit, each given
native audio and about one video frame per second, against the two modular
pipelines above (Whisper plus timestamped stills). Four counterfactual
controls came first: picture order, spoken negation, a spoken code, and exact
silence. A model that failed a control had its film judgments withheld. Then
three film excerpts: a speaker handover, the same pictures with the audio
delayed 2 s, and a protected uncovered hold.

| Route | Controls | Film judgments | Notes |
|---|---|---|---|
| Gemma 4 12B (omni, MLX) | 3/4 | diagnostic only | the default video-first template gave wrong colours and lost the negation; an audio-first template fixed those; it invented a spoken instruction for exact silence; with identical decoded video it said "revise" for the original and "keep, well-synchronized" for the delayed version, describing different visuals; reproduced byte-identically with case order reversed |
| Qwen3-Omni 30B-A3B | 2/4 | none usable | all three film requests emitted repeated `<\|im_start\|>` tokens until the 6,000-token budget ran out (about 90 s each); runtime, preprocessing or weights: undiagnosed |
| Nemotron 3 Nano Omni 30B-A3B | 4/4 | keep / keep / keep | needed the MLX-VLM model-name mapping (the checkpoint's class name bypassed the adapter), the unpruned video path (efficient video sampling not implemented) and an exact millisecond timestamp label; 6–9 s per request, about 39 GB peak memory; missed the delayed audio; its transcript of the delayed version omitted a whole passage Whisper still found |
| Whisper + Gemma 26B stills | 4/4 (pipeline) | keep ×3 | no native audio; cannot measure sync by design |
| Whisper + Qwen 27B text | 4/4 | keep ×3 | see above |

What the pilot established:

1. **Test the actual audiovisual route.** A colour trap does not catch a
   template that works on images or sound separately but fails when they are
   combined. Use paired controls that change one input at a time.
2. **Validate fixtures first.** macOS `say` returned success in a sandbox and
   wrote zero-sample files, invalidating a whole run.
3. **"Keep" proves nothing.** No route caught the 2 s delay. Intersect
   protected intervals with picture coverage in code.
4. **A modular pipeline is a serious comparator.** Local ASR, deterministic
   audio measurements and timed stills beat a newer omni model at specific
   tasks.
5. **Promote nothing automatically.** Input controls, then defect sensitivity
   and false-alarm tests on unseen excerpts, then blind pairwise comparison
   against the editor's preferences.

### Where an omni model may still fit

A follow-up on two short graded excerpts used Nemotron as an *event logger*
(time-ordered actions, speech and sound) with no names, transcript or story
hints. Its observations, fused text-only by Gemma with existing Whisper text
and context, produced searchable "moment records". Much of the added meaning
came from Whisper and existing context rather than the omni model, so its
marginal value is unproven. Its uncertainty entries were weak, and it
described laughter that nobody verified. Recommended division of work:

| Job | Route |
|---|---|
| Literal composition, people count, objects, readable detail | Gemma on graded stills |
| Words and timing | Whisper, with targeted second passes |
| Action sequence, speech/action relationship | an omni model, selectively, as discovery |
| Laughter, hesitation, sound changes | an omni model as discovery; confirm locally |
| Names, places, history | a curated registry; never guessed from clothing or filenames |
| What a clip is about | text-only fusion of the above |
| Theme, motif, setup and payoff | the story model over linked evidence and the editor's corrections |

## Species and birds

- **BioCLIP 2.** A shortlist softmax always has a winner (0.99998 on one
  frame). The open-ended tree-of-life ranking is the check: a claim
  *agrees* when the named species, or for a genus the named genus, is in its
  top five, *disagrees* when it is not and the top answer is confident, and
  otherwise *cannot tell*. On one cut, 17 of 23 claims read under 0.1. It is
  uncalibrated on this footage, so a person decides every label.
- **BirdNET 2.4.** The location model limits species to those plausible at
  the shoot location in that week. The acoustic model reads 3 s windows every
  1.5 s. Detections are evidence, not labels. Lifting quiet camera audio by
  30 dB changed no detection.

## Hosted models

Hosted audiovisual APIs were researched as capability context, then rejected.
The rule is that footage never leaves the machine. Frontier models (Claude,
GPT) orchestrate and reason over text and metadata. An agent that extracts
frames with tools is not receiving native audiovisual input.
