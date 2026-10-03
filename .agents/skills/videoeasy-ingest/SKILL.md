---
name: videoeasy-ingest
description: Run, calibrate, or debug the videoeasy ingest pipeline — probe, shots, transcribe, segment, frames, motion, annotate, audit, selects, assemble, and lav-mic waveform sync. Use when starting an ingest run, changing the vision model or grading config, tuning shot detection, investigating shallow or hallucinated annotations, or diagnosing a failing stage. Not for story work over a bible that already exists.
---

# videoeasy ingest

Ingest turns footage into `data/films/<film>/out/bible-context.md`. Every
stage is idempotent and writes its JSON incrementally, so re-running skips
completed work. Every command takes `--config config.<film>.yaml`.

## Never skip the preflight

```bash
lms ps                                        # vision model loaded and served? CONTEXT must be >= 16384
uv run videoeasy check --config config.<film>.yaml   # ffmpeg, LUTs, vision trap test, context window, whisper
```

`check` sends a synthetic image whose colours the model must name exactly. A
**SUSPECT** result means images are not reaching the model, so every
annotation would be invented. Stop and fix the backend before running
anything else. It is not a soft warning.

Run `check` after *any* change to `models.vision`, `models.vision_url` or the
serving backend.

## Calibrate before committing a night

`videoeasy all` is an overnight job. Never launch it to test a change.

```bash
uv run videoeasy probe       --config ...
uv run videoeasy shots       --config ...   # prints the shot count: sanity-check it
uv run videoeasy transcribe  --config ...
uv run videoeasy segment     --config ...   # windows long takes; prints how many will be tagged
uv run videoeasy frames      --config ... --only <id>   # one shot or window
uv run videoeasy annotate    --config ... --only <id>
```

Judge the result on two surfaces together:

1. the record in `out/annotations.json`;
2. the matching contact sheet `work/sheets/<id>.jpg`. **Open it as an
   image.** Judging annotation quality without the frames is guesswork.

Show the editor 2–3 calibrated shots and get their read on depth before the
full run.

### What a bad annotation looks like

- **Repeated phrasing across shots.** The same metaphor or theme wording on
  unrelated shots means the prompt is leaking examples. It has happened: an
  example phrase from the prompt appeared word for word across a whole
  corpus. Fix `src/videoeasy/prompts.py`, and never put concrete example
  phrases in the metaphor or theme fields or in `film.yaml` `context`. A
  bible whose shots all read alike cannot tell them apart, which defeats its
  only purpose.
- **Confident description that does not match the contact sheet.** Images are
  being dropped. Re-run `check`.
- **Mood and colour read as washed out or melancholy whatever the subject.**
  Frames reached the model ungraded. Check the `grade:` match rules.
- **Measurements the model was never given.** "No significant optical flow is
  detected" on a film with no `motion.json` means the prompt stayed silent
  about absent statistics. `prompts.py` must say when there are none.
- **Asserted sound, time of day, crew roles or camera technique.** Stills
  carry none of these. Tighten the evidence-limit lines in the system prompt.
- **Empty content with `finish_reason=length`.** A thinking model burned the
  budget before answering. Raise `max_tokens` in `src/videoeasy/vlm.py`
  (default 6000), or load with a larger context.

After a prompt or model change, re-annotate with `--force`. Otherwise
completed shots are skipped and you will be reading stale records.

## Long takes: windows

A single tag over a 358-second walk-and-talk summarised four frames ninety
seconds apart and invented a transition that was not in evidence. `segment`
cuts any take at or over `segments.min_source_seconds` into windows, each
framed and tagged on its own and nested under the whole-take record, which
is preserved.

- An edge is an **index, not a cut**: a real transcript silence where one is
  within reach (`speech_pause`, with the silence length recorded), and an even
  division where none is (`even_split`). Never write a prompt, tag or note
  that calls a window edge an edit point.
- A window's transcript text is for SEARCH. It never goes to the vision
  model: a spoken topic is not a visible fact, and once it enters the prompt
  the tags start describing what the model "heard".
- Re-running `segment` leaves existing windows alone. `--force` recomputes
  and prints the ids it orphaned.
- Locked-off takes gain nothing from per-window tags. List them in
  `segments.index_only_sources` after checking the contact sheet: they are
  still windowed for search, just not sent to the vision model. Tag single
  windows by name when wanted.

Denser sampling does not make camera technique knowable: tags still assert
"the camera is stationary" from stills, and shot scale still comes back
wrong. Run `audit --only <id>` on a representative window and read it as a
warning system, never as verification. Shake and drone moves are measured by
`steadiness` and `moves`, not judged.

## Grading

Every frame the vision model sees is graded first. Profiles and
first-match-wins filename rules live under `grade:` in the config. Adding a
camera is a profile block plus a `match:` rule. Log footage goes through the
LUT that approximates the intended final grade, so annotations describe how
the *finished* film feels.

## Stage notes

| Stage | Needs footage drive | Notes |
|---|---|---|
| `probe` | yes | records embedded start timecode for the Resolve conform |
| `shots` | yes | PySceneDetect on B-roll; A-roll files stay whole |
| `transcribe` | yes | mlx-whisper with word timestamps; runs before `segment`; `--broll` for B-roll audio |
| `segment` | no | windows long takes → `out/segments.json`; needs the transcript |
| `frames` | yes | graded frames and a contact sheet per shot and window; `--only` bounds it |
| `motion` | yes | Farneback optical flow → motion_energy, camera_ratio |
| `annotate` | no | works from extracted frames |
| `audit` | no | opt-in; see `docs/evidence-audits.md` |
| `selects` | no | parses the docx XML directly (python-docx sees neither `w:shd` shading nor comments) |
| `assemble` | no | rebuilds the bible from artifacts |

## B-roll audio

`transcribe --broll` forces English, keeps Whisper's `no_speech_prob`,
`avg_logprob` and `compression_ratio`, drops segments over 0.6 / 2.4 and
records the drops. It writes a track under −50 dB peak as silent and a source
with no track as such, so no B-roll unit is ever "not checked". Whisper
invents speech over wind and birds without these guards. A-roll entries are
never touched.

## Audio sync

Camera and mic clocks can disagree by hours, so only the calendar day is used
as a prior, and cross-correlation of 100 Hz RMS envelopes finds the offset.

```bash
uv run videoeasy audiosync cache
uv run videoeasy audiosync match  --clip <camera file>
uv run videoeasy audiosync verify --clip <camera file>   # word-overlap check
uv run videoeasy audiosync report                        # → out/audio_sync.json
```

`transcribe` prefers synced lav audio over camera scratch audio when the sync
was transcript-verified. Known defect: a mic session shorter than the clip
can yield an impossible lag (docs/lessons.md §9).

## Troubleshooting

- **`check` fails on vision.** LM Studio is not serving or the model is
  unloaded. `lms ps`, reload, `lms server start`.
- **Context TOO SMALL, or tags hitting the token budget.**
  `lms unload --all && lms load google/gemma-4-26b-a4b --context-length 16384`,
  then re-run `annotate` (only error entries are retried).
- **SUSPECT.** The backend is dropping images. Do not proceed. Stay on LM
  Studio.
- **Missing footage in the inventory.** The drive is unmounted, or the
  extension is not in `VIDEO_EXTS` (`src/videoeasy/config.py`). Check the
  inventory count before continuing: an empty source directory produces an
  empty inventory, which later stages will happily use.
- **Wrong shot boundaries.** Tune `shots.adaptive_threshold` (lower is more
  sensitive, and log footage is low-contrast), delete `out/shots.json`,
  re-run `shots` and `frames`.
- **Connection refused on localhost:1234.** The server is down or a sandbox
  blocked the request. Check `lms ps` before concluding it is a bug.

## Do not

- Modify anything on the footage drive or under `inputs/`.
- Rename or remount the footage drive under another name: absolute source
  paths are baked into the artifacts.
- Regenerate `out/bible-synthesis.md`. It is conversational judgment, not a
  pipeline output. `assemble` does not touch it, and neither should you.
- Run Ollama and LM Studio jobs at the same time.
