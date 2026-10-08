# Facial register moments

`videoeasy.faces`, `videoeasy.sounds` and `videoeasy.register` tag each clip at
the timecodes where a face visibly shows a register (laughter, joy, sadness,
grief, contemplation, anger, fear, effort, or the project's own vocabulary).
The result is a map an editor can search and a placement step can use: "every
silent, steady moment of grief from this session", or "cover for this
testimony line from the same person's laughter in the workshop".

It is the face modality of the register design in
[sibling-project.md](sibling-project.md), applied to field and B-roll clips
rather than to a speaker's utterances. It follows the rules in
[lessons.md](lessons.md): measure what can be measured, judge only the rest,
read modalities apart and fuse them in code, and trust no number until the
editor has calibrated it.

## What a tag means

A tag says what a face **shows** at a timecode. It never says what a person
feels, and nothing here names anyone: a face track (`t03`) is an anonymous
run of detections inside one clip. Tags are stored against footage, never
collected into a profile of a person, and never leave the machine. The
review page is a local file and is never published: it shows people's faces
in vulnerable moments.

## Pipeline

```
faces (measured) ──┐
                   ├─► register moments ─► register read (seen) ─► register map (fused) ─► review ─► calibrate
sounds (heard) ────┘
```

| Stage | What it does | Output |
|---|---|---|
| `faces` | Decodes each source at 5 fps (long side 1920 px). YuNet finds faces; MediaPipe's FaceLandmarker reads 52 blendshapes and head pose for every face of 40 px or more. Detections are linked into tracks (overlap, or a near centre at a similar size, up to 1 s apart). A sample under `MIN_FACE_PX`, turned past `MAX_YAW`, or without landmarks is **unreadable**, never neutral. Blendshapes collapse into cue channels (smile, cheek raise, inner brow, lowered brow, eyes closed, jaw open, frown, press, stretch, look down …). A track with 3 s of readable samples gets its own baseline (its median per cue); a shorter one uses the population's. | `out/faces/<id>.json`, `out/faces.json`, manifest; raw evidence in `work/faces/` keyed on source content |
| `sounds` | YAMNet over each source's audio in ~0.96 s windows. AudioSet classes are grouped into families (laughter, crying, breath, shout, cheer, song, drum). A run of windows at or over `EVENT_MIN` is an event. Speech and music shares are kept as context. **Heard, not seen:** an event is never attributed to a face. | `out/sounds.json`, manifest |
| `register moments` | Candidates, no model. A cue **run** is a cue at or over its floor **and** this far over the track's baseline, held for 0.8 s (eye closure for 1.5 s, so blinks don't count). Runs on one track within 0.6 s join into one moment. Also: still stretches on a readable face (no expressive cue, lips still) when the vocabulary has a term that `still` supports; sound events no face moment covers; and **controls**, a deterministic share of readable stretches with no cue run, so calibration can measure what the candidate step misses. Each moment carries its face size, yaw, cue peaks against the baseline, heard events, and `lips_moving` (spread of jaw opening). | `out/register-moments.json` |
| `register read` | The local vision model gets the middle frame with the face boxed, plus five graded crops of that face in time order. A sound-led moment gets four frames instead. The prompt holds the vocabulary's one-line definitions and nothing else: no transcript, no name, no session, no measured cue. The answer is JSON: `visible` (own words), `register` (a term, `none` or `unreadable`), `strength` 0–3, `tears`, `speaking`, `same_face`. Anything outside the schema is `invalid`; a failed call is `unchecked`. Readings are cached on the moment, its span and the prompt. | `out/register-seen.json` |
| `register map` | Fusion in code. **`agreed`**: the seen register is supported by a measured cue or a heard family listed for it (`SUPPORT`, or `cues`/`heard` in the profile). **`seen_only`**: the model named a register that nothing measured backs. **`measured_only`**: cues or sounds fired, the model saw no expression. Also `none`, `unreadable`, `unchecked` and `invalid`. A moment is `speaking` when the lips measurably move or the model says so: that is sync, not silent cover. The map also reports coverage (readable face time vs footage) and **parroting** (the share of `visible` texts that repeat four words of a definition). | `out/register.json`, `out/register-map.md`, manifest |
| `register review` | A local, blind labelling page: a stratified sample across status, register and kind, controls included. For each moment it shows a short graded clip and the face crops, with the vocabulary as buttons (up to two terms, or none / unreadable / wrong face). Labels stay in the browser until downloaded. The model's reading is hidden. | `work/reviews/register-<date>/index.html` |
| `register calibrate` | The editor's labels against the fused label and the seen register: precision and recall per term, what the controls held, and a confusion list. | `out/register-calibration.md`, manifest |

## Commands

```bash
uv run python -m videoeasy.faces  --config config.<p>.yaml [--only SRC] [--workers 4]
uv run python -m videoeasy.sounds --config config.<p>.yaml [--only SRC] [--workers 6]
uv run python -m videoeasy.register moments   --config config.<p>.yaml
uv run python -m videoeasy.register read      --config config.<p>.yaml [--limit N] [--only SRC] [--kinds face,heard,still,control] [--workers 3] [--retry]
uv run python -m videoeasy.register map       --config config.<p>.yaml
uv run python -m videoeasy.register find      --config config.<p>.yaml [--register grief] [--min-strength 2] [--silent] [--status agreed,seen_only] [--sources A,B] [--person LABEL]
uv run python -m videoeasy.register review    --config config.<p>.yaml [--n 120]
uv run python -m videoeasy.register showcase  --config config.<p>.yaml [--per 6]   # local page: numbers, most confident tags (face close-ups), disagreements
uv run python -m videoeasy.register calibrate --config config.<p>.yaml --labels <downloaded labels>.json
```

`read` goes in priority order: silent face moments with the strongest cues
first, then sound-led, still and control moments. A partial run with
`--limit` therefore reads what matters most. Start LM Studio first (see
AGENTS.md: gemma-4, 16k context), and never run an Ollama job at the same
time.

### The measurement environment

MediaPipe 0.10.21 pins numpy below 2 and protobuf 4, which the project has
outgrown. MediaPipe 1.0.1 aborts on macOS: its face detector graph opens a
Metal helper even with the CPU delegate. So `src/videoeasy/faceworker.py` is a
PEP 723 script with its own pinned dependencies. `faces` and `sounds` sync its
environment once (only when it cannot import MediaPipe) and run it on the CPU.
Parallel `uv run --script` calls race on that shared environment, and 66
sources once failed to import MediaPipe that way, so the interpreter is
resolved once per run. The project's own environment never imports MediaPipe.

Models download on first use into `~/.cache/videoeasy/models` (or
`$VIDEOEASY_CACHE/models`) and are refused when their sha256 differs from the
pinned one.

## Vocabulary

The default vocabulary is generic and defines each term by what a face
visibly does. A project replaces it in its `film.yaml`:

```yaml
face_registers:
  grief: "crying or on the edge of tears, the face crumpling, eyes wet or reddened, a hand to the face"
  awe: {definition: "eyes wide, mouth open, looking up", cues: [eye_wide, brow_outer_up], heard: []}
```

Definitions describe the face, never a story. Never put example phrases in
them (the model copies them; the map reports how often), and never start one
with a gloss word and a colon: the model answers in the same shape ("effort:
straining") and the term becomes unreachable ([lessons.md](lessons.md) 2.12). `cues` and `heard`
say which measured cues and sound families may support a term. For the
default terms they come from `register.SUPPORT`; a new term without them can
only ever be `seen_only`.

## What the first event run showed (4 Oct 2026)

- Measurement ran at 5–8x real time on one CPU worker. A frontal to-camera
  face (about 280 px) read in every frame. Observational B-roll faces in a
  group were mostly 30–60 px or turned 40–60°, and many read in few or no
  frames. Expect large unreadable shares on observational footage, and
  report them.
- On a pilot of 93 candidates from arrival and opening clips, every reading
  parsed (0 invalid) and parroting was 0.0. On the full run, 4 of the first
  300 answers were `invalid` with one repeated reason, which traced to the
  shape of one definition (lessons 2.12). The vocabulary was reworded and the
  read restarted. Status: 25 agreed (15
  contemplation, 6 joy, 4 laughter), 19 seen-only, 28 measured-only, 19 none,
  2 unreadable.
- The model sees more than the cue floors. Subtle closed-mouth smiles at
  strength 1 rarely tripped the smile floor, and 5 of 7 controls were read as
  joy or laughter, one a strength-2 laugh with the head tilted down (pitch
  weakens the smile blendshape).
- The lowered-brow cue fired most often (37 of 57 face candidates), and the
  model read most of those faces as neutral or smiling: probably sun squint
  and small-face noise. Its floor is a calibration question.
- Contemplation at strength 1 was mostly people looking down while
  listening. Whether that counts is the editor's call.

None of these are results until the editor's blind sample is scored.

## Placement (next)

The map is shaped like the other usable-parts measurements: moments are
source-second spans, as steady runs (`steadiness.py`) and clean drone moves
(`moves.py`) are. The intended wiring:

- **Catalogue.** brollmatch's catalogue line for a unit gains its agreed,
  silent moments inside its steady parts ("REGISTER MOMENTS: 12.4–16.0
  laughter, 31.0–35.5 grief"), shown as seen, never as said.
- **Proposer.** The words of each pick (a talk bite's own word-only register
  from `talks.py`, or a testimony line) are matched to moments of that
  register, preferring the same session, and the same person once the editor
  links tracks to people (see below).
- **Builder.** A row placed on a register moment is clamped into it, as rows
  are clamped into steady runs, and a `speaking` moment is never silent cover
  (`speech.talking_cover` rule).
- **Scorer.** cuteval re-reads the face in every register-covered stretch of
  the render.

Linking a face track to a person (so a participant's before/after video gets
their own moments) is `videoeasy.identity`: face embeddings group the tracks
(or match them to reference photos), and the editor confirms every link
([identity.md](identity.md)). `register find --person <label>` then keeps
only that person's confirmed faces. It stays local. The recognition model is
downloaded at setup and runs in its own worker environment: its licence is
its own.

## Licences

YuNet (OpenCV zoo): MIT. MediaPipe FaceLandmarker and YAMNet: Apache-2.0. The
vision model is whatever the config names (gemma-4 by default). None of these
recognise who a face belongs to. Identity has its own model and licence
([identity.md](identity.md)).
