# Long talks: beats, hooks, bites and cover briefs

`python -m videoeasy.talks` turns one long talk into what an editor needs to
cut 60–90 s features and testimonials from it. The talk is a speaker's
body-worn recording (a lapel transmitter recording internally, typically in
32-bit float). The output is:

- the talk's **beats**: contiguous stretches, each with one purpose;
- **hooks** (3–12 s): lines that could open a video;
- **bites** (6–30 s): passages that stand on their own;
- **closers** (3–15 s): lines that land an ending.

Every candidate carries a **cover brief**: what b-roll would complement the
words, or `face` when the speaker should carry them. It also carries its
in/out in the original recording files.

It follows the rules in [lessons.md](lessons.md):

- Measure what can be measured: levels, overs, edge room, durations, and
  the voice that is not the wearer's.
- Judge only the rest, through a proposer and an independent verifier.
- A failed check is `unchecked`, never a score.
- One ASR pass is a question, not a finding.
- No number is trusted until it is calibrated against the editor
  ([Calibration](#calibration)).

This is milestone 1 of the sound-bite job in
[sibling-project.md](sibling-project.md), built inside videoeasy because event
projects need it now. It has no diarization, pairwise ranking or padded
recheck yet (see [Limits](#limits)).

## Commands

```bash
# the whole analysis of one recorder session, written under the project
uv run python -m videoeasy.talks all --mic-dir /path/to/MIC --session 06 \
    --project data/projects/<p> --talk-id talk-01 --speaker "Speaker A" --title "Opening talk"

# named files instead of a session (a single file, or chunks whose names carry start times)
uv run python -m videoeasy.talks all --audio A.WAV B.WAV --out data/projects/<p>/talks/talk-01

# levels and transcript only, no text model (stages 1-3, then the report)
uv run python -m videoeasy.talks all --skip-model --mic-dir ... --session ... --out ...

# any one stage, from the cached earlier ones
uv run python -m videoeasy.talks {prepare|transcribe|sentences|beats|bites|verify|report} --out <dir>

# ranked candidates as a cut for brollmatch, plus a transcript keyed by chunk
uv run python -m videoeasy.talks export-cut --out <dir> [--top 20]

# a selects timeline in Resolve: dry run by default
uv run python -m videoeasy.talks resolve --out <dir> [--timeline "talk-01 selects"] [--bin Talks/talk-01] [--top 30]
uv run python -m videoeasy.talks resolve --out <dir> --apply --resolve-project "<open project>"
```

Options:

- `--session`: a session number, or the file name of the chunk to start
  from. `--mic-dir` is searched along with its immediate sub-folders (one per
  transmitter). The same session number in two folders is refused: name the
  folder instead.
- `--name-pattern`: a regex with named groups `session`, `date` (YYYYMMDD) and
  `time` (HHMMSS). The default matches DJI mic names
  (`DJI_<nn>_<YYYYMMDD>_<HHMMSS>.WAV`). Newer recorders may name files
  differently, so check yours.
- `--assume-contiguous`: take `--audio` files in the given order when their
  names carry no start time. This is recorded as `given order (unchecked)`.
- `--channel N`: analyse channel N of a multi-channel file (default 1). A
  channel is never mixed.
- `--registers a,b,c`: the register vocabulary. With `--project`, it defaults
  to the project's `deliverables.yaml` `registers`.
- `--broll-context FILE`: a short text (up to 6000 characters) of what b-roll
  exists, so cover briefs prefer material that is actually there.
- `--include-off-mic`: allow off-mic sentences inside a candidate.
- `--overwrite`: replace a previous run whose inputs differ, and ask again
  where the last answer was `invalid`.
- Models are set by environment variable:
  - `VIDEOEASY_TEXT_MODEL` / `VIDEOEASY_TEXT_URL` (Ollama; the default is the
    eval loop's text model);
  - `VIDEOEASY_WHISPER_MODEL` (default `mlx-community/whisper-large-v3-mlx`).

## 32-bit float, and why the working file is built the way it is

A 32-bit float recorder does not clip internally: a shout or a knock on the
transmitter is stored above 0 dBFS. Converting that to integer audio at unity
gain clips it, and Whisper wants 16 kHz anyway. So `prepare`:

1. **Proves the chunks are one recording.** The recorder splits a session
   into contiguous chunks (DJI: about 30.7 min each). Chunks are ordered by
   the start time in their names, and each must begin where the last ended,
   within 2 s, because names carry whole seconds. A gap or overlap is refused
   and named: a talk with a hole would map every later in/out to the wrong
   place. Chunks from two sessions are refused.
2. **Measures each chunk from the float data.** It records the sample peak in
   dBFS (positive values allowed) and the count of samples over 0 dBFS. It
   also records integrated loudness and true peak (ffmpeg `ebur128`).
3. **Builds one working file.** It is 16 kHz, mono, signed 16-bit, made with a
   **single linear gain** that puts the talk's peak at −1 dBFS. If resampling
   raises a peak above the sample peak (inter-sample overshoot), that peak
   decides instead, so the integer conversion cannot clip. The clip count is
   recorded and must be 0. There is no compression and no loudness
   normalisation, because level is evidence: it is what tells the wearer
   from another voice, and speech from room.
4. **Keeps the way back.** Every talk time maps to (chunk file, seconds into
   it), from the samples actually written per chunk. Every in/out the tool
   reports is in the original float files, for the editor to lay.

Originals are never written. File hashes are cached in the output folder.
`evalrun.fingerprint` would write a `.sha256` sidecar beside large files, and
the recordings' folder must never be touched.

**What real recordings showed.** These are DJI mic files from an earlier
shoot, run read-only:

- **Short chunks held only handling noise.** The 17.6 s and 9.6 s chunks
  were start/stop handling noise with no speech: sample peaks of +6.0 and
  +6.7 dBFS (30 and 24 overs), integrated −33.5 and −46.8 LUFS. Whisper
  returned one segment for each, and both were dropped by its own
  confidence thresholds. A single bump decides the gain (−7.0 and −7.8 dB
  there). On a long talk that means the speech sits correspondingly lower in
  the working file. Sixteen bits still leave about 60 dB of resolution under
  quiet speech at −35 dBFS, which is why a recorded linear gain is preferred
  to anything that reshapes the dynamics.
- **A 60 s excerpt of a long chunk carried real speech.** Its peak was
  −9.1 dBFS with 0 overs, integrated −39.9 LUFS, so the gain was +8.1 dB and
  the working file peaked at −1.09 dBFS (resampling lowered it), with 0
  clipped samples. Whisper kept 34 segments and dropped 1; prepare,
  transcribe and sentences took 5 s together, transcription 4.8 s of that.

## Stages and outputs

All outputs sit in the talk's folder (`--out`, or
`<project>/talks/<talk-id>/`). Each JSON output has a `<name>.manifest.json`
beside it, recording:

- input hashes,
- the model,
- prompt hashes,
- what was checked and what was left unchecked.

| Stage | Writes | Model | What it does |
|---|---|---|---|
| prepare | `prepare.json`, `work.wav`, `hashes.json` | none | grouping, measurement, the working file and the talk-time map (above) |
| transcribe | `transcript.json` | Whisper | word timestamps, `condition_on_previous_text=False`, `--language` (default `en`); segments over `no_speech_prob` 0.6 or `compression_ratio` 2.4 are dropped and kept in `dropped` (the B-roll pass's thresholds: Whisper invents speech in quiet stretches) |
| sentences | `sentences.json` | none | sentence units, measured edge room and the off-mic flag (below) |
| beats | `beats.json` | text model, one call | contiguous, ordered beats with title, summary, energy and turn; a talk too long for the context is split into overlapping windows and merged at the middle of each overlap (recorded) |
| bites | `bites.json` | text model, one call per beat | proposals: kind, sentence span, why, register (from the vocabulary, or `none`), cover brief |
| verify | `verdicts.json` | text model, one call per candidate | the candidate's words ONLY, no context, no proposer reasoning: `standalone` 0–3, `refers_to_unsaid`, `strength` 0–3 (hook: would it stop a scroll; bite: specific, quotable, felt; closer: does it land), a reason |
| report | `analysis.json`, `analysis.md`, `selects.csv` | none | everything joined; ranked hooks, bites per beat and closers with talk time and source chunk + chunk time; flags; what is not established |
| export-cut | `cut-selects.json`, `transcripts-by-chunk.json` | none | see [Into the cut loop](#into-the-cut-loop) |
| resolve | `resolve-plan.json` or `resolve-report.json` | none | see [Resolve selects timeline](#resolve-selects-timeline) |

Ranking is deterministic:

- score = standalone + strength, out of 6;
- ties go to the candidate with more room at its tighter edge, then to talk
  order;
- within a kind, an overlapping lower candidate is marked `superseded_by`
  the better one;
- hooks, bites and closers are ranked separately; a hook may sit inside a
  bite.

`selects.csv` has one row per candidate per chunk it touches (a candidate
across a chunk join gets two rows, `part` 1/2 and 2/2). Its columns:

- `source_file`, `source_in_s`, `source_out_s`, `talk_in_s`, `talk_out_s`;
- the exact word times;
- `text`, `cover_brief`, `register`, `flags`.

Its in/out points carry up to 0.3 s of *measured* room as handles, never
more than was measured.

## Measured, judged, and neither

| Measured (code) | Judged (local text model) | Not established here |
|---|---|---|
| chunk contiguity, sample peak, overs, loudness, true peak, working-file clipping | the beats | who is speaking (no diarization) |
| sentence times, pauses, durations | which spans are hooks, bites, closers | the speaker's voice or face register |
| edge room from energy, noise floor, signal-to-noise | standalone and strength scores | whether a word is right (one ASR pass) |
| off-mic level | the register reading of the words | whether a score means what the editor means (uncalibrated) |
| ids, beat bounds, duration limits, dangling starts | the cover brief | |

The register on a candidate is a reading of the **words only**. Displayed
register (voice, face) needs the measurements in
[sibling-project.md](sibling-project.md) §4.1. A cover brief is a request for
picture, not a claim that such picture exists. Pass `--broll-context` so it
leans on what does exist, then let `brollmatch` and its verifier decide.

## Statuses and caching

- **`ok`**: the check ran and its result is usable.
- **`unchecked`**: it did not run, or the model did not answer. There is no
  score and nothing is established, so the next run simply retries it.
- **`invalid`**: the answer was unusable, for example beats with a gap, an
  id that doesn't exist, or a score of 9. Nothing from it is partly
  accepted. `--overwrite` asks again.
- **`excluded`** (candidates only): the measured gates keep it out of
  ranking. It is listed in the report under "Gated out" with the reason.

Every stage is keyed on the content hash of its inputs, its parameters, the
model and the prompts. Caching works as follows:

- **Unchanged inputs.** A re-run with unchanged inputs does no model work.
  Per-beat proposals and per-candidate verdicts are cached individually, so
  a retry after a timeout asks only what failed.
- **Changed inputs.** A previous run whose inputs differ is evidence: it is
  replaced only with `--overwrite`.

## Flags and thresholds (all provisional)

| What | Value | Effect |
|---|---|---|
| chunk contiguity | ±2 s | refuse |
| working-file peak | −1 dBFS | single linear gain |
| sentence break | terminal punctuation, or a pause ≥ 0.6 s | |
| longest sentence | 30 s, split at its longest internal pause | |
| noise floor | 10th percentile of all 10 ms frames | |
| speech level of a stretch | 90th percentile of its word frames | |
| quiet (for edge room) | within 6 dB of the noise floor | |
| room measurable | speech ≥ 12 dB over the floor; else room is `None` and the flag is `room_unmeasured` | |
| room cap | 2 s (more is reported as 2 s) | |
| `tight_in` / `tight_out` | measured room < 0.15 s | flag |
| off-mic | sentence level ≥ 10 dB under the median sentence level | excluded from candidates unless `--include-off-mic` |
| durations | hook 3–12 s, bite 6–30 s, closer 3–15 s | excluded outside them |
| `dangling_start:<word>` | first word in {and, but, so, because, which, that, this, it, they, he, she, also, then, …} | flag |
| `spans_chunks` | the candidate crosses a chunk join | flag; two source rows |
| handles on selects | up to 0.3 s of measured room | |
| proposals per beat | at most 6 | |

Why quiet is measured against the noise floor, from the same real excerpt:

- **The rule first tried failed.** Quiet was defined as 20 dB under the
  median speech level. That put the threshold at −67 dBFS, below the
  −58 dBFS floor of wind and birds, and it measured 0 s of room at all 33
  sentence edges.
- **What the floor rule measures.** With the floor rule the same excerpt
  measures a floor of −57.2, speech at −28.4 and an SNR of 28.8 dB. Still,
  26 of 33 starts and 31 of 33 ends have under 0.15 s of room: it is
  conversational speech with short pauses, and Whisper's word ends tend to
  fall early. Expect the tight flags to be common. They are flags, not
  gates.
- **Off-mic split cleanly.** 10 of the 33 sentences sat at −43 to −47 dB
  against the wearer's −32.6 dB: a second voice the lav heard. That reading
  comes from the levels alone; nobody listened to confirm it.

## Calibration

None of these numbers means anything to the editor until it has been
compared with the editor's own judgment. Follow
[lessons.md](lessons.md) §6.12.

1. **Freeze.** Fix the working file, the transcript, the sentence ids, the
   prompts and the model (the manifests record all of them). Develop on one
   talk and confirm on a second that the prompts were not developed on.
2. **Make a baseline.** Use the 20 longest pause-bounded spans of 6–30 s
   (from `sentences.json`), so that "beats the baseline" means something.
3. **Score blind.** Shuffle the top 20 candidates and the 20 baseline spans
   into one packet with ids hidden. The editor marks each keep, maybe or
   reject while listening from the source in/out. Keep the key in a separate
   file until scoring is done.
4. **Write the decision rule before scoring.** For example: at least 10 of
   the top 20 keep or maybe; at least 4 more keeps than the baseline; no bite
   whose in/out lands inside a word; and three repeated runs sharing at
   least 15 of the top 20.
5. **Calibrate the measured thresholds by listening.** Check edge room
   (does a 0.15 s head cut a breath or a word?) and off-mic (is the flagged
   sentence another voice?) on a sample of each.

Record the editor's time from a finished `prepare` to a marked selects
timeline. That is the number this tool exists to shrink.

## Resolve selects timeline

`resolve` plans a timeline of every ranked candidate: hooks, then bites, then
closers, each in rank order. Candidates sit on A1, about 1 s apart, cut from
the **original float chunks**, and a candidate across a chunk join is two
items butted together. There is one marker per candidate:

- colour by kind: hook Red, bite Blue, closer Green (marker colours include
  Red, clip colours do not);
- name: kind, rank and score;
- note: the words plus the cover brief.

The default is a **dry run** that writes `resolve-plan.json`. `--apply` does
the rest:

- refuses when the open project is not the named one, or when Resolve is on
  no page (a modal dialog, under which every write silently fails);
- refuses an existing timeline name;
- imports missing chunks into the bin and verifies them by re-reading it;
- creates the timeline;
- places every item with `AppendToTimeline` + `recordFrame`, the only
  placement that does not ripple;
- adds the markers, reads everything back and verifies it: items by name,
  track and record frame within ±2 frames, then duration; markers by frame
  and name; unexpected items listed. It exits 2 unless every check passes.

**Not checked live.** Nobody has checked live which frame units
`AppendToTimeline` uses for an audio-only WAV. For video, startFrame and
endFrame are the source's own frames
([resolve-scripting.md](resolve-scripting.md)). The plan assumes a WAV is
addressed in frames of the timeline's rate, and the live timeline's rate
replaces `--fps` at apply time. A wrong unit shows up in the read-back as
durations off by the rate ratio, so the first real run is its own test: make
it on a scratch timeline and read the verify table before trusting it.

## Into the cut loop

`export-cut` writes `cut-selects.json`, a cut in the format `brollmatch`
reads:

- one beat per ranked candidate, with title `<kind> #<rank> (<score>/6)`;
- the audio picks from the original chunks (`clip` = chunk stem, `in_s`,
  `out_s`, `duration_s`, `text`, `source_path`), and no video rows;
- the cover brief as the beat **note**, which `brollmatch` hands its
  proposer as the editor's note;
- for a `face` brief the note is "no cover: …", which `brollmatch` keeps
  bare by rule.

It also writes `transcripts-by-chunk.json`, the same words in the ingest's
transcript schema, keyed by chunk stem with chunk-local times.

What each part of the loop can do with it:

- **`brollmatch`** reads it as it is, given a film directory with the
  b-roll catalogue (`out/`) and `--cut` pointing at the file.
- **The film profile needs an entry for the chunks.** Mic chunk stems start
  with `DJI_`, like drone clips, so a profile with the prefix `[DJI_, drone]`
  labels a talk pick as drone. Both start `DJI_0`, so no prefix alone
  separates them: list the chunk stems under `clips:` with a role of their
  own.
- **`layin` does not work yet, for two reasons.** It names each audio row
  `<clip>.MOV`, so a WAV chunk in the media pool (`<stem>.WAV`) is refused
  as missing. And its word-bound lookups read the film's
  `out/transcripts.json`, so the chunk transcripts must be merged there
  first; that is an ingest artifact this tool does not write. The WAV frame
  question above applies too.
- **The selects carry no picture.** A feature needs the speaker's picture
  under the lav audio: camera clips synced to the chunks
  (`videoeasy audiosync`).

## How this fits an event project

See [event-projects.md](event-projects.md).

- **Recordings to deliverables.** Speaker and facilitator talks recorded on
  lapel transmitters become `analysis.md` and `selects.csv` per talk. The
  hooks and bites are the raw material for 60–90 s features and testimonial
  cuts; the beats say where in a long talk each part lives.
- **Registers.** `--project data/projects/<p>` takes the register
  vocabulary from `deliverables.yaml`, so the words' register lines up with
  the b-roll package's bins.
- **Cover briefs to b-roll.** The briefs say what the b-roll package must
  supply. Pass a short summary of what was shot (per register bin, or from
  the ingest annotations) as `--broll-context`. `export-cut` →
  `brollmatch` turns briefs into verified cover proposals.

## Limits

- **No diarization.** Off-mic is a level, not an identity. An interviewer on
  their own transmitter is not separated from the wearer.
- **No padded recheck yet.** Before a bite is used, listen; or rerun the
  words through `speech.transcribe_excerpt` and `speech.analyse` as the
  story check does.
- **No pairwise ranking.** Absolute 0–3 scores drift. The sibling design's
  pairwise step is not built.
- **Fillers and restarts.** Whisper drops or smooths them, so disfluency is
  not measured.
- **Uncalibrated.** Every threshold and both scores, until the editor's
  ratings say otherwise.
