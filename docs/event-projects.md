# Event projects: a shot list and many short deliverables

videoeasy was built for one short documentary: about an hour of interview
and a few hundred shots, cut into one film. An event shoot is a different
shape. Several operators film a multi-day programme from a shot list, and
the editor owes dozens of short videos: testimonials, before/after
interviews, speaker or facilitator features, a B-roll package organised by
mood, and full session recordings. This doc covers using the same tools on
that shape, and is honest about which parts are automated, which are
agent-assisted, and which aren't built.

Set up the machine first: [setup.md](setup.md).

> **Status.** The project file (`deliverables.yaml`), the item tracker
> (`items.yaml`) and the `videoeasy.deliverables` commands are new and
> unit-tested, but not yet run on a full event's footage. Where this doc and
> `src/videoeasy/deliverables.py` disagree, the code is authoritative;
> `deliverables.example.yaml` passes `validate` and shows every feature.

## How it differs from the documentary case

- **Many deliverables, one footage pool.** A B-roll take from day 2 can
  serve a testimonial, a speaker feature and the B-roll package. Ingest runs
  once over the whole shoot, and every deliverable draws on the same
  artifacts.
- **The schedule is metadata.** The shot list says who filmed what, when.
  A clip's recorded start time places it in a session, and the session says
  which deliverables it feeds and which moods it was meant to capture. That
  is a measurement (a time inside a window), not a judgment, so it comes
  before any model looks at a frame.
- **The unit of work is the item.** "10 speaker features" is ten items, each
  with its own status, people, sessions and files. Progress is counted
  against evidence on disk, not memory.

## The project directory

Everything about one event lives under `data/projects/<project>/`, which git
ignores like the rest of `data/`. Names of the people filmed, the shot list
and the client stay there and never enter the repository.

```
data/projects/<project>/
  deliverables.yaml       hand-written: schedule, cameras, people, moods, deliverable types
  items.yaml              one entry per deliverable item, with its status
  out/                    pipeline artifacts for the whole shoot (transcripts, annotations, ...)
    sessions.json         written by `assign`: each clip's session, feeds and mood candidates
  work/                   frames, contact sheets, caches (regenerable)
  film.yaml               the project's profile: roles, aliases, register paragraph (items fall back to it)
  items/<item-id>/        one item laid out as a film directory (`deliverables item-dir <item-id>` makes it):
    film.yaml             optional: only when this item needs its own roles or aliases
    editorial/            cut-vN.json, intent.json, renders/, ...
    out -> ../../out      the shared artifacts (relative link)
    work -> ../../work    shared frames and caches (relative link)
```

`deliverables item-dir <item-id>` creates `editorial/` and the two links and
never replaces anything that exists. An item without its own `film.yaml`
uses the project's, and reports say which file they read. `brollmatch`,
`cutbuild`, `cuteval`, `storycheck`, `intent` and `layin` resolve their
`out/` and `work/` paths through the links, and source paths come from the
ingest inventory, so `.MP4` sources and footage outside `inputs/` work too.
That is checked by reading the loaders and by unit tests, not yet by a full
loop on event footage. Two things to know:

- The default synthesis path (`out/bible-synthesis.md`) resolves to the
  project's shared file. Story work for one item should keep its own
  synthesis in `editorial/` and pass `--synthesis editorial/<file>.md`.
- `deliver` needs a `delivery_loudness` rule in the item's own
  `editorial/intent.json`.

## From shot list to `deliverables.yaml`

Go through the shot list line by line. Each line usually says *when*
(session), *who filmed* (operator), *what* (interview, B-roll of a mood,
drone, static recording) and *for which deliverable*. Map each part onto a
field:

| Shot list says | Goes to |
|---|---|
| day and time of a session, its title | `sessions[].day`, `start`, `end`, `title` |
| operator names, their cameras | `cameras[]` (`name`, `operator`, how to recognise the files) |
| "operator X: 3 before/after interviews" | `sessions[].capture[]` as `{operator, text, feeds}` |
| "B-roll of joy and celebration" | `sessions[].capture[].registers` or session `registers` |
| "10 × 60–90 s features" | `deliverable_types[]` (`count`, `duration_s`, `template`) |
| whose editing scope it is | `deliverable_types[].scope` |
| "every video includes the edit, grade and mix" | `deliverable_types[].includes` |
| named participants | `people` (private: this file never leaves `data/`) |

An abridged example of the shape (the full, annotated version is
[`deliverables.example.yaml`](../deliverables.example.yaml)):

```yaml
project: harbour-festival-2026
client: "Example Arts Trust"
timezone: Australia/Sydney

cameras:
  - {name: a-cam,  prefix: DSCF, path_contains: /op-a/,   operator: op-a, clock: creation_time_local, clock_offset_s: 0}
  - {name: b-cam,  prefix: DSCF, path_contains: /op-b/,   operator: op-b, clock: creation_time_local, clock_offset_s: 0}
  - {name: drone,  prefix: DJI_,                         operator: op-c, clock: creation_time_utc,   clock_offset_s: 0}
  - {name: static, path_contains: /static/,              operator: op-d, clock: timecode,            clock_offset_s: 0}

registers: [focus, laughter, celebration, stillness]

sessions:
  - id: d1-keynote
    day: 1
    start: "19:00"
    end: "20:30"
    title: Opening keynote
    cameras: [a-cam, static]
    capture:
      - {operator: op-a, text: "60–90 s feature of the keynote speaker", feeds: [speaker-feature]}
      - {operator: op-d, text: "static camera and audio of the full keynote", feeds: [session-recording]}
  - id: d2-workshop-wood
    day: 2
    start: "10:00"
    end: "12:00"
    title: Woodwork workshop
    cameras: [a-cam]
    registers: [focus, laughter]
    capture:
      - {operator: op-a, text: "3 before/after interviews with attendees", feeds: [before-after]}
  - id: d2-workshop-glass          # runs at the same time as the woodwork workshop
    day: 2
    start: "10:00"
    end: "12:00"
    title: Glass workshop
    cameras: [b-cam]               # disjoint from d2-workshop-wood, so the overlap is allowed
    capture:
      - {operator: op-b, text: "B-roll of concentration and stillness", registers: [focus, stillness]}

deliverable_types:
  - {id: speaker-feature, title: "Speaker features", count: 6, duration_s: [60, 90],
     template: "speaker to the room → B-roll → speaker to camera", scope: editing, includes: [edit, grade, mix]}
  - {id: before-after, title: "Attendee before/after", count: 9,
     template: "before interview → B-roll of the workshop → after interview", scope: editing, includes: [edit, grade, mix]}
  - {id: session-recording, title: "Full session recordings", count: 4, scope: recordings}
```

Some tips:

- **Capture who filmed what.** Event shot lists put several operators in one
  session ("A films the feature, B films attendee interviews, both get
  B-roll of laughter"). Write each as its own `capture` entry with its
  `operator`. `assign` then gives a clip only the feeds and moods of the
  entries for *its* operator, falling back to the session-level `feeds` and
  `registers` when nothing matches. A plain string is also accepted, for a
  line with no operator.
- **Give each operator their own folder.** Two operators on the same camera
  brand write the same filename prefix (two Fuji bodies both write `DSCF`).
  A camera entry matches on `prefix` (the filename), `path_contains` (a
  substring of the source path, such as the operator's card folder) or both.
  Copy each card into a folder named for its operator and match on that.
- **Parallel tracks are normal.** Two sessions may overlap in time only when
  they list disjoint `cameras:`. Each lists camera *names*. A clip matches
  only sessions that include its camera.
- **Fill in `items.yaml` from the types.** `plan` lays out the items each
  type implies. You then attach people and sessions as they become known.

Validate after every edit:

```bash
uv run python -m videoeasy.deliverables validate --project data/projects/<project>
```

## Clocks

Camera clocks lie. Bodies drift, some were never set, drones record UTC,
and some cameras write local time into a field labelled UTC. The `clock`
field says which source to trust per camera (`creation_time_utc`,
`creation_time_local` or `timecode`). `clock_offset_s` corrects a body that
is wrong.

```bash
uv run python -m videoeasy.deliverables clocks --config config.<project>.yaml --project data/projects/<project>
```

`clocks` shows, per camera, the recorded start times it reads, so that a
camera whose clips sit an hour outside every session stands out. Measure the
offset rather than guessing it. The best evidence is two cameras that heard
the same sound: a clap, a bell, the first word of a keynote. Cross-correlate
the audio (the `audiosync` commands do this for lav recordings against
camera clips) and set the difference as `clock_offset_s`. This is the same
lesson as lav sync: [sync by waveform, not by clock](lessons.md#510-sync-by-waveform-not-by-clock).

## Ingest, then assign

The ingest config points at the project's footage and writes into the
project:

```yaml
# config.<project>.yaml (git-ignored)
paths:
  aroll_dir: data/projects/<project>/inputs/interviews   # interviews, to-camera pieces, session recordings
  broll_dir: data/projects/<project>/inputs/broll        # B-roll and drone
  work_dir:  data/projects/<project>/work
  out_dir:   data/projects/<project>/out
```

Run [`check`](setup.md#3-check-the-whole-chain), calibrate on 2–3 clips, then
ingest as for a film ([user guide §4](user-guide.md#4-verify-calibrate-then-commit-the-night)).
`probe` records each clip's creation time along with its timecode. Then:

```bash
uv run python -m videoeasy.deliverables assign --config config.<project>.yaml --project data/projects/<project>
```

`assign` writes `out/sessions.json`: for every clip, its camera, its corrected
start time, the session that contains it, and `feeds_candidates` and
`register_candidates` from that session's capture entries. A `null` session
is information, not an error:

- **outside every session**: the clip was recorded between sessions or
  before the schedule starts, or the camera's clock is off. Check `clocks`.
- **`ambiguous: <ids>`**: two sessions still match after the camera filter.
  Give one of them a `cameras:` list, or fix overlapping times.
- **no camera matched**: no `prefix`/`path_contains` fits the file. Add a
  camera entry.
- **no clock**: the file carries neither the creation time nor the timecode
  its camera's `clock` names.

Candidates are what the shot list *intended* to be filmed then. They are not
a description of the clip. A clip in a "joy" session may show someone
setting up a tripod.

## Deliverables, one kind at a time

How much of each job the repository does today:

| Job | Automated (measured) | Agent-assisted over text | Manual (the editor) | Not built yet |
|---|---|---|---|---|
| Testimonials | transcription with word times; edge room; ASR recheck | bite proposals from transcripts | rating and choosing bites; final trim | — |
| Long talks (lav recordings) | float gain staging; word times; edge room vs noise floor; off-mic flags (`videoeasy.talks`) | beats, hooks, bites, closers with cover briefs; standalone check | rating; calibration; picture sync | voice/face register; diarization |
| Before/after pairs | session times | pairing proposals from names spoken and session times | confirming every pair | face recognition (deliberately absent) |
| Speaker / facilitator features | the whole eval loop's measurements | radio cut, cover proposals | approving the radio cut; review | — |
| B-roll package by mood | session mood candidates; shake; drone moves; grading | shortlists from annotations | confirming each label; final selection | mood classification constrained to session candidates |
| Session recordings | transcription for search | finding passages | the recording edit itself | — |
| Delivery and tracking | loudness master with manifest; status against files | status reports | sign-off | — |

### Testimonials

1. Transcribe the interviews: `videoeasy transcribe --config …` writes
   word-timed transcripts to `out/transcripts.json`.
2. In a Claude Code session, the agent reads the transcripts and proposes
   bites with word-level in and out times: self-contained, starting at the
   start of a thought, not leaning on the question for meaning. You rate
   them, and the agent records your decisions.
3. In and out points come from **measured room**, not the stored
   transcript. The same per-channel energy measure the lay-in uses
   (`max_head_s`/`max_tail_s`) finds how much silence a cut point really
   has. A transcript once missed the word right beside a cut
   ([lessons 5.8](lessons.md#58-make-cut-points-from-measured-room-not-stored-transcripts)).
4. A word that matters (a name, a number, a "not") gets a padded second
   transcription before anyone trusts it. One ASR pass is a question, not a
   finding ([lessons 5.4](lessons.md#54-one-asr-pass-is-a-question-not-a-finding)).

Short interviews stay a conversation over transcripts, which works well at
the scale of a few hours of interviews.

### Long talks on lav recorders

Facilitators and speakers who wore a lav recorder (DJI Mic and similar,
often 32-bit float, split into ~30 min chunks) get a talk of their own:
[`videoeasy.talks`](talks.md) joins the chunks, stages the gain so float overs
never clip on the way into Whisper, transcribes with word times, measures
each sentence's room against the noise floor, flags the quiet sentences that
are probably someone else heard by the lav, and has the local text model
outline the beats and propose hooks, bites and closers. Each one carries a
**cover brief**: what b-roll would complement the words, or `face` when the
speaker should carry them. An independent pass reads each candidate with no
context and asks whether it stands alone. Unverified candidates are listed,
never ranked.

```bash
uv run python -m videoeasy.talks all --project data/projects/<project> \
    --mic-dir <lav folder> --session <first chunk> --talk-id <id> --speaker "<name>" \
    --broll-context data/projects/<project>/broll-context.txt     # optional: what b-roll exists
```

Outputs per talk: `analysis.md` to read, `selects.csv` for any NLE (source
chunk and time per candidate), and optionally a Resolve selects timeline
with a marker per candidate (`talks resolve`, dry run by default).
`talks export-cut` writes the selects as a cut the b-roll proposer can read.
Thresholds are provisional until the editor calibrates them
([talks.md](talks.md)).

### Before/after pairs

A before/after video needs the same person's "before" and "after"
interviews. The tool does **not** recognise faces, and won't: it would
mean biometric processing of the people filmed. Pairing comes from:

- names spoken on camera ("I'm …"), found in the transcripts;
- names the operator logged on set (ask operators to say or slate the name
  at the start of each interview);
- session times: "before" in the arrival session, "after" in the closing one.

The agent proposes pairs from those. **Every pair is the editor's
confirmation**, recorded in `items.yaml` (`people`, `sessions`). The B-roll
in the middle comes from the sessions the person attended, not from picking
the person out of a crowd.

### Speaker and facilitator features

A 60–90 s feature of one speaker is a small documentary: the speaker to the
room, B-roll, the speaker to camera, sometimes a testimonial to close. That
is the full loop this repository was built for:

```
radio cut (story room) → brollmatch → cutbuild → layin → cuteval / storycheck → deliver
```

Run it per item from `items/<item-id>/` (see
[the project directory](#the-project-directory)). The project's `film.yaml`
(or the item's own, when it differs) should have `roles` that mark the to-camera clips (`to_camera: true`, `never_cover:
true`) and whose `aliases` hold the names ASR gets wrong. Record decisions
in its `editorial/intent.json`. Details:
[user guide §6–7](user-guide.md#6-the-story-room) and
[eval-design.md](eval-design.md).

### B-roll package by mood

The deliverable is a graded, organised set of B-roll in bins: one per mood
from the shot list, plus categories like leadership moments and drone.

- **Candidates come from the schedule.** `assign` gives each clip the moods
  its session intended to capture (`register_candidates`). That is a
  measurement of *when*, not of *what*.
- **The annotation adds what was seen.** `annotate` writes mood, emotional
  metaphor, themes and cut notes per shot, from graded frames.
- **Measurements decide usability.** `steadiness` keeps only the steady
  parts of handheld takes. `moves` keeps only the clean moves of drone takes
  ([lessons 4.2–4.4](lessons.md#42-camera-shake-as-a-stabiliser-residual)).
- **The editor confirms every label.** The agent builds shortlists per mood
  from candidates plus annotations; you accept, move or reject.
- **Organise in Resolve:**
  ```bash
  uv run python -m videoeasy.deliverables bins --project data/projects/<project> --config config.<project>.yaml \
      --resolve-project "<Resolve project>" --source-bin "Master/<shoot>/Broll"          # dry run: what would move where
  uv run python -m videoeasy.deliverables bins ... --apply                               # after you've read the dry run
  ```
  `MoveClips`' return value is unreliable, so the result is verified by
  re-reading the bins ([resolve-scripting.md](resolve-scripting.md#media-pool)).
- **Grade** through the same per-source LUT profiles as ingest (config
  `grade:`, `film.yaml` `layin.grades`).

**Planned, not built:** a pass in which the local vision model picks a mood
for each clip, constrained to its session's candidates, with "none of these"
allowed and an honest `unchecked` when it can't tell. It would be calibrated
against the editor's labels the way
[lessons 6.12](lessons.md#612-compare-prompt-variants-the-way-you-would-compare-treatments)
describes, before any of its labels are used.

### Session recordings

Full recordings from a static camera are mostly outside the tool: they are
edited by hand. Transcribing them still pays off. The transcript makes a
90-minute keynote searchable for a quote a feature needs, and its pauses
are where clean chapter points are.

### Delivery and tracking

- **Masters:** `videoeasy.deliver` normalises a render to −16 LUFS
  integrated and −1 dBTP with two-pass linear loudnorm, then **measures the
  result** and writes a manifest saying honoured or broken. The picture is
  stream-copied.
  ```bash
  uv run python -m videoeasy.deliver --film data/projects/<project>/items/<item-id> --render editorial/renders/<r>.mp4
  ```
- **Status:** each item moves through
  `planned → footage → selects → cut → review → graded → mixed → delivered`.
  Record a step with `set`:
  ```bash
  uv run python -m videoeasy.deliverables set <item-id> status=selects --project data/projects/<project>
  uv run python -m videoeasy.deliverables status --project data/projects/<project> [--scope editing] [--md progress.md]
  ```
  `status` counts items per type against the shot list's totals. It also
  checks the recorded status against what is on disk, and reports an item
  as **claimed, no evidence** when its record is ahead of its files (say,
  `delivered` with no measured master). `--md` writes a checklist you can
  share with a producer.

## Consent and labels

- Mood labels describe **what a clip displays** ("laughter at the table",
  "a still, downcast moment"), never a person's inner state. "He is
  grieving" is not something footage establishes. Write labels about the
  picture.
- People's names, the shot list, the client and every transcript stay under
  `data/`. They never go into the repository, a prompt example or a test
  fixture.
- Only local models see pixels or hear audio. A hosted frontier model, if
  you use one, sees text: transcripts and annotations.
