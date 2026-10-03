---
name: videoeasy-deliverables
description: Set up or run an event / shot-list deliverables project with videoeasy — turning a shot list into data/projects/<project>/deliverables.yaml, checking camera clocks, assigning clips to sessions, tracking many short deliverables (testimonials, before/after interviews, speaker features, a B-roll package by mood, session recordings) in items.yaml, proposing testimonial bites from transcripts, organising Resolve bins, and reporting status. Use when an editor has a shot list and a deliverables list rather than one film. Not for single-film story work (videoeasy-story-room) or ingest debugging (videoeasy-ingest).
---

# videoeasy deliverables

You are helping an editor deliver many short videos from one event shoot.
The editor owns every editorial decision. You set things up, run the checks,
read the output with them, and propose; they confirm. The guide is
`docs/event-projects.md`; machine setup is `docs/setup.md`.

## Before anything

1. Read `data/projects/<project>/deliverables.yaml` and `items.yaml` if they
   exist. They are the state of the project. Don't work from memory of an
   earlier session.
2. **Never commit anything under `data/`**, any `config.<project>.yaml`,
   `film.yaml`, `.mcp.json`, media or LUTs. They are git-ignored; keep it
   that way. People's names, the shot list, the client and transcripts never
   go into the repository, a code comment, a prompt example or a test
   fixture.
3. If the machine is new, start with `uv run videoeasy doctor` and walk
   through `docs/setup.md` with the editor, one "check:" line at a time.

## Building `deliverables.yaml` from a shot list

Work through the shot list *with* the editor; don't infer what you can ask.
Ask for:

- the **dates** of each day and the **timezone**;
- every **operator** and their **cameras**: brand, filename prefix, and the
  folder their cards were copied into. Two bodies of the same brand write the
  same prefix, so recommend one folder per operator and match on
  `path_contains`;
- each camera's **clock** source (`creation_time_utc`,
  `creation_time_local` or `timecode`); drones usually record UTC;
- the **deliverable totals** and whose **scope** each type is in.

Then map the shot list line by line:

- one `sessions[]` entry per scheduled block (`id`, `day`, `start`, `end`,
  `title`);
- **one `capture` entry per operator per line**:
  `{operator, text, feeds, registers}`. Shot lists routinely give several
  operators different jobs in the same session; `assign` uses the operator
  to give a clip only its own operator's feeds and moods;
- `cameras:` on sessions that run in parallel. Overlap is allowed only with
  disjoint camera sets;
- the moods the shot list names go in `registers` (top level, and per
  session or capture entry);
- one `deliverable_types[]` entry per line of the deliverables list (`count`,
  `duration_s`, `template`, `scope`, `includes`).

Run `validate` after every edit and fix what it reports before moving on.

## Order of operations

```bash
uv run videoeasy doctor
uv run videoeasy check --config config.<project>.yaml
uv run videoeasy probe --config config.<project>.yaml
uv run python -m videoeasy.deliverables clocks --config config.<project>.yaml --project data/projects/<project>
uv run python -m videoeasy.deliverables assign --config config.<project>.yaml --project data/projects/<project>
uv run python -m videoeasy.deliverables plan   --project data/projects/<project>
uv run python -m videoeasy.deliverables status --project data/projects/<project>
```

- **Clocks before assign.** If a camera's clips sit outside every session,
  measure its offset. The best evidence is a sound two cameras both heard,
  matched by waveform (`audiosync`). Never set an offset by guessing from
  the schedule.
- **Read `assign`'s nulls with the editor.** Outside every session, `ambiguous:
  <ids>`, no camera matched and no clock are each a different fix
  (`docs/event-projects.md`, "Ingest, then assign").
- **Calibrate before ingesting the whole shoot.** Run 2–3 clips through
  `annotate --only` and review them with the editor before `videoeasy all`,
  which is an overnight job you never launch to test something.

## Testimonials and before/after interviews

- Propose bites from `out/transcripts.json` with **word-level in and out
  times**, the clip, and why the bite stands alone. Prefer bites that start
  at the start of a thought and don't need the interviewer's question to
  make sense.
- Present a shortlist per interview. The editor rates and chooses, and you
  record their choice. Never mark a bite chosen on your own judgment.
- Cut points come from measured room (the lay-in's per-channel energy,
  `max_head_s`/`max_tail_s`), not from the gap in the stored transcript.
- A bite whose meaning hangs on one word (a name, a number, a "not") gets a
  padded second transcription before you rely on it (`speech.py` recheck).
- **Before/after pairs** come from names spoken on camera, names the
  operators logged, and session times. Propose pairs with that evidence
  quoted. The editor confirms every pair. Never identify people by face.

## Long talks on lav recorders

When a speaker wore a lav recorder (32-bit float WAV chunks, e.g. DJI Mic),
analyse the talk with `videoeasy.talks` (read `docs/talks.md` first):

```bash
uv run python -m videoeasy.talks all --project data/projects/<project> --mic-dir <dir> --session <first chunk> --talk-id <id> --speaker "<name>" [--broll-context <file>]
```

- Run `--skip-model` first on a new recorder model and read the prepare and
  sentences report: chunk grouping, peak and overs, gain, noise floor, how
  many sentences are flagged off-mic. If the filename pattern differs from
  `DJI_<nn>_<date>_<time>.WAV`, pass `--name-pattern`.
- Present the ranked hooks, bites and closers from `analysis.md` with their
  talk times, source chunk times and cover briefs. The editor rates them;
  record keep/reject in the item's notes or `editorial/` and never re-rank
  by your own reading.
- Unverified candidates are listed, not ranked: do not promote them.
- Off-mic sentences are probably another voice; a bite needing one is the
  editor's call (`--include-off-mic`).
- Thresholds are provisional; the editor's ratings are the calibration
  (lessons 6.12). Say so when presenting numbers.
- The Resolve selects timeline is a dry run until the editor confirms;
  audio frame units for WAV items are not yet verified live, so check the
  read-back report.

## Recording decisions

Record every decision in `items.yaml` through `set`, not by hand-editing in
a way that loses history:

```bash
uv run python -m videoeasy.deliverables set <item-id> status=selects people=<id> --project data/projects/<project>
```

Status moves `planned → footage → selects → cut → review → graded → mixed →
delivered`. Set a status only when the evidence for it exists. `status`
flags **claimed, no evidence** when a record is ahead of its files. Treat
that as a finding to resolve with the editor, not something to paper over.

Speaker features run the documentary loop per item (`videoeasy-story-room`
skill). Their editorial decisions go in that item's
`editorial/intent.json`.

## Pictures and moods

- **Local models validate pictures.** Don't judge frames by looking at them
  yourself. To judge a clip, a cut or a grade, send graded frames to the
  local vision model with a written prompt and read its scored answer. Open
  a contact sheet only when one specific shot matters to an argument.
- Mood shortlists combine the session's `register_candidates` (what was
  *meant* to be filmed then) with the annotation (what the frames show). Say
  which is which. The editor confirms each label.
- Labels describe what a clip displays, never a person's inner state.
- Use only the steady parts of handheld takes (`steadiness`) and the clean
  moves of drone takes (`moves`).

## Resolve

- Every write needs the editor's OK, after you've said what will change and
  how many clips it touches.
- Dry run first (`bins` without `--apply`; `layin --plan-only`), then apply,
  then **verify by re-reading**. Write return values lie: `MoveClips`
  reports success unreliably, and with a modal dialog open every write
  silently does nothing (`GetCurrentPage()` is `None`). Ask the editor to
  close the dialog.
- Guard the project name on every mutating call. One Resolve project may
  hold other work.
- Details: `videoeasy-resolve` skill and `docs/resolve-scripting.md`.

## Delivery

`videoeasy.deliver` writes a −16 LUFS / −1 dBTP master and measures it. Mark
an item `delivered` only when that manifest says honoured and the editor has
signed off. `status --md` writes a checklist the editor can share.
