# Cut evaluation: the local-model loop

How videoeasy judges a rendered cut and rebuilds its pictures without the
frontier model ever looking at a frame. This is the design and its known
weaknesses. The reasons behind most rules are in [lessons.md](lessons.md),
and the Resolve side is in [resolve-scripting.md](resolve-scripting.md).

## 1. The rule this exists to enforce

The frontier model never judges a render, a grade or a cut by looking at
frames. It writes prompts, sends graded frames to the local vision model and
transcripts to the local text model, reads the scores, and fixes the
generator. Judging by eye produced a first draft the editor called "awful".
The loop replaces taste-at-a-glance with three kinds of evidence:

- **measurements** (ffmpeg, OpenCV, waveforms);
- **local-model judgments** (scored, with their variance not yet measured);
- **explicit `unchecked` results** wherever a check failed or did not run.

Some faults are categorical or unresolved, and are not numbers. The loop
changes **pictures only**. Story and radio-cut changes are a separate, manual
operation, and their results must pass the same evidence, technical and
intent checks.

Models (all local, on one Mac, never run concurrently):

| Job | Model | Endpoint |
|---|---|---|
| Vision judge and verifier | `google/gemma-4-26b-a4b`, 16k context | LM Studio `:1234/v1` |
| Text judge, proposer, story model | 27B Qwen text model, `think: false`, `format: json` | Ollama `/api/chat` |
| Transcription of the render | `whisper-large-v3-mlx`, chunked per beat | in-process |
| Everything measurable | ffmpeg / ffprobe | `signalstats`, `blackdetect`, `silencedetect`, `ebur128`, `loudnorm` |

## 2. One iteration

```
cut-vN.json + render-vN.mp4
   │
   ├─ cuteval ───────────► eval/cut-vN-core-auto.{json,md}   (scorecard)
   ├─ storycheck ────────► eval/cut-vN-story.{json,md}       (words + story)
   │
   ▼
brollmatch --eval <scorecard>  (propose, text model)  ──► broll-plan-vN+1.json
brollmatch --verify-only       (verify, vision model) ──► broll-plan-vN+1.json (+ verdicts)
   │
   ▼
cutbuild ────────────────────────────────────────────► cut-vN+1.{json,md}  (audio unchanged, video rows rebuilt)
   │
   ▼
layin (Resolve API): plan ► lay ► readback ► conform ► render ► back to the top
```

Before proposing on a film for the first time, run the measurement passes
once: `steadiness` (shake per catalogue unit, steady runs), `moves` (drone
moves) and `transcribe --broll` (what each B-roll unit's own audio says).

### 2.1 Shared layout rule

`cuteval.layout()` reproduces the lay-in, so every scorer knows what is on
screen at every second without reading the timeline back. For each beat, the
tier-1 audio picks are butted in order, followed by `gap_after_s` (0.5 s) of
room tone. Each video row carries its destination interval and pick (`pick`,
`dest_in_s`, `dest_out_s`); a BARE row is the speaker's own picture.
`pick_spans()` gives each pick a render-time span with `id=b{beat}p{n}` and a
stable `key=clip@in_s` that carries feedback between versions. The lay-in
implements the same rule, plus crossfade extensions that stay inside the
audio pick's span, so the render and the model agree within ±2 frames.

### 2.2 `cuteval`: score a render

Inputs: the cut, the render and the ingest transcripts. Outputs: per-stretch
JSON, a markdown report and a manifest.

- **Stretches.** The union of picture-row and spoken-pick boundaries, so
  every stretch sits under exactly one pick. Each has a `kind` (video, bare,
  placeholder), its words, and a `to_camera` flag when the bare picture is a
  presenter piece. Recording roles come from the film profile
  (`film.yaml` `roles`), never from a filename guess in the scorer.
- **Picture vs words (vision model).** One or two JPEGs grabbed from the render
  inside the stretch, so the judge sees the graded, rendered picture, go to
  the vision model with the words spoken over them. It returns `on_screen`; a
  `relation` in {illustrates, supports, generic, mismatch, distracting}; a
  `score` 0–3 mapped from the relation; a `technical` list (underexposed,
  overexposed, soft focus, motion blur, shaky, colour cast, dead frame, blank
  card, none); and `would_rather_see`. Bare stretches get a rule: 3 when the
  words are personal and the face carries them, 1 when they name something
  showable. Presenter stretches: greetings, hand-overs and sign-offs are the 3
  case. Placeholder cards are fixed at 1 and asked what real picture they
  promise. Protected stretches (from `intent.json`) are scored too, with the
  judge told they stay uncovered. The rule finding is what counts there. The
  headline is the duration-weighted mean, with and without placeholders.
- **Exposure (ffmpeg).** `signalstats` on the extracted JPEG: YAVG under 45
  or over 185, clipped highlights, crushed blacks, low saturation.
- **Rhythm (deterministic).** Bare % of spoken time, longest bare run per
  beat, mean hold, jump cuts.
- **Joins (ffmpeg).** `blackdetect d=0.04:pix_th=0.10` and
  `silencedetect n=-50dB:d=0.15`. Any black is a lay-in fault.
- **Cover rules (measured).** Talking cover from the covered source's
  transcript, shake from the render's own frames, and the drone move each
  drone stretch shows. Listed whatever the picture score says (2.9, 2.11).
- **Levels.** Integrated loudness and true peak per pick (ebur128). Mono picks
  are checked left against right.
- **Viewer read (text model).** The spoken text by beat with no other
  context, under a first-time-viewer prompt: logline, who is who,
  clarity/pull/drag per beat, where attention drifted, what is missing,
  reorder, what to cut first, does it end.

### 2.3 `brollmatch`: propose and verify cover

- **Catalogue.** Every B-roll shot or window from the bible, with its tags,
  `people`, `heard` (the unit's own audio: words, seconds, or silent / no
  track / not checked), shake and steady runs, clean drone moves, the
  synthesis sections that cite the source, and 400 characters of cut notes.
  Units whose role is `never_cover` (the sit-down, presenter pieces) are
  excluded, so a pick cannot be covered with its own A-roll. Talking units and
  units with no steady part are hidden. The heard words are printed as
  "said, not necessarily visible".
- **Propose (text model, one call per beat).** The prompt states the film's
  register (`film.yaml` `context`), the beat note, available placeholder
  cards, the picks with ids and durations, a review block, and the catalogue
  with ALREADY USED marks. For each pick it returns `need` in {illustrate,
  support, bare} and up to four candidates, best first. Hallucinated ids are
  dropped.
- **Feedback from the last scorecard (`--eval`).** A stretch that scored 0–1
  is listed in the REVIEW block with what the judge would rather see, and that
  shot is barred for that pick. A pick whose single stretch scored 3 keeps its
  shot. A shot scored 0 with a technical flag in {motion blur, shaky, soft
  focus, dead frame} is barred film-wide.
- **Verify (vision model, one call per candidate × pick).** Up to four graded
  frames of the candidate, from inside its steady runs or clean moves, sent
  with the words. The verdict records which files and source times were sent,
  and returns `shows`, `score` 0–3, `usable` frame numbers and `reason`.
- **Choose (deterministic).** Rank candidates with an accepted verdict by
  score minus 0.75 per use beyond the first; placeholders count 2.5; barred
  candidates are dropped; a best score under 1 falls to bare. A missing,
  errored or invalid verdict is not eligible. When no candidate is eligible,
  the row is `unchecked`, with no shot and no score.
- **Editorial rules that override the models.** A beat whose title or note
  says "no cover" is bare whatever the proposer says. A sync-walk beat never
  takes another walk take (role `sync_walk`) as cover. A pick with no
  candidates and a covered neighbour inherits the neighbour's shot.

### 2.4 `cutbuild`: rebuild the video rows

Audio is copied unchanged. **Invariant: the rows emitted for a pick sum to
the pick's duration.**

- The chosen shot is held for the pick's length, starting at the recorded
  source time of the first frame the verifier accepted, and never earlier.
- A shot reused later continues from where its last use ended. When no room
  is left, it repeats certified footage and the row's note says so.
- Long picks split across the chosen shot and verified alternatives scoring
  2 or more. A short piece hands its deficit to the next. A short last piece
  lets earlier pieces run on inside their steady part or clean move before
  the speaker fills the rest.
- Every row and merge is clamped into one steady run or clean move
  (`clamp_to_runs`); the rest of the pick is bare.
- Adjacent picks on the same shot merge only when the second pick's verdict
  for that shot is accepted. Only an accepted verdict can certify cover, and
  one whose reason complains of blur with fewer than two usable frames
  cannot.
- An `unchecked` plan row becomes a bare row marked provisional, and the cut
  says so at the top.
- Placeholder cards the proposer dropped are re-attached to the pick whose
  words best match the card (keyword order from `film.yaml`
  `layin.placeholders`).
- A hand-set plan row may carry `segments` (`[{until_s, shot or none}]`,
  seconds from the pick's start, set from word times): the picture follows
  them instead of the even split, with none meaning the speaker. A segment's
  shot passes the same gates as a chosen shot, or that segment goes to the
  speaker with the reason.
- `check_cut` refuses a cut whose rows are not contiguous per beat or do not
  end where the words end, and `intent.check_cut` refuses one that breaks a
  rule.

### 2.5 Lay-in, read-back and conform (`layin.py`)

`plan` places every row at its destination (a cut without `dest_in_s` is
refused). `lay` writes and verifies each write. `readback` reads every item
on every track. `conform` matches planned rows to items by name, track and
record frame within two frames, and compares duration, source in-point
(converted to timeline frames), fades, gain and channel map. A-roll picks
alternate tracks with 12-frame crossfades and handles capped at neighbouring
words, beat gaps take the outgoing take's own tail, cover goes on V3, gains
are measured per pick, and LUTs, CDL, exposure trims and markers are set per
item. Details and the API's traps are in
[resolve-scripting.md](resolve-scripting.md).

### 2.6 `storycheck`: did the words play, and does the story hold

1. **What was actually said.** The render is transcribed beat by beat
   (cached under the render hash, beat spans and model). Each pick's intended
   words are aligned against the words heard inside its span, as complete
   token-edit operations (`speech.py`): omissions, substitutions, insertions
   inside the matched run, and leaks at either boundary that are not a
   neighbouring pick's words. Spelling variants go through the film's alias
   table, with a fuzzy fallback that never touches a token under four letters
   or any critical token. Any edit on a polarity, quantity, modality, name or
   pronoun token flags the pick. So do coverage under 0.8 and two or more
   leaks.
2. **Padded recheck.** Every flagged pick is transcribed again from a 2 s
   padded excerpt of the render and from the source at the pick's in and out:
   - `cleared_on_recheck`: the padded pass hears the pick as intended (an ASR
     miss);
   - `confirmed`: both render passes report the same edits, the source
     carries the words, *and* the render's waveform stops matching the
     source's (`speech.waveform_match`: correlation per quarter-second speech
     window at a locally searched lag; under 0.6 is lost);
   - `waveform_intact`: the same ASR disagreement, but all of the source's
     speech is in the render;
   - `intent_mismatch`: the source itself says something else (fix the cut
     text);
   - `unresolved`: anything else, so listen.

   No state moves a trim or drops a pick. Both attempts are kept in
   `<render>.recheck.json`.
3. **Recorded intent, resolved.** `intent.py` checks the active rules on the
   heard transcript (forbidden words, framing rules) and on the cut (2.7).
4. **Story against the spine (text model).** The film as it plays, each pick
   labelled by recording role, judged against the synthesis's spine plus the
   additions later decisions require, with one resolved brief from
   `intent.py`. The model walks the spine's numbered beats, marking each
   present/partial/absent/out_of_order with the cut beats that carry it. It
   marks each rule honoured/broken/not_testable_from_words, and lists
   repetition, orphans, joins where the words do not lead on, voice seconds by
   role group (`film.yaml` `story.voices`), any film-specific questions
   (`story.questions`), and its top fixes. Seconds by recording role are
   computed deterministically, not by the model.

### 2.7 `intent.py`: decisions checked as rules, not scores

`editorial/intent.json` is edited by hand and holds every decision the cut
must honour, each with an origin, who decided, its source, a date and what
it supersedes. `resolve()` retires superseded rules and keeps the latest per
topic. The rule kinds and their fields are in the
[user guide](user-guide.md#5-editorial-intent-intentjson).

`check_cut` judges rows by their destination intervals: rows without them are
`not_testable`, never `honoured`. `check_readback` judges what Resolve holds.
The same file drives the builder (protected picks held bare), the proposer
(protected picks marked bare) and the scorer (the judge is told; rules are
reported separately from scores). `cutbuild` refuses to write a cut with a
broken rule, and `layin` exits 2 when the read-back breaks one.

### 2.8 Dialogue level and delivery

`cuteval` measures integrated loudness and true peak for every pick and
reports picks more than 3 LU under the −24 LUFS working target or over −1
dBTP. `layin`'s plan lists picks whose source needs more than the +30 dB
clip-gain cap and those held under target by the −1.5 dBTP ceiling. The
remedy (the Dialogue Leveler in "optimize moderate levels" mode with an
output gain equal to the shortfall, up to 6 dB) is written, read back and
conformed. The residual spread is an editorial decision recorded in
`intent.json`. `deliver` masters to the `delivery_loudness` rule with
two-pass linear `loudnorm`, video stream-copied, then measures the master
and writes `honoured` or `broken`.

### 2.9 Cover rules measured without a model

- **Talking cover** (`speech.talking_cover`): a cover row whose source range
  holds at least 1 s of transcribed words. An untranscribed unit is unknown,
  not clean. Off-camera speech (a tag saying nobody is visible) and clips
  named in `talking_cover_exempt` are excepted.
- **Camera shake** (`steadiness.py`): the stabilisation residual as a
  percentage of frame width, per unit and in 2 s windows every 0.5 s. A unit
  with any over-limit window (provisional `MAX_SHAKE_PCT` 2.0) is usable only
  inside its steady runs (3 s or more that no over-limit window touches), and
  is not cover at all when it has none. The speaker's own sync picture is
  exempt.

All three tools (proposer, builder, scorer) act on both rules. A shot that
slips past the catalogue is still caught on the render.

### 2.10 B-roll audio and what the proposer is told

B-roll audio is transcribed with confidence filters and explicit silent /
no-track states, so no unit is "not checked". The catalogue says what was
heard separately from what was seen. Removing shaky and talking units cut one
film's visible catalogue from 193 units to 90, which brought the proposer
prompt from about 26k tokens to about 14k of a 40k context.

### 2.11 Drone moves, measured

`moves.py` segments each drone take into moves (pan, slide, orbit, tilt,
rise, descend, push-in, pull-out, rotate, hover) and calls a move clean when
it keeps one direction (mean cosine ≥ 0.90), never dips mid-move by more than
25 % of its top speed, stays under the shake limit, and lasts 3 s or more.
Clean moves become the unit's usable parts. The proposer sees each move by
name, and `cuteval` lists any drone stretch that is not at least 90 % one
clean move.

## 3. Thresholds that trip a fix

| Signal | Threshold | Action |
|---|---|---|
| Picture vs words | any stretch at 0 | new cover (barred for that pick next turn) |
| Picture vs words | beat mean < 1.5 | re-propose the beat |
| Technical flag with score 0 | blur, shake, soft focus, dead frame | shot barred film-wide |
| Exposure | YAVG < 45 or > 185, clipped, crushed, low saturation | grade fix on that source |
| Joins | any black; silence > 0.15 s | lay-in fix |
| Bare % | tracked, not thresholded | editorial call |
| Word alignment | coverage < 0.8, ≥ 2 leaks, or any critical-token edit | padded recheck, then waveform; act by state |
| Forbidden word heard | any (from `intent.json`) | drop or re-pick |
| Intent rule broken | any (`check_cut` / `check_readback`) | build refused; fix the plan or record a new decision |
| Dialogue level | pick > 3 LU under −24 LUFS, or > −1 dBTP | listed; needs an agreed target and a measured fix |
| Talking cover | ≥ 1 s of the covered source's speech inside a cover row | not cover (hidden, dropped, listed); named exemptions only |
| Camera shake | residual > 2.0 % of width (provisional), per unit and per 2 s window | usable only inside steady runs; none means not cover; sync picture exempt |
| Drone move | stretch < 90 % one clean move | listed; builder keeps drone rows inside clean moves |
| One-channel pick | mono pick L/R differ by > 3 LU on the render | render `broken`; reopen the timeline and re-render |

## 4. What successive iterations showed

Over about ten iterations on one 18-minute film, the mean picture score
stayed between 1.75 and 2.15 out of 3, while the things that mattered most
were fixed by rules, not by the score. Talking cover went from 27 rows
(269 s) to 0 in one turn once it was measured. Shaky cover went to 0 once
units were restricted to their steady parts. Black frames went to 0 once
joins were planned in placed frames. Unflagged picks rose from 55 of 70 to
61 of 81 as the speech checks grew stricter and rechecks cleared ASR misses.
Two scorecards from different methods, or from before and after a re-lay,
were never comparable, and the early rows had no manifests. That is why
every report now carries one. Read a mean-score delta of 0.05 as noise until
run-to-run variance has been measured.

## 5. Known weaknesses, and the roadmap

The eval is the product. Read this list as the backlog, in this order:

1. **Honest failure, frozen evidence.** *Done*: status-bearing results, no
   fallback scores, content-hash cache keys, a manifest per report,
   overwrite refusal (`evalrun.py`, `tests/test_eval_status.py`).
2. **Timing conservation and the actual edit.** *Done*: destination
   intervals on every row, deficits filled in place, verdicts checked before
   merging, placeholder ids by content hash, scoring split at pick
   boundaries, lay-in with read-back conform.
3. **Speech and intent.** *Done*: token edits with critical-token escalation,
   padded recheck with waveform confirmation, resolved intent enforced in
   the builder, proposer, scorer and read-back, and per-pick level
   diagnostics.
4. **Calibration.** *Open*: an editor-rated stratified sample with audio,
   held out from rubric development; repeated identical requests for
   variance; blind pairwise comparison on matched units.

Remaining weaknesses:

**Judges and calibration**
- No ground truth. The 0–3 scale is self-consistent, but its absolute level
  means nothing, and deltas need variance and comparability first.
- One frame, sometimes two, per stretch. Lips and shake are now measured, but
  motion blur or a pan onto the subject can be missed or over-punished
  depending on where the grab lands.
- Verify frames are ingest frames, not render frames. Only the next `cuteval`
  sees the sub-range actually used.
- The judge scores protected stretches low even when told.
- Placeholder cards can exhaust the token budget and stay unchecked.
- Temperature 0.2 is not zero, and variance has not been measured.
- The proposer is text-only over vision tags, so a wrong tag propagates to a
  wrong proposal.
- The viewer read cannot tell who is speaking unless the words say so.
- The story model has needed patches. Each is a prompt note, not a test.
- The rubric favours literal illustration over purposeful contrast, motif or
  breathing room. Visible relation, editorial purpose and intent compliance
  should be separate judgments.

**Deterministic parts**
- Jump-cut detection is a heuristic on source names and bare stretches, and
  it also flags joins where a bare stretch merely begins or ends.
- Repeated footage is not measured.
- The alignment thresholds (0.8 coverage, 2 leaks, ±0.2 s window) and the
  critical-token lists were tuned on one render. A change of actor or action
  expressed by swapping a verb is not caught.
- The reuse penalty 0.75, unverified weight 1.5 and placeholder weight 2.5
  are guesses that nothing tests.
- A pick whose every candidate is barred falls to bare instead of being
  re-proposed.
- No automatic check confirms that the layout model and the rendered timeline
  agree beyond the joins measurement and the ±2 frame conform.

**Process**
- Unit tests cover the invariants (`test_eval_status`, `test_placement`,
  `test_speech`, `test_intent`, `test_deliver`, `test_steadiness`,
  `test_moves`). The model judgments are checked only by running the loop and
  reading the reports.
- No held-out beat: the scorecard that drives the proposer is the one that
  reports improvement. The story check is the only independent signal, and it
  measures words, not pictures.

## 6. Where things live

- Code: `src/videoeasy/{cuteval,brollmatch,cutbuild,storycheck,layin,speech,intent,steadiness,moves,deliver,sources,evalrun}.py`
- Film profile: `data/films/<film>/film.yaml` (see `film.example.yaml`)
- Decisions: `data/films/<film>/editorial/intent.json`
- Artifacts: `data/films/<film>/editorial/{cut-vN.json, broll-plan-vN.json, eval/, renders/, lay/}`
- Operating procedure: `.agents/skills/videoeasy-story-room/SKILL.md`
  ("Judging a cut") and `.agents/skills/videoeasy-resolve/SKILL.md`
