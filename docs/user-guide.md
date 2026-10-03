# User guide

How to take a film from raw footage to a story-room conversation, a measured
cut and a Resolve timeline. For an event shoot with a shot list and many
short deliverables, read this first and then
[event-projects.md](event-projects.md).

## 1. Requirements

Setting up a new machine, with memory and disk guidance and a check after
every step, is covered in [setup.md](setup.md). Once it's set up,
`uv run videoeasy doctor` re-checks the machine without any config. In
short:

- macOS on Apple Silicon (mlx-whisper and the MLX model builds are
  Apple-only).
- [uv](https://docs.astral.sh/uv/), Python 3.11+, `ffmpeg` and `ffprobe` on
  the path.
- [LM Studio](https://lmstudio.ai/) for the vision model, and
  [Ollama](https://ollama.com/) for the text judge and proposer.
- DaVinci Resolve **Studio** with external scripting enabled, for the lay-in
  (Preferences → System → General → External scripting using: Local).
- LUTs for your cameras' log profiles. The repository ships none.

```bash
uv sync                       # core
uv sync --extra birds         # optional: BirdNET
uv sync --extra species       # optional: BioCLIP
```

## 2. Lay out a film

Everything about a film lives under `data/films/<film>/`, which git ignores:

```
data/films/<film>/
  film.yaml                     the film profile (section 3)
  inputs/aroll  inputs/broll    symlinks to the footage (originals are never written)
  editorial/
    transcript-selects.docx     the marked-up transcript (Google Docs → File → Download → .docx)
    intent.json                 recorded decisions (section 5)
    cut-vN.json  broll-plan-vN.json  eval/  renders/  lay/  placeholders/
  out/                          pipeline artifacts + bible-context.md + bible-synthesis.md
  work/                         frames, contact sheets, caches (regenerable)
```

Then a per-film ingest config at the repo root, copied from
[`config.example.yaml`](../config.example.yaml), as `config.<film>.yaml`
(also git-ignored, because it holds machine paths):

- `paths:` points at `inputs/`, the selects docx, `work/` and `out/`.
- `grade:` lists profiles and first-match filename rules. A profile is a
  `lut:` (a `.cube` file) or raw ffmpeg `filters:`. Every frame a vision
  model sees is graded first.
- `models:` sets the vision model and URL and the Whisper model.
- `shots`, `frames`, `motion` and `segments` are tuning blocks. Locked-off
  long takes go in `segments.index_only_sources`.
- `location:` (lat, lon, shoot date) for BirdNET.
- `film_dir:` is optional. By default the profile is found at the parent of
  `out_dir` when a `film.yaml` sits there.

Keep drive mount names stable: absolute source paths are baked into the
artifacts and the Resolve media pool.

## 3. The film profile (`film.yaml`)

Everything specific to one film that is not footage: what the film is,
which recording context each clip comes from, the spellings ASR gets wrong,
the story model's film-specific questions, and the Resolve constants. A
missing file or key falls back to generic defaults, so every tool runs on a
new film before you write a profile. The authoritative reference is
[`film.example.yaml`](../film.example.yaml) and the docstring of
`src/videoeasy/film.py`. In outline:

```yaml
name: "My film"
context: >-
  One paragraph: what the film is and its register (pace, warmth, what the
  picture should do). Goes into the annotation and proposer prompts.
  Never put example metaphor or theme phrases here: the model parrots them.
roles:                         # recording contexts
  sitdown:   {label: "sit-down interview", never_cover: true}
  presenter: {label: "presenter to camera", to_camera: true, never_cover: true}
  walk:      {label: "walk-and-talk, sync sound", sync_walk: true}
  drone:     {label: "drone", drone: true}
default_role: walk             # anything not listed below
clips: {C0001: sitdown}        # exact base clip name -> role (overrides prefixes)
prefixes: [[DJI_, drone]]      # first match wins
aliases: {intended: [asr_variant]}   # names, species, jargon the transcriber mishears
names: [intended]              # tokens whose edit changes who is meant (a semantic risk)
story:
  voices: {portrait: "who carries the film", method: "who explains the method"}
  questions: {chapter_shape: "does chapter one end where the spine says?"}
layin:
  project: "Resolve project name"      # guarded: writes refuse any other project
  bin: [Films, "My film"]
  clip_bins: [Aroll, Broll, Drone, AI Drafts]
  drafts_bin: AI Drafts
  grades:                              # per timeline item, first filename-prefix match wins
    - {name: log, match: C, lut: /path/to/log_to_rec709.cube}
    - {name: drone, match: DJI_, cdl: {NodeIndex: 1, Slope: "1 1 1", Offset: "0 0 0", Power: "1 1 1", Saturation: 1.0}}
  placeholders: {keyword: card_file.mov}   # checked in the listed order
  render_dir: editorial/renders
  grade_trims: editorial/grade-trims.json
  render_preset: "YouTube - 1080p"
  aroll_dir: inputs/aroll
```

What the role flags do:

- `to_camera`: the uncovered picture is a person addressing the audience,
  which changes the bare-picture rubric (greetings, hand-overs and sign-offs
  score 3).
- `never_cover`: no unit from this role is ever proposed as cover.
- `sync_walk`: a beat of this sync sound is never covered by another take of
  the same kind.
- `drone`: measured by `moves.py`; usable only inside clean moves.

The role table records **where the sound was recorded**. It is not a list of
who is in the shot. Keep person names out of filename rules.

## 4. Verify, calibrate, then commit the night

```bash
lms load google/gemma-4-26b-a4b --context-length 16384 && lms server start
uv run videoeasy check --config config.<film>.yaml
```

`check` verifies ffmpeg, the LUTs, Whisper, the loaded context window
(at least 16384 tokens) and a **vision trap**: a synthetic image whose colours
the model must name exactly. `SUSPECT` means images are not reaching the
model. Stop there. Run `check` after any model or backend change.

Calibrate before the overnight run:

```bash
uv run videoeasy probe      --config config.<film>.yaml
uv run videoeasy shots      --config config.<film>.yaml   # sanity-check the shot count
uv run videoeasy transcribe --config config.<film>.yaml
uv run videoeasy segment    --config config.<film>.yaml   # windows long takes
uv run videoeasy frames     --config config.<film>.yaml --only <id>
uv run videoeasy annotate   --config config.<film>.yaml --only <id>
```

Read the record in `out/annotations.json` next to its contact sheet
`work/sheets/<id>.jpg`. Do 2–3 shots and review their depth before
committing. Then:

```bash
uv run videoeasy all --config config.<film>.yaml          # overnight
```

Stages are idempotent and resumable. `--force` on `annotate` or `transcribe`
redoes from scratch; use it after a prompt or model change.

Optional passes:

```bash
uv run videoeasy transcribe --config config.<film>.yaml --broll   # B-roll audio, confidence-filtered
uv run python -m videoeasy.steadiness --config config.<film>.yaml # camera shake, steady runs
uv run python -m videoeasy.moves --config config.<film>.yaml      # drone moves
uv run --extra birds python -m videoeasy.birds --config config.<film>.yaml
uv run videoeasy audit --config config.<film>.yaml --only <id>    # frame evidence audit (evidence-audits.md)
uv run videoeasy audiosync cache|match --clip <file>|verify --clip <file>|report   # lav ↔ camera sync
```

## 5. Editorial intent (`intent.json`)

`data/films/<film>/editorial/intent.json` is the one place a decision lives.
You edit it by hand, and the builder, proposer, scorer, story check and
lay-in all read it.

```json
{
  "film": "my-film",
  "about": "what this file is for",
  "rules": [
    {
      "id": "no-cover-admission",
      "topic": "admission beat",
      "kind": "no_cover",
      "scope": {"beat_title_contains": "admission"},
      "decision": "The admission plays on the speaker's face, uncovered.",
      "origin": "editor",
      "by": "editor",
      "source": "review notes, cut-v3",
      "decided": "2026-09-20",
      "supersedes": []
    }
  ]
}
```

Common fields: `id`, `topic`, `kind`, `decision` (in words), `origin` (the
subject on camera, the editor, or a story-room proposal the editor accepted),
`by`, `source`, `decided` (ISO date), `supersedes` (ids it retires), and
optional `status: "superseded"` and `story` (text added to the story brief).
`intent.resolve()` retires anything superseded and keeps the latest
`decided` per `topic`.

| Kind | Extra fields | Checked how |
|---|---|---|
| `words_forbidden` | `words` | heard transcript of the render |
| `words_allowed_history` | `words`, `forbidden_phrases` | allowed only in the stated framing |
| `clip_span_excluded` | `clip`, `in_s`, `out_s` or `spans: [{clip, in_s, out_s}]` | no pick overlaps (meeting at an edge is fine) |
| `clips_excluded` | `clips` | none used |
| `must_be_present` | `clips` | each used |
| `no_cover` | `scope: {beat_title_contains}` | beat stays bare |
| `no_cover_pick` | `scope: {clip, text_contains}` | that pick stays bare |
| `cover_only_named_detail` | `scope: {beat_notes_contains}`, `never_from_prefix` | sync beat not covered by another take of its kind |
| `cover_required` | `scope: {beat_title_contains, pick}`, `clips_any` | one of the clips covers it |
| `placeholder_required` | `keys` (placeholder keywords) | card present (`review`: still missing material) |
| `talking_cover_exempt` | `clips` | named exception to the talking-cover rule |
| `delivery_loudness` | `target_lufs`, `max_true_peak_dbtp`, `tolerance_lu` | `deliver` measures the master |
| `open` | n/a | an unresolved call, reported as `open` |
| `note` | n/a | a recorded decision about method, `not_testable` |

Finding statuses: `honoured`, `broken`, `review` (allowed only conditionally,
so a person should look), `open`, `not_testable`. `cutbuild` refuses to write
a cut with a broken rule, and `layin` exits 2 when the Resolve read-back
breaks one. Grant exceptions by naming clips, never with a pattern.

## 6. The story room

Open Claude Code in the repo. The agent reads `out/bible-context.md`,
`out/bible-synthesis.md` (if present) and the film's private brief and
register reference under `data/films/<film>/`. See the
`videoeasy-story-room` skill. The first session builds the synthesis layer
(themes ↔ shots ↔ beats, motifs, coverage gaps). Argue with it: corrections
persist there. A rebuild is a new file beside the old one
(`bible-synthesis-v2.md`), never an overwrite.

## 7. The eval loop

One iteration, from the film directory:

```bash
F=data/films/<film>
uv run python -m videoeasy.cuteval   --cut $F/editorial/cut-vN.json --render $F/editorial/renders/<r>.mp4 \
    --transcripts $F/out/transcripts.json --out $F/editorial/eval/cut-vN-core-auto --frames-dir $F/work/eval/cut-vN-core
uv run python -m videoeasy.storycheck --film $F --cut editorial/cut-vN.json --render editorial/renders/<r>.mp4 \
    --out editorial/eval/cut-vN-story [--synthesis out/bible-synthesis-v2.md]
uv run python -m videoeasy.brollmatch --film $F --cut editorial/cut-vN.json --out editorial/broll-plan-vN+1.json \
    --eval editorial/eval/cut-vN-core-auto.json --skip-verify
uv run python -m videoeasy.brollmatch --film $F --cut editorial/cut-vN.json --out editorial/broll-plan-vN+1.json --verify-only
uv run python -m videoeasy.cutbuild  --film $F --cut editorial/cut-vN.json --plan editorial/broll-plan-vN+1.json --version N+1
uv run python -m videoeasy.intent    --film $F --cut editorial/cut-vN+1.json
uv run python -m videoeasy.layin     --film $F --cut editorial/cut-vN+1.json --name "<film> cut-vN+1 (AI)" --render <stem>
uv run python -m videoeasy.deliver   --film $F --render editorial/renders/<stem>.mp4
```

Every eval tool writes `<out>.manifest.json` and refuses to overwrite a
previous run without `--overwrite`. Do not run the proposer (Ollama) while
the scorer (LM Studio) runs. Design and thresholds:
[eval-design.md](eval-design.md).

## 8. What comes out

| Artifact | What it is |
|---|---|
| `out/bible-context.md` | the whole film in one document, regenerated by `assemble` |
| `out/bible-synthesis.md` | curated judgment; never written by a script |
| `out/annotations.json`, `segments.json`, `transcripts.json`, `selects.json` | per-unit records |
| `out/steadiness.json`, `moves.json`, `birds.json` | measurements |
| `work/sheets/*.jpg` | one contact sheet per shot or window |
| `editorial/eval/*` | scorecards, story checks and their manifests |
| `editorial/lay/*` | lay-in plan, read-back and conform |

## 9. Tests

```bash
uv run python -m unittest discover tests
```

The tests cover the invariants (status handling, placement and timing,
speech alignment, intent, delivery, steadiness, moves, segments, bible
assembly) with synthetic fixtures. They need no models, no footage and no
Resolve.

## 10. Troubleshooting

- **`check` fails on vision.** LM Studio is not serving or the model is
  unloaded: `lms ps`, reload, `lms server start`.
- **`SUSPECT`.** The backend is dropping images. Do not proceed. Stay on LM
  Studio.
- **Context TOO SMALL, or tags failing at the token budget.**
  `lms unload --all && lms load <model> --context-length 16384`, then re-run
  `annotate` (only error entries are retried).
- **Empty answers with `finish_reason=length`.** A thinking model spent its
  budget reasoning. Raise `max_tokens` in `vlm.py`, or disable reasoning.
- **Connection refused on localhost.** The server is down or a sandbox
  blocked the request. Check `lms ps` before suspecting the code.
- **Missing footage in the inventory.** The drive is unmounted, or the
  extension is not in `VIDEO_EXTS` (`config.py`).
- **Wrong shot boundaries.** Lower `shots.adaptive_threshold` (log footage is
  low-contrast), delete `out/shots.json`, re-run `shots` and `frames`.
- **Resolve writes do nothing.** A modal dialog or the Project Manager is
  open (`GetCurrentPage()` is `None`). See
  [resolve-scripting.md](resolve-scripting.md).
- **Mono picks on one side of the render.** Reopen the timeline before
  rendering; `layin` does this, and `verify_channels` checks it.
