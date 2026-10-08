# videoeasy

AI-assisted documentary editing. Local models ingest footage, and only local
models ever see pixels or hear audio. A frontier model does story work over
the resulting **footage bible** in a coding-agent session. Films live under
`data/films/<film>/` (never committed): before any story work, read that
film's private brief and register reference there. Event shoots with a shot
list and many short deliverables live under `data/projects/<project>/`
(never committed): read its `deliverables.yaml` and `items.yaml` first and
use the `videoeasy-deliverables` skill. A new machine starts with
[docs/setup.md](docs/setup.md).

> This file is the single source of agent instructions for this repo.
> `CLAUDE.md` is a symlink to it, so Claude Code, Codex and anything else that
> reads `AGENTS.md` load identical content. **Edit `AGENTS.md`.**

## Commands

```bash
uv run videoeasy doctor                              # machine check, no config: tools, models and servers, Resolve scripting (optional items warn)
uv run videoeasy check --config config.<film>.yaml   # verify the whole chain BEFORE any run (includes the vision trap test)
uv run videoeasy all --config config.<film>.yaml     # full ingest: probe → shots → transcribe → segment → frames → motion → annotate → selects → assemble
uv run videoeasy segment --config ...                # window long takes into searchable moments (out/segments.json)
uv run videoeasy annotate --config ... --only <id>   # single-shot/window calibration
uv run videoeasy assemble --config ...               # rebuild out/bible-context.md from artifacts
uv run videoeasy transcribe --config ... --broll     # B-roll audio: forced English, Whisper confidence kept, silence recorded; A-roll untouched
uv run videoeasy audit --config ... --only <id>      # opt-in frame evidence audit (docs/evidence-audits.md)
uv run python -m videoeasy.cuteval ...               # score a rendered cut with local models
uv run python -m videoeasy.brollmatch ...            # propose + verify cover per spoken pick
uv run python -m videoeasy.cutbuild ...              # rebuild a cut's video rows from the verified plan
uv run python -m videoeasy.storycheck ...            # transcribe a render and judge it against the synthesis spine
uv run python -m videoeasy.layin ...                 # lay a cut into Resolve, read it back, conform (videoeasy-resolve skill)
uv run python -m videoeasy.intent --film data/films/<film> --cut editorial/cut-vN.json [--readback ... --plan ...]   # recorded decisions vs the cut / the timeline
uv run python -m videoeasy.deliver --film data/films/<film> --render editorial/renders/<r>.mp4   # measured loudness master
uv run python -m videoeasy.steadiness --config ...   # camera shake per catalogue unit → out/steadiness.json
uv run python -m videoeasy.moves --config ...        # drone moves and which are clean → out/moves.json
uv run --extra birds python -m videoeasy.birds --config ...   # BirdNET over source audio → out/birds.json (heard, not seen)
uv run python -m videoeasy.faces --config ...            # face tracks + blendshape cues per source (pinned MediaPipe worker env) → out/faces/
uv run python -m videoeasy.sounds --config ...           # laughter, crying, shouts, cheers heard per source (YAMNet) → out/sounds.json
uv run python -m videoeasy.register {moments|read|map|find|review|showcase|calibrate} --config ...   # facial register moments: measured, heard, seen, fused (docs/register.md)
uv run python -m videoeasy.identity setup                                          # downloads InsightFace buffalo_l (pinned hash; NON-COMMERCIAL model licence, never in the repo) + its worker env
uv run python -m videoeasy.identity {run|review|confirm|export|import} --config ...  # which face tracks are one person: proposals, then the editor confirms (docs/identity.md)
uv run python -m unittest discover tests             # invariants: no models, no footage, no Resolve
```

Event projects (`docs/event-projects.md`; `--project data/projects/<project>` on every command):

```bash
uv run python -m videoeasy.deliverables validate     # check deliverables.yaml and items.yaml
uv run python -m videoeasy.deliverables plan         # the items each deliverable type implies
uv run python -m videoeasy.deliverables clocks --config config.<project>.yaml   # per-camera recorded times vs the schedule
uv run python -m videoeasy.deliverables assign --config config.<project>.yaml   # clips → sessions, feeds and mood candidates → out/sessions.json
uv run python -m videoeasy.deliverables status [--scope NAME] [--md PATH]       # counts vs the shot list; "claimed, no evidence"
uv run python -m videoeasy.deliverables set <item> key=value ...                # record a decision or status in items.yaml
uv run python -m videoeasy.deliverables item-dir <item>                         # items/<item>/ as a film dir: editorial/ + links to the shared out/ and work/
uv run python -m videoeasy.deliverables bins --config config.<project>.yaml --resolve-project NAME --source-bin PATH [--apply]   # Resolve bins; dry run without --apply
```

Long talks recorded on lav recorders (`docs/talks.md`; 32-bit float safe, originals never written):

```bash
uv run python -m videoeasy.talks all --project data/projects/<p> --mic-dir <dir> --session <first chunk> --talk-id <id>   # prepare → transcribe → sentences → beats → bites → verify → report
uv run python -m videoeasy.talks all ... --skip-model     # measurements and transcript only, no LLM
uv run python -m videoeasy.talks resolve ... --resolve-project NAME --timeline NAME [--apply]   # selects timeline + markers; dry run by default
uv run python -m videoeasy.talks export-cut ...           # ranked selects as a cut brollmatch can read
```

Waveform sync between lav-mic recordings and camera clips (`transcribe`
prefers synced mic audio over camera scratch audio when a match is verified):

```bash
uv run videoeasy audiosync cache | match --clip <file> | verify --clip <file> | report
```

Start the vision server first:
`lms load google/gemma-4-26b-a4b --context-length 16384 && lms server start`.
The context flag matters (see below). The model comes from config
`models.vision`.

Every eval tool writes a `<out>.manifest.json` (input hashes, prompt and
model, what was checked versus `unchecked`) and refuses to overwrite a
previous run without `--overwrite`. A result whose status is not `ok` carries
no score (`src/videoeasy/evalrun.py`). Editorial decisions live in
`data/films/<film>/editorial/intent.json`, and the builder, proposer, scorer
and lay-in read the same file. Film specifics (recording roles, ASR aliases,
the register paragraph, Resolve constants) live in
`data/films/<film>/film.yaml` (`film.example.yaml`, `src/videoeasy/film.py`).
Never hard-code a film into the code.

## Hard-won constraints: do not regress these

Each has a full entry in [docs/lessons.md](docs/lessons.md).

- **Never point vision at Ollama.** Its MLX backend silently drops images and
  the model invents plausible descriptions. Vision goes through LM Studio's
  OpenAI-compatible API (port 1234). `videoeasy check` traps this. Run it after
  any model or backend change.
- **Load the vision model with at least 16k context.** At LM Studio's default
  4096, eight frames plus the prompt leave no room for the answer, and busy
  shots come back `finish_reason=length` whatever `max_tokens` is. `check`
  fails under `vlm.MIN_CONTEXT_TOKENS`.
- **Thinking models spend the budget first.** Keep `max_tokens` generous (vlm.py
  default 6000, one retry at double). An empty answer with
  `finish_reason=length` means raise it or turn reasoning off.
- **Grade every frame before a vision model sees it** (config `grade:`,
  matched per source by filename substring). Flat log frames poison mood and
  colour annotation.
- **Never put concrete example phrases in interpretive prompt fields.** The
  model copies them verbatim across shots and destroys the bible's ability to
  tell shots apart. The film's register goes in `film.yaml` `context` as a
  description, never as samples.
- **Never let a prompt stay silent about missing measurements.** With no
  `motion.json`, the model reported optical flow it never saw.
- **A window is an index, not a cut.** Window edges land on transcript
  silences where one is in reach and on even divisions otherwise, and every
  edge records which. Nothing may describe a window edge as an edit point. A
  window's transcript is for search only and never enters the vision prompt:
  a spoken topic is not a visible fact.
- **Measure, don't judge, what stills cannot show.** Talking cover
  (`speech.talking_cover`), camera shake (steady runs in
  `out/steadiness.json`), drone moves (`out/moves.json`), joins, loudness and
  channels are measurements and rules. A vision score never overrides them.
  Provisional thresholds stay provisional until the editor reviews them.
- **One ASR pass is a question.** A flagged pick gets a padded second
  transcription, and is `confirmed` lost only when the render's waveform
  also stops matching the source's. Nothing is trimmed on one pass.
- **Failed checks are `unchecked`, never a middling score.** Caches are keyed
  on content hashes, never on file existence.
- **Never run Ollama and LM Studio jobs at the same time.**
- **Resolve API writes fail silently** when a modal dialog is open
  (`GetCurrentPage()` is `None`). `MoveClips`' return value is unreliable.
  There is no "Red" clip colour (use `AddFlag("Red")`). `AppendToTimeline`
  with `recordFrame` is the only non-rippling placement, and title inserts
  ripple. Verify every write by re-reading. Details:
  [docs/resolve-scripting.md](docs/resolve-scripting.md) and the
  `videoeasy-resolve` skill.

## Architecture in one paragraph

A short documentary's material (about an hour of transcript plus its shots)
fits in one frontier-model context, so understanding lives in a legible,
correctable document (`out/bible-context.md`), not in vector search. Ingest
produces per-shot and per-window records (subject, movement, emotional
metaphor, visual metaphor, themes, plus measured motion) and a word-level
timestamped transcript. The synthesis layer (themes ↔ shots ↔ beats) is built
and maintained in conversation and saved to `out/bible-synthesis.md`.
Details: [docs/architecture.md](docs/architecture.md).

## Workflow expectations

- **Validation goes through local models, not the frontier model's eyes.** To
  judge a render, a grade or a cut: write the prompt, send graded frames to
  the local vision model and the transcript to the local text model, read the
  scores, fix the generator, re-run. `cuteval` → `brollmatch` → `cutbuild`
  is one iteration (`videoeasy-story-room` skill). Don't open contact sheets
  to grade by eye when a scorer can do it. Design, thresholds and weaknesses
  are in [docs/eval-design.md](docs/eval-design.md); its weakness list is the
  roadmap.
- **Before the first overnight run,** calibrate on 2–3 shots
  (`annotate --only`) and review annotation depth with the editor.
- **Story-room sessions:** read `out/bible-context.md`,
  `out/bible-synthesis.md` (and any later build beside it), and the film's
  private brief and register reference under `data/films/<film>/`. Builds sit
  side by side, so a new build is a new file, never a rewrite. Contact sheets
  in `work/sheets/` are JPEGs. Open them when a specific shot matters to an
  argument, and never guess a shot's content from its id.
- **Transcript markup and the editor's draft radio cut are prior editorial
  intent** already extracted into the bible. Treat them as decisions, not
  suggestions to re-derive.
- **Record decisions in `intent.json`** with who decided, the source and what
  it supersedes. Never encode a decision only in a beat note or a prompt.
  Grant exceptions by name, never by pattern.

## Agent environment notes

- **Hand-curated vs regenerable.** `out/bible-synthesis.md` (and any later
  build) and `editorial/intent.json` are human judgment: never regenerate or
  clobber them from a script. Everything else in `out/` (`*.json`,
  `bible-context.md`) can be rebuilt by the pipeline, and `work/` is entirely
  regenerable. Before overwriting anything under `data/`, look at the file
  and say what you are about to replace.
- **Originals are never modified.** Nothing on the footage drive or under
  `inputs/` is written to, ever.
- **Footage lives on external drives** (symlinks or absolute config paths).
  Source paths are baked into artifacts, so mount names must stay stable.
  Stages that need the drive: probe, shots, frames, motion, transcribe.
  Annotate and assemble work from extracted frames.
- **Network.** `check` and `annotate` talk to LM Studio on
  `http://localhost:1234`, and the text judges to Ollama on `:11434`. In a
  sandboxed session these fail with a connection error, which means the
  sandbox blocked it or the server is down, not that the code is broken.
  Check `lms ps` first.
- **`videoeasy all` is an overnight job.** Never launch it to test a change.
  Use `check`, then a single-shot `annotate --only <id>`.
- **Stages are idempotent and resumable.** Re-running skips completed work.
  `--force` on `annotate`/`transcribe` redoes from scratch (use it after a
  prompt or model change).
- **Never commit film or project material.** `data/` (including
  `data/films/` and `data/projects/`), `config.<film>.yaml`,
  `config.<project>.yaml`, `film.yaml`, `.mcp.json`, media and LUTs are
  git-ignored. Keep it that way. Names of people filmed, shot lists, clients
  and transcripts never go into code, comments, prompt examples or test
  fixtures.

## Repo skills

Reusable procedures live in `.agents/skills/` (read natively by Codex, and
surfaced to Claude Code through per-skill symlinks in `.claude/skills/`):

| Skill | Use when |
|---|---|
| `videoeasy-ingest` | running or debugging the ingest pipeline, calibrating annotation, changing models or grading |
| `videoeasy-story-room` | story work over the bible: radio cuts, B-roll matching, beats, the synthesis layer, judging a cut |
| `videoeasy-resolve` | anything touching the DaVinci Resolve scripting API or MCP |
| `videoeasy-deliverables` | an event / shot-list project: building `deliverables.yaml`, clocks, session assignment, testimonial bites, tracking items, Resolve bins |

Edit the real files under `.agents/skills/`.
