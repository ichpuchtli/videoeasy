# videoeasy

AI-assisted documentary editing, built around one rule: **only local models
ever see the pixels or hear the audio.** Local vision, speech and text models
ingest the footage into a **footage bible**, a single legible document
describing every shot, window and word. A frontier model (Claude, in a Claude
Code session) then does story work over that text: radio cuts, B-roll
matching, coverage gaps, structure. The cut is checked by a local-model
evaluation loop and laid into DaVinci Resolve through its scripting API,
then read back and conformed.

Most of what is here was learned the hard way. The code is half of this
repository; the other half is what went wrong: [docs/lessons.md](docs/lessons.md).

## What it does

- **Ingest** (`videoeasy all`): ffprobe inventory with timecode → scene
  detection → word-timestamped Whisper transcripts → long takes windowed at
  speech pauses → graded frames and contact sheets → optical flow → a vision
  model's per-shot annotation → marked-up transcript selects → `bible-context.md`.
- **Measurements the vision model can't make from stills:** camera shake as a
  stabiliser residual with the steady runs of each take, drone moves
  (pan, orbit, rise, push-in…) and which of them are clean, whether a cover
  shot holds someone else talking, B-roll audio transcribed with confidence
  filters, loudness and channel balance per pick, joins (black and silence),
  bird calls (BirdNET), plant species (BioCLIP, advisory).
- **A local eval loop:** `cuteval` scores a render (picture vs words,
  exposure, rhythm, joins, cover rules, levels) → `brollmatch` proposes and
  verifies cover → `cutbuild` rebuilds picture rows while conserving every
  pick's timing → `layin` lays the cut into Resolve, reads it back and
  conforms it → `storycheck` transcribes the render and checks that every
  word played and the story holds. Every result is `ok`, `unchecked` or
  `invalid`, and only `ok` carries a score. Every report has a manifest of
  content hashes.
- **Editorial decisions as rules:** `intent.json` records who decided what,
  and when, and what it supersedes. Code checks it against the cut and the
  Resolve read-back with no model in the loop, so a quality score can never
  override a decision.
- **Delivery:** a loudness-normalised master, measured against the recorded
  target.

## Status

A personal research tool used on real documentary work, published for its
code and methodology. Expect sharp edges:

- macOS on Apple Silicon only (mlx-whisper, MLX model builds).
- Needs [LM Studio](https://lmstudio.ai/) (vision model), [Ollama](https://ollama.com/)
  (text judge and proposer), `ffmpeg`, [uv](https://docs.astral.sh/uv/), and
  DaVinci Resolve **Studio** for the lay-in.
- Thresholds marked provisional are working numbers, not calibrated ones, and
  the judges have no human ground truth yet ([eval-design.md §5](docs/eval-design.md#5-known-weaknesses-and-the-roadmap)).

## Quickstart

```bash
uv sync
cp config.example.yaml config.myfilm.yaml          # paths, grading LUTs, models
mkdir -p data/films/myfilm && cp film.example.yaml data/films/myfilm/film.yaml
lms load google/gemma-4-26b-a4b --context-length 16384 && lms server start
uv run videoeasy check --config config.myfilm.yaml   # includes the vision trap test
uv run videoeasy annotate --config config.myfilm.yaml --only <shot_id>   # calibrate first
uv run videoeasy all --config config.myfilm.yaml     # overnight
uv run python -m unittest discover tests             # invariants; no models, no footage
```

Then open Claude Code in the repo for the story room. The full walk-through
is in the [user guide](docs/user-guide.md).

## Documentation

| Doc | What's in it |
|---|---|
| [docs/lessons.md](docs/lessons.md) | every hard-won lesson: failure, evidence, rule, guard |
| [docs/architecture.md](docs/architecture.md) | why a document and not RAG, split of labour, pipeline, decisions |
| [docs/eval-design.md](docs/eval-design.md) | the cut-evaluation loop, thresholds, known weaknesses and roadmap |
| [docs/resolve-scripting.md](docs/resolve-scripting.md) | DaVinci Resolve Python API and native MCP: what holds and what lies |
| [docs/local-models.md](docs/local-models.md) | which local models did which job, and how they failed |
| [docs/user-guide.md](docs/user-guide.md) | setup, config and film profile, `intent.json`, commands, troubleshooting |
| [docs/evidence-audits.md](docs/evidence-audits.md) | the opt-in frame evidence and tag consistency audit |
| [AGENTS.md](AGENTS.md) | instructions for coding agents (`CLAUDE.md` is a symlink to it) |
| [.agents/skills/](.agents/skills/) | agent skills: ingest, story room, Resolve |

## Privacy

Footage, transcripts, annotations, cuts, renders and editorial decisions all
live under `data/`, which is never committed. So are the per-film configs
(`config.<film>.yaml`) and profiles (`film.yaml`). The repository contains
code, generic examples and documentation only. Nothing here sends media to a
hosted model. The frontier model receives text.

## License

MIT. See [LICENSE](LICENSE).
