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
- **Long talks** (`videoeasy.talks`): an hour of lav-recorder audio (32-bit
  float safe) → word-timed transcript → sentences with measured edge room and
  off-mic flags → the talk's beats → hooks, bites and closers, each with a
  cover brief for the b-roll and an independent standalone check → a readable
  report, `selects.csv` and an optional Resolve selects timeline.
- **Event projects:** a multi-day shoot with a shot list and many short
  deliverables (testimonials, before/after interviews, speaker features, a
  B-roll package by mood). The schedule goes in a private
  `deliverables.yaml`: clips are assigned to sessions by their corrected
  recording time, each item is tracked against evidence on disk, and media
  can be organised into Resolve bins
  ([docs/event-projects.md](docs/event-projects.md)).

## Status

A personal research tool used on real documentary work, published for its
code and methodology. Expect sharp edges:

- macOS on Apple Silicon only (mlx-whisper, MLX model builds).
- Needs [LM Studio](https://lmstudio.ai/) (vision model), [Ollama](https://ollama.com/)
  (text judge and proposer), `ffmpeg`, [uv](https://docs.astral.sh/uv/), and
  DaVinci Resolve **Studio** for the lay-in.
- Thresholds marked provisional are working numbers, not calibrated ones, and
  the judges have no human ground truth yet ([eval-design.md §5](docs/eval-design.md#5-known-weaknesses-and-the-roadmap)).

## Models used

Every result in the docs came from these offline models. To reproduce the
work, use the same builds: the judges' scores, the proposer's choices and
the failure modes in [lessons.md](docs/lessons.md) all depend on them. Each
eval report's manifest records the model id, URL and prompt hash that
produced it.

| Job | Model | Build | Size | Runtime | Settings |
|---|---|---|---|---|---|
| Shot and window annotation; picture-vs-words judge; cover verifier | [`google/gemma-4-26b-a4b`](https://lmstudio.ai/models/google/gemma-4-26b-a4b) | GGUF Q8_0, 26B total / 4B active (MoE), vision | 28.1 GB | LM Studio 0.4.12, OpenAI-compatible server on `:1234/v1` | loaded with `--context-length 16384` (the 4096 default truncates busy shots); temperature 0.3 annotating, 0.2 judging; `max_tokens` 6000 (annotation, one retry at 12000), 1500 (judge), 1200 (verifier) |
| Text judge, cover proposer, story check, talk beats, bites and verification | [`qwen3.8:27b-mtp-q8_0`](https://ollama.com/library/qwen3.8) | Q8_0, 27.3B, manifest digest `8a1582877303` | 29 GB | Ollama 0.33.0 (the tag needs ≥ 0.32.12), `/api/chat` on `:11434` | `think: false`, `format: json`, temperature 0.2 (viewer read 0.3), `num_ctx` 40000 (viewer read 32768) |
| Transcription: sources, B-roll audio, renders, long talks | [`mlx-community/whisper-large-v3-mlx`](https://huggingface.co/mlx-community/whisper-large-v3-mlx) | revision `49e6aa286ad60c14352c404340ded53710378a11` | 2.9 GB | mlx-whisper 0.4.3 on MLX 0.31.2, in-process | word timestamps; `condition_on_previous_text=False` on renders and talks; B-roll pass forces English and drops segments past no-speech 0.6 / compression 2.4 |
| Vision, smaller alternative | [`zai-org/glm-4.6v-flash`](https://lmstudio.ai/models/zai-org/glm-4.6v-flash) | MLX 4-bit, 9B, vision, thinking model | 7.1 GB | LM Studio | passes `videoeasy check`; was the first annotator until July 2026 calibration chose Gemma (deeper cut notes, about 1.7× faster); keep `max_tokens` generous, as it reasons before answering |
| Bird calls heard (optional `birds` extra) | BirdNET acoustic 2.4 + geo 2.4 | TFLite (LiteRT) | | `birdnet` 1.1.1 | location and week from the config |
| Plant species on screen (optional `species` extra) | BioCLIP 2 | | | `pybioclip` 2.1.6 | advisory only |
| Story work over the footage bible (text only) | Claude, in Claude Code | frontier, hosted | | | receives text only, never pixels or audio; not pinned to one version |
| Face identity (optional) | InsightFace [`buffalo_l`](https://github.com/deepinsight/insightface/tree/master/python-package) (SCRFD detection, ArcFace `w600k_r50`) | release v0.7 zip, sha256 `80ffe37d…`, downloaded by `identity setup`, never in this repository | 288 MB | `insightface` 1.0.1 + onnxruntime 1.28 (CPU) in the PEP 723 worker's own environment | **non-commercial research use only** (InsightFace's licence for its pretrained models); a model of your own can answer instead ([identity.md](docs/identity.md)) |
| Everything measured | ffmpeg 8.1, OpenCV | | | | `signalstats`, `blackdetect`, `silencedetect`, `ebur128`, `loudnorm`, Lucas–Kanade + RANSAC |

Reference machine: Apple M5 Max, 128 GB unified memory, macOS 26.6.2. The
two production models together need about 60 GB. **Run one model job at a
time:** with Ollama and LM Studio both busy, each slowed by about 4×, and
the OS killed a background process for low memory. Python dependencies are
pinned in `uv.lock` (`uv sync --frozen`).

Getting the same builds:

```bash
lms get google/gemma-4-26b-a4b          # choose the Q8_0 GGUF build
lms load google/gemma-4-26b-a4b --context-length 16384 && lms server start
ollama pull qwen3.8:27b-mtp-q8_0        # `ollama list` should show ID 8a1582877303
uv run python -c "from huggingface_hub import snapshot_download; \
  snapshot_download('mlx-community/whisper-large-v3-mlx', revision='49e6aa286ad60c14352c404340ded53710378a11')"
uv run videoeasy doctor                 # confirms the servers, the loaded context and the text model
```

Using a different model is a one-line change (`models.vision` in the config,
`VIDEOEASY_TEXT_MODEL` / `VIDEOEASY_TEXT_URL` for the text model,
`VIDEOEASY_WHISPER_MODEL` for long talks), but treat it as a new
experiment. Re-run `videoeasy check`, which includes the image trap test,
and expect scores not to be comparable with earlier runs. **Never serve the
vision model through Ollama:** its MLX backend silently dropped images while
the model described shots it never saw
([lessons 1.1](docs/lessons.md#11-a-vision-backend-can-drop-images-silently)).
Smaller-machine options, and how each model failed, are in
[docs/setup.md](docs/setup.md) and [docs/local-models.md](docs/local-models.md).

## Quickstart

New to the repo or to local models? Follow [docs/setup.md](docs/setup.md)
step by step; every step has a check. The short version:

```bash
uv sync
uv run videoeasy doctor                              # machine check: tools, models, servers, Resolve; no config needed
cp config.example.yaml config.myfilm.yaml            # paths, grading LUTs, models
mkdir -p data/films/myfilm && cp film.example.yaml data/films/myfilm/film.yaml
lms load google/gemma-4-26b-a4b --context-length 16384 && lms server start
uv run videoeasy check --config config.myfilm.yaml   # includes the vision trap test
uv run videoeasy annotate --config config.myfilm.yaml --only <shot_id>   # calibrate first
uv run videoeasy all --config config.myfilm.yaml     # overnight
uv run python -m unittest discover tests             # invariants; no models, no footage
```

Then open Claude Code in the repo for the story room. The full walk-through
is in the [user guide](docs/user-guide.md). For an event shoot with a shot
list, continue with [docs/event-projects.md](docs/event-projects.md).

## Documentation

| Doc | What's in it |
|---|---|
| [docs/setup.md](docs/setup.md) | setting up a new Mac step by step: memory and disk, installs, models, Resolve, checks |
| [docs/lessons.md](docs/lessons.md) | every hard-won lesson: failure, evidence, rule, guard |
| [docs/architecture.md](docs/architecture.md) | why a document and not RAG, split of labour, pipeline, decisions |
| [docs/eval-design.md](docs/eval-design.md) | the cut-evaluation loop, thresholds, known weaknesses and roadmap |
| [docs/resolve-scripting.md](docs/resolve-scripting.md) | DaVinci Resolve Python API and native MCP: what holds and what lies |
| [docs/local-models.md](docs/local-models.md) | which local models did which job, and how they failed |
| [docs/user-guide.md](docs/user-guide.md) | setup, config and film profile, `intent.json`, commands, troubleshooting |
| [docs/talks.md](docs/talks.md) | long talks on lav recorders: 32-bit float handling, beats, hooks, bites, cover briefs, selects |
| [docs/event-projects.md](docs/event-projects.md) | event shoots: shot list → `deliverables.yaml`, clocks, session assignment, per-deliverable workflows, tracking |
| [docs/register.md](docs/register.md) | facial register moments: face cues measured, sounds heard, faces seen by the local vision model, fused in code, calibrated blind |
| [docs/identity.md](docs/identity.md) | face identity: which tracks are one person (InsightFace, downloaded at setup), the editor confirms; licences; bring-your-own-model files |
| [docs/sibling-project.md](docs/sibling-project.md) | design for a sibling project: mood of what's displayed, sound bites in long talks, related-dialogue search |
| [docs/evidence-audits.md](docs/evidence-audits.md) | the opt-in frame evidence and tag consistency audit |
| [AGENTS.md](AGENTS.md) | instructions for coding agents (`CLAUDE.md` is a symlink to it) |
| [.agents/skills/](.agents/skills/) | agent skills: ingest, story room, Resolve, event deliverables |

## Privacy

Footage, transcripts, annotations, cuts, renders and editorial decisions all
live under `data/` (films in `data/films/`, event projects in
`data/projects/`), which is never committed. So are the per-film configs
(`config.<film>.yaml`) and profiles (`film.yaml`). The repository contains
code, generic examples and documentation only. Nothing here sends media to a
hosted model. The frontier model receives text. Face identity runs locally,
and only links the editor confirms name anyone ([identity.md](docs/identity.md)).

## License

MIT. See [LICENSE](LICENSE).
