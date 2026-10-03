# Setting up on your own Mac

For an editor who knows Resolve and their footage but is new to this
repository and to running models locally. Every step ends with a **check:**
line. Don't move on until it passes. Claude Code can walk through this with
you (see [Let Claude Code help](#let-claude-code-help)).

The model sizes and disk figures below were measured on the 128 GB Apple
Silicon machine this was developed on (October 2026).

## 1. What you need

### A Mac with Apple Silicon

mlx-whisper and the MLX model builds run only on Apple Silicon. Intel Macs,
Linux and Windows are not supported.

### Memory

Three models do the work, and **only one should run at a time** (see
[GPU contention](#troubleshooting)):

| Job | Model | Served by | Size on disk |
|---|---|---|---|
| Vision: shot annotation, picture-vs-words judge, cover verifier | `google/gemma-4-26b-a4b` (Q8) | LM Studio | 28.05 GB |
| Text: cover proposer, viewer read, story check | `qwen3.8:27b-mtp-q8_0` (Q8) | Ollama | 29 GB |
| Transcription | `mlx-community/whisper-large-v3-mlx` | in-process | 2.9 GB |
| Vision alternative (smaller) | `zai-org/glm-4.6v-flash` (9B) | LM Studio | 7.09 GB |

A loaded model needs roughly its disk size in memory, plus its context
cache: 16k tokens for vision, 32–40k for the text model.

- **64 GB or more:** the production setup. One 28–29 GB model at a time,
  with Resolve open. Unload one before loading the other: `lms unload --all`
  for LM Studio, `ollama stop qwen3.8:27b-mtp-q8_0` for Ollama.
- **96–128 GB:** both models fit at once, but still don't run their jobs
  concurrently. Running both at once slowed each about 4x. With both
  resident on the 128 GB machine, macOS killed a background process for low
  memory.
- **32 GB:** the 28–29 GB models leave nothing for macOS and Resolve. Use
  GLM-4.6V-flash for vision. It passes the same image trap test, but in
  calibration it wrote shallower cut notes than Gemma 4 26B and was about
  1.7x slower. A thinking model, it needs a generous `max_tokens`
  ([lessons 1.2](lessons.md#12-thinking-models-spend-the-budget-before-they-answer)).
  The text jobs (eval loop and story check) have **not been tested with a
  smaller model.** Expect weaker JSON adherence, and check a few beats by
  hand before trusting it. Transcription and every measurement (shake,
  drone moves, loudness, joins) run fine on 32 GB.

The text model defaults to `qwen3.8:27b-mtp-q8_0` at `http://localhost:11434`.
To use another, set `VIDEOEASY_TEXT_MODEL` (and `VIDEOEASY_TEXT_URL` for a
different server) in your shell. Every text job (proposer, viewer read, story
check, talk analysis) and `videoeasy doctor` read it, and each run's manifest
records the model it actually used:

```bash
export VIDEOEASY_TEXT_MODEL=qwen3:14b      # example; put it in your shell profile to keep it
```

### Disk

- **Models:** about 60 GB for the production three, plus 7 GB for GLM if
  you want the small vision model.
- **Ingest working files:** 78 GB of source footage produced 788 MB of
  `work/`, about 1%: 328 MB of graded frames, 87 MB of contact sheets,
  366 MB of judge frames from the eval loop. The footage itself stays where
  it is; `inputs/` holds symlinks.
- **Renders:** 46 review renders of an 18-minute cut took 21 GB, about
  460 MB each with the YouTube 1080p preset. Short deliverables are smaller,
  but every iteration is a new file.
- **Diagnostics:** review and diagnostic directories grew to 85 GB over two
  weeks of eval iteration. Put `data/` on a volume with room, and prune old
  review runs.

### DaVinci Resolve Studio

The lay-in, read-back and media-pool organisation use Resolve's external
scripting, which needs **Studio**. Everything before the lay-in (ingest,
transcripts, measurements, the story room) works without Resolve.

## 2. Install

### Homebrew

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

check: `brew --version` prints a version.

### ffmpeg, uv, git, gh

```bash
brew install ffmpeg uv git gh
```

check: `ffmpeg -hide_banner -filters | grep -cE ' (lut3d|ebur128|loudnorm|blackdetect|silencedetect) '`
prints `5`. Those are the filters the grading, loudness and joins checks use.

### GitHub access

The repository may be private. Its owner invites your GitHub account as a
collaborator, and you accept the invitation (from GitHub's email, or at
`https://github.com/<owner>/videoeasy/invitations`).

```bash
gh auth login                       # GitHub.com, SSH or HTTPS, log in with a browser
gh repo clone <owner>/videoeasy
cd videoeasy
```

check: `git log --oneline | head -1` shows a commit.

### Python environment

```bash
uv sync
uv run python -m unittest discover tests
```

check: the last line reads `OK`. These tests need no models, footage or
Resolve.

### LM Studio (vision)

1. Install [LM Studio](https://lmstudio.ai/) and open it once.
2. Put its command-line tool on your path, then open a new terminal:
   ```bash
   ~/.lmstudio/bin/lms bootstrap
   ```
3. Download the vision model:
   ```bash
   lms get google/gemma-4-26b-a4b        # or zai-org/glm-4.6v-flash on 32 GB
   ```
4. Load it **with a 16k context** and start the server:
   ```bash
   lms load google/gemma-4-26b-a4b --context-length 16384
   lms server start
   ```

check: `lms ps` lists the model, and `curl -s localhost:1234/v1/models`
returns JSON naming it.

The context flag is not optional. At LM Studio's default 4096 tokens, 9 of
the first 130 annotations in a batch failed with `finish_reason=length`
([lessons 1.3](lessons.md#13-the-default-context-window-truncates-the-busiest-shots)).

**Never serve the vision model from Ollama.** Its MLX backend silently drops
images, and the model then describes shots it never saw
([lessons 1.1](lessons.md#11-a-vision-backend-can-drop-images-silently)).

### Ollama (text)

1. Install [Ollama](https://ollama.com/) (the app, or `brew install ollama`
   and then `ollama serve`).
2. Pull the text model. This tag needs Ollama 0.32.12 or later:
   ```bash
   ollama pull qwen3.8:27b-mtp-q8_0
   ```
   This is the tag used during development. If your Ollama can't find it,
   pull the nearest 27B Qwen text model at Q8 and set `VIDEOEASY_TEXT_MODEL`
   to its tag (see [Memory](#memory)).

check: `ollama list` shows the model, and `curl -s localhost:11434/api/tags`
returns JSON.

### Whisper

The model downloads on first use (2.9 GB). To fetch it now instead of in the
middle of a run:

```bash
uv run python -c "from huggingface_hub import snapshot_download; snapshot_download('mlx-community/whisper-large-v3-mlx', revision='49e6aa286ad60c14352c404340ded53710378a11')"
```

check: `uv run python -c "import mlx_whisper; print('ok')"` prints `ok`.

### DaVinci Resolve Studio

1. Install Resolve Studio and open a project.
2. Preferences → System → General → **External scripting using: Local**.
   Restart Resolve.

check: with Resolve open on a normal page (no dialogs),

```bash
uv run python -c "from videoeasy.resolve import get_resolve; print(get_resolve().GetVersionString())"
```

prints Resolve's version. If it hangs or prints nothing, close any open
dialog or the Project Manager: while one is open, Resolve reports no page and
every write silently does nothing ([resolve-scripting.md](resolve-scripting.md#silent-write-failures)).

### Optional: Resolve's native MCP for Claude Code

Resolve Studio 21.1 ships its own MCP server. It lets Claude Code search the
scripting API and run scripts in Resolve. Setup is in
[resolve-scripting.md](resolve-scripting.md#native-mcp). The bundle is
Blackmagic's, so extract it from your own install; it is not in this
repository.

check: in Claude Code, `/mcp` lists `davinci-resolve` as connected.

## 3. Check the whole chain

### `videoeasy doctor` (no config needed)

```bash
uv run videoeasy doctor
```

This checks the machine before any project exists: the platform, Python,
`ffmpeg`/`ffprobe`, `uv`, LM Studio and the vision model's context window,
`mlx_whisper`, Ollama and its text model, Resolve scripting and free disk.
Missing required items fail the run with a fix printed beside each one.
Optional items (the eval loop's text model, Resolve, cached Whisper weights,
disk space) print `WARN`. `--config config.<project>.yaml` takes the vision
model and URL from a config instead of the defaults.

check: the last line says all required items are present.

### `videoeasy check` (with a config)

Make a config for your project from the example, point its paths at your
footage, and set its LUTs. Configs hold machine paths and are git-ignored:

```bash
cp config.example.yaml config.myproject.yaml
uv run videoeasy check --config config.myproject.yaml
```

`check` verifies ffmpeg, every LUT the grade profiles name, Whisper, the
loaded context window, and a **vision trap**: a synthetic image whose colours
the model must name exactly.

check: every line reads `ok`. `SUSPECT` on the vision line means images are
not reaching the model. Stop and fix that first.

## 4. Calibrate before any overnight run

Run the stages on 2–3 representative clips (an interview, a busy B-roll
take, a drone shot) and read what comes out next to the contact sheets. The
commands are in the [user guide §4](user-guide.md#4-verify-calibrate-then-commit-the-night).
Only then start `videoeasy all`, which is an overnight job.

check: the 2–3 annotations describe what the contact sheets show, and you'd
be happy to search 500 of them.

For an event shoot with a shot list and many deliverables, continue with
[event-projects.md](event-projects.md).

## Let Claude Code help

Install [Claude Code](https://claude.com/claude-code), open a terminal in the
repository, run `claude`, and ask for what you want, for example:

- "Walk me through docs/setup.md and run each check with me."
- "Run `videoeasy doctor` and explain anything that fails."
- "Help me write config.myproject.yaml for footage in /Volumes/MyDrive/Shoot."
- "I have a shot list in a Google Doc. Help me turn it into
  `data/projects/<project>/deliverables.yaml`."

The agent reads [AGENTS.md](../AGENTS.md) and the skills in `.agents/skills/`
first. It will:

- run the checks and read their output with you;
- write configs, project files and scripts under `data/` and explain them;
- propose transcript selects and cuts as text, for you to accept or change.

It won't:

- commit anything under `data/`, configs, `film.yaml` or media. These are
  git-ignored, and it is told never to change that;
- write to Resolve without telling you what will change and getting your
  OK. It does a dry run first and verifies every write by reading the
  timeline back;
- judge pictures by looking at frames itself. Picture judgments go through
  the local vision model, with prompts and scores you can read;
- run `videoeasy all` to test something. It checks a single clip first.

## Troubleshooting

| Symptom | Cause | Fix |
|---|---|---|
| Annotations describe things that aren't in the shot; `check` says `SUSPECT` | images aren't reaching the vision model (Ollama's MLX backend drops them silently) | serve vision from LM Studio; re-run `check` after any model or backend change |
| Busy shots come back `finish_reason=length` whatever `max_tokens` is | LM Studio loaded the model at its default 4096-token context | `lms unload --all && lms load <model> --context-length 16384` |
| Empty answer with `finish_reason=length` | a thinking model (GLM-4.6V) spent the budget reasoning | raise `max_tokens` (default 6000, one retry at double) |
| Everything is ~4x slower than usual; a background job was killed | Ollama and LM Studio both running jobs, fighting over the GPU and memory | one at a time: `lms unload --all` or `ollama stop <model>` |
| Resolve script runs "successfully" but nothing changed | a modal dialog or the Project Manager is open, so `GetCurrentPage()` is `None` and writes silently fail | close it; the tools refuse to write when there's no page |
| Read-back says every gain is missing | read during a render (item properties read `None`) | read back only when no render is in progress |
| `check`/`annotate` fail with a connection error inside Claude Code | the sandbox blocked localhost, or the server is down | `lms ps`; approve the network call; it is not a code bug |
| Annotation mentions "no optical flow detected" on a project that never ran `motion` | the prompt didn't say measurements were missing | run `motion`; the prompt now states when statistics are absent |

More entries, each with its evidence, are in [lessons.md](lessons.md).
