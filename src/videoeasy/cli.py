from __future__ import annotations

import io
import json
from pathlib import Path

import typer

from .config import load_config

app = typer.Typer(help="Footage ingest pipeline: probe -> shots -> transcribe -> segment -> frames -> motion -> annotate -> selects -> assemble")

from .audiosync import app as audiosync_app  # noqa: E402
app.add_typer(audiosync_app, name="audiosync")

CONFIG_OPT = typer.Option("config.yaml", "--config", "-c")


@app.command()
def check(config: Path = CONFIG_OPT, audit: bool = typer.Option(False, "--audit", help="also test audit's native image route")):
    """Verify tooling: ffmpeg, LUT, vision model (sends a test image), whisper import.

    The vision test is a trap for silent image-dropping (Ollama MLX does this):
    the model must name the actual colors, not plausible ones.
    """
    import subprocess

    from PIL import Image, ImageDraw

    from .vlm import chat_vision, extract_json

    cfg = load_config(config)
    ok = True

    for tool in ("ffmpeg", "ffprobe"):
        found = subprocess.run(["which", tool], capture_output=True).returncode == 0
        print(f"{'ok ' if found else 'MISSING'} {tool}")
        ok &= found

    for profile in cfg.grade_profiles.values():
        if profile.lut is not None:
            found = profile.lut.is_file()
            print(f"{'ok ' if found else 'MISSING'} grade '{profile.name}': LUT {profile.lut}")
            ok &= found
        else:
            print(f"ok  grade '{profile.name}': filters `{profile.filters}`")

    img = Image.new("RGB", (1024, 1024), (200, 60, 30))
    draw = ImageDraw.Draw(img)
    draw.ellipse([300, 300, 724, 724], fill=(240, 220, 40))
    draw.rectangle([50, 50, 250, 250], fill=(30, 90, 200))
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    try:
        raw = chat_vision(
            cfg.vision_url, cfg.vision_model,
            "Answer with only a JSON object.",
            'Respond with JSON {"shapes": [{"shape": ..., "color": ...}], "background_color": ...}',
            [buf.getvalue()],
        )
        answer = extract_json(raw)
        # Both shape colors named correctly is conclusive that images arrive;
        # background may come back as a hex code, so don't string-match it.
        text = json.dumps(answer).lower()
        vision_ok = "yellow" in text and "blue" in text
        print(f"{'ok ' if vision_ok else 'SUSPECT (images may not be reaching the model)'} "
              f"vision model {cfg.vision_model} -> {answer}")
        ok &= vision_ok
    except Exception as e:
        print(f"FAILED vision model {cfg.vision_model} at {cfg.vision_url}: {e}\n"
              f"  is LM Studio serving? `lms load {cfg.vision_model} && lms server start`")
        ok = False

    from .vlm import MIN_CONTEXT_TOKENS, loaded_context_length
    context = loaded_context_length(cfg.vision_url, cfg.vision_model)
    if context is None:
        print("??  vision context length not reported by the server (not LM Studio, or model not loaded)")
    elif context < MIN_CONTEXT_TOKENS:
        print(f"TOO SMALL vision context {context} tokens: eight frames + prompt leave no room for an "
              f"answer and busy shots fail with finish_reason=length.\n"
              f"  reload with `lms unload --all && lms load {cfg.vision_model} "
              f"--context-length {MIN_CONTEXT_TOKENS}`")
        ok = False
    else:
        print(f"ok  vision context {context} tokens")

    if audit:
        from .evidence import check_native
        try:
            native = check_native(cfg, buf.getvalue())
            native_ok = not native["errors"]
            print(f"{'ok ' if native_ok else 'FAILED'} native audit vision -> "
                  f"{native['errors'] if native['errors'] else native['parsed']}")
            ok &= native_ok
        except Exception as exc:
            print(f"FAILED native audit vision: {exc}")
            ok = False

    try:
        import mlx_whisper  # noqa: F401
        print("ok  mlx-whisper importable")
    except ImportError as e:
        print(f"MISSING mlx-whisper: {e}")
        ok = False

    raise typer.Exit(0 if ok else 1)


@app.command()
def probe(config: Path = CONFIG_OPT):
    """Inventory all source files (duration, fps, timecode)."""
    from . import probe as stage
    cfg = load_config(config)
    inv = stage.run(cfg)
    print(f"{len(inv['aroll'])} a-roll files, {len(inv['broll'])} b-roll files -> out/inventory.json")


@app.command()
def shots(config: Path = CONFIG_OPT):
    """Split b-roll files into shots via scene detection."""
    from . import shots as stage
    cfg = load_config(config)
    result = stage.run(cfg)
    broll = [s for s in result if s["role"] == "broll"]
    print(f"{len(broll)} b-roll shots -> out/shots.json (verify boundaries on contact sheets)")


@app.command()
def segment(
    config: Path = CONFIG_OPT,
    force: bool = typer.Option(False, "--force", help="recompute windows, including takes already windowed"),
):
    """Window long takes into searchable moments -> out/segments.json.

    Short takes are untouched. A window is an index into a continuous take —
    a pause in the transcript where there is one, an even division where there
    is not — never a claimed cut.
    """
    from . import segments as stage
    cfg = load_config(config)
    before = {s["shot_id"] for s in stage.load(cfg)}
    result = stage.build(cfg, force=force)
    sources = {s["parent_shot_id"] for s in result}
    tagged = [s for s in result if s["visual_tag"]]
    print(f"{len(result)} windows across {len(sources)} long takes -> out/segments.json "
          f"({len(tagged)} to annotate, {len(result) - len(tagged)} indexed for search only)")
    orphaned = before - {s["shot_id"] for s in result}
    if orphaned:
        print(f"WARNING: {len(orphaned)} previous windows no longer exist; their frames and "
              f"annotations are now orphaned: {', '.join(sorted(orphaned))}")


@app.command()
def frames(config: Path = CONFIG_OPT, only: str = typer.Option(None, "--only", help="single shot_id or segment id")):
    """Extract LUT-graded frames + contact sheets per shot and segment."""
    from . import frames as stage
    stage.run(load_config(config), only=only)
    print("frames + contact sheets done -> work/frames, work/sheets")


@app.command()
def motion(config: Path = CONFIG_OPT):
    """Compute optical-flow motion stats per b-roll shot and b-roll window."""
    from . import motion as stage
    stage.run(load_config(config))
    print("motion stats -> out/motion.json")


@app.command()
def transcribe(config: Path = CONFIG_OPT, force: bool = typer.Option(False, "--force"),
               broll: bool = typer.Option(False, "--broll", help="camera audio of the B-roll instead (forced English, Whisper confidence kept; a-roll entries untouched)"),
               only: str = typer.Option(None, "--only", help="one source id (with --broll)")):
    """Transcribe a-roll with word-level timestamps (mlx-whisper); --broll does the B-roll camera audio."""
    from . import transcribe as stage
    cfg = load_config(config)
    if broll:
        stage.run_broll(cfg, force=force, only=only)
    else:
        stage.run(cfg, force=force)
    print("transcripts -> out/transcripts.json")


@app.command()
def annotate(
    config: Path = CONFIG_OPT,
    force: bool = typer.Option(False, "--force", help="re-annotate everything"),
    only: str = typer.Option(None, "--only", help="single shot_id"),
):
    """Annotate every shot and window with the local VLM."""
    from . import annotate as stage
    stage.run(load_config(config), force=force, only=only)


@app.command()
def audit(
    config: Path = CONFIG_OPT,
    only: str = typer.Option(..., "--only", help="one exact shot_id; required to bound local-model cost"),
    force: bool = typer.Option(False, "--force", help="redo observations and checker; retain attempt history"),
):
    """Opt-in frame evidence + text consistency audit; never rewrites tags.

    Run `check` first. Exit 0=no issue detected, 1=unchecked/error,
    2=needs review. No status means footage has been verified.
    """
    from . import evidence
    try:
        result = evidence.run(load_config(config), only=only, force=force)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        typer.echo(f"Audit failed: {exc}", err=True)
        raise typer.Exit(1)
    raise typer.Exit({"no_issue_detected": 0, "unchecked": 1, "needs_review": 2}[result["status"]])


@app.command()
def selects(config: Path = CONFIG_OPT):
    """Extract highlighted passages from the transcript .docx."""
    from . import selects as stage
    stage.run(load_config(config))


@app.command()
def assemble(config: Path = CONFIG_OPT):
    """Assemble all artifacts into out/bible-context.md."""
    from . import bible as stage
    stage.run(load_config(config))


@app.command("all")
def run_all(config: Path = CONFIG_OPT):
    """Run the full pipeline in order (idempotent stages; safe to re-run)."""
    from . import annotate as annotate_stage
    from . import bible as bible_stage
    from . import frames as frames_stage
    from . import motion as motion_stage
    from . import probe as probe_stage
    from . import segments as segments_stage
    from . import selects as selects_stage
    from . import shots as shots_stage
    from . import transcribe as transcribe_stage

    cfg = load_config(config)
    probe_stage.run(cfg)
    shots_stage.run(cfg)
    # transcribe moved ahead of frames: a window boundary prefers a real pause
    # in the transcript, so the transcript has to exist before takes are
    # windowed and windows are what frames are then extracted for.
    transcribe_stage.run(cfg)
    segments_stage.build(cfg)
    frames_stage.run(cfg)
    motion_stage.run(cfg)
    annotate_stage.run(cfg)
    selects_stage.run(cfg)
    bible_stage.run(cfg)


if __name__ == "__main__":
    app()
