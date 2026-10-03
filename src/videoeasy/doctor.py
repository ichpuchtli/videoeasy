"""Can this machine run videoeasy? One line per requirement, each with its fix.

Needs no film and no config: it is the first thing to run on a new machine.
Required items fail the run; optional ones (Ollama for the eval loop's text
judge, DaVinci Resolve for the lay-in and bins) only warn. Every network probe
has a 2 s timeout, so a server that is down reads as down rather than as a
hang.

Two failure modes this exists to catch early, both learned the hard way:
LM Studio serving the vision model with its default 4096-token window (the
busiest shots then fail with finish_reason=length however high max_tokens is),
and a vision backend that is up but silently drops images (that one needs a
real image round trip: `videoeasy check`, which this does not replace).
"""
from __future__ import annotations

import importlib.util
import os
import platform
import shutil
import subprocess
import sys
import threading
from dataclasses import dataclass
from pathlib import Path

import httpx
import yaml

OK, MISSING, WARN = "ok", "MISSING", "WARN"
TIMEOUT_S = 2.0
VISION_MODEL = "google/gemma-4-26b-a4b"
VISION_URL = "http://localhost:1234/v1"
WHISPER_REPO = "mlx-community/whisper-large-v3-mlx"
MIN_FREE_GB = 100   # one film's frames, analysis caches and renders reached ~90 GB


@dataclass
class Check:
    status: str
    name: str
    detail: str
    fix: str = ""
    required: bool = True


class Env:
    """The outside world, one method per question; tests replace it."""

    def platform(self) -> tuple[str, str]:
        return sys.platform, platform.machine()

    def python(self) -> tuple[int, int, int]:
        return sys.version_info[:3]

    def which(self, tool: str) -> str | None:
        return shutil.which(tool)

    def first_line(self, argv: list[str]) -> str | None:
        try:
            p = subprocess.run(argv, capture_output=True, text=True, timeout=5)
        except (OSError, subprocess.TimeoutExpired):
            return None
        out = (p.stdout or p.stderr).strip().splitlines()
        return out[0] if p.returncode == 0 and out else None

    def get_json(self, url: str) -> dict | None:
        try:
            r = httpx.get(url, timeout=TIMEOUT_S)
            r.raise_for_status()
            return r.json()
        except (httpx.HTTPError, ValueError):
            return None

    def context_length(self, url: str, model: str) -> int | None:
        from .vlm import loaded_context_length
        return loaded_context_length(url, model, timeout=TIMEOUT_S)

    def find_spec(self, name: str) -> bool:
        return importlib.util.find_spec(name) is not None

    def exists(self, path: str | Path) -> bool:
        return Path(path).exists()

    def disk_free(self, path: str | Path) -> int:
        return shutil.disk_usage(path).free

    def resolve_answers(self) -> bool | None:
        """True when Resolve answers the scripting API, False when it does not, None when it did not reply in time."""
        result: list[bool] = []

        def probe():
            try:
                from .resolve import get_resolve
                get_resolve()
                result.append(True)
            except Exception:  # noqa: BLE001 - any failure means "not answering"
                result.append(False)

        t = threading.Thread(target=probe, daemon=True)
        t.start()
        t.join(TIMEOUT_S)
        return result[0] if result else None


def lms_path(env: Env) -> str | None:
    return env.which("lms") or next((p for p in [str(Path.home() / ".lmstudio/bin/lms")] if env.exists(p)), None)


def checks(env: Env, vision_model: str = VISION_MODEL, vision_url: str = VISION_URL, ollama_url: str = "http://localhost:11434",
           ollama_models: tuple[str, ...] = (), repo_root: Path | None = None) -> list[Check]:
    from .resolve import _API, _LIB
    from .vlm import MIN_CONTEXT_TOKENS
    out: list[Check] = []
    add = out.append

    plat, machine = env.platform()
    if plat == "darwin" and machine == "arm64":
        add(Check(OK, "Apple Silicon macOS", f"{plat} {machine}"))
    else:
        add(Check(MISSING, "Apple Silicon macOS", f"{plat} {machine}: mlx-whisper (transcription) runs only on Apple Silicon",
                  "run videoeasy on an M-series Mac"))
    py = env.python()
    add(Check(OK if py >= (3, 11) else MISSING, "Python", ".".join(map(str, py)), "" if py >= (3, 11) else "uv python install 3.12"))

    for tool in ("ffmpeg", "ffprobe"):
        if env.which(tool):
            add(Check(OK, tool, env.first_line([tool, "-version"]) or "found"))
        else:
            add(Check(MISSING, tool, "not on PATH", "brew install ffmpeg"))
    add(Check(OK, "uv", env.first_line(["uv", "--version"]) or "found") if env.which("uv")
        else Check(MISSING, "uv", "not on PATH", "brew install uv"))

    lms = lms_path(env)
    add(Check(OK, "lms (LM Studio CLI)", lms) if lms else
        Check(WARN, "lms (LM Studio CLI)", "not found", "open LM Studio once, then run ~/.lmstudio/bin/lms bootstrap", required=False))

    base = vision_url.rsplit("/v1", 1)[0]
    models = env.get_json(f"{base}/api/v0/models")
    if models is None:
        add(Check(MISSING, "LM Studio server", f"no answer at {base}", "open LM Studio, then: lms server start"))
    else:
        loaded = [m.get("id") for m in models.get("data", []) if m.get("state") == "loaded"]
        add(Check(OK, "LM Studio server", f"{base}; loaded: {', '.join(loaded) or 'none'}"))
        load_fix = f"lms load {vision_model} --context-length {MIN_CONTEXT_TOKENS}"
        if vision_model not in loaded:
            add(Check(MISSING, "vision model", f"{vision_model} is not loaded", load_fix))
        else:
            ctx = env.context_length(vision_url, vision_model)
            if ctx is None:
                add(Check(WARN, "vision model", f"{vision_model} loaded; context length not reported", load_fix, required=False))
            elif ctx < MIN_CONTEXT_TOKENS:
                add(Check(MISSING, "vision model", f"{vision_model} loaded with a {ctx}-token window: busy shots will come back "
                          f"finish_reason=length", f"lms unload {vision_model} && {load_fix}"))
            else:
                add(Check(OK, "vision model", f"{vision_model}, {ctx}-token window (images: confirm with `videoeasy check`)"))

    tags = env.get_json(f"{ollama_url}/api/tags")
    if tags is None:
        add(Check(WARN, "Ollama", f"no answer at {ollama_url} (the eval loop's text judge and proposer need it)",
                  "install Ollama, then: ollama serve", required=False))
    else:
        have = {m.get("name") for m in tags.get("models", [])}
        add(Check(OK, "Ollama", f"{ollama_url}; {len(have)} models"))
        for m in ollama_models:
            add(Check(OK, f"Ollama model {m}", "pulled") if m in have else
                Check(WARN, f"Ollama model {m}", "not pulled (eval loop text judge; VIDEOEASY_TEXT_MODEL picks another)", f"ollama pull {m}", required=False))

    if env.find_spec("mlx_whisper"):
        add(Check(OK, "mlx-whisper", "importable"))
    else:
        add(Check(MISSING, "mlx-whisper", "not importable in this environment", "uv sync"))
    hub = Path.home() / ".cache/huggingface/hub" / ("models--" + WHISPER_REPO.replace("/", "--"))
    add(Check(OK, "whisper weights", WHISPER_REPO) if env.exists(hub) else
        Check(WARN, "whisper weights", f"{WHISPER_REPO} not cached: the first transcription downloads several GB",
              "", required=False))

    module = os.path.join(_API, "Modules", "DaVinciResolveScript.py")
    if not (env.exists(module) and env.exists(_LIB)):
        add(Check(WARN, "DaVinci Resolve scripting", "scripting module not found at the standard install path "
                  "(only the lay-in and bins need it)", "install DaVinci Resolve Studio", required=False))
    else:
        answers = env.resolve_answers()
        if answers:
            add(Check(OK, "DaVinci Resolve", "answers the scripting API", required=False))
        else:
            add(Check(WARN, "DaVinci Resolve", "installed, not answering" + (" (timed out)" if answers is None else ""),
                      "start Resolve; Preferences → System → General → External scripting using: Local", required=False))

    root = repo_root or Path.cwd()
    try:
        free_gb = env.disk_free(root) / 1e9
        add(Check(OK if free_gb >= MIN_FREE_GB else WARN, "free disk", f"{free_gb:.0f} GB on {root}",
                  "" if free_gb >= MIN_FREE_GB else f"frames, caches and renders for one film reached ~90 GB; free space or "
                  "point work_dir at a bigger volume", required=False))
    except OSError as e:
        add(Check(WARN, "free disk", f"unknown: {e}", required=False))
    return out


def report(results: list[Check]) -> tuple[str, int]:
    """The printable report and the exit code: 0 only when no required check is missing."""
    lines = []
    for c in results:
        lines.append(f"{c.status:<8} {c.name:<26} {c.detail}")
        if c.fix and c.status != OK:
            lines.append(f"{'':<8} {'':<26} fix: {c.fix}")
    failed = [c for c in results if c.required and c.status == MISSING]
    lines.append("")
    lines.append(f"{len(failed)} required item(s) missing" if failed else "all required items present"
                 + ("; see the warnings above" if any(c.status == WARN for c in results) else ""))
    return "\n".join(lines), (1 if failed else 0)


def vision_from_config(config: Path | None) -> tuple[str, str]:
    """Vision model and URL from a config file, read without load_config (which creates work directories)."""
    if config and Path(config).is_file():
        raw = yaml.safe_load(Path(config).read_text()) or {}
        m = raw.get("models") or {}
        return m.get("vision", VISION_MODEL), str(m.get("vision_url", VISION_URL)).rstrip("/")
    return VISION_MODEL, VISION_URL


def main(config: Path | None = None, env: Env | None = None) -> int:
    from .textmodel import text_model, text_url
    model, url = vision_from_config(config)
    text, code = report(checks(env or Env(), model, url, ollama_url=text_url(), ollama_models=(text_model(),),
                               repo_root=Path.cwd()))
    print(text)
    return code
