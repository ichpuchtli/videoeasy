"""Which local text model the eval tools call, in one place.

The proposer (brollmatch), the viewer read (cuteval) and the story check
(storycheck) all use one Ollama text model. The default is the model the loop
was tuned with; another machine may have a different one pulled. The
environment overrides it:

  VIDEOEASY_TEXT_MODEL   e.g. qwen3:32b
  VIDEOEASY_TEXT_URL     e.g. http://localhost:11434

Resolved when a tool starts, and recorded in every manifest that tool writes,
so a score always names the model that produced it. Prompts and thresholds
were tuned on the default: a different model is a new judge, not the same one.
"""
from __future__ import annotations

import os

DEFAULT_TEXT_MODEL = "qwen3.8:27b-mtp-q8_0"
DEFAULT_TEXT_URL = "http://localhost:11434"


def text_model() -> str:
    return os.environ.get("VIDEOEASY_TEXT_MODEL", "").strip() or DEFAULT_TEXT_MODEL


def text_url() -> str:
    return (os.environ.get("VIDEOEASY_TEXT_URL", "").strip() or DEFAULT_TEXT_URL).rstrip("/")
