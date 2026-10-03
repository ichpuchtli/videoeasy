"""Vision-model client: OpenAI-compatible chat API (LM Studio).

Ollama 0.31's MLX backend silently drops images (verified July 2026: prompt
token counts stay text-only and the model hallucinates), so the pipeline
talks to LM Studio instead. Any OpenAI-compatible server works — change
models.vision_url / models.vision in config.yaml.
"""
from __future__ import annotations

import base64
import json
import re

import httpx


def chat_vision(
    url: str,
    model: str,
    system: str,
    user_text: str,
    images: list[bytes],
    # Thinking models (GLM-4.6V) spend most of the budget on reasoning before
    # any visible content appears — keep this generous.
    max_tokens: int = 6000,
    temperature: float = 0.3,
    timeout: float = 600.0,
) -> str:
    content: list[dict] = [{"type": "text", "text": user_text}]
    content += [
        {"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(img).decode()}}
        for img in images
    ]
    # Thinking models occasionally ruminate through any budget on a hard
    # image; retry once with double before giving up.
    for attempt_max_tokens in (max_tokens, max_tokens * 2):
        response = httpx.post(
            f"{url}/chat/completions",
            json={
                "model": model,
                "temperature": temperature,
                "max_tokens": attempt_max_tokens,
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": content},
                ],
            },
            timeout=timeout,
        )
        response.raise_for_status()
        choice = response.json()["choices"][0]
        text = choice["message"]["content"] or ""
        # A length-cut answer is truncated JSON — as useless as no answer.
        if choice.get("finish_reason") != "length":
            return text
    raise RuntimeError(
        f"model hit the token budget before finishing an answer, "
        f"even at max_tokens={attempt_max_tokens}"
    )


def extract_json(text: str) -> dict:
    """Parse JSON out of model output that may carry GLM box tokens,
    code fences, or leading prose."""
    cleaned = re.sub(r"<\|[^|]+\|>", "", text)
    cleaned = re.sub(r"```(?:json)?", "", cleaned).strip()
    start = cleaned.find("{")
    if start == -1:
        raise json.JSONDecodeError("no JSON object in response", cleaned, 0)
    decoder = json.JSONDecoder()
    obj, _ = decoder.raw_decode(cleaned[start:])
    return obj


# Eight graded frames plus the annotation prompt use most of a 4096-token
# window on their own, so the model's answer is cut off (finish_reason=length)
# no matter how large max_tokens is. LM Studio loads at 4096 by default; the
# busiest shots of one film's batch (Sept 2026) failed for exactly this.
MIN_CONTEXT_TOKENS = 16384


def loaded_context_length(url: str, model: str, timeout: float = 10.0) -> int | None:
    """Context window the model is currently loaded with, or None if the
    server does not expose LM Studio's /api/v0/models endpoint."""
    base = url.rsplit("/v1", 1)[0]
    try:
        response = httpx.get(f"{base}/api/v0/models", timeout=timeout)
        response.raise_for_status()
    except httpx.HTTPError:
        return None
    for entry in response.json().get("data", []):
        if entry.get("id") == model:
            return entry.get("loaded_context_length")
    return None
