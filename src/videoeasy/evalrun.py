"""Provenance and status helpers for the cut-evaluation loop.

Every report written by cuteval, brollmatch, cutbuild and storycheck carries a
manifest: content hashes of its inputs, the prompt and model behind every
judgment, the source version of the tool, and what was checked versus left
unchecked. Caches are keyed on these hashes, never on file existence.

Status vocabulary shared by the four tools:

  ok         the check ran and its result is usable
  unchecked  the check did not run or failed to run (missing file, ffmpeg
             error, model timeout, truncated response): no score, no finding,
             and never "no fault"
  invalid    the check ran but the response is unusable (bad JSON, score out
             of range, relation and score disagree): no score

A result with any status other than ok carries no score and cannot make a
candidate eligible, a stretch accepted, or a report claim improvement.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import subprocess
from pathlib import Path

OK = "ok"
UNCHECKED = "unchecked"
INVALID = "invalid"

SIDECAR_MIN_BYTES = 50_000_000  # large files are hashed once; the digest is cached (see fingerprint_cache_dir)


def fingerprint_cache_dir() -> Path:
    """Where large-file digests are cached: $VIDEOEASY_CACHE/fingerprints, else ~/.cache/videoeasy/fingerprints.

    Never beside the file. The cache once lived in a `<name>.sha256` next to each file, and steadiness and moves
    fingerprint SOURCE footage: a film whose inputs were symlinks got 123 sidecars beside the links; one whose
    config points straight at the footage (or at a synced cloud folder) would have had them written into the
    originals' folder and uploaded with it. Originals are never written."""
    root = os.environ.get("VIDEOEASY_CACHE")
    return (Path(root) if root else Path.home() / ".cache" / "videoeasy") / "fingerprints"


def fingerprint(path: str | Path) -> str:
    """First 16 hex of the sha256 of the file's content.

    Large files keep their digest in the fingerprint cache (keyed on the resolved path; trusted only while size
    and mtime still match), so a 1.6 GB render or a source take is hashed once per version. A legacy
    `<name>.sha256` beside the file is still read, never written."""
    path = Path(path)
    st = path.stat()
    large = st.st_size >= SIDECAR_MIN_BYTES
    resolved = str(path.resolve())
    cached = fingerprint_cache_dir() / (hashlib.sha256(resolved.encode("utf-8")).hexdigest()[:32] + ".json")
    if large:
        for side in (cached, path.with_name(path.name + ".sha256")):
            try:
                c = json.loads(side.read_text())
                if c.get("size") == st.st_size and c.get("mtime") == st.st_mtime:
                    return c["sha256"][:16]
            except (OSError, ValueError, KeyError):
                pass
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    digest = h.hexdigest()
    if large:
        try:
            cached.parent.mkdir(parents=True, exist_ok=True)
            cached.write_text(json.dumps({"path": resolved, "size": st.st_size, "mtime": st.st_mtime, "sha256": digest}))
        except OSError:
            pass
    return digest[:16]


def text_hash(*parts: str) -> str:
    return hashlib.sha256("\n\x00\n".join(parts).encode("utf-8")).hexdigest()[:16]


def bytes_hash(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()[:16]


def source_hash(module_file: str) -> str:
    return fingerprint(Path(module_file))


def ffprobe_duration(path: str | Path) -> float | None:
    """Container duration in seconds, or None when ffprobe fails or the file is missing."""
    if not Path(path).exists():
        return None
    p = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(path)],
                       capture_output=True, text=True)
    if p.returncode != 0:
        return None
    try:
        return float(p.stdout.strip())
    except ValueError:
        return None


def refuse_overwrite(paths: list[Path], overwrite: bool) -> None:
    """A prior run is evidence; it is replaced only when asked."""
    existing = [str(p) for p in paths if Path(p).exists()]
    if existing and not overwrite:
        raise SystemExit("refusing to overwrite a previous run (pass --overwrite): " + ", ".join(existing))


def write_manifest(stem: Path, **fields) -> dict:
    """Write `<stem>.manifest.json` and return it. `stem` is the report path without suffix."""
    m = dict(created=dt.datetime.now().astimezone().isoformat(timespec="seconds"), **fields)
    Path(str(stem) + ".manifest.json").write_text(json.dumps(m, indent=1, default=str))
    return m
