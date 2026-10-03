"""Long talks on body-worn recorders -> beats, hooks, bites and closers, each with a cover brief.

A facilitator's lapel transmitter records the whole talk internally in 32-bit
float, split into contiguous chunks (DJI mics: ~30.7 min each). This tool
turns one talk into what an editor needs to build a 60-90 s feature or a
testimonial: the talk's structure (beats), the lines that could open a video
(hooks), passages that stand alone (bites), lines that land an ending
(closers), and for each a brief of what b-roll would complement the words.

Stages (each cached by the content hash of its inputs, one manifest per output):

  prepare      group contiguous chunks (refuse a gap), measure each chunk's
               sample peak (float may exceed 0 dBFS: overs are counted), its
               integrated loudness and true peak; build ONE 16 kHz mono s16
               working file with a single linear gain that puts the talk's peak
               at -1 dBFS. No compression, no clipping, originals never written.
               Every talk time maps back to (chunk file, chunk time).
  transcribe   mlx-whisper, word timestamps, no conditioning on previous text;
               segments past the no-speech / compression thresholds are dropped
               and recorded (Whisper invents speech in quiet passages).
  sentences    sentence units from word times and punctuation; MEASURED edge
               room before and after each from the working file's energy, and
               an off-mic flag for sentences far quieter than the wearer.
  beats        one local-LLM call over the compact transcript: contiguous,
               ordered beats. Validated in code; any violation -> invalid.
  bites        per beat, the LLM proposes hooks, bites and closers with a
               register reading of the WORDS and a cover brief; code gates them
               (ids, durations, off-mic, dangling starts, tight edges).
  verify       an independent call per candidate with ONLY its words:
               standalone 0-3 and strength 0-3. No verdict -> unchecked, never
               ranked, never given a fallback score.
  report       analysis.json, analysis.md and selects.csv (source chunk + time
               for every in and out, usable in any NLE).
  export-cut   ranked candidates as a cut JSON for brollmatch (one beat per
               candidate, its cover brief as the beat note) plus a transcript
               keyed by chunk.
  resolve      optional selects timeline in DaVinci Resolve (talks_resolve.py).

Usage:
    uv run python -m videoeasy.talks all --mic-dir /path/to/DJI_MIC --session 06 \\
        --project data/projects/<p> [--talk-id talk-01] [--speaker "Speaker A"] [--title "Opening talk"]
    uv run python -m videoeasy.talks all --audio a.WAV b.WAV --out data/projects/<p>/talks/talk-01
    uv run python -m videoeasy.talks report --out <dir>       # any single stage, from the cached earlier ones
    uv run python -m videoeasy.talks resolve --out <dir> --resolve-project "<project>" --timeline "talk-01 selects" [--apply]

Every threshold here is provisional until the editor has rated a sample
(docs/talks.md, "Calibration"). The register on a candidate is a reading of
the words only: no voice or face register is measured here.
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import math
import os
import re
import subprocess
import sys
import wave
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .evalrun import INVALID, OK, UNCHECKED, refuse_overwrite, source_hash, text_hash, write_manifest
from .textmodel import text_model, text_url
from .transcribe import COMPRESSION_MAX, NO_SPEECH_MAX

ANALYSIS_VERSION = "talks-v1"     # bump when a stage's rules change; every cache key carries it
EXCLUDED = "excluded"             # a candidate the measured gates keep out of ranking (listed, with the reason)

# ---- prepare
WORK_SR = 16000
TARGET_PEAK_DB = -1.0
CONTIGUITY_TOL_S = 2.0            # chunk names carry whole seconds; DJI chunks butt to within one
QUIET_GAIN_WARN_DB = 40.0
DEFAULT_NAME_PATTERN = r"^DJI_(?P<session>\d+)_(?P<date>\d{8})_(?P<time>\d{6})\.wav$"   # case-insensitive; Mic 3 naming unconfirmed
WHISPER_DEFAULT = "mlx-community/whisper-large-v3-mlx"

# ---- sentences (all provisional)
ENV_HZ = 100
SENTENCE_PAUSE_S = 0.6
SENTENCE_MAX_S = 30.0
# Quiet is judged against the recording's own noise floor, not against speech: on a field lav the first rule
# tried (quiet = 20 dB under the median speech level) put the threshold at -67 dBFS, under a -58 dBFS floor of
# wind and birds, and measured 0 s of room at every one of 33 sentence edges of a real 60 s excerpt.
NOISE_PCT = 10                    # noise floor = this percentile of all 10 ms frames
SPEECH_PCT = 90                   # a stretch's speech level = this percentile of its word frames (word spans carry quiet frames)
QUIET_ABOVE_FLOOR_DB = 6.0        # quiet = within this much of the noise floor
MIN_SNR_DB = 12.0                 # speech less than this far over the floor: edge room is unmeasurable, said so
OFF_MIC_BELOW_DB = 10.0           # a sentence this far under the wearer's typical sentence level is probably another voice
ROOM_CAP_S = 2.0                  # edge room is measured up to this much; more is reported as the cap
TIGHT_EDGE_S = 0.15
HANDLE_S = 0.3                    # selects in/out: word times plus up to this much measured room

# ---- candidates
KINDS = ("hook", "bite", "closer")
DURATIONS = {"hook": (3.0, 12.0), "bite": (6.0, 30.0), "closer": (3.0, 15.0)}
DANGLING = {"and", "but", "so", "because", "cause", "which", "that", "that's", "this", "it", "it's", "they", "they're",
            "he", "she", "also", "then", "or", "plus", "anyway", "like", "these", "those", "there", "as"}
MAX_PER_BEAT = 6

# ---- local text model
NUM_CTX = 40000
CTX_RESERVE_TOKENS = 9000         # prompt scaffolding + the answer
CHARS_PER_TOKEN = 3.5
WINDOW_OVERLAP_SENTENCES = 15
TEMPERATURE = 0.2


class TalkError(SystemExit):
    """A refusal with a reason (gap between chunks, ambiguous session, nothing to do)."""


# ===================================================================== helpers
def fmt_t(s: float) -> str:
    s = max(0.0, float(s))
    h, rem = divmod(int(s), 3600)
    m, sec = divmod(rem, 60)
    return f"{h}:{m:02d}:{sec:02d}" if h else f"{m:02d}:{sec:02d}"


def db(x: float) -> float:
    return 20.0 * math.log10(x) if x > 0 else float("-inf")


def file_sha(path: Path, cache: dict) -> str:
    """sha256[:16] of a file, cached by (size, mtime) in the OUTPUT dir's cache: evalrun.fingerprint writes a
    sidecar beside large files, and the originals' folder is never written."""
    st = path.stat()
    k = str(path.resolve())
    c = cache.get(k)
    if c and c.get("size") == st.st_size and c.get("mtime") == st.st_mtime:
        return c["sha"]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    cache[k] = dict(size=st.st_size, mtime=st.st_mtime, sha=h.hexdigest()[:16])
    return cache[k]["sha"]


def _read(path: Path) -> dict | None:
    return json.loads(path.read_text()) if path.exists() else None


def _write(path: Path, doc: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=1, default=str))


def cached(path: Path, key: str, overwrite: bool) -> dict | None:
    """The stage's previous output when its key still matches. A run that came back `unchecked` (a model or
    tool that did not answer) produced no evidence and is simply retried; any other previous run with a
    different key is evidence and is replaced only with --overwrite. Unchanged inputs never redo work, except
    that --overwrite asks again where the last answer was `invalid`."""
    doc = _read(path)
    if doc is None or doc.get("status") == UNCHECKED:
        return None
    if doc.get("key") == key:
        return None if (overwrite and doc.get("status") == INVALID) else doc
    refuse_overwrite([path], overwrite)
    return None


# ===================================================================== chunks
@dataclass
class Chunk:
    path: str
    name: str
    session: str | None = None
    start: dt.datetime | None = None
    duration_s: float | None = None
    channels: int | None = None
    sample_rate: int | None = None
    extra: dict = field(default_factory=dict)


def parse_name(name: str, pattern: str = DEFAULT_NAME_PATTERN) -> tuple[str, dt.datetime] | None:
    m = re.search(pattern, name, re.IGNORECASE)
    if not m:
        return None
    try:
        return m.group("session"), dt.datetime.strptime(m.group("date") + m.group("time"), "%Y%m%d%H%M%S")
    except (IndexError, ValueError):
        return None


def probe_audio(path: str) -> dict:
    p = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type,codec_name,sample_fmt,sample_rate,channels",
                        "-of", "json", path], capture_output=True, text=True)
    if p.returncode != 0:
        raise TalkError(f"ffprobe failed on {path}: {p.stderr.strip()[:200]}")
    info = json.loads(p.stdout)
    a = next((s for s in info.get("streams", []) if s.get("codec_type") == "audio"), None)
    if a is None:
        raise TalkError(f"{path}: no audio stream")
    return dict(duration_s=float(info["format"]["duration"]), channels=int(a.get("channels") or 1),
                sample_rate=int(a.get("sample_rate") or 0), codec=a.get("codec_name"), sample_fmt=a.get("sample_fmt"))


def group_contiguous(chunks: list[Chunk], tol: float = CONTIGUITY_TOL_S, assume_contiguous: bool = False) -> dict:
    """Order chunks by the start time in their names and prove each one starts where the last ended.
    A gap or an overlap beyond `tol` is refused, listing it: a talk with a hole would map every later
    in/out to the wrong place. Names without a start time are refused for more than one chunk unless the
    caller vouches for the order (`assume_contiguous`), which is recorded."""
    if not chunks:
        raise TalkError("no audio chunks")
    if any(c.start is None for c in chunks):
        if len(chunks) > 1 and not assume_contiguous:
            raise TalkError("chunk names carry no start time (pattern did not match): " + ", ".join(c.name for c in chunks if c.start is None)
                            + " -- pass --name-pattern, or --assume-contiguous to take the given order on trust")
        return dict(chunks=list(chunks), method="given order (unchecked)" if len(chunks) > 1 else "single chunk", gaps=[])
    ordered = sorted(chunks, key=lambda c: c.start)
    sessions = {c.session for c in ordered}
    if len(sessions) > 1:
        raise TalkError(f"chunks from more than one recorder session: {sorted(s or '?' for s in sessions)}")
    gaps, bad = [], []
    for a, b in zip(ordered, ordered[1:]):
        expected = a.start + dt.timedelta(seconds=a.duration_s or 0.0)
        gap = (b.start - expected).total_seconds()
        gaps.append(dict(after=a.name, before=b.name, gap_s=round(gap, 3)))
        if abs(gap) > tol:
            bad.append(f"{a.name} ends {expected:%H:%M:%S}, {b.name} starts {b.start:%H:%M:%S} ({gap:+.1f} s)")
    if bad:
        raise TalkError("chunks are not contiguous (a talk with a hole maps every later time to the wrong place): " + "; ".join(bad))
    return dict(chunks=ordered, method="start times in names", gaps=gaps)


def find_session(mic_dir: Path, session: str, pattern: str = DEFAULT_NAME_PATTERN) -> list[Chunk]:
    """Chunks of one recorder session under `mic_dir` (and its immediate sub-folders, e.g. one per
    transmitter). `session` is a chunk file name (that chunk and the later ones of its session) or a session
    number. The same number in two folders is refused: name the folder instead."""
    dirs = [mic_dir] + sorted(p for p in mic_dir.iterdir() if p.is_dir() and not p.name.startswith("."))
    found: dict[Path, list[Chunk]] = {}
    for d in dirs:
        for f in sorted(d.iterdir()):
            if not f.is_file() or f.name.startswith(".") or f.suffix.lower() != ".wav":
                continue
            parsed = parse_name(f.name, pattern)
            if parsed:
                found.setdefault(d, []).append(Chunk(path=str(f), name=f.name, session=parsed[0], start=parsed[1]))
    by_name = [(d, c) for d, cs in found.items() for c in cs if session in (c.name, Path(c.name).stem)]
    if by_name:
        d, first = by_name[0]
        return sorted([c for c in found[d] if c.session == first.session and c.start >= first.start], key=lambda c: c.start)
    want = session.lstrip("0") or "0"
    hits = {d: [c for c in cs if (c.session or "").lstrip("0") == want] for d, cs in found.items()}
    hits = {d: cs for d, cs in hits.items() if cs}
    if not hits:
        raise TalkError(f"no chunks of session {session!r} under {mic_dir} (pattern {pattern})")
    if len(hits) > 1:
        raise TalkError(f"session {session!r} exists in more than one folder: {', '.join(str(d) for d in hits)} -- pass --mic-dir <that folder>")
    return sorted(next(iter(hits.values())), key=lambda c: c.start)


# ===================================================================== prepare
def _pcm_stream(path: str, channel: int, rate: int | None = None, block: int = 1 << 20):
    """float32 blocks of one channel, at the file's own rate or `rate`; float decode keeps overs intact."""
    af = f"pan=mono|c0=c{channel - 1}" + (f",aresample={rate}" if rate else "")
    cmd = ["ffmpeg", "-v", "error", "-i", path, "-vn", "-af", af, "-f", "f32le", "-acodec", "pcm_f32le", "pipe:1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    carry = b""
    while True:
        buf = proc.stdout.read(block * 4)
        if not buf:
            break
        buf = carry + buf
        n = len(buf) // 4 * 4
        carry = buf[n:]
        if n:
            yield np.frombuffer(buf[:n], dtype=np.float32)
    err = proc.stderr.read().decode(errors="replace")
    if proc.wait() != 0:
        raise TalkError(f"ffmpeg decode failed on {path}: {err.strip()[:200]}")


def measure_chunk(path: str, channel: int) -> dict:
    """Sample peak of the float data (above 1.0 is an over, kept, never clipped here) and the over count."""
    peak, overs, n = 0.0, 0, 0
    for x in _pcm_stream(path, channel):
        if x.size:
            a = np.abs(x)
            peak = max(peak, float(a.max()))
            overs += int((a > 1.0).sum())
            n += x.size
    return dict(peak=peak, overs=overs, samples=n)


def loudness(path: str, channel: int) -> dict:
    """Integrated loudness and true peak (ffmpeg ebur128). Unmeasurable -> None with the reason."""
    p = subprocess.run(["ffmpeg", "-v", "info", "-nostats", "-i", path, "-vn", "-af", f"pan=mono|c0=c{channel - 1},ebur128=peak=true",
                        "-f", "null", "-"], capture_output=True, text=True)
    tail = p.stderr[p.stderr.rfind("Summary:"):] if "Summary:" in p.stderr else ""
    mi = re.search(r"I:\s+(-?[\d.]+|-inf)\s+LUFS", tail)
    mp = re.search(r"True peak:\s+Peak:\s+(-?[\d.]+|-inf)\s+dBFS", tail)
    if p.returncode != 0 or not mi:
        return dict(lufs=None, true_peak_dbtp=None, status=UNCHECKED, reason="ebur128 failed or printed no summary")
    f = lambda m: None if m is None or m.group(1) == "-inf" else float(m.group(1))  # noqa: E731
    return dict(lufs=f(mi), true_peak_dbtp=f(mp), status=OK)


def plan_gain(native_peak: float, resampled_peak: float, target_db: float = TARGET_PEAK_DB) -> tuple[float, str]:
    """One linear gain for the whole talk. The talk's sample peak goes to `target_db`; if resampling to the
    working rate raised a peak further (inter-sample overshoot), that peak decides instead, so the integer
    conversion can never clip. Returns (gain_db, reason)."""
    if native_peak <= 0 and resampled_peak <= 0:
        raise TalkError("the talk is digitally silent: nothing to analyse")
    ref = max(native_peak, resampled_peak)
    reason = "talk sample peak" if native_peak >= resampled_peak else "resampled peak (inter-sample overshoot) above the sample peak"
    return target_db - db(ref), reason


def write_working(chunks: list[dict], channel: int, gain_db: float, dest: Path, stream=_pcm_stream) -> dict:
    """Concatenate the chunks at WORK_SR with one gain into a s16 mono WAV. Returns per-chunk sample counts
    (the talk-time map is built from what was actually written) and the clip count, which must be 0."""
    g = 10 ** (gain_db / 20.0)
    counts, clipped, peak = [], 0, 0.0
    tmp = dest.with_suffix(".partial.wav")
    with wave.open(str(tmp), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(WORK_SR)
        for c in chunks:
            n = 0
            for x in stream(c["path"], channel, WORK_SR):
                y = x.astype(np.float64) * g
                a = np.abs(y)
                if a.size:
                    peak = max(peak, float(a.max()))
                clipped += int((a > 32767 / 32768).sum())
                w.writeframes(np.clip(np.round(y * 32768), -32768, 32767).astype("<i2").tobytes())
                n += x.size
            counts.append(n)
    tmp.replace(dest)
    return dict(samples=counts, clipped_samples=clipped, peak_dbfs=round(db(peak), 3) if peak > 0 else None)


def resampled_peak(path: str, channel: int, stream=_pcm_stream) -> float:
    peak = 0.0
    for x in stream(path, channel, WORK_SR):
        if x.size:
            peak = max(peak, float(np.abs(x).max()))
    return peak


def talk_to_source(t: float, chunks: list[dict]) -> tuple[dict, float]:
    """(chunk, seconds into that chunk's original file) for a talk time."""
    for c in chunks:
        if t < c["talk_t1"] or c is chunks[-1]:
            return c, max(0.0, t - c["talk_t0"])
    raise ValueError("no chunks")


def source_spans(t0: float, t1: float, chunks: list[dict]) -> list[dict]:
    """The original-file spans covering talk [t0, t1): one per chunk it touches."""
    out = []
    for c in chunks:
        lo, hi = max(t0, c["talk_t0"]), min(t1, c["talk_t1"])
        if hi - lo > 1e-3:
            out.append(dict(file=c["path"], name=c["name"], in_s=round(lo - c["talk_t0"], 3), out_s=round(hi - c["talk_t0"], 3)))
    return out


def prepare(chunks_in: list[Chunk], out: Path, channel: int = 1, assume_contiguous: bool = False, overwrite: bool = False,
            meta: dict | None = None) -> dict:
    hashes_path = out / "hashes.json"
    hashes = _read(hashes_path) or {}
    for c in chunks_in:
        info = probe_audio(c.path)
        c.duration_s, c.channels, c.sample_rate = info["duration_s"], info["channels"], info["sample_rate"]
        c.extra = dict(codec=info["codec"], sample_fmt=info["sample_fmt"])
        if channel > (c.channels or 1):
            raise TalkError(f"{c.name} has {c.channels} channel(s); --channel {channel} does not exist")
    grouped = group_contiguous(chunks_in, assume_contiguous=assume_contiguous)
    chunks = grouped["chunks"]
    shas = [file_sha(Path(c.path), hashes) for c in chunks]
    _write(hashes_path, hashes)
    key = text_hash(ANALYSIS_VERSION, "prepare", str(channel), str(TARGET_PEAK_DB), *shas)
    path = out / "prepare.json"
    work = out / "work.wav"
    prev = cached(path, key, overwrite)
    if prev is not None and work.exists():
        if meta and any(v and prev.get("meta", {}).get(k) != v for k, v in meta.items()):   # labels, not evidence
            prev["meta"] = {**prev.get("meta", {}), **{k: v for k, v in meta.items() if v}}
            _write(path, prev)
        return prev
    rows = []
    for c, sha in zip(chunks, shas):
        m = measure_chunk(c.path, channel)
        lo = loudness(c.path, channel)
        rows.append(dict(path=c.path, name=c.name, session=c.session, start=c.start.isoformat() if c.start else None,
                         duration_s=round(c.duration_s, 3), channels=c.channels, channel_used=channel, sample_rate=c.sample_rate, sha=sha,
                         **c.extra, peak_dbfs=round(db(m["peak"]), 3) if m["peak"] > 0 else None, peak_linear=m["peak"],
                         overs_samples=m["overs"], overs_s=round(m["overs"] / (c.sample_rate or 48000), 4), loudness=lo))
    native = max(r["peak_linear"] for r in rows)
    rpeak = max(resampled_peak(r["path"], channel) for r in rows)
    gain_db, reason = plan_gain(native, rpeak)
    written = write_working(rows, channel, gain_db, work)
    t = 0.0
    for r, n in zip(rows, written["samples"]):
        r["talk_t0"] = round(t, 4)
        t += n / WORK_SR
        r["talk_t1"] = round(t, 4)
    status, notes = OK, []
    if written["clipped_samples"]:
        status = INVALID
        notes.append(f"{written['clipped_samples']} samples would clip in the working file")
    if gain_db > QUIET_GAIN_WARN_DB:
        notes.append(f"very quiet recording: +{gain_db:.1f} dB to reach {TARGET_PEAK_DB} dBFS")
    doc = dict(key=key, status=status, notes=notes, version=ANALYSIS_VERSION, meta=meta or {}, contiguity=dict(method=grouped["method"], gaps=grouped["gaps"]),
               chunks=rows, talk_peak_dbfs=round(db(native), 3), resampled_peak_dbfs=round(db(rpeak), 3), overs_samples=sum(r["overs_samples"] for r in rows),
               gain_db=round(gain_db, 3), gain_reason=reason, target_peak_dbfs=TARGET_PEAK_DB,
               working=dict(path=str(work), sample_rate=WORK_SR, format="s16 mono", duration_s=round(t, 3), peak_dbfs=written["peak_dbfs"],
                            clipped_samples=written["clipped_samples"], sha=file_sha(work, hashes)))
    _write(hashes_path, hashes)
    _write(path, doc)
    write_manifest(out / "prepare", tool="videoeasy.talks", stage="prepare", version=ANALYSIS_VERSION, source=source_hash(__file__), key=key,
                   inputs=[dict(path=r["path"], sha=r["sha"]) for r in rows], checked=["contiguity", "sample peak and overs", "loudness", "working-file clipping"],
                   unchecked=[] if all(r["loudness"]["status"] == OK for r in rows) else ["loudness of some chunks"])
    return doc


# ===================================================================== transcribe
def _mlx_transcribe(path: str, model: str, language: str | None) -> dict:
    import mlx_whisper  # deferred: slow import, downloads the model on first use

    return mlx_whisper.transcribe(path, path_or_hf_repo=model, word_timestamps=True, condition_on_previous_text=False, language=language)


def whisper_model() -> str:
    return os.environ.get("VIDEOEASY_WHISPER_MODEL", "").strip() or WHISPER_DEFAULT


def split_segments(raw: dict) -> tuple[list[dict], list[dict]]:
    """Kept and dropped segments, Whisper's own confidence kept on each (thresholds from transcribe.py's B-roll pass)."""
    kept, dropped = [], []
    for seg in raw.get("segments", []):
        if not str(seg.get("text", "")).strip():
            continue
        q = dict(no_speech_prob=round(float(seg.get("no_speech_prob", 0.0)), 3), avg_logprob=round(float(seg.get("avg_logprob", 0.0)), 3),
                 compression_ratio=round(float(seg.get("compression_ratio", 0.0)), 3))
        row = dict(start=round(float(seg["start"]), 3), end=round(float(seg["end"]), 3), text=seg["text"].strip(), quality=q,
                   words=[dict(w=w["word"].strip(), s=round(float(w["start"]), 3), e=round(float(w["end"]), 3),
                               p=round(float(w.get("probability", 0.0)), 3)) for w in seg.get("words", []) if str(w.get("word", "")).strip()])
        (dropped if q["no_speech_prob"] > NO_SPEECH_MAX or q["compression_ratio"] > COMPRESSION_MAX else kept).append(row)
    return kept, dropped


def transcribe(out: Path, language: str | None = "en", overwrite: bool = False, transcriber=_mlx_transcribe) -> dict:
    prep = _need(out / "prepare.json", "prepare")
    model = whisper_model()
    key = text_hash(ANALYSIS_VERSION, "transcribe", prep["working"]["sha"], model, str(language))
    path = out / "transcript.json"
    prev = cached(path, key, overwrite)
    if prev is not None:
        return prev
    t0 = dt.datetime.now()
    try:
        raw = transcriber(prep["working"]["path"], model, language)
    except Exception as e:  # noqa: BLE001 - a failed pass is recorded, never a silent empty transcript
        doc = dict(key=key, status=UNCHECKED, error=f"{type(e).__name__}: {e}", model=model, language=language, segments=[], dropped=[])
        _write(path, doc)
        return doc
    kept, dropped = split_segments(raw)
    doc = dict(key=key, status=OK, model=model, language=language, detected_language=raw.get("language"), segments=kept, dropped=dropped,
               thresholds=dict(no_speech_max=NO_SPEECH_MAX, compression_max=COMPRESSION_MAX),
               elapsed_s=round((dt.datetime.now() - t0).total_seconds(), 1))
    _write(path, doc)
    write_manifest(out / "transcript", tool="videoeasy.talks", stage="transcribe", version=ANALYSIS_VERSION, source=source_hash(__file__), key=key,
                   inputs=dict(working=prep["working"]["sha"]), model=model, language=language,
                   checked=["word timestamps", "segment confidence thresholds"], unchecked=["words themselves (one ASR pass is a question, not a finding)"])
    return doc


# ===================================================================== sentences
TERMINAL = re.compile(r"[.?!…]['\")\]]*$")


def build_sentences(words: list[dict], pause_s: float = SENTENCE_PAUSE_S, max_s: float = SENTENCE_MAX_S) -> list[dict]:
    """Sentence units: a word ending in terminal punctuation, or a pause of `pause_s` or more, closes one.
    A run longer than `max_s` is split at its longest internal pause until it fits (or has no pause left)."""
    runs, cur = [], []
    for i, w in enumerate(words):
        cur.append(w)
        nxt = words[i + 1] if i + 1 < len(words) else None
        if nxt is None or TERMINAL.search(w["w"]) or nxt["s"] - w["e"] >= pause_s:
            runs.append(cur)
            cur = []

    def split(run):
        if run[-1]["e"] - run[0]["s"] <= max_s or len(run) < 2:
            return [run]
        gaps = [run[i + 1]["s"] - run[i]["e"] for i in range(len(run) - 1)]
        k = max(range(len(gaps)), key=lambda i: (gaps[i], -abs(i - len(gaps) / 2)))
        return split(run[:k + 1]) + split(run[k + 1:])

    out = []
    for run in (piece for r in runs for piece in split(r)):
        out.append(dict(t0=run[0]["s"], t1=run[-1]["e"], text=" ".join(w["w"] for w in run), words=run))
    for i, s in enumerate(out):
        s["id"] = f"S{i + 1:04d}"
        s["pause_before"] = round(s["t0"] - out[i - 1]["t1"], 3) if i else round(s["t0"], 3)
        s["pause_after"] = round(out[i + 1]["t0"] - s["t1"], 3) if i + 1 < len(out) else None
    return out


def envelope_db(work: Path, hz: int = ENV_HZ) -> np.ndarray:
    """RMS level per 1/hz s of the working file, in dBFS (floor -200)."""
    hop = WORK_SR // hz
    vals, carry = [], np.zeros(0, dtype=np.float64)
    with wave.open(str(work), "rb") as w:
        while True:
            b = w.readframes(hop * 4096)
            if not b:
                break
            x = np.concatenate([carry, np.frombuffer(b, dtype="<i2").astype(np.float64) / 32768.0])
            n = len(x) // hop * hop
            carry = x[n:]
            if n:
                vals.append(np.sqrt((x[:n].reshape(-1, hop) ** 2).mean(axis=1)))
    rms = np.concatenate(vals) if vals else np.zeros(0)
    return np.maximum(20 * np.log10(np.maximum(rms, 1e-10)), -200.0).astype(np.float32)


def _frames(w: dict, hz: int = ENV_HZ) -> range:
    return range(int(math.floor(w["s"] * hz)), max(int(math.ceil(w["e"] * hz)), int(math.floor(w["s"] * hz)) + 1))


def speech_level(env: np.ndarray, words: list[dict], pct: float = SPEECH_PCT) -> float | None:
    idx = [i for w in words for i in _frames(w) if 0 <= i < len(env)]
    return float(np.percentile(env[idx], pct)) if idx else None


def edge_room(env: np.ndarray, t0: float, t1: float, quiet_db: float, cap_s: float = ROOM_CAP_S, hz: int = ENV_HZ) -> tuple[float, float]:
    """Seconds of measured quiet immediately before t0 and after t1 (up to cap_s). A loud frame right at the
    edge gives 0: the transcript's word time is not the cut point, the energy is."""
    cap = int(round(cap_s * hz))
    i, n = int(math.floor(t0 * hz)) - 1, 0
    while i >= 0 and n < cap and env[i] < quiet_db:
        n, i = n + 1, i - 1
    head = n / hz
    j, m = int(math.ceil(t1 * hz)), 0
    while j < len(env) and m < cap and env[j] < quiet_db:
        m, j = m + 1, j + 1
    return round(head, 2), round(m / hz, 2)


def measure_sentences(sentences: list[dict], env: np.ndarray) -> dict:
    """Edge room, level and the off-mic flag on every sentence; returns the levels used.
    Room is None (not 0) when the speech stands too little above the noise floor to tell a gap from a word."""
    all_words = [w for s in sentences for w in s["words"]]
    speech = speech_level(env, all_words)
    if speech is None or not len(env):
        for s in sentences:
            s.update(head_room=None, tail_room=None, level_db=None, off_mic=False)
        return dict(noise_floor_db=None, speech_db=None, wearer_db=None, quiet_db=None, snr_db=None, room_status=UNCHECKED, status=UNCHECKED,
                    reason="no words kept by the transcript")
    floor = float(np.percentile(env, NOISE_PCT))
    for s in sentences:
        lvl = speech_level(env, s["words"])
        s["level_db"] = None if lvl is None else round(lvl, 1)
    levels = [s["level_db"] for s in sentences if s["level_db"] is not None]
    wearer = float(np.median(levels))
    quiet = floor + QUIET_ABOVE_FLOOR_DB
    room_ok = speech - floor >= MIN_SNR_DB
    for s in sentences:
        s["head_room"], s["tail_room"] = edge_room(env, s["t0"], s["t1"], quiet) if room_ok else (None, None)
        s["off_mic"] = s["level_db"] is not None and s["level_db"] <= wearer - OFF_MIC_BELOW_DB
    return dict(noise_floor_db=round(floor, 1), speech_db=round(speech, 1), wearer_db=round(wearer, 1), quiet_db=round(quiet, 1),
                snr_db=round(speech - floor, 1), room_status=OK if room_ok else UNCHECKED, status=OK,
                reason=None if room_ok else f"speech only {speech - floor:.1f} dB over the noise floor: edge room unmeasurable")


def sentences_stage(out: Path, overwrite: bool = False) -> dict:
    prep = _need(out / "prepare.json", "prepare")
    tr = _need(out / "transcript.json", "transcribe")
    key = text_hash(ANALYSIS_VERSION, "sentences", tr["key"], prep["working"]["sha"], str(SENTENCE_PAUSE_S), str(SENTENCE_MAX_S),
                    str(NOISE_PCT), str(SPEECH_PCT), str(QUIET_ABOVE_FLOOR_DB), str(MIN_SNR_DB), str(OFF_MIC_BELOW_DB))
    path = out / "sentences.json"
    prev = cached(path, key, overwrite)
    if prev is not None:
        return prev
    if tr.get("status") != OK:
        doc = dict(key=key, status=UNCHECKED, reason="transcript not ok", sentences=[])
        _write(path, doc)
        return doc
    words = [w for seg in tr["segments"] for w in seg["words"]]
    sents = build_sentences(words)
    env = envelope_db(Path(prep["working"]["path"]))
    levels = measure_sentences(sents, env)
    doc = dict(key=key, status=levels["status"], levels=levels, sentences=sents,
               thresholds=dict(pause_s=SENTENCE_PAUSE_S, max_s=SENTENCE_MAX_S, noise_pct=NOISE_PCT, speech_pct=SPEECH_PCT,
                               quiet_above_floor_db=QUIET_ABOVE_FLOOR_DB, min_snr_db=MIN_SNR_DB, off_mic_below_db=OFF_MIC_BELOW_DB,
                               room_cap_s=ROOM_CAP_S, provisional=True),
               counts=dict(sentences=len(sents), off_mic=sum(bool(s["off_mic"]) for s in sents)))
    _write(path, doc)
    write_manifest(out / "sentences", tool="videoeasy.talks", stage="sentences", version=ANALYSIS_VERSION, source=source_hash(__file__), key=key,
                   inputs=dict(transcript=tr["key"], working=prep["working"]["sha"]), checked=["edge room from energy", "off-mic level"],
                   unchecked=["who is speaking (no diarization: off-mic is a level, not an identity)"])
    return doc


# ===================================================================== local text model
def chat_json(prompt: str, model: str | None = None, url: str | None = None, timeout: float = 1500) -> dict:
    import httpx

    r = httpx.post(f"{url or text_url()}/api/chat", json={"model": model or text_model(), "stream": False, "think": False, "format": "json",
                                                          "options": {"temperature": TEMPERATURE, "num_ctx": NUM_CTX},
                                                          "messages": [{"role": "user", "content": prompt}]}, timeout=timeout)
    r.raise_for_status()
    return json.loads(r.json()["message"]["content"])


def ask(prompt: str, chat) -> tuple[str, dict | None, str | None]:
    """(status, answer, error). Transport failures are unchecked; an unparsable answer is invalid."""
    try:
        ans = chat(prompt)
    except json.JSONDecodeError as e:
        return INVALID, None, f"unparsable JSON: {e}"
    except Exception as e:  # noqa: BLE001 - recorded as unchecked, never as an empty answer
        return UNCHECKED, None, f"{type(e).__name__}: {e}"
    if not isinstance(ans, dict):
        return INVALID, None, "answer is not a JSON object"
    return OK, ans, None


def compact_lines(sentences: list[dict]) -> list[str]:
    return [f"[{s['id']} {fmt_t(s['t0'])}] {s['text']}" + (" (off-mic)" if s.get("off_mic") else "") for s in sentences]


def est_tokens(text: str) -> int:
    return int(len(text) / CHARS_PER_TOKEN) + 1


# ===================================================================== beats
BEATS_PROMPT = """You are a documentary editor reading the transcript of {scope}, recorded on the speaker's lapel microphone{who}. Each line is one sentence: [sentence id, time] words. A line marked (off-mic) is probably another voice (an audience member, an interviewer) heard faintly by the speaker's microphone.

Divide {part} into its beats: contiguous stretches in order, every sentence in exactly one beat, the first beat starting at {first} and the last ending at {last}. A beat is a stretch with one purpose: a story, an argument, an exercise, an instruction, a question and its answer, a turn. For {minutes:.0f} minutes, expect roughly {lo} to {hi} beats.

TRANSCRIPT
{lines}

Answer with one JSON object:
{{"beats": [ {{"title": "at most 8 words", "first": "<sentence id>", "last": "<sentence id>", "summary": "what happens in this beat, at most 25 words", "energy": "low|medium|high", "turn": "what changes as this beat begins, at most 15 words"}} ]}}
Use only sentence ids that appear above. Do not quote words that are not in the transcript."""


def validate_beats(beats, ids: list[str]) -> tuple[list[dict] | None, list[str]]:
    """Contiguous, ordered, complete coverage of `ids`; any violation rejects the whole answer."""
    pos = {s: i for i, s in enumerate(ids)}
    reasons = []
    if not isinstance(beats, list) or not beats:
        return None, ["no beats"]
    expect, out = 0, []
    for k, b in enumerate(beats, 1):
        if not isinstance(b, dict):
            return None, [f"beat {k} is not an object"]
        f, l = b.get("first"), b.get("last")
        if f not in pos or l not in pos:
            reasons.append(f"beat {k}: unknown sentence id {f if f not in pos else l!r}")
            continue
        if pos[f] > pos[l]:
            reasons.append(f"beat {k}: first {f} after last {l}")
        if pos[f] != expect:
            reasons.append(f"beat {k}: starts at {f}, expected {ids[expect] if expect < len(ids) else 'end of talk'} (gap or overlap)")
        if not str(b.get("title") or "").strip():
            reasons.append(f"beat {k}: no title")
        expect = pos[l] + 1
        out.append(dict(n=k, title=str(b.get("title") or "").strip(), first=f, last=l, summary=str(b.get("summary") or "").strip(),
                        energy=str(b.get("energy") or "").strip().lower(), turn=str(b.get("turn") or "").strip()))
    if not reasons and expect != len(ids):
        reasons.append(f"beats end at {ids[expect - 1]}, the talk ends at {ids[-1]}")
    return (None, reasons) if reasons else (out, [])


def windows_for(sentences: list[dict], budget_tokens: int, overlap: int = WINDOW_OVERLAP_SENTENCES) -> list[tuple[int, int]]:
    """[lo, hi) sentence index windows whose compact lines fit the budget, overlapping by `overlap` sentences."""
    lines = compact_lines(sentences)
    toks = [est_tokens(x) + 1 for x in lines]
    if sum(toks) <= budget_tokens:
        return [(0, len(sentences))]
    out, lo = [], 0
    while lo < len(sentences):
        hi, t = lo, 0
        while hi < len(sentences) and t + toks[hi] <= budget_tokens:
            t += toks[hi]
            hi += 1
        hi = max(hi, lo + 1)
        out.append((lo, hi))
        if hi >= len(sentences):
            break
        lo = max(hi - overlap, lo + 1)
    return out


def merge_windows(parts: list[tuple[tuple[int, int], list[dict]]], ids: list[str]) -> list[dict]:
    """Join per-window beats: in each overlap the cut is the middle sentence; the earlier window keeps beats up
    to it (its last kept beat ends there), the later one starts just after it."""
    merged: list[dict] = []
    for k, ((lo, hi), beats) in enumerate(parts):
        pos = {s: i for i, s in enumerate(ids)}
        start_cut = lo if k == 0 else (lo + parts[k - 1][0][1] - 1) // 2 + 1
        end_cut = hi - 1 if k == len(parts) - 1 else (parts[k + 1][0][0] + hi - 1) // 2
        keep = [dict(b) for b in beats if pos[b["last"]] >= start_cut and pos[b["first"]] <= end_cut]
        if not keep:
            continue
        keep[0]["first"] = ids[start_cut] if pos[keep[0]["first"]] < start_cut else keep[0]["first"]
        keep[-1]["last"] = ids[end_cut] if pos[keep[-1]["last"]] > end_cut else keep[-1]["last"]
        merged.extend(keep)
    for i, b in enumerate(merged, 1):
        b["n"] = i
    return merged


def beats_stage(out: Path, chat=None, overwrite: bool = False, model: str | None = None) -> dict:
    sd = _need(out / "sentences.json", "sentences")
    prep = _need(out / "prepare.json", "prepare")
    model = model or text_model()
    sents = sd["sentences"]
    ids = [s["id"] for s in sents]
    meta = prep.get("meta") or {}
    budget = NUM_CTX - CTX_RESERVE_TOKENS
    wins = windows_for(sents, budget)
    prompts = []
    for k, (lo, hi) in enumerate(wins):
        part = sents[lo:hi]
        minutes = (part[-1]["t1"] - part[0]["t0"]) / 60 if part else 0
        prompts.append(BEATS_PROMPT.format(scope=f"one long talk{(' titled ' + repr(meta['title'])) if meta.get('title') else ''}",
                                           who=f" ({meta['speaker']})" if meta.get("speaker") else "",
                                           part="the WHOLE talk" if len(wins) == 1 else f"THIS PART ({k + 1} of {len(wins)}) of the talk",
                                           first=part[0]["id"] if part else "-", last=part[-1]["id"] if part else "-", minutes=minutes,
                                           lo=max(2, round(minutes / 8)), hi=max(3, round(minutes / 3)), lines="\n".join(compact_lines(part))))
    key = text_hash(ANALYSIS_VERSION, "beats", sd["key"], model, *prompts)
    path = out / "beats.json"
    prev = cached(path, key, overwrite)
    if prev is not None:
        return prev
    if sd.get("status") != OK or not sents:
        doc = dict(key=key, status=UNCHECKED, reason="sentences not ok or empty", beats=None)
        _write(path, doc)
        return doc
    chat = chat or (lambda p: chat_json(p, model))
    parts, raw = [], []
    for (lo, hi), prompt in zip(wins, prompts):
        status, ans, err = ask(prompt, chat)
        raw.append(dict(window=[lo, hi], status=status, error=err, answer=ans))
        if status != OK:
            doc = dict(key=key, status=status, reason=f"window {lo}-{hi}: {err}", beats=None, raw=raw, model=model, windows=len(wins))
            _write(path, doc)
            return doc
        beats, reasons = validate_beats(ans.get("beats"), ids[lo:hi])
        if beats is None:
            doc = dict(key=key, status=INVALID, reason="; ".join(reasons), beats=None, raw=raw, model=model, windows=len(wins))
            _write(path, doc)
            return doc
        parts.append(((lo, hi), beats))
    beats = parts[0][1] if len(parts) == 1 else merge_windows(parts, ids)
    final, reasons = validate_beats(beats, ids)
    status = OK if final is not None else INVALID
    for b in final or []:
        b["t0"] = sents[ids.index(b["first"])]["t0"]
        b["t1"] = sents[ids.index(b["last"])]["t1"]
    doc = dict(key=key, status=status, reason="; ".join(reasons) or None, beats=final, raw=raw, model=model, windows=len(wins),
               merged_windows=len(wins) > 1)
    _write(path, doc)
    write_manifest(out / "beats", tool="videoeasy.talks", stage="beats", version=ANALYSIS_VERSION, source=source_hash(__file__), key=key,
                   inputs=dict(sentences=sd["key"]), model=model, prompts=[text_hash(p) for p in prompts], windows=len(wins),
                   checked=["ids exist, contiguous, ordered, complete"], unchecked=["whether the beats are the right ones (editor's call)"])
    return doc


# ===================================================================== bites
BITES_PROMPT = """You are cutting short videos (60-90 second features and testimonials) from a long talk. Below is ONE beat of the talk, "{title}" ({summary}). Each line is one sentence: [sentence id, time] words. A line marked (off-mic) is another voice: never use it.

Propose the candidates that are genuinely there (an empty list is a valid answer):
- "hook": a 3-12 second line that would make someone stop scrolling and could open a video;
- "bite": a 6-30 second passage that stands on its own: specific, felt, quotable;
- "closer": a 3-15 second line that lands an ending.
Start each candidate on the sentence where its thought starts, not mid-thought.

For each candidate give: "kind"; "first" and "last" (sentence ids from this beat, first not after last); "why" (at most 20 words); "register": one of {registers} or "none" - your reading of the WORDS only; "cover_brief": what the audience should see while these words play - concrete visible things, actions, or a b-roll register - or exactly "face" when the words are personal and the speaker's face should carry them.{broll}

BEAT
{lines}

Answer with one JSON object, at most {max_n} candidates:
{{"candidates": [ {{"kind": "hook|bite|closer", "first": "<id>", "last": "<id>", "why": "...", "register": "...", "cover_brief": "..."}} ]}}"""


def first_token(text: str) -> str:
    m = re.match(r"\W*([\w']+)", text.lower())
    return m.group(1) if m else ""


def gate(cand: dict, sents_by_id: dict, order: dict, beat: dict, vocab: list[str], include_off_mic: bool = False,
         durations: dict = DURATIONS) -> dict:
    """Code checks on one proposed candidate. Returns it with status ok / invalid / excluded, reasons and flags."""
    c = dict(kind=str(cand.get("kind") or "").strip().lower(), first=cand.get("first"), last=cand.get("last"), why=str(cand.get("why") or "").strip(),
             register=str(cand.get("register") or "none").strip().lower(), cover_brief=str(cand.get("cover_brief") or "").strip(),
             beat=beat["n"], flags=[], reasons=[])
    if c["kind"] not in KINDS:
        c["reasons"].append(f"unknown kind {c['kind']!r}")
    if c["first"] not in sents_by_id or c["last"] not in sents_by_id:
        c["reasons"].append("unknown sentence id")
    elif not (order[beat["first"]] <= order[c["first"]] <= order[c["last"]] <= order[beat["last"]]):
        c["reasons"].append("ids out of order or outside the beat")
    if not vocab:                     # no vocabulary given: nothing to read the words against
        c["register"] = "none"
    elif c["register"] not in set(vocab) | {"none"}:
        c["reasons"].append(f"register {c['register']!r} not in the vocabulary")
    if not c["cover_brief"]:
        c["reasons"].append("no cover brief")
    if c["reasons"]:
        c["status"] = INVALID
        return c
    span = [s for s in sents_by_id.values() if order[c["first"]] <= order[s["id"]] <= order[c["last"]]]
    c["sentences"] = [s["id"] for s in span]
    c["t0"], c["t1"] = span[0]["t0"], span[-1]["t1"]
    c["duration_s"] = round(c["t1"] - c["t0"], 2)
    c["text"] = " ".join(s["text"] for s in span)
    c["head_room"], c["tail_room"] = span[0].get("head_room"), span[-1].get("tail_room")
    c["id"] = f"{c['kind'][0]}-{c['first']}-{c['last']}"
    tok = first_token(c["text"])
    if tok in DANGLING:
        c["flags"].append(f"dangling_start:{tok}")
    if c["head_room"] is None or c["tail_room"] is None:
        c["flags"].append("room_unmeasured")
    if c["head_room"] is not None and c["head_room"] < TIGHT_EDGE_S:
        c["flags"].append("tight_in")
    if c["tail_room"] is not None and c["tail_room"] < TIGHT_EDGE_S:
        c["flags"].append("tight_out")
    lo, hi = durations[c["kind"]]
    if any(s.get("off_mic") for s in span) and not include_off_mic:
        c["reasons"].append("contains an off-mic sentence")
    if c["duration_s"] < lo:
        c["reasons"].append(f"{c['duration_s']} s is under the {c['kind']} minimum {lo} s")
    if c["duration_s"] > hi:
        c["reasons"].append(f"{c['duration_s']} s is over the {c['kind']} maximum {hi} s")
    c["status"] = EXCLUDED if c["reasons"] else OK
    return c


def vocabulary(project: Path | None, registers: str | None) -> list[str]:
    if registers:
        return [r.strip().lower() for r in registers.split(",") if r.strip()]
    if project and (project / "deliverables.yaml").exists():
        import yaml

        doc = yaml.safe_load((project / "deliverables.yaml").read_text()) or {}
        return [str(r).strip().lower() for r in doc.get("registers") or []]
    return []


def bites_stage(out: Path, vocab: list[str], broll_context: str | None = None, chat=None, overwrite: bool = False,
                include_off_mic: bool = False, model: str | None = None) -> dict:
    sd = _need(out / "sentences.json", "sentences")
    bd = _need(out / "beats.json", "beats")
    model = model or text_model()
    sents = sd["sentences"]
    by_id = {s["id"]: s for s in sents}
    order = {s["id"]: i for i, s in enumerate(sents)}
    broll = (f"\nB-roll that exists (prefer briefs this material can meet):\n{broll_context}" if broll_context else "")
    prompts = []
    for b in bd.get("beats") or []:
        part = sents[order[b["first"]]: order[b["last"]] + 1]
        prompts.append(BITES_PROMPT.format(title=b["title"], summary=b.get("summary") or "", registers=", ".join(vocab) if vocab else "(no vocabulary given)",
                                           broll=broll, lines="\n".join(compact_lines(part)), max_n=MAX_PER_BEAT))
    key = text_hash(ANALYSIS_VERSION, "bites", bd["key"], model, str(include_off_mic), json.dumps(DURATIONS), *prompts)
    path = out / "bites.json"
    prev = cached(path, key, overwrite)
    if prev is not None:
        return prev
    if bd.get("status") != OK:
        doc = dict(key=key, status=UNCHECKED, reason=f"beats are {bd.get('status')}: {bd.get('reason')}", beats=[], candidates=[])
        _write(path, doc)
        return doc
    chat = chat or (lambda p: chat_json(p, model))
    calls = ((_read(path) or {}).get("calls") or {})       # per-beat answers by prompt hash: a retry asks only what failed
    per_beat, cands, seen = [], [], {}
    for b, prompt in zip(bd["beats"], prompts):
        ph = text_hash(model, prompt)
        hit = calls.get(ph)
        if hit is None or hit["status"] == UNCHECKED or (overwrite and hit["status"] == INVALID):
            status, ans, err = ask(prompt, chat)
            if status == OK and not isinstance(ans.get("candidates"), list):
                status, err = INVALID, "no candidates list"
            hit = calls[ph] = dict(status=status, answer=ans, error=err)
        row = dict(beat=b["n"], status=hit["status"], error=hit["error"], proposed=0, prompt=ph)
        if row["status"] == OK:
            for raw in hit["answer"]["candidates"][:MAX_PER_BEAT * 2]:
                if not isinstance(raw, dict):
                    continue
                row["proposed"] += 1
                c = gate(raw, by_id, order, b, vocab, include_off_mic)
                span = (c.get("first"), c.get("last"))
                if c["status"] == OK and span in seen:             # the same span offered twice: one candidate, both kinds noted
                    seen[span].setdefault("also_proposed_as", []).append(c["kind"])
                    continue
                if c["status"] == OK:
                    seen[span] = c
                cands.append(c)
        per_beat.append(row)
    status = OK if all(r["status"] == OK for r in per_beat) else UNCHECKED
    doc = dict(key=key, status=status, beats=per_beat, candidates=cands, vocabulary=vocab, model=model, durations=DURATIONS,
               broll_context=bool(broll_context), calls=calls)
    _write(path, doc)
    write_manifest(out / "bites", tool="videoeasy.talks", stage="bites", version=ANALYSIS_VERSION, source=source_hash(__file__), key=key,
                   inputs=dict(beats=bd["key"], sentences=sd["key"]), model=model, prompts=[text_hash(p) for p in prompts],
                   checked=["ids, order, beat bounds", "durations", "off-mic", "dangling start", "edge room"],
                   unchecked=[f"beat {r['beat']}: {r['error']}" for r in per_beat if r["status"] != OK]
                   + ["register is a reading of the words only; voice and face are not measured"])
    return doc


# ===================================================================== verify
STRENGTH = {"hook": "would it stop someone scrolling in its first seconds and make them want the rest?",
            "bite": "is it specific, quotable and felt, rather than generic or abstract?",
            "closer": "does it land an ending: would a video feel finished on it?"}

VERIFY_PROMPT = """Below are a few words of speech, cut from a longer talk. You have no other context: judge them as a viewer meeting them cold.

"{text}"

Answer with one JSON object:
{{"standalone": <0-3: 3 = makes complete sense on its own, 0 = cannot be followed without what came before>,
  "refers_to_unsaid": [<anything it refers to that is not said in it: a person, "this", "that", an earlier point>],
  "strength": <0-3: {strength}>,
  "reason": "<one line, at most 25 words>"}}"""


def validate_verdict(v) -> tuple[str, dict | None, str | None]:
    if not isinstance(v, dict):
        return INVALID, None, "not an object"
    out = {}
    for k in ("standalone", "strength"):
        x = v.get(k)
        if isinstance(x, bool) or not isinstance(x, (int, float)) or x != int(x) or not 0 <= x <= 3:
            return INVALID, None, f"{k} {x!r} is not an integer 0-3"
        out[k] = int(x)
    reason = str(v.get("reason") or "").strip()
    if not reason:
        return INVALID, None, "no reason"
    refs = v.get("refers_to_unsaid") or []
    out.update(reason=reason, refers_to_unsaid=[str(r) for r in refs] if isinstance(refs, list) else [str(refs)])
    return OK, out, None


def verify_stage(out: Path, chat=None, overwrite: bool = False, model: str | None = None) -> dict:
    bd = _need(out / "bites.json", "bites")
    model = model or text_model()
    path = out / "verdicts.json"
    store = _read(path) or {"entries": {}}            # a content-keyed cache: entries are never discarded
    chat = chat or (lambda p: chat_json(p, model))
    results, calls = {}, 0
    for c in bd.get("candidates", []):
        if c["status"] != OK:
            continue
        prompt = VERIFY_PROMPT.format(text=c["text"], strength=STRENGTH[c["kind"]])
        k = text_hash(ANALYSIS_VERSION, "verify", model, prompt)
        hit = store["entries"].get(k)
        if hit is None or hit["status"] == UNCHECKED or (hit["status"] == INVALID and overwrite):
            status, ans, err = ask(prompt, chat)
            calls += 1
            if status == OK:
                status, ans, err = validate_verdict(ans)
            hit = dict(status=status, verdict=ans, error=err, model=model, prompt=text_hash(prompt))
            store["entries"][k] = hit
        results[c["id"]] = dict(key=k, **hit)
    key = text_hash(ANALYSIS_VERSION, "verdicts", bd["key"], *sorted(f"{cid}:{r['key']}:{r['status']}" for cid, r in results.items()))
    store.update(key=key, status=OK if all(r["status"] == OK for r in results.values()) else UNCHECKED, results=results, model=model, model_calls=calls)
    _write(path, store)
    write_manifest(out / "verdicts", tool="videoeasy.talks", stage="verify", version=ANALYSIS_VERSION, source=source_hash(__file__), key=key,
                   inputs=dict(bites=bd["key"]), model=model, model_calls=calls,
                   checked=["each candidate's words judged alone by an independent call"],
                   unchecked=[cid for cid, r in results.items() if r["status"] != OK] + ["scores are uncalibrated against the editor"])
    return store


def rank(cands: list[dict]) -> dict:
    """Verified candidates ranked per kind (score = standalone + strength, ties by the tighter edge's room, then
    talk order); overlaps within a kind keep the better one. Unverified candidates are listed, never ranked."""
    ranked, unchecked = {k: [] for k in KINDS}, []
    for c in cands:
        if c["status"] != OK:
            continue
        v = c.get("verification") or {}
        if v.get("status") != OK:
            unchecked.append(c["id"])
            continue
        c["score"] = v["verdict"]["standalone"] + v["verdict"]["strength"]
    for kind in KINDS:
        pool = [c for c in cands if c["status"] == OK and c["kind"] == kind and "score" in c]
        pool.sort(key=lambda c: (-c["score"], -min(c.get("head_room") or 0.0, c.get("tail_room") or 0.0), c["t0"], c["id"]))
        kept: list[dict] = []
        for c in pool:
            clash = next((k for k in kept if c["t0"] < k["t1"] and k["t0"] < c["t1"]), None)
            if clash:
                c["superseded_by"] = clash["id"]
                continue
            kept.append(c)
        for i, c in enumerate(kept, 1):
            c["rank"] = i
        ranked[kind] = [c["id"] for c in kept]
    return dict(ranked=ranked, unchecked=sorted(unchecked))


# ===================================================================== report
def selects_rows(cands: list[dict], ranked: dict, chunks: list[dict], handle: float = HANDLE_S) -> list[dict]:
    by_id = {c["id"]: c for c in cands if c.get("id")}
    rows = []
    for kind in KINDS:
        for cid in ranked.get(kind, []):
            c = by_id[cid]
            tin = c["t0"] - min(c.get("head_room") or 0.0, handle)
            tout = c["t1"] + min(c.get("tail_room") or 0.0, handle)
            spans = source_spans(tin, tout, chunks)
            for i, sp in enumerate(spans):
                rows.append(dict(kind=kind, rank=c["rank"], id=cid, part=f"{i + 1}/{len(spans)}", score=c["score"],
                                 standalone=c["verification"]["verdict"]["standalone"], strength=c["verification"]["verdict"]["strength"],
                                 source_file=sp["name"], source_in_s=sp["in_s"], source_out_s=sp["out_s"], source_path=sp["file"],
                                 talk_in_s=round(tin, 3), talk_out_s=round(tout, 3), words_in_s=c["t0"], words_out_s=c["t1"],
                                 text=c["text"], cover_brief=c["cover_brief"], register=c["register"], flags=" ".join(c["flags"])))
    return rows


def analyse(out: Path) -> dict:
    """Join every stage into one record (no model work)."""
    prep = _need(out / "prepare.json", "prepare")
    tr = _read(out / "transcript.json") or {}
    sd = _read(out / "sentences.json") or {}
    bd = _read(out / "beats.json")
    bt = _read(out / "bites.json")
    vd = _read(out / "verdicts.json") or {}
    cands = [dict(c) for c in (bt or {}).get("candidates", [])]
    for c in cands:
        if c.get("id") in (vd.get("results") or {}):
            c["verification"] = vd["results"][c["id"]]
    ranking = rank(cands)
    by_id = {c["id"]: c for c in cands if c.get("id")}
    for c in cands:
        if c.get("t0") is not None:
            c["source"] = source_spans(c["t0"], c["t1"], prep["chunks"])
            if len(c["source"]) > 1:
                c["flags"].append("spans_chunks")
    stage = lambda d, why: dict(status=(d or {}).get("status", UNCHECKED), reason=(d or {}).get("reason") or (None if d else why))  # noqa: E731
    stages = dict(beats=stage(bd, "not run"), bites=stage(bt, "not run"), verify=stage(vd if vd.get("results") is not None else None, "not run"))
    status = OK if prep.get("status") == OK and tr.get("status") == OK and sd.get("status") == OK and all(s["status"] == OK for s in stages.values()) else UNCHECKED
    return dict(version=ANALYSIS_VERSION, status=status, meta=prep.get("meta") or {}, prepare=prep,
                transcript=dict(status=tr.get("status", UNCHECKED), model=tr.get("model"), language=tr.get("language"),
                                segments=len(tr.get("segments", [])), dropped=len(tr.get("dropped", [])), error=tr.get("error")),
                sentences=dict(status=sd.get("status", UNCHECKED), counts=sd.get("counts"), levels=sd.get("levels"), thresholds=sd.get("thresholds")),
                stages=stages,
                beats=(bd or {}).get("beats"), candidates=cands, ranked=ranking["ranked"], unchecked=ranking["unchecked"],
                selects=selects_rows(cands, ranking["ranked"], prep["chunks"]) if by_id else [],
                keys=dict(prepare=prep["key"], transcript=tr.get("key"), sentences=sd.get("key"), beats=(bd or {}).get("key"),
                          bites=(bt or {}).get("key"), verdicts=vd.get("key")))


def _src(c: dict) -> str:
    return "; ".join(f"{s['name']} {fmt_t(s['in_s'])}-{fmt_t(s['out_s'])} ({s['in_s']:.2f}-{s['out_s']:.2f} s)" for s in c.get("source", []))


def render_md(a: dict, top_hooks: int = 8, top_per_beat: int = 3) -> str:
    p, meta = a["prepare"], a["meta"]
    L = [f"# Talk analysis: {meta.get('title') or meta.get('talk_id') or 'talk'}", ""]
    if meta.get("speaker"):
        L.append(f"Speaker: {meta['speaker']}  ")
    L += [f"Duration {fmt_t(p['working']['duration_s'])} in {len(p['chunks'])} chunk(s) ({p['contiguity']['method']}). "
          f"Peak {p['talk_peak_dbfs']} dBFS, {p['overs_samples']} samples over 0 dBFS; one linear gain of {p['gain_db']:+.2f} dB "
          f"({p['gain_reason']}) for the working file, which clipped {p['working']['clipped_samples']} samples.", ""]
    L += ["| chunk | start | duration | peak dBFS | overs | LUFS | talk time |", "|---|---|---|---|---|---|---|"]
    for c in p["chunks"]:
        L.append(f"| {c['name']} | {c['start'] or '-'} | {fmt_t(c['duration_s'])} | {c['peak_dbfs']} | {c['overs_samples']} | "
                 f"{c['loudness'].get('lufs')} | {fmt_t(c['talk_t0'])}-{fmt_t(c['talk_t1'])} |")
    t = a["transcript"]
    L += ["", f"Transcript: {t['status']}, {t['segments']} segments kept, {t['dropped']} dropped by Whisper's own confidence "
          f"(model {t['model']}, language {t['language']}). One ASR pass is a question, not a finding: listen before trusting a word.", ""]
    s = a["sentences"]
    if (s.get("counts") or {}).get("sentences"):
        L.append(f"Sentences: {s['counts']['sentences']}, of which {s['counts']['off_mic']} flagged off-mic (≥ {OFF_MIC_BELOW_DB:g} dB under the "
                 f"wearer's typical sentence level {s['levels']['wearer_db']} dB; provisional). Noise floor {s['levels']['noise_floor_db']} dB, "
                 f"speech {s['levels']['speech_db']} dB" + (f"; {s['levels']['reason']}." if s['levels'].get('reason') else "."))
    for name, st in a["stages"].items():
        if st["status"] != OK:
            L.append(f"- **{name}: {st['status']}** {st.get('reason') or ''}")
    by_id = {c["id"]: c for c in a["candidates"] if c.get("id")}
    if a.get("beats"):
        L += ["", "## Beats", ""]
        for b in a["beats"]:
            L.append(f"{b['n']}. **{b['title']}** [{fmt_t(b['t0'])}-{fmt_t(b['t1'])}] {b['summary']}"
                     + (f" *(energy {b['energy']}; {b['turn']})*" if b.get("energy") or b.get("turn") else ""))

    def block(c):
        v = c["verification"]["verdict"]
        out = [f"- **{c['kind']} #{c['rank']}** score {c['score']}/6 (standalone {v['standalone']}, strength {v['strength']}) - talk "
               f"{fmt_t(c['t0'])}-{fmt_t(c['t1'])} ({c['duration_s']} s); source {_src(c)}",
               f"  > {c['text']}",
               f"  Cover: {c['cover_brief']} · register (words): {c['register']} · why: {c['why']}"]
        extra = [*c["flags"]] + ([f"refers to unsaid: {', '.join(v['refers_to_unsaid'])}"] if v.get("refers_to_unsaid") else [])
        if c.get("also_proposed_as"):
            extra.append("also proposed as " + ", ".join(c["also_proposed_as"]))
        if extra:
            out.append("  Flags: " + "; ".join(extra))
        return out

    if a["ranked"]["hook"]:
        L += ["", "## Top hooks", ""]
        for cid in a["ranked"]["hook"][:top_hooks]:
            L += block(by_id[cid])
    if a["ranked"]["bite"] and a.get("beats"):
        L += ["", "## Top bites per beat", ""]
        for b in a["beats"]:
            mine = [by_id[cid] for cid in a["ranked"]["bite"] if by_id[cid]["beat"] == b["n"]][:top_per_beat]
            if mine:
                L += [f"### {b['n']}. {b['title']}", ""]
                for c in mine:
                    L += block(c)
                L.append("")
    if a["ranked"]["closer"]:
        L += ["", "## Closers", ""]
        for cid in a["ranked"]["closer"][:top_hooks]:
            L += block(by_id[cid])
    if a["unchecked"]:
        L += ["", "## Not verified (no score, not ranked)", ""]
        for cid in a["unchecked"]:
            c = by_id[cid]
            L.append(f"- {cid} {c['kind']} {fmt_t(c['t0'])}-{fmt_t(c['t1'])}: {c['verification'].get('error') if c.get('verification') else 'no verdict'}")
    excl = [c for c in a["candidates"] if c["status"] in (EXCLUDED, INVALID)]
    if excl:
        L += ["", f"## Gated out ({len(excl)})", ""]
        for c in excl:
            L.append(f"- {c.get('id') or (str(c.get('first')) + '-' + str(c.get('last')))} {c['kind']}: {c['status']}: {'; '.join(c['reasons'])}")
    L += ["", "## What this does not establish", "",
          "- Scores are a local model's reading and are uncalibrated against the editor's ratings (docs/talks.md, Calibration).",
          "- The register is read from the words alone; voice and face are not measured here.",
          "- Off-mic is a level, not an identity: there is no diarization.",
          f"- Edge room is measured from energy, quiet = within {QUIET_ABOVE_FLOOR_DB:g} dB of the noise floor (provisional); "
          f"selects carry up to {HANDLE_S} s of it as handles.", ""]
    return "\n".join(L)


CSV_FIELDS = ["kind", "rank", "id", "part", "score", "standalone", "strength", "source_file", "source_in_s", "source_out_s", "talk_in_s", "talk_out_s",
              "words_in_s", "words_out_s", "text", "cover_brief", "register", "flags", "source_path"]


def report_stage(out: Path, overwrite: bool = False) -> dict:
    a = analyse(out)
    key = text_hash(ANALYSIS_VERSION, "report", json.dumps(a["keys"], sort_keys=True), source_hash(__file__))
    path = out / "analysis.json"
    prev = cached(path, key, overwrite)
    if prev is not None:
        return prev
    a["key"] = key
    _write(path, a)
    (out / "analysis.md").write_text(render_md(a))
    with open(out / "selects.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=CSV_FIELDS)
        w.writeheader()
        for r in a["selects"]:
            w.writerow({k: r.get(k) for k in CSV_FIELDS})
    write_manifest(out / "analysis", tool="videoeasy.talks", stage="report", version=ANALYSIS_VERSION, source=source_hash(__file__), key=key,
                   inputs=a["keys"], checked=[k for k, v in a["stages"].items() if v["status"] == OK],
                   unchecked=[f"{k}: {v['status']}" for k, v in a["stages"].items() if v["status"] != OK] + a["unchecked"])
    return a


# ===================================================================== export to the cut format
def export_cut(out: Path, top: int | None = None, gap_after_s: float = 1.0) -> tuple[dict, dict]:
    """Ranked candidates as a cut brollmatch can read: one beat per candidate (hooks, then bites, then closers,
    each in rank order), its cover brief as the beat note ("no cover" when the brief is the speaker's face,
    which brollmatch keeps bare), the audio picks from the original chunks. Plus a transcript keyed by chunk
    stem in the ingest schema, with chunk-local word times, for tools that look words up by clip."""
    a = _need(out / "analysis.json", "report")
    by_id = {c["id"]: c for c in a["candidates"] if c.get("id")}
    rows = {}
    for r in a["selects"]:
        rows.setdefault(r["id"], []).append(r)
    order = [cid for kind in KINDS for cid in a["ranked"][kind]]
    if top:
        order = order[:top]
    chunks = a["prepare"]["chunks"]
    sd = _need(out / "sentences.json", "sentences")
    words_of = {s["id"]: s["words"] for s in sd["sentences"]}
    beats = []
    for cid in order:
        c = by_id[cid]
        brief = c["cover_brief"]
        note = "no cover: the speaker's face carries these words" if brief.strip().lower() == "face" else f"Cover brief: {brief}"
        audio = []
        for r in rows[cid]:
            chunk = next(ch for ch in chunks if ch["name"] == r["source_file"])
            words = [w for s in c["sentences"] for w in words_of.get(s, []) if chunk["talk_t0"] <= w["s"] < chunk["talk_t1"]]
            audio.append(dict(clip=Path(r["source_file"]).stem, in_s=r["source_in_s"], out_s=r["source_out_s"],
                              duration_s=round(r["source_out_s"] - r["source_in_s"], 3), text=" ".join(w["w"] for w in words) or c["text"],
                              source_path=r["source_path"], tier=1, talk_candidate=cid))
        beats.append(dict(title=f"{c['kind']} #{c['rank']} ({c['score']}/6)", notes=note, audio=audio, video=[], gap_after_s=gap_after_s))
    cut = dict(source="videoeasy.talks", talk=a["meta"], analysis_key=a["key"], beats=beats)
    by_chunk: dict = {}
    for s in sd["sentences"]:
        groups: dict = {}
        for w in s["words"]:                       # a sentence across a chunk join becomes one segment per chunk
            ch, _ = talk_to_source(w["s"], chunks)
            groups.setdefault(ch["name"], (ch, []))[1].append(w)
        for ch, ws in groups.values():
            o = ch["talk_t0"]
            seg = dict(start=round(ws[0]["s"] - o, 3), end=round(ws[-1]["e"] - o, 3), text=" ".join(w["w"] for w in ws),
                       words=[dict(w=w["w"], s=round(w["s"] - o, 3), e=round(w["e"] - o, 3)) for w in ws])
            by_chunk.setdefault(Path(ch["name"]).stem, dict(segments=[], audio_source="body-worn recorder", role="aroll"))["segments"].append(seg)
    return cut, by_chunk


# ===================================================================== CLI
def _need(path: Path, stage: str) -> dict:
    doc = _read(path)
    if doc is None:
        raise TalkError(f"{path.name} not found: run the {stage} stage first")
    return doc


def out_dir_for(args) -> Path:
    if args.out:
        return Path(args.out)
    if args.project and args.talk_id:
        return Path(args.project) / "talks" / args.talk_id
    raise TalkError("say where the talk lives: --out DIR, or --project data/projects/<p> with --talk-id (or --audio/--mic-dir to derive it)")


def chunks_from_args(args) -> list[Chunk]:
    pattern = args.name_pattern or DEFAULT_NAME_PATTERN
    if args.audio:
        out = []
        for f in args.audio:
            p = Path(f)
            if not p.is_file():
                raise TalkError(f"{f}: not a file")
            parsed = parse_name(p.name, pattern)
            out.append(Chunk(path=str(p), name=p.name, session=parsed[0] if parsed else None, start=parsed[1] if parsed else None))
        return out
    if args.mic_dir:
        if not args.session:
            raise TalkError("--mic-dir needs --session (a chunk file name or a session number)")
        return find_session(Path(args.mic_dir), args.session, pattern)
    raise TalkError("prepare needs --audio F1 F2 ... or --mic-dir DIR --session S")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m videoeasy.talks", description="Long talks -> beats, hooks, bites, closers, cover briefs")
    ap.add_argument("stage", choices=["prepare", "transcribe", "sentences", "beats", "bites", "verify", "report", "all", "export-cut", "resolve"])
    ap.add_argument("--audio", nargs="+")
    ap.add_argument("--mic-dir")
    ap.add_argument("--session")
    ap.add_argument("--name-pattern", help="regex with named groups session, date (YYYYMMDD), time (HHMMSS); default: DJI mic naming")
    ap.add_argument("--assume-contiguous", action="store_true", help="take --audio files in the given order when their names carry no start time")
    ap.add_argument("--channel", type=int, default=1, help="1-based channel to analyse in a multi-channel file")
    ap.add_argument("--project", help="data/projects/<p>: default output under talks/<talk-id>, registers from deliverables.yaml")
    ap.add_argument("--talk-id")
    ap.add_argument("--out")
    ap.add_argument("--speaker")
    ap.add_argument("--title")
    ap.add_argument("--language", default="en")
    ap.add_argument("--registers", help="comma-separated register vocabulary (default: the project's deliverables.yaml registers)")
    ap.add_argument("--broll-context", help="text file: a short summary of the b-roll that exists, so briefs prefer showable material")
    ap.add_argument("--include-off-mic", action="store_true")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--skip-model", action="store_true", help="prepare, transcribe and sentences only, then the report")
    ap.add_argument("--top", type=int, help="export-cut / resolve: at most this many candidates")
    ap.add_argument("--resolve-project")
    ap.add_argument("--timeline")
    ap.add_argument("--bin", help="media-pool path for the chunks (default Talks/<talk-id>)")
    ap.add_argument("--fps", type=float, default=24.0, help="resolve dry run: timeline fps to plan at (the live timeline's rate wins on --apply)")
    ap.add_argument("--apply", action="store_true", help="resolve: write to Resolve (default is a dry run)")
    args = ap.parse_args(argv)

    if args.stage in ("prepare", "all") and not args.out and not args.talk_id and (args.audio or args.mic_dir):
        first = chunks_from_args(args)
        args.talk_id = Path(sorted(first, key=lambda c: (c.start or dt.datetime.min, c.name))[0].name).stem
    out = out_dir_for(args)
    out.mkdir(parents=True, exist_ok=True)
    meta = dict(talk_id=args.talk_id or out.name, speaker=args.speaker, title=args.title)
    vocab = vocabulary(Path(args.project) if args.project else None, args.registers)
    broll = Path(args.broll_context).read_text()[:6000] if args.broll_context else None

    def run(stage):
        if stage == "prepare":
            d = prepare(chunks_from_args(args), out, args.channel, args.assume_contiguous, args.overwrite, meta)
            print(f"prepare: {d['status']}, {len(d['chunks'])} chunk(s), {fmt_t(d['working']['duration_s'])}, peak {d['talk_peak_dbfs']} dBFS, "
                  f"{d['overs_samples']} overs, gain {d['gain_db']:+.2f} dB, clipped {d['working']['clipped_samples']}" + "".join(f"\n  note: {n}" for n in d["notes"]))
        elif stage == "transcribe":
            d = transcribe(out, args.language, args.overwrite)
            print(f"transcribe: {d['status']}, {len(d['segments'])} segments kept, {len(d['dropped'])} dropped" + (f" ({d.get('error')})" if d.get("error") else ""))
        elif stage == "sentences":
            d = sentences_stage(out, args.overwrite)
            print(f"sentences: {d['status']}, {d.get('counts')}")
        elif stage == "beats":
            d = beats_stage(out, overwrite=args.overwrite)
            print(f"beats: {d['status']}, {len(d.get('beats') or [])} beats" + (f" ({d.get('reason')})" if d.get("reason") else ""))
        elif stage == "bites":
            d = bites_stage(out, vocab, broll, overwrite=args.overwrite, include_off_mic=args.include_off_mic)
            from collections import Counter
            print(f"bites: {d['status']}, " + ", ".join(f"{k} {v}" for k, v in Counter(c['status'] for c in d['candidates']).items()))
        elif stage == "verify":
            d = verify_stage(out, overwrite=args.overwrite)
            print(f"verify: {d['status']}, {len(d['results'])} candidates, {d['model_calls']} model calls")
        elif stage == "report":
            d = report_stage(out, args.overwrite)
            print(f"report: {out / 'analysis.md'} ({sum(len(v) for v in d['ranked'].values())} ranked, {len(d['unchecked'])} unverified)")

    if args.stage == "all":
        for st in (["prepare", "transcribe", "sentences"] + ([] if args.skip_model else ["beats", "bites", "verify"]) + ["report"]):
            run(st)
    elif args.stage == "export-cut":
        cut, by_chunk = export_cut(out, args.top)
        refuse_overwrite([out / "cut-selects.json", out / "transcripts-by-chunk.json"], args.overwrite)
        _write(out / "cut-selects.json", cut)
        _write(out / "transcripts-by-chunk.json", by_chunk)
        write_manifest(out / "cut-selects", tool="videoeasy.talks", stage="export-cut", version=ANALYSIS_VERSION, source=source_hash(__file__),
                       analysis=cut["analysis_key"], beats=len(cut["beats"]))
        print(f"export-cut: {len(cut['beats'])} beats -> {out / 'cut-selects.json'}; chunk transcripts -> {out / 'transcripts-by-chunk.json'}")
    elif args.stage == "resolve":
        from . import talks_resolve
        return talks_resolve.main_from(args, out)
    else:
        run(args.stage)
    return 0


if __name__ == "__main__":
    sys.exit(main())
