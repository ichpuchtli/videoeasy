"""Waveform sync between DJI Mic 32-bit-float recordings and camera clips.

Resolve-style audio sync: RMS energy envelopes (100 Hz) cross-correlated via
FFT. Clocks on the cameras and mic disagree by hours (verified), so the only
time prior used is the calendar day; the correlation finds the true offset.

Mic sessions: DJI_<nn>_<YYYYMMDD>_<HHMMSS>.WAV under <mic_root>/{Left,Right}/,
recorded in contiguous ~30.7-min chunks — chunks of one session are
concatenated and matched as a whole. The mic folder is the config's
`audiosync.mic_root` (absolute, or relative to the config file); with none
configured there are no sessions and nothing is matched.
"""
from __future__ import annotations

import datetime as dt
import json
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import typer

from .config import Config, load_config

ENV_HZ = 100          # envelope frame rate
SR = 8000             # decode sample rate
HOP = SR // ENV_HZ
NAME_RE = re.compile(r"DJI_(\d+)_(\d{8})_(\d{6})\.WAV$", re.IGNORECASE)

app = typer.Typer(help="Sync DJI mic recordings to camera clips by waveform")


def envelope(path: str, start: float | None = None, dur: float | None = None) -> np.ndarray:
    """Stream-decode audio to a 100 Hz RMS envelope (log-compressed)."""
    cmd = ["ffmpeg", "-v", "error"]
    if start is not None:
        cmd += ["-ss", f"{start:.3f}"]
    if dur is not None:
        cmd += ["-t", f"{dur:.3f}"]
    cmd += ["-i", path, "-vn", "-f", "f32le", "-ac", "1", "-ar", str(SR), "pipe:1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    frames = []
    block = HOP * 4096
    carry = b""
    while True:
        buf = proc.stdout.read(block * 4)
        if not buf:
            break
        buf = carry + buf
        n = (len(buf) // (HOP * 4)) * HOP * 4
        carry = buf[n:]
        if n:
            x = np.frombuffer(buf[:n], dtype=np.float32).reshape(-1, HOP)
            frames.append(np.sqrt((x.astype(np.float64) ** 2).mean(axis=1)))
    proc.wait()
    if not frames:
        return np.zeros(0)
    return np.log1p(np.concatenate(frames) * 1000.0).astype(np.float32)


@dataclass
class Session:
    key: str                 # e.g. "Left_06"
    day: str                 # "20250128"
    start: dt.datetime
    files: list[dict]        # [{path, start_env, n_env, start_time}]
    env_path: Path


def mic_root(cfg: Config) -> Path | None:
    """The DJI mic folder (holding Left/ and Right/) from the config's `audiosync.mic_root`."""
    v = (cfg.raw.get("audiosync") or {}).get("mic_root")
    if not v:
        return None
    p = Path(v)
    return p if p.is_absolute() else cfg.root / p


def mic_sessions(cfg: Config) -> list[Session]:
    root = mic_root(cfg)
    if root is None:
        return []
    cache_dir = cfg.work_dir / "audiosync"
    cache_dir.mkdir(parents=True, exist_ok=True)
    groups: dict[str, list] = {}
    for side in ("Left", "Right"):
        for wav in sorted((root / side).glob("*.WAV")):
            m = NAME_RE.search(wav.name)
            if m:
                groups.setdefault(f"{side}_{m.group(1)}", []).append(
                    (dt.datetime.strptime(m.group(2) + m.group(3), "%Y%m%d%H%M%S"), wav)
                )
    sessions = []
    for key, files in groups.items():
        files.sort()
        sessions.append(Session(
            key=key,
            day=files[0][0].strftime("%Y%m%d"),
            start=files[0][0],
            files=[{"path": str(p), "start_time": t.isoformat()} for t, p in files],
            env_path=cache_dir / f"{key}.npy",
        ))
    return sessions


def build_session_env(session: Session) -> None:
    if session.env_path.exists():
        return
    parts = []
    for f in session.files:
        parts.append(envelope(f["path"]))
    env = np.concatenate(parts)
    np.save(session.env_path, env)
    # chunk boundary metadata for mapping offsets back to files
    meta, pos = [], 0
    for f, p in zip(session.files, parts):
        meta.append({**f, "start_env": pos, "n_env": len(p)})
        pos += len(p)
    session.env_path.with_suffix(".json").write_text(json.dumps(meta, indent=1))


def _znorm(x: np.ndarray) -> np.ndarray:
    s = x.std()
    return (x - x.mean()) / (s if s > 1e-9 else 1.0)


def xcorr_match(clip_env: np.ndarray, sess_env: np.ndarray) -> tuple[float, float, float]:
    """Return (offset_s, peak_r, margin) of clip within session."""
    a, b = _znorm(sess_env), _znorm(clip_env)
    n = len(a) + len(b)
    nfft = 1 << (n - 1).bit_length()
    corr = np.fft.irfft(np.fft.rfft(a, nfft) * np.conj(np.fft.rfft(b, nfft)), nfft)
    corr = corr[: len(a) - len(b) + 1] / len(b)  # valid lags only
    if len(corr) < 1:
        return 0.0, 0.0, 0.0
    peak = int(np.argmax(corr))
    peak_r = float(corr[peak])
    guard = ENV_HZ * 60  # ignore ±60 s around the peak for the runner-up
    masked = corr.copy()
    masked[max(0, peak - guard): peak + guard] = -np.inf
    runner = float(masked.max()) if np.isfinite(masked).any() else 0.0
    return peak / ENV_HZ, peak_r, peak_r - runner


def locate(session: Session, offset_s: float) -> dict:
    meta = json.loads(session.env_path.with_suffix(".json").read_text())
    pos = offset_s * ENV_HZ
    for f in meta:
        if pos < f["start_env"] + f["n_env"]:
            return {"file": f["path"], "offset_in_file_s": round((pos - f["start_env"]) / ENV_HZ, 3)}
    last = meta[-1]
    return {"file": last["path"], "offset_in_file_s": round((pos - last["start_env"]) / ENV_HZ, 3)}


def _session_spans(session: Session, offset_s: float, dur_s: float) -> list[tuple[str, float, float]]:
    """(file, local_start, local_dur) segments covering [offset, offset+dur) in a session."""
    meta = json.loads(session.env_path.with_suffix(".json").read_text())
    spans = []
    for f in meta:
        c_start, c_len = f["start_env"] / ENV_HZ, f["n_env"] / ENV_HZ
        lo = max(offset_s, c_start)
        hi = min(offset_s + dur_s, c_start + c_len)
        if hi > lo:
            spans.append((f["path"], lo - c_start, hi - lo))
    return spans


def extract_mic_wav(cfg: Config, clip: str, clip_dur_s: float) -> Path | None:
    """Build a 48k mono wav of the mic audio matching a synced camera clip.

    Uses every side whose match was transcript-verified (mixed if both were);
    returns None when no side is verified. Cached in work/micaudio/.
    """
    rec = _load_clip(cfg, clip)
    if rec is None:
        return None
    verified = {side: m for side, m in rec["matches"].items() if m.get("verified")}
    if not verified:
        return None
    out_dir = cfg.work_dir / "micaudio"
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"{clip}.wav"
    if dest.exists():
        return dest

    sessions = {s.key: s for s in mic_sessions(cfg)}
    side_wavs = []
    for side, m in verified.items():
        spans = _session_spans(sessions[m["session"]], m["offset_s"], clip_dur_s)
        seg_paths = []
        for i, (path, start, dur) in enumerate(spans):
            seg = out_dir / f".{clip}.{side}.{i}.wav"
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", f"{start:.3f}",
                            "-t", f"{dur:.3f}", "-i", path, "-ac", "1", "-ar", "48000",
                            str(seg)], check=True, capture_output=True)
            seg_paths.append(seg)
        side_wav = out_dir / f".{clip}.{side}.wav"
        if len(seg_paths) == 1:
            seg_paths[0].rename(side_wav)
        else:
            lst = out_dir / f".{clip}.{side}.txt"
            lst.write_text("".join(f"file '{p}'\n" for p in seg_paths))
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0",
                            "-i", str(lst), "-c", "copy", str(side_wav)],
                           check=True, capture_output=True)
            lst.unlink()
            for p in seg_paths:
                p.unlink()
        side_wavs.append(side_wav)

    if len(side_wavs) == 1:
        side_wavs[0].rename(dest)
    else:
        subprocess.run(["ffmpeg", "-v", "error", "-y",
                        "-i", str(side_wavs[0]), "-i", str(side_wavs[1]),
                        "-filter_complex", "amix=inputs=2:normalize=1",
                        str(dest)], check=True, capture_output=True)
        for p in side_wavs:
            p.unlink()
    return dest


def _clip_result_path(cfg: Config, clip: str) -> Path:
    d = cfg.out_dir / "audiosync"
    d.mkdir(parents=True, exist_ok=True)
    return d / f"{clip}.json"


def _load_clip(cfg: Config, clip: str) -> dict | None:
    p = _clip_result_path(cfg, clip)
    return json.loads(p.read_text()) if p.exists() else None


def _save_clip(cfg: Config, clip: str, rec: dict) -> None:
    # one file per clip: concurrent matchers never clobber each other
    _clip_result_path(cfg, clip).write_text(json.dumps(rec, indent=1))


@app.command()
def cache(session: str = typer.Option(None, "--session", help="one session key, e.g. Left_06"),
          config: str = typer.Option("config.yaml", "--config", "-c")):
    """Build (or reuse) envelope caches for mic sessions."""
    cfg = load_config(config)
    for s in mic_sessions(cfg):
        if session and s.key != session:
            continue
        build_session_env(s)
        print(f"{s.key}: cached ({s.env_path.name})")


@app.command()
def match(clip: str = typer.Option(..., "--clip", help="camera clip filename, e.g. C0001.MOV"),
          config: str = typer.Option("config.yaml", "--config", "-c")):
    """Match one camera clip against all same-day mic sessions."""
    cfg = load_config(config)
    src = next((p for p in [cfg.aroll_dir / clip, cfg.broll_dir / clip] if p.exists()), None)
    if src is None:
        raise SystemExit(f"clip {clip} not found in source dirs")
    day = dt.datetime.fromtimestamp(src.stat().st_mtime).strftime("%Y%m%d")

    clip_env = envelope(str(src))
    if len(clip_env) < ENV_HZ * 3:
        raise SystemExit(f"clip too short / no audio ({len(clip_env)} env frames)")

    candidates = [s for s in mic_sessions(cfg) if s.day == day] or mic_sessions(cfg)
    best = {}
    for s in candidates:
        if not s.env_path.exists():
            build_session_env(s)
        off, r, margin = xcorr_match(clip_env, np.load(s.env_path))
        side = s.key.split("_")[0]
        rec = {"session": s.key, "offset_s": round(off, 3), "peak_r": round(r, 4),
               "margin": round(margin, 4), **locate(s, off)}
        if side not in best or r > best[side]["peak_r"]:
            best[side] = rec

    rec = {"day": day, "matches": best, "clip_env_frames": int(len(clip_env))}
    _save_clip(cfg, clip, rec)
    print(json.dumps(rec, indent=1))


@app.command()
def report(config: str = typer.Option("config.yaml", "--config", "-c")):
    """Merge per-clip results into <out_dir>/audio_sync.json and print a table."""
    cfg = load_config(config)
    merged = {p.stem: json.loads(p.read_text())
              for p in sorted((cfg.out_dir / "audiosync").glob("*.json"))}
    (cfg.out_dir / "audio_sync.json").write_text(json.dumps(merged, indent=1))
    print(f"{'clip':<16}{'side':<7}{'session':<10}{'peak_r':>7}{'margin':>8}{'overlap':>9}  verdict")
    for clip, rec in merged.items():
        for side, m in rec.get("matches", {}).items():
            verdict = ("VERIFIED" if m.get("verified")
                       else "strong-waveform" if m["peak_r"] > 0.5 and m["margin"] > 0.3
                       else "no-coverage?" if m["peak_r"] < 0.25
                       else "ambiguous")
            print(f"{clip:<16}{side:<7}{m['session']:<10}{m['peak_r']:>7.2f}{m['margin']:>8.2f}"
                  f"{m.get('verify_overlap', float('nan')):>9.2f}  {verdict}")


@app.command()
def verify(clip: str = typer.Option(..., "--clip"),
           window_at: float = typer.Option(0.5, help="fraction into the clip to sample"),
           config: str = typer.Option("config.yaml", "--config", "-c")):
    """Transcribe 20s of camera audio and the matched mic audio; report word overlap."""
    import mlx_whisper

    cfg = load_config(config)
    rec = _load_clip(cfg, clip)
    if rec is None:
        raise SystemExit(f"no match recorded for {clip} — run match first")
    src = next(p for p in [cfg.aroll_dir / clip, cfg.broll_dir / clip] if p.exists())

    def words(path, start):
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(start), "-t", "20",
                            "-i", str(path), "-vn", "-ac", "1", "-ar", "16000", tmp.name],
                           check=True, capture_output=True)
            res = mlx_whisper.transcribe(tmp.name, path_or_hf_repo="mlx-community/whisper-tiny",
                                         condition_on_previous_text=False)
        return {w.strip(".,?!\"'").lower() for w in res["text"].split() if len(w) > 2}

    probe = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                            "-of", "csv=p=0", str(src)], capture_output=True, text=True)
    clip_dur = float(probe.stdout.strip())
    t_clip = max(0.0, clip_dur * window_at - 10)
    cam_words = words(src, t_clip)
    sessions = {s.key: s for s in mic_sessions(cfg)}

    for side, m in rec["matches"].items():
        # map session-time (clip start offset + sample point) back to the
        # right chunk file — clips can span chunk boundaries
        spot = locate(sessions[m["session"]], m["offset_s"] + t_clip)
        mic_words = words(spot["file"], spot["offset_in_file_s"])
        inter = len(cam_words & mic_words)
        union = len(cam_words | mic_words) or 1
        overlap = round(inter / union, 3)
        if overlap >= m.get("verify_overlap", -1):
            m["verify_overlap"] = overlap
            m["verified"] = bool(overlap >= 0.3 and inter >= 5)
    _save_clip(cfg, clip, rec)
    print(json.dumps(rec["matches"], indent=1))
