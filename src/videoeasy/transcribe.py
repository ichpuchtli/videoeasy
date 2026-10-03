"""Stage 6: transcribe all a-roll with word-level timestamps.

Word timestamps are what make surgical soundbite trimming possible later —
the radio cut selects words, not whole segments. mlx-whisper runs natively
on Apple Silicon.
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import Config


def transcribe_file(path: str, model: str) -> dict:
    import mlx_whisper  # deferred: slow import, downloads model on first use

    result = mlx_whisper.transcribe(
        path,
        path_or_hf_repo=model,
        word_timestamps=True,
        # VAD-adjacent guard: without it Whisper hallucinates during silence
        condition_on_previous_text=False,
    )
    return {
        "language": result.get("language"),
        "segments": [
            {
                "start": round(seg["start"], 3),
                "end": round(seg["end"], 3),
                "text": seg["text"].strip(),
                "words": [
                    {"w": w["word"], "s": round(w["start"], 3), "e": round(w["end"], 3)}
                    for w in seg.get("words", [])
                ],
            }
            for seg in result["segments"]
            if seg["text"].strip()
        ],
    }


def run(cfg: Config, force: bool = False) -> None:
    from .audiosync import extract_mic_wav

    inventory = json.loads((cfg.out_dir / "inventory.json").read_text())
    out_path = cfg.out_dir / "transcripts.json"
    transcripts = json.loads(out_path.read_text()) if out_path.exists() and not force else {}

    for clip in inventory["aroll"]:
        if clip["id"] in transcripts:
            continue
        # Prefer the waveform-synced DJI lav audio (32-bit float, on-body)
        # over camera scratch audio whenever the sync was transcript-verified.
        clip_name = Path(clip["path"]).name
        mic_wav = None
        try:
            mic_wav = extract_mic_wav(cfg, clip_name, clip["duration_s"])
        except Exception as e:
            print(f"  mic extract failed for {clip_name} ({e}); using camera audio")
        if mic_wav is not None:
            print(f"transcribing {clip['id']} ({clip['duration_s']:.0f}s) from lav mix...")
            transcripts[clip["id"]] = transcribe_file(str(mic_wav), cfg.whisper_model)
            transcripts[clip["id"]]["audio_source"] = "dji_mic"
        elif not clip["has_audio"]:
            transcripts[clip["id"]] = {"segments": [], "note": "no audio stream"}
        else:
            print(f"transcribing {clip['id']} ({clip['duration_s']:.0f}s) from camera audio...")
            transcripts[clip["id"]] = transcribe_file(clip["path"], cfg.whisper_model)
            transcripts[clip["id"]]["audio_source"] = "camera"
        out_path.write_text(json.dumps(transcripts, indent=2))


# ------------------------------------------------------------------ B-roll audio
# The editor's v6 review (26 Sep 2026): a cover row is "unknown" for the talking-cover
# rule until its source has a transcript, and 92 B-roll sources had none. B-roll
# audio is mostly natural sound, on which Whisper invents speech and misreads the
# language (a 20 s wind-and-birds probe came back "nn"), so this pass forces
# English, keeps Whisper's own per-segment confidence, drops segments over the
# no-speech / compression thresholds, and records what it dropped. A source that
# is digitally silent is written as an empty transcript with the level measured.
NO_SPEECH_MAX = 0.6
COMPRESSION_MAX = 2.4
SILENT_MAX_DB = -50.0


def peak_db(path: str) -> float | None:
    import re
    import subprocess
    p = subprocess.run(["ffmpeg", "-v", "info", "-i", path, "-vn", "-af", "volumedetect", "-f", "null", "-"], capture_output=True, text=True)
    m = re.search(r"max_volume:\s*(-?[\d.]+) dB", p.stderr)
    return float(m.group(1)) if m else None


def transcribe_broll_file(path: str, model: str) -> dict:
    import mlx_whisper

    result = mlx_whisper.transcribe(path, path_or_hf_repo=model, word_timestamps=True, condition_on_previous_text=False, language="en")
    kept, dropped = [], []
    for seg in result["segments"]:
        if not seg["text"].strip():
            continue
        q = dict(no_speech_prob=round(float(seg.get("no_speech_prob", 0.0)), 3), avg_logprob=round(float(seg.get("avg_logprob", 0.0)), 3),
                 compression_ratio=round(float(seg.get("compression_ratio", 0.0)), 3))
        row = {"start": round(seg["start"], 3), "end": round(seg["end"], 3), "text": seg["text"].strip(),
               "words": [{"w": w["word"], "s": round(w["start"], 3), "e": round(w["end"], 3)} for w in seg.get("words", [])], "quality": q}
        if q["no_speech_prob"] > NO_SPEECH_MAX or q["compression_ratio"] > COMPRESSION_MAX:
            dropped.append(row)
        else:
            kept.append(row)
    return {"language": "en (forced)", "segments": kept,
            "quality": dict(no_speech_max=NO_SPEECH_MAX, compression_max=COMPRESSION_MAX, dropped=dropped,
                            detected_language=result.get("language"))}


def run_broll(cfg: Config, force: bool = False, only: str | None = None) -> dict:
    """Camera audio of every B-roll source that has an audio track -> transcripts.json (same schema, role 'broll').
    A-roll entries are never touched. Returns a summary."""
    inventory = json.loads((cfg.out_dir / "inventory.json").read_text())
    out_path = cfg.out_dir / "transcripts.json"
    transcripts = json.loads(out_path.read_text()) if out_path.exists() else {}
    summary = dict(sources=0, no_audio=0, silent=0, transcribed=0, with_speech=0, failed=0)
    for clip in inventory["broll"]:
        if only and clip["id"] != only:
            continue
        summary["sources"] += 1
        if not clip.get("has_audio"):
            summary["no_audio"] += 1
            if clip["id"] not in transcripts or force:   # recorded, so a missing entry never reads as "not checked"
                transcripts[clip["id"]] = {"language": None, "segments": [], "audio_source": None, "role": "broll",
                                           "quality": dict(status="silent", reason="no audio track")}
                out_path.write_text(json.dumps(transcripts, indent=1))
            continue
        if clip["id"] in transcripts and not force and transcripts[clip["id"]].get("role") == "broll":
            summary["transcribed"] += 1
            summary["with_speech"] += bool(transcripts[clip["id"]]["segments"])
            continue
        peak = peak_db(clip["path"])
        if peak is not None and peak < SILENT_MAX_DB:
            transcripts[clip["id"]] = {"language": None, "segments": [], "audio_source": "camera", "role": "broll",
                                       "quality": dict(status="silent", peak_db=peak, silent_max_db=SILENT_MAX_DB)}
            summary["silent"] += 1
        else:
            try:
                rec = transcribe_broll_file(clip["path"], cfg.whisper_model)
            except Exception as e:  # noqa: BLE001
                print(f"  {clip['id']}: transcription failed: {e}")
                summary["failed"] += 1
                continue
            rec.update(audio_source="camera", role="broll")
            rec["quality"].update(status="ok", peak_db=peak)
            transcripts[clip["id"]] = rec
            summary["transcribed"] += 1
            summary["with_speech"] += bool(rec["segments"])
            print(f"  {clip['id']}: peak {peak} dB, {len(rec['segments'])} speech segments kept, {len(rec['quality']['dropped'])} dropped"
                  + (f": {rec['segments'][0]['text'][:70]!r}" if rec["segments"] else ""))
        out_path.write_text(json.dumps(transcripts, indent=1))
    print(f"B-roll audio: {summary}")
    return summary
