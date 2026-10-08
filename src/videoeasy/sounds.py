"""Vocal and crowd sounds heard on a clip, as a measurement: YAMNet over the camera audio.

Laughter, crying, a sigh or a shout carry a register that a face turned away
cannot show, and cheering or a chant marks a moment of the event itself. YAMNet
(AudioSet's 521 classes, run by faceworker.py in its own environment) reads
each source's audio in ~0.96 s windows. Here its classes are grouped into
FAMILIES, and a run of windows where a family scores at or over EVENT_MIN is
an event with its time span and peak.

A sound HEARD is not a face SEEN: an event is never attributed to a person on
screen (the laugh may be off camera), and register.py keeps it as its own
modality. B-roll audio at an event is often music or a facilitator's voice;
the share of speech and music windows is kept per source so a quiet
"no laughter" on a music-heavy clip is read for what it is. EVENT_MIN is
provisional until the editor's calibration.

    uv run python -m videoeasy.sounds --config config.<p>.yaml [--only SRC] [--workers 4] [--force]
    -> <out_dir>/sounds.json + sounds.manifest.json
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

from .evalrun import OK, UNCHECKED, fingerprint, source_hash, write_manifest
from .faces import MODELS, ensure_model, run_worker, sources, worker_python

FAMILIES = {
    "laughter": ("Laughter", "Giggle", "Snicker", "Belly laugh", "Chuckle, chortle", "Baby laughter"),
    "crying": ("Crying, sobbing", "Whimper", "Wail, moan", "Baby cry, infant cry"),
    "breath": ("Sigh", "Gasp", "Groan", "Grunt", "Pant", "Breathing"),
    "shout": ("Shout", "Yell", "Battle cry", "Screaming", "Children shouting", "Whoop"),
    "cheer": ("Cheering", "Applause", "Clapping"),
    "song": ("Singing", "Chant", "Choir", "Humming"),
    "drum": ("Drum", "Drum roll", "Bass drum", "Tabla"),
}
CONTEXT = ("Speech", "Music")
EVENT_MIN = 0.2       # provisional: a family at or over this in a window is heard
CONTEXT_MIN = 0.3     # a window counts as speech / music at or over this


def family_scores(window: dict) -> dict[str, float]:
    return {f: max((window.get(c, 0.0) for c in cs), default=0.0) for f, cs in FAMILIES.items()}


def events(windows: list, hop_s: float | None = None) -> list[dict]:
    """Runs of consecutive windows per family at or over EVENT_MIN: [{family, t0, t1, peak, classes}]."""
    if not windows:
        return []
    ts = [w[0] for w in windows]
    hop = hop_s or (min(b - a for a, b in zip(ts, ts[1:])) if len(ts) > 1 else 0.96)
    out = []
    for fam, cls in FAMILIES.items():
        run = None
        for t, w in windows:
            s = max((w.get(c, 0.0) for c in cls), default=0.0)
            if s >= EVENT_MIN:
                hit = sorted((c for c in cls if w.get(c, 0.0) >= EVENT_MIN), key=lambda c: -w[c])
                if run and t - run["t1"] <= hop * 1.5:
                    run["t1"] = round(t + hop, 2)
                    run["peak"] = max(run["peak"], s)
                    run["classes"] = sorted(set(run["classes"]) | set(hit))
                else:
                    run = dict(family=fam, t0=round(t, 2), t1=round(t + hop, 2), peak=s, classes=hit)
                    out.append(run)
            elif run and t - run["t1"] > hop * 1.5:
                run = None
    return sorted(out, key=lambda e: (e["t0"], e["family"]))


def has_audio(src) -> bool:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", str(src)],
                       capture_output=True, text=True)
    return bool(r.stdout.strip())


def source_record(raw: dict) -> dict:
    ws = raw["windows"]
    n = max(1, len(ws))
    return dict(status=OK, audio_s=raw["audio_s"], rms_dbfs=raw["rms_dbfs"], windows=len(ws),
                speech_frac=round(sum(1 for _, w in ws if w.get("Speech", 0) >= CONTEXT_MIN) / n, 2),
                music_frac=round(sum(1 for _, w in ws if w.get("Music", 0) >= CONTEXT_MIN) / n, 2),
                events=events(ws))


def run(cfg, only: str | None = None, workers: int = 4, force: bool = False) -> dict:
    srcs, dup = sources(cfg)
    if only:
        srcs = {k: v for k, v in srcs.items() if k == only}
    work = cfg.work_dir / "sounds"
    work.mkdir(parents=True, exist_ok=True)
    out = cfg.out_dir / "sounds.json"
    doc = json.loads(out.read_text()) if out.exists() else {}
    units = doc.setdefault("sources", {})
    for sid, paths in dup.items():
        units[sid] = dict(status=UNCHECKED, why="two files share this id: " + "; ".join(paths))
    model = ensure_model("yamnet")
    worker_python()   # synced once, before the workers start

    def one(sid, src):
        if not has_audio(src):   # a drone clip, an HFR take: nothing to hear is a fact, not a failed check
            return sid, dict(status=OK, no_audio=True, audio_s=0.0, rms_dbfs=None, windows=0, speech_frac=None, music_frac=None,
                             events=[], path=str(src))
        try:
            raw = work / f"{sid}.{fingerprint(src)}.raw.json"
            if force or not raw.exists():
                run_worker("sounds", src, raw, yamnet=model)
            rec = source_record(json.loads(raw.read_text()))
        except Exception as e:  # noqa: BLE001
            return sid, dict(status=UNCHECKED, why=str(e)[:300], path=str(src))
        rec.update(path=str(src))
        return sid, rec

    todo = sorted(srcs.items())
    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        for k, f in enumerate(as_completed([ex.submit(one, s, p) for s, p in todo]), 1):
            sid, rec = f.result()
            units[sid] = rec
            ev = ", ".join(f"{e['family']} {e['t0']:.0f}-{e['t1']:.0f}s {e['peak']:.2f}" for e in rec.get("events", [])[:5])
            line = ("no audio track" if rec.get("no_audio") else ev or "no events") if rec["status"] == OK else f"UNCHECKED {rec['why']}"
            print(f"[{k}/{len(todo)}] {sid}: {line}", file=sys.stderr)
    doc["method"] = dict(model=dict(file="yamnet.tflite", sha256=MODELS["yamnet"]["sha256"]),
                         families=FAMILIES, event_min=EVENT_MIN, context_min=CONTEXT_MIN)
    out.write_text(json.dumps(doc, indent=1))
    ok = [u for u in units.values() if u.get("status") == OK]
    counts: dict[str, int] = {}
    for u in ok:
        for e in u["events"]:
            counts[e["family"]] = counts.get(e["family"], 0) + 1
    write_manifest(cfg.out_dir / "sounds", tool="sounds", tool_sha=source_hash(__file__), method=doc["method"], sources=len(units),
                   measured=len(ok), unchecked=sorted(k for k, u in units.items() if u.get("status") != OK), events=counts)
    return doc


def load(cfg_out) -> dict[str, dict]:
    from pathlib import Path
    p = Path(cfg_out) / "sounds.json"
    return (json.loads(p.read_text()).get("sources") or {}) if p.exists() else {}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="vocal and crowd sound events per source (heard, not seen)")
    ap.add_argument("--config", required=True)
    ap.add_argument("--only")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args(argv)
    from .config import load_config
    doc = run(load_config(a.config), a.only, a.workers, a.force)
    ok = [u for u in doc["sources"].values() if u.get("status") == OK]
    counts: dict[str, int] = {}
    for u in ok:
        for e in u["events"]:
            counts[e["family"]] = counts.get(e["family"], 0) + 1
    print(f"{len(doc['sources'])} sources, {len(ok)} measured; events: " + (", ".join(f"{k} {v}" for k, v in sorted(counts.items())) or "none"),
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
