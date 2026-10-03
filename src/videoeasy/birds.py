"""Birds heard, as a measurement: BirdNET over the camera audio.

The editor (29 Sep 2026): run BirdNET over the B-roll audio, and later put small
text labels in the film where a bird call is heard. A bird HEARD on a clip is
not a bird SEEN in it (the catalogue already keeps heard and seen apart), and
B-roll audio is not in the film's mix: labels will come from what the film
actually plays, so both roles are measured here.

  1. BirdNET's location model (2.4) gives the species plausible at the shoot
     location in the shoot week (config `location`); the acoustic model is
     limited to that list;
  2. each source's audio, summed to mono at 48 kHz, is read in 3 s windows
     every 1.5 s by the BirdNET 2.4 acoustic model (TFLite on ai-edge-litert,
     local, no network after the model download);
  3. every detection at DETECT_MIN or above is kept as evidence, with its
     confidence; CONFIDENT marks the ones worth a listen. Nothing is labelled
     on this alone: a species name on screen in a nature film has
     to be right, so a person confirms each one first.

Camera audio here is quiet (about -50 dBFS mean, the bird band near -62);
lifting it 30 dB changed no detection on the probes, so it is measured as
recorded.

    uv run --extra birds python -m videoeasy.birds --config config.<film>.yaml [--role broll|aroll|all] [--only SRC]
    -> <out_dir>/birds.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import subprocess
import sys
import tempfile
from pathlib import Path

from .evalrun import OK, UNCHECKED, source_hash

MODEL = ("acoustic", "2.4", "tf")
GEO = ("geo", "2.4", "tf")
LIBRARY = "litert"
DETECT_MIN = 0.1        # kept as evidence
CONFIDENT = 0.5         # provisional: worth a listen; nothing is labelled on BirdNET alone
OVERLAP_S = 1.5         # 3 s windows every 1.5 s
GEO_MIN = 0.03          # BirdNET's default location threshold
TOP_K = 3


def birdnet_week(date: str) -> int:
    """BirdNET's week of the year: four per month, 1-48."""
    d = dt.date.fromisoformat(date)
    return (d.month - 1) * 4 + min(4, (d.day - 1) // 7 + 1)


def split_name(name: str) -> tuple[str, str]:
    sci, _, common = name.partition("_")
    return sci, common


def extract(src: Path, dst: Path) -> None:
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vn", "-af", "pan=mono|c0=c0+c1", "-ar", "48000",
                    "-c:a", "pcm_s16le", str(dst)], check=True)


def has_audio(src: Path) -> bool:
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "a", "-show_entries", "stream=index", "-of", "csv=p=0", str(src)],
                       capture_output=True, text=True)
    return bool(r.stdout.strip())


def summarise(sources: dict) -> list[dict]:
    """Per species: detections, sources, best confidence, and how many clear CONFIDENT."""
    by: dict[str, dict] = {}
    for sid, s in sources.items():
        for d in s.get("detections", []):
            e = by.setdefault(d["sci"], dict(sci=d["sci"], common=d["common"], detections=0, confident=0, best=0.0, sources=set()))
            e["detections"] += 1
            e["confident"] += d["confidence"] >= CONFIDENT
            e["best"] = max(e["best"], d["confidence"])
            e["sources"].add(sid)
    out = [dict(e, best=round(e["best"], 3), sources=sorted(e["sources"])) for e in by.values()]
    return sorted(out, key=lambda e: (-e["confident"], -e["best"]))


def run(cfg, role: str = "all", only: str | None = None, workers: int = 4) -> dict:
    import birdnet
    from . import segments as segments_mod
    loc = (cfg.raw.get("location") or {})
    if not {"lat", "lon", "shoot_date"} <= set(loc):
        raise SystemExit("config has no location (lat, lon, shoot_date): BirdNET's species list depends on place and week")
    week = birdnet_week(str(loc["shoot_date"]))
    geo = birdnet.load(*GEO, library=LIBRARY)
    species = geo.predict(float(loc["lat"]), float(loc["lon"]), week=week, min_confidence=GEO_MIN).to_set()
    srcs: dict[str, dict] = {}
    for u in segments_mod.units(cfg):
        if (role == "all" or u["role"] == role) and (only is None or u["source_id"] == only):
            srcs.setdefault(u["source_id"], dict(path=u["source_path"], role=u["role"]))
    out_path = cfg.out_dir / "birds.json"
    doc = json.loads(out_path.read_text()) if out_path.exists() else {}
    doc["method"] = dict(model=f"BirdNET {MODEL[0]} {MODEL[1]} ({LIBRARY})", geo=f"BirdNET geo {GEO[1]}", place=loc.get("place"),
                         lat=loc["lat"], lon=loc["lon"], shoot_date=str(loc["shoot_date"]), week=week, species_list=len(species),
                         geo_min=GEO_MIN, detect_min=DETECT_MIN, confident=CONFIDENT, overlap_s=OVERLAP_S, top_k=TOP_K,
                         tool_sha=source_hash(__file__))
    sources = doc.setdefault("sources", {})
    with tempfile.TemporaryDirectory() as td:
        wavs: dict[str, Path] = {}
        for sid, s in sorted(srcs.items()):
            p = Path(s["path"])
            if not has_audio(p):
                sources[sid] = dict(status=UNCHECKED, role=s["role"], why="no audio track")
                continue
            w = Path(td) / f"{sid}.wav"
            extract(p, w)
            wavs[sid] = w
        print(f"{len(wavs)} sources with audio, {len(species)} species plausible at {loc.get('place')} in week {week}", file=sys.stderr)
        model = birdnet.load(*MODEL, library=LIBRARY)
        res = model.predict([str(w) for w in wavs.values()], top_k=TOP_K, overlap_duration_s=OVERLAP_S,
                            custom_species_list=species, default_confidence_threshold=DETECT_MIN, n_workers=workers)
        df = res.to_dataframe()
        by_file: dict[str, list[dict]] = {}
        for r in df.itertuples(index=False):
            sci, common = split_name(str(r.species_name))
            by_file.setdefault(Path(str(r.input)).stem, []).append(
                dict(t0=float(r.start_time), t1=float(r.end_time), sci=sci, common=common, confidence=round(float(r.confidence), 3)))
        for sid in wavs:
            dets = sorted(by_file.get(sid, []), key=lambda d: (d["t0"], -d["confidence"]))
            sources[sid] = dict(status=OK, role=srcs[sid]["role"], detections=dets,
                                confident=sum(1 for d in dets if d["confidence"] >= CONFIDENT))
    doc["species"] = summarise({k: v for k, v in sources.items() if v.get("status") == OK})
    out_path.write_text(json.dumps(doc, indent=1))
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--role", choices=["all", "broll", "aroll"], default="all")
    ap.add_argument("--only", default=None, help="one source id")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args(argv)
    from .config import load_config
    doc = run(load_config(args.config), args.role, args.only, args.workers)
    sp = doc["species"]
    print(f"{len(sp)} species detected at {DETECT_MIN}+; confident (>= {CONFIDENT}): "
          + (", ".join(f"{e['common']} {e['confident']}x (best {e['best']})" for e in sp if e["confident"]) or "none"), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
