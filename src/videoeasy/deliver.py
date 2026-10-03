"""Delivery master: raise a render to the delivery loudness and limit its peaks, then measure the result.

The edit keeps dialogue at -24 LUFS (the lay-in's working target). A master
for YouTube is delivered at the level recorded in `editorial/intent.json`
(`delivery_loudness`: -16 LUFS integrated, -1 dBTP). This is ffmpeg's
two-pass `loudnorm` on the audio with the video stream copied untouched, so
the picture the scorers judged is byte-identical to the picture delivered.

    uv run python -m videoeasy.deliver --film data/films/<film> \
        --render editorial/renders/<film>-cut-v6.mp4
    -> editorial/renders/<film>-cut-v6.master.mp4 (+ .manifest.json)

The result is measured again with ebur128 and compared with the rule; the
manifest says honoured or broken with the numbers. Nothing here changes a
timeline or a source file.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

from .evalrun import OK, UNCHECKED, ffprobe_duration, fingerprint, refuse_overwrite, source_hash, write_manifest


def measure(path: str | Path) -> dict:
    """Integrated loudness, range and true peak of the whole programme."""
    p = subprocess.run(["ffmpeg", "-v", "info", "-i", str(path), "-vn", "-af", "ebur128=peak=true:framelog=quiet", "-f", "null", "-"],
                       capture_output=True, text=True)
    if p.returncode != 0:
        return dict(status=UNCHECKED, error=p.stderr[-160:])
    I = re.search(r"\n\s+I:\s+(-?[\d.]+) LUFS", p.stderr)
    LRA = re.search(r"LRA:\s+(-?[\d.]+) LU", p.stderr)
    P = re.search(r"Peak:\s+(-?[\d.]+) dBFS", p.stderr)
    if not (I and P):
        return dict(status=UNCHECKED, error="ebur128 printed no summary")
    return dict(status=OK, lufs=float(I.group(1)), lra=float(LRA.group(1)) if LRA else None, true_peak=float(P.group(1)))


def loudnorm_pass1(path: str | Path, target: float, tp: float, lra: float = 11.0) -> dict:
    p = subprocess.run(["ffmpeg", "-v", "info", "-i", str(path), "-vn", "-af", f"loudnorm=I={target}:TP={tp}:LRA={lra}:print_format=json",
                        "-f", "null", "-"], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"loudnorm pass 1 failed: {p.stderr[-160:]}")
    m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", p.stderr, re.S)
    if not m:
        raise RuntimeError("loudnorm pass 1 printed no JSON")
    return json.loads(m.group(0))


def master(render: Path, out: Path, target: float = -16.0, tp: float = -1.0, lra: float = 11.0) -> dict:
    """Two-pass loudnorm on the audio, video stream-copied. Returns pass-1 stats and the measured result."""
    stats = loudnorm_pass1(render, target, tp, lra)
    # linear gain to the target (no dynamics change), then a limiter for the true-peak ceiling with 1.5 dB of margin (the sample-peak limiter overshoots true peak by about 1 dB)
    lim = 10 ** ((tp - 1.5) / 20)
    af = (f"loudnorm=I={target}:TP={tp}:LRA={lra}:measured_I={stats['input_i']}:measured_TP={stats['input_tp']}:"
          f"measured_LRA={stats['input_lra']}:measured_thresh={stats['input_thresh']}:offset={stats['target_offset']}:linear=true,"
          f"alimiter=limit={lim:.4f}:attack=5:release=50:level=false")
    p = subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(render), "-c:v", "copy", "-af", af, "-ar", "48000",
                        "-c:a", "aac", "-b:a", "256k", "-movflags", "+faststart", str(out)], capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"loudnorm pass 2 failed: {p.stderr[-200:]}")
    return dict(pass1=stats, filter=af, result=measure(out))


def verdict(meas: dict, target: float, tp: float, tol: float) -> str:
    if meas.get("status") != OK:
        return "unchecked"
    return "honoured" if abs(meas["lufs"] - target) <= tol and meas["true_peak"] <= tp + 0.1 else "broken"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--film", required=True)
    ap.add_argument("--render", required=True, help="render path relative to --film")
    ap.add_argument("--out", default=None, help="master path (default <render stem>.master.mp4 beside it)")
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args(argv)
    film = Path(args.film)
    render = film / args.render
    out = Path(args.out) if args.out else render.with_name(render.stem + ".master.mp4")
    refuse_overwrite([out], args.overwrite)
    if ffprobe_duration(render) is None:
        raise SystemExit(f"render missing or unreadable: {render}")
    from . import intent as intent_mod
    rule = next((r for r in intent_mod.resolve(intent_mod.load(film))["active"] if r["kind"] == "delivery_loudness"), None)
    if rule is None:
        raise SystemExit("no active delivery_loudness rule in editorial/intent.json; nothing delivered")
    target, tp, tol = float(rule["target_lufs"]), float(rule["max_true_peak_dbtp"]), float(rule.get("tolerance_lu", 1.0))
    before = measure(render)
    rep = master(render, out, target, tp)
    v = verdict(rep["result"], target, tp, tol)
    write_manifest(out.with_suffix(""), tool="deliver", tool_sha=source_hash(__file__),
                   render=dict(path=str(render), sha=fingerprint(render), measured=before),
                   master=dict(path=str(out), sha=fingerprint(out), measured=rep["result"]),
                   rule=dict(id=rule["id"], target_lufs=target, max_true_peak_dbtp=tp, tolerance_lu=tol, status=v),
                   loudnorm=dict(pass1=rep["pass1"], filter=rep["filter"], video="stream copy", audio="aac 256k 48k"))
    print(f"master {out.name}: {before.get('lufs')} LUFS / {before.get('true_peak')} dBTP -> {rep['result'].get('lufs')} LUFS / "
          f"{rep['result'].get('true_peak')} dBTP; rule {rule['id']} {v}", file=sys.stderr)
    return 0 if v == "honoured" else 1


if __name__ == "__main__":
    sys.exit(main())
