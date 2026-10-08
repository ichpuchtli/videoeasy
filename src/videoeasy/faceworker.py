# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["mediapipe==0.10.21", "numpy<2"]
# ///
"""Raw face and sound measurement of ONE source, run in its own environment.

faces.py and sounds.py start this file with `uv run --script`, never import it.
MediaPipe 0.10.21 pins numpy<2 and protobuf 4, which the project's environment
has outgrown, and MediaPipe 1.0.1 aborts on macOS: its face detector graph
opens a Metal helper even with the CPU delegate ("Check failed: service_
Service is unavailable", 4 Oct 2026). So the measurement runs on the CPU in a
pinned environment of its own (uv caches it), and only stdlib, numpy, OpenCV
(MediaPipe's opencv-contrib) and MediaPipe are imported here. The output is
raw evidence; linking, baselines and every judgment happen in the project.

    uv run --script faceworker.py faces SRC OUT.json --yunet M.onnx --landmarker M.task [--fps 5] [--long 1920]
    uv run --script faceworker.py sounds SRC OUT.json --yamnet M.tflite

faces: the source decoded at --fps (VideoToolbox when it is there), long side
--long px; YuNet (OpenCV zoo, MIT) finds faces on each frame; every face at or
over --min-px is cropped with margin and MediaPipe's FaceLandmarker
(Apache-2.0) reads its 52 blendshapes and head pose from the crop. A crop can
hold a neighbour's face, so of up to two faces found in it the one nearest the
crop's centre is kept.

sounds: the source's audio, mono at 16 kHz, read by YAMNet (AudioSet's 521
classes, Apache-2.0) in its ~0.96 s windows; every class at or over 0.05 in a
window's top 25 is kept.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

VERSION = 1
CROP_MARGIN = 1.8      # crop side, in face-box sides
CROP_PX = 384          # crops are resized to this width before the landmarker


def probe(path: str) -> tuple[int, int, float]:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries", "stream=width,height:format=duration",
                          "-of", "json", path], capture_output=True, text=True, check=True).stdout
    d = json.loads(out)
    s = d["streams"][0]
    return int(s["width"]), int(s["height"]), float(d["format"].get("duration") or 0.0)


def decode(path: str, fps: float, long_px: int):
    """(t, BGR frame) at `fps`. VideoToolbox first; plain decode when it fails before the first frame."""
    import numpy as np
    w, h, dur = probe(path)
    scale = min(1.0, long_px / max(w, h))
    W, H = int(round(w * scale / 2) * 2), int(round(h * scale / 2) * 2)
    n = W * H * 3
    for hw in (["-hwaccel", "videotoolbox"], []):
        p = subprocess.Popen(["ffmpeg", "-v", "error", *hw, "-i", path, "-an", "-vf", f"fps={fps},scale={W}:{H}",
                              "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
        i = 0
        while True:
            buf = p.stdout.read(n)
            if len(buf) < n:
                break
            yield i / fps, np.frombuffer(buf, np.uint8).reshape(H, W, 3)
            i += 1
        p.wait()
        if i:
            return
    raise RuntimeError(f"ffmpeg decoded no frames ({dur:.2f} s long; one sample every {1 / fps:.2f} s)")


def euler(m) -> tuple[float, float, float]:
    import numpy as np
    r = np.asarray(m)[:3, :3]
    yaw = np.degrees(np.arctan2(-r[2, 0], np.hypot(r[0, 0], r[1, 0])))
    pitch = np.degrees(np.arctan2(r[2, 1], r[2, 2]))
    roll = np.degrees(np.arctan2(r[1, 0], r[0, 0]))
    return round(float(yaw), 1), round(float(pitch), 1), round(float(roll), 1)


def faces(src: str, out: str, yunet: str, landmarker: str, fps: float, long_px: int, min_px: int) -> dict:
    import cv2
    import mediapipe as mp
    import numpy as np
    det = cv2.FaceDetectorYN.create(yunet, "", (320, 320), 0.6, 0.3, 5000)
    lm = mp.tasks.vision.FaceLandmarker.create_from_options(mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=landmarker, delegate=mp.tasks.BaseOptions.Delegate.CPU),
        running_mode=mp.tasks.vision.RunningMode.IMAGE, num_faces=2, min_face_detection_confidence=0.4,
        output_face_blendshapes=True, output_facial_transformation_matrixes=True))
    names: list[str] | None = None
    rows = []
    t0 = time.time()
    nframes = 0
    W = H = 0
    for t, img in decode(src, fps, long_px):
        nframes += 1
        H, W = img.shape[:2]
        det.setInputSize((W, H))
        _, found = det.detect(img)
        for f in ([] if found is None else found):
            x, y, w, h, score = (float(v) for v in (f[0], f[1], f[2], f[3], f[14]))
            px = min(w, h)
            row = [round(t, 3), round(x, 1), round(y, 1), round(w, 1), round(h, 1), round(score, 3), None, None, None]
            if px >= min_px:
                cx, cy, s = x + w / 2, y + h / 2, max(w, h) * CROP_MARGIN
                x0, y0, x1, y1 = int(max(0, cx - s / 2)), int(max(0, cy - s / 2)), int(min(W, cx + s / 2)), int(min(H, cy + s / 2))
                crop = img[y0:y1, x0:x1]
                if crop.size:
                    k = CROP_PX / crop.shape[1]
                    crop = cv2.resize(crop, (CROP_PX, max(1, int(round(crop.shape[0] * k)))))
                    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    row[8] = round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 1)   # sharpness of the face crop
                    res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))))
                    if res.face_blendshapes:
                        ch, cw = crop.shape[:2]
                        best = min(range(len(res.face_landmarks)), key=lambda i: (
                            (np.mean([p.x for p in res.face_landmarks[i]]) - 0.5) ** 2 * cw * cw
                            + (np.mean([p.y for p in res.face_landmarks[i]]) - 0.5) ** 2 * ch * ch))
                        cats = res.face_blendshapes[best]
                        if names is None:
                            names = [c.category_name for c in cats]
                        row[6] = [round(float(c.score), 3) for c in cats]
                        row[7] = list(euler(res.facial_transformation_matrixes[best]))
            rows.append(row)
    doc = dict(tool="faceworker", version=VERSION, kind="faces", source=src, fps=fps, long_px=long_px, min_px=min_px,
               frame_w=W, frame_h=H, frames=nframes, seconds=round(time.time() - t0, 1),
               columns=["t", "x", "y", "w", "h", "det", "blendshapes", "pose_ypr", "sharp"], blendshape_names=names or [],
               detections=rows)
    Path(out).write_text(json.dumps(doc, separators=(",", ":")))
    return doc


def sounds(src: str, out: str, yamnet: str) -> dict:
    import mediapipe as mp
    import numpy as np
    sr = 16000
    p = subprocess.run(["ffmpeg", "-v", "error", "-i", src, "-vn", "-ac", "1", "-ar", str(sr), "-f", "f32le", "-"], capture_output=True)
    if p.returncode != 0:
        raise RuntimeError(p.stderr.decode(errors="replace")[:200])
    x = np.frombuffer(p.stdout, np.float32)
    windows = []
    if len(x) >= sr // 2:
        clf = mp.tasks.audio.AudioClassifier.create_from_options(mp.tasks.audio.AudioClassifierOptions(
            base_options=mp.tasks.BaseOptions(model_asset_path=yamnet, delegate=mp.tasks.BaseOptions.Delegate.CPU),
            running_mode=mp.tasks.audio.RunningMode.AUDIO_CLIPS, max_results=25, score_threshold=0.05))
        for r in clf.classify(mp.tasks.components.containers.AudioData.create_from_array(x.astype(float), sr)):
            windows.append([round(r.timestamp_ms / 1000, 3), {c.category_name: round(float(c.score), 3) for c in r.classifications[0].categories}])
    rms = float(np.sqrt(np.mean(x.astype(np.float64) ** 2))) if len(x) else 0.0
    doc = dict(tool="faceworker", version=VERSION, kind="sounds", source=src, sample_rate=sr, audio_s=round(len(x) / sr, 2),
               rms_dbfs=round(20 * np.log10(rms), 1) if rms > 0 else None, windows=windows)
    Path(out).write_text(json.dumps(doc, separators=(",", ":")))
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["faces", "sounds"])
    ap.add_argument("src")
    ap.add_argument("out")
    ap.add_argument("--yunet")
    ap.add_argument("--landmarker")
    ap.add_argument("--yamnet")
    ap.add_argument("--fps", type=float, default=5.0)
    ap.add_argument("--long", type=int, default=1920)
    ap.add_argument("--min-px", type=int, default=40)
    a = ap.parse_args(argv)
    if a.kind == "faces":
        d = faces(a.src, a.out, a.yunet, a.landmarker, a.fps, a.long, a.min_px)
        print(f"{Path(a.src).name}: {d['frames']} frames, {len(d['detections'])} faces, "
              f"{sum(1 for r in d['detections'] if r[6])} read, {d['seconds']} s", file=sys.stderr)
    else:
        d = sounds(a.src, a.out, a.yamnet)
        print(f"{Path(a.src).name}: {len(d['windows'])} sound windows", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
