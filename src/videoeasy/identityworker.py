# /// script
# requires-python = ">=3.11,<3.13"
# dependencies = ["insightface==1.0.1", "onnxruntime==1.28.0"]
# ///
"""Face embeddings of a list of images, run in its own environment.

identity.py starts this file with the interpreter of its script environment,
never imports it: the project's environment never holds a recognition
library. The model is InsightFace's buffalo_l pack (SCRFD detection, ArcFace
recognition), which identity.py downloads at setup into the videoeasy model
cache and checks against a pinned hash. The InsightFace code is MIT; its
pretrained models are for non-commercial research only. Only stdlib, numpy,
OpenCV and InsightFace are imported here, and the output is raw evidence:
which images held a face and that face's embedding. Grouping, matching and
every judgment happen in the project.

    python identityworker.py LIST.json OUT.npz --root MODELROOT

LIST.json: [{"file": "/abs/crop.jpg", "pick": "centre" | "largest"}, ...]. A
video crop keeps the face nearest its centre (a crop can hold a neighbour); a
reference photo keeps its largest face. A tight crop with no room around the
face (a gallery thumbnail, an aligned 112 px face) is padded and detected
again before it counts as faceless.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

VERSION = 1


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("items")
    ap.add_argument("out")
    ap.add_argument("--root", required=True, help="model root: <root>/models/buffalo_l/*.onnx")
    a = ap.parse_args(argv)
    import cv2
    import numpy as np
    from insightface.app import FaceAnalysis
    app = FaceAnalysis(name="buffalo_l", root=a.root, providers=["CPUExecutionProvider"], allowed_modules=["detection", "recognition"])
    app.prepare(ctx_id=-1, det_size=(640, 640))
    items = json.loads(Path(a.items).read_text())
    E = np.zeros((len(items), 512), np.float32)
    found = np.zeros(len(items), np.int8)
    t0 = time.time()
    for i, it in enumerate(items):
        img = cv2.imread(it["file"])
        if img is None:
            continue
        faces = app.get(img)
        if not faces:
            pad = max(img.shape[:2]) // 2
            img = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0))
            faces = app.get(img)
        if not faces:
            continue
        h, w = img.shape[:2]
        if it.get("pick") == "largest":
            f = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        else:
            f = min(faces, key=lambda f: ((f.bbox[0] + f.bbox[2]) / 2 - w / 2) ** 2 + ((f.bbox[1] + f.bbox[3]) / 2 - h / 2) ** 2)
        E[i] = f.normed_embedding
        found[i] = 1
        if (i + 1) % 500 == 0:
            print(f"[{i + 1}/{len(items)}] embedded, {time.time() - t0:.0f} s", file=sys.stderr)
    np.savez(a.out, files=np.array([it["file"] for it in items]), embeddings=E, found=found, version=VERSION)
    print(f"{len(items)} images, {int(found.sum())} with a face, {time.time() - t0:.0f} s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
