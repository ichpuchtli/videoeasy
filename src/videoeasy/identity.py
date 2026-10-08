"""Face identity: which face tracks show the same person, and who the editor says that is.

The editor (8 Oct 2026, the event shoot): a participant's before/after reel
needs that man's own moments, so face tracks have to be linked to people.
The recognition model is never part of this repository, which is MIT: its
licence is its own. `setup` downloads InsightFace's buffalo_l pack (pinned
hash) into the videoeasy model cache, whose pretrained models are for
non-commercial research only, and shows that licence; identityworker.py runs
it in a script environment of its own, so the project's environment never
holds a recognition library. An editor who needs another model runs their
own tool on the exported request and `import`s its answer. docs/identity.md
has the licences and both files.

  setup    download and check the model, build the worker environment, load the model once
  export   each face track with a recognisable view gets up to CROPS_PER_TRACK graded crops of its most
           frontal, largest readable samples -> <work>/identity/request.json + crops/
  run      export, then the worker embeds every new crop (cached per crop), and here the tracks are matched
           to reference photos (--people) or grouped (average linkage on cosine distance, cut at 1 - SAME)
           -> <work>/identity/answer.json, imported as below
  import   an answer (the built-in one, or another tool's), checked against the request (its id, every key,
           each source's current faces measurement) -> <out>/identity.json: PROPOSED links under the
           answer's own ids, never names
  review   a local page (never published: these are people's faces): each proposed person's tracks,
           a label field and a keep/drop toggle per track; the editor downloads the decisions
  confirm  merges the downloaded decisions into <film>/editorial/identity-decisions.json

Only a `confirmed` link names a person downstream (`register find --person`). A proposal is the tool's
guess and never stands in for a name. A decision is bound to the faces measurement it was made on:
track ids are positions within a source (lessons 6.5), so when a source is measured again its
decisions stop applying instead of naming whoever holds that id now.

    uv run python -m videoeasy.identity setup
    uv run python -m videoeasy.identity run     --config C [--people DIR] [--same 0.45] [--only SRC] [--workers 4]
    uv run python -m videoeasy.identity export  --config C [--only SRC] [--workers 4]      # for another tool
    uv run python -m videoeasy.identity import  --config C --answer ANSWER.json             # that tool's answer
    uv run python -m videoeasy.identity review  --config C
    uv run python -m videoeasy.identity confirm --config C --decisions DOWNLOADED.json [--by NAME]
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from . import faces as faces_mod
from .evalrun import INVALID, OK, UNCHECKED, source_hash, text_hash, write_manifest

REQUEST_SCHEMA = "videoeasy.identity.request/1"
ANSWER_SCHEMA = "videoeasy.identity.answer/1"
DECISIONS_SCHEMA = "videoeasy.identity.decisions/1"
MIN_PX = 80             # provisional: a recognition model needs about this many pixels across a face
CROPS_PER_TRACK = 3
CROP_GAP_S = 1.0        # crops of one track are at least this far apart
CROP_PX = 320
CROP_MARGIN = 1.8       # crop side, in face-box sides (as faceworker crops for the landmarker)
OVERLAP_S = 0.5         # two tracks of one clip sharing this much time cannot be one person
PROPOSED, CONFIRMED, REJECTED, NONE = "proposed", "confirmed", "rejected", "none"
SAME = 0.45             # provisional: cosine similarity at or over which two tracks (or a track and a reference) are one person
WORKER = Path(__file__).with_name("identityworker.py")
MODEL = dict(file="buffalo_l.zip", url="https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip",
             sha256="80ffe37d8a5940d59a7384c201a2a38d4741f2f3c51eef46ebb28218a7b0ca2f", members=("det_10g.onnx", "w600k_r50.onnx"),
             name="InsightFace buffalo_l (SCRFD det_10g, ArcFace w600k_r50), CPU",
             licence="InsightFace pretrained models: non-commercial research use only (the InsightFace code is MIT)")
NOTICE = ("Face identity uses InsightFace's buffalo_l models, downloaded from InsightFace's own release into the videoeasy model "
          "cache. They are not part of videoeasy (MIT): InsightFace licenses its pretrained models for non-commercial research "
          "only. For other work, use a model whose licence allows it (docs/identity.md: export, then import its answer).")


# ----------------------------------------------------------------------- model
def model_root() -> Path:
    """<cache>/models/insightface, laid out as InsightFace expects: <root>/models/buffalo_l/*.onnx."""
    return faces_mod.model_dir() / "insightface"


def ensure_recognition() -> Path:
    """The pinned buffalo_l pack, downloaded on first use, with the two models used unpacked beside it."""
    import zipfile
    zp = faces_mod.ensure_model("buffalo_l", MODEL)
    d = model_root() / "models" / "buffalo_l"
    if not all((d / m).exists() for m in MODEL["members"]):
        d.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zp) as z:
            for m in MODEL["members"]:
                z.extract(m, d)
    return model_root()


def embed_files(files: list[str], pick: str, work: Path) -> list[np.ndarray | None]:
    """The worker's embedding of the chosen face in each image, None where it found no face."""
    work.mkdir(parents=True, exist_ok=True)
    items, out = work / "embed-items.json", work / "embed.part.npz"
    items.write_text(json.dumps([dict(file=f, pick=pick) for f in files]))
    args = [faces_mod.worker_python(WORKER, "import insightface, onnxruntime"), str(WORKER), str(items), str(out),
            "--root", str(ensure_recognition())]
    tail: list[str] = []
    with subprocess.Popen(args, stderr=subprocess.PIPE, stdout=subprocess.DEVNULL, text=True) as p:
        for ln in p.stderr:
            if ln.startswith("[") or " with a face" in ln:
                print("  " + ln.rstrip(), file=sys.stderr)
            elif ln.strip():
                tail = (tail + [ln.strip()])[-4:]
    if p.returncode != 0 or not out.exists():
        raise SystemExit(f"{WORKER.name} failed: " + " | ".join(tail)[:400])
    z = np.load(out)
    got = {f: (e if ok else None) for f, e, ok in zip(z["files"].tolist(), z["embeddings"], z["found"])}
    out.unlink()
    items.unlink()
    return [got[f] for f in files]


def embeddings(cfg, req: dict) -> dict[str, np.ndarray | None]:
    """Every crop's embedding, cached by crop file (crop names carry their track and sample) and by model."""
    root = request_dir(cfg)
    cache_p = root / "embeddings.npz"
    cache: dict[str, np.ndarray | None] = {}
    if cache_p.exists():
        z = np.load(cache_p)
        if str(z["model"]) == MODEL["sha256"]:
            cache = {f: (e if ok else None) for f, e, ok in zip(z["files"].tolist(), z["embeddings"], z["found"])}
    need = sorted({c["file"] for t in req["tracks"] for c in t["crops"]} - set(cache))
    if need:
        print(f"embedding {len(need)} new crops ({len(cache)} cached)", file=sys.stderr)
        for f, e in zip(need, embed_files([str(root / f) for f in need], "centre", root)):
            cache[f] = e
        files = sorted(cache)
        np.savez(cache_p, files=np.array(files), model=MODEL["sha256"], found=np.array([cache[f] is not None for f in files], np.int8),
                 embeddings=np.stack([cache[f] if cache[f] is not None else np.zeros(512, np.float32) for f in files]))
    return cache


def references(people: Path, work: Path) -> dict[str, np.ndarray]:
    """{label: unit centroid} from one folder of photos per person (the folder's name is the label)."""
    photos = [(d.name, f) for d in sorted(Path(people).iterdir()) if d.is_dir()
              for f in sorted(d.iterdir()) if f.suffix.lower() in (".jpg", ".jpeg", ".png")]
    es = embed_files([str(f) for _, f in photos], "largest", work) if photos else []
    by: dict[str, list] = {}
    for (label, _), e in zip(photos, es):
        if e is not None:
            by.setdefault(label, []).append(e)
    for d in sorted(Path(people).iterdir()):
        if d.is_dir() and d.name not in by:
            print(f"reference {d.name}: no face found in its photos", file=sys.stderr)
    return {k: _unit(np.mean(v, axis=0)) for k, v in by.items()}


def _unit(v: np.ndarray) -> np.ndarray:
    return v / (np.linalg.norm(v) + 1e-9)


def propose(req: dict, emb: dict[str, np.ndarray | None], refs: dict[str, np.ndarray], same: float = SAME) -> dict:
    """An answer to the request: tracks matched to a reference at `same` or more, the rest grouped by average linkage
    on cosine distance cut at 1 - `same`. A group of one track is no proposal."""
    keys, rows, entries = [], [], {}
    for t in req["tracks"]:
        es = [emb[c["file"]] for c in t["crops"] if emb.get(c["file"]) is not None]
        if es:
            keys.append(t["key"])
            rows.append(_unit(np.mean(es, axis=0)))
        else:
            entries[t["key"]] = dict(key=t["key"], person=None, why="no face found in the crops")
    X = np.stack(rows) if rows else np.zeros((0, 512), np.float32)
    people: dict[str, dict] = {}
    rest = list(range(len(keys)))
    if refs and len(keys):
        names = sorted(refs)
        S = X @ np.stack([refs[n] for n in names]).T
        rest = []
        for i, k in enumerate(keys):
            j = int(np.argmax(S[i]))
            if S[i, j] >= same:
                people[f"ref:{names[j]}"] = dict(label=names[j])
                entries[k] = dict(key=k, person=f"ref:{names[j]}", score=round(float(S[i, j]), 3))
            else:
                rest.append(i)
    groups: list[list[int]] = []
    if len(rest) >= 2:
        from scipy.cluster.hierarchy import fcluster, linkage
        lab = fcluster(linkage(X[rest], method="average", metric="cosine"), t=1 - same, criterion="distance")
        by: dict[int, list[int]] = {}
        for i, g in zip(rest, lab):
            by.setdefault(int(g), []).append(i)
        groups = sorted((m for m in by.values() if len(m) >= 2), key=lambda m: (-len(m), keys[m[0]]))
    for n, members in enumerate(groups, 1):
        c = _unit(X[members].mean(axis=0))
        people[f"g{n:03d}"] = dict(label=None, tracks=len(members))
        for i in members:
            entries[keys[i]] = dict(key=keys[i], person=f"g{n:03d}", score=round(float(X[i] @ c), 3))
    for i in rest:
        entries.setdefault(keys[i], dict(key=keys[i], person=None, why="no other track matched"))
    return dict(schema=ANSWER_SCHEMA, request_id=req["request_id"],
                tool=dict(name="videoeasy identity (built in)", model=MODEL["name"], model_sha256=MODEL["sha256"],
                          model_licence=MODEL["licence"], same=same, references=sorted(refs)),
                people=people, tracks=[entries[t["key"]] for t in req["tracks"]])


# ---------------------------------------------------------------------- export
def pick_samples(tr: dict, n: int = CROPS_PER_TRACK) -> list[list]:
    """The track's most frontal, largest readable samples, at least CROP_GAP_S apart, in time order: [t, x, y, w, h]."""
    pool = [s for s in tr["samples"] if s[5] and min(s[3], s[4]) >= MIN_PX]
    pool.sort(key=lambda s: (abs(s[6] or 0.0), -min(s[3], s[4]), s[0]))
    out: list[list] = []
    for s in pool:
        if all(abs(s[0] - o[0]) >= CROP_GAP_S for o in out):
            out.append(list(s[:5]))
            if len(out) == n:
                break
    return sorted(out)


def crop_image(img, box):
    x, y, w, h = box
    cx, cy, s = x + w / 2, y + h / 2, max(w, h) * CROP_MARGIN
    return img.crop((int(cx - s / 2), int(cy - s / 2), int(cx + s / 2), int(cy + s / 2))).resize((CROP_PX, CROP_PX))


def request_dir(cfg) -> Path:
    return cfg.work_dir / "identity"


def request_id(sources: dict, tracks: list[dict]) -> str:
    return hashlib.sha256(json.dumps([sources, tracks], sort_keys=True).encode()).hexdigest()[:16]


def current_raws(cfg) -> dict[str, str]:
    """Each measured source's faces evidence file (its name carries the source's content fingerprint)."""
    p = cfg.out_dir / "faces.json"
    if not p.exists():
        return {}
    return {sid: Path(s["raw"]).name for sid, s in json.loads(p.read_text())["sources"].items() if s.get("status") == OK and s.get("raw")}


def export(cfg, only: str | None = None, workers: int = 4) -> Path:
    """Crops of every track with a recognisable view, and the request that lists them. Crops are cached by name."""
    from .register import _grab, _jpeg, frame_size
    root = request_dir(cfg)
    raws = current_raws(cfg)
    recs, planned, skipped = {}, [], 0
    for sid in sorted(raws):
        if only and sid != only:
            continue
        rec = faces_mod.load_source(cfg.out_dir, sid)
        if rec is None:
            continue
        recs[sid] = rec
        for tr in rec["tracks"]:
            picks = pick_samples(tr)
            if not picks:
                skipped += 1
                continue
            crops = [dict(file=f"crops/{sid}/{tr['id']}-{text_hash(raws[sid], tr['id'], json.dumps(p))[:10]}.jpg", t=p[0], box=p[1:5])
                     for p in picks]
            planned.append(dict(key=f"{sid}/{tr['id']}", source=sid, track=tr["id"], t0=tr["t0"], t1=tr["t1"], px=tr["px"], crops=crops))

    todo: dict[str, list[dict]] = {}
    for t in planned:
        for c in t["crops"]:
            if not (root / c["file"]).exists():
                todo.setdefault(t["source"], []).append(c)

    def one(sid: str) -> tuple[str, int]:
        rec = recs[sid]
        size, grade, failed = frame_size(dict(path=rec["path"]), rec), cfg.grade_for(rec["path"]), 0
        for c in todo[sid]:
            try:
                img = _grab(rec["path"], c["t"], size, grade)
            except Exception:  # noqa: BLE001
                failed += 1
                continue
            (root / c["file"]).parent.mkdir(parents=True, exist_ok=True)
            (root / c["file"]).write_bytes(_jpeg(crop_image(img, c["box"])))
        return sid, failed

    with ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = [ex.submit(one, sid) for sid in sorted(todo)]
        for k, f in enumerate(as_completed(futs), 1):
            sid, failed = f.result()
            print(f"[{k}/{len(futs)}] {sid}: {len(todo[sid]) - failed} crops" + (f", {failed} frames failed" if failed else ""), file=sys.stderr)

    tracks, no_frame = [], 0
    for t in planned:
        t["crops"] = [c for c in t["crops"] if (root / c["file"]).exists()]
        if t["crops"]:
            tracks.append(t)
        else:
            no_frame += 1
    sources = {sid: raws[sid] for sid in recs}
    req = dict(schema=REQUEST_SCHEMA, request_id=request_id(sources, tracks),
               created=dt.datetime.now().astimezone().isoformat(timespec="seconds"), crop_px=CROP_PX, min_px=MIN_PX,
               note="Face crops of anonymous tracks, for a local identity tool. Answer with " + ANSWER_SCHEMA
                    + " (videoeasy docs/identity.md). Never publish or upload these crops.",
               sources=sources, tracks=tracks)
    root.mkdir(parents=True, exist_ok=True)
    (root / "request.json").write_text(json.dumps(req, indent=1))
    write_manifest(root / "request", tool="identity export", tool_sha=source_hash(__file__), request_id=req["request_id"],
                   sources=len(sources), tracks=len(tracks), crops=sum(len(t["crops"]) for t in tracks),
                   skipped=dict(no_recognisable_view=skipped, no_frame=no_frame), min_px=MIN_PX, crop_px=CROP_PX)
    print(f"{len(tracks)} tracks in {len(sources)} sources ({skipped} with no view of {MIN_PX} px or more, frontal and readable; "
          f"{no_frame} with no frame decoded)", file=sys.stderr)
    return root / "request.json"


# ---------------------------------------------------------------------- import
def _entry_problem(e) -> str | None:
    if not isinstance(e, dict) or not isinstance(e.get("key"), str):
        return "an entry without a key"
    p, s = e.get("person"), e.get("score")
    if p is not None and not (isinstance(p, (str, int)) and not isinstance(p, bool) and str(p).strip()):
        return "person is not a non-empty string, integer or null"
    if s is not None and not (isinstance(s, (int, float)) and not isinstance(s, bool)):
        return "score is not a number or null"
    return None


def overlaps(links: dict[str, dict]) -> set[str]:
    """Tracks proposed as one person that share more than OVERLAP_S of one clip's time: one face can't be in two places."""
    by: dict[tuple[str, str], list[tuple[str, dict]]] = {}
    for k, v in links.items():
        if v["status"] == PROPOSED:
            by.setdefault((v["proposal"], v["source"]), []).append((k, v))
    bad: set[str] = set()
    for group in by.values():
        for i, (ka, a) in enumerate(group):
            for kb, b in group[i + 1:]:
                if min(a["t1"], b["t1"]) - max(a["t0"], b["t0"]) > OVERLAP_S:
                    bad |= {ka, kb}
    return bad


def resolve_answer(req: dict, ans: dict, raws: dict[str, str]) -> dict:
    """The tool's answer as links, one per requested track. No names: a proposal carries the tool's id."""
    entries, problems = {}, []
    for e in ans.get("tracks") or []:
        why = _entry_problem(e)
        if why and not (isinstance(e, dict) and isinstance(e.get("key"), str)):
            problems.append(why)
            continue
        entries[e["key"]] = (e, why)
    known = {t["key"] for t in req["tracks"]}
    links = {}
    for t in req["tracks"]:
        k = t["key"]
        base = dict(source=t["source"], track=t["track"], t0=t["t0"], t1=t["t1"], raw=req["sources"][t["source"]],
                    crops=[c["file"] for c in t["crops"]])
        e, why = entries.get(k, (None, None))
        if raws.get(t["source"]) != base["raw"]:
            links[k] = dict(base, status=UNCHECKED, why="the source's faces were measured again after the export")
        elif e is None:
            links[k] = dict(base, status=UNCHECKED, why="the tool gave no answer for this track")
        elif why:
            links[k] = dict(base, status=INVALID, why=why)
        elif e.get("person") is None:
            links[k] = dict(base, status=NONE, why=str(e.get("why") or "")[:200] or None)
        else:
            links[k] = dict(base, status=PROPOSED, proposal=str(e["person"]),
                            score=round(float(e["score"]), 3) if e.get("score") is not None else None)
    conflicts = overlaps(links)
    for k in conflicts:
        links[k]["conflict"] = True
    counts: dict[str, int] = {}
    for v in links.values():
        counts[v["status"]] = counts.get(v["status"], 0) + 1
    people = {str(p): (v.get("label") if isinstance(v, dict) else None) for p, v in (ans.get("people") or {}).items()}
    return dict(schema="videoeasy.identity/1", request_id=req["request_id"], tool=ans.get("tool") or {}, tool_people=people,
                links=links, counts=counts, proposed_people=len({v["proposal"] for v in links.values() if v["status"] == PROPOSED}),
                conflicts=sorted(conflicts), unknown_keys=sorted(k for k in entries if k not in known), malformed=problems)


def import_answer(cfg, answer_path: Path) -> dict:
    req = json.loads((request_dir(cfg) / "request.json").read_text())
    return record(cfg, req, json.loads(Path(answer_path).read_text()), answer_path)


def record(cfg, req: dict, ans: dict, answer_path: Path) -> dict:
    if ans.get("schema") != ANSWER_SCHEMA:
        raise SystemExit(f"not an identity answer (schema {ANSWER_SCHEMA}): {answer_path}")
    if ans.get("request_id") != req["request_id"]:
        raise SystemExit("this answer was made for another export (request_id differs): run the tool again on the current request.json")
    doc = resolve_answer(req, ans, current_raws(cfg))
    doc.update(imported=dt.datetime.now().astimezone().isoformat(timespec="seconds"), answer=str(answer_path))
    (cfg.out_dir / "identity.json").write_text(json.dumps(doc, indent=1))
    tool = doc["tool"]
    write_manifest(cfg.out_dir / "identity", tool="identity import", tool_sha=source_hash(__file__), answer=str(answer_path),
                   request_id=req["request_id"], identity_tool=tool.get("name"), identity_model=tool.get("model"),
                   identity_model_licence=tool.get("model_licence", "not stated by the tool"), counts=doc["counts"],
                   conflicts=len(doc["conflicts"]), unknown_keys=len(doc["unknown_keys"]), malformed=len(doc["malformed"]),
                   unchecked=sorted(k for k, v in doc["links"].items() if v["status"] in (UNCHECKED, INVALID)),
                   note="proposals only: a link names a person once the editor confirms it (identity review, confirm)")
    return doc


def run(cfg, people: Path | None = None, same: float = SAME, only: str | None = None, workers: int = 4) -> dict:
    """The built-in tool: export, embed what is new, match or group, import."""
    print(NOTICE, file=sys.stderr)
    ensure_recognition()
    req_path = export(cfg, only, workers)
    req = json.loads(req_path.read_text())
    emb = embeddings(cfg, req)
    refs = references(people, request_dir(cfg)) if people else {}
    ans = propose(req, emb, refs, same)
    answer_path = request_dir(cfg) / "answer.json"
    answer_path.write_text(json.dumps(ans, indent=1))
    return record(cfg, req, ans, answer_path)


def setup() -> Path:
    """Download and check the model, build the worker environment, and load the model once on no images."""
    import tempfile
    print(NOTICE, file=sys.stderr)
    root = ensure_recognition()
    with tempfile.TemporaryDirectory() as d:
        embed_files([], "centre", Path(d))
    return root


# ------------------------------------------------------------------- decisions
def decisions_path(cfg) -> Path:
    return Path(cfg.film_dir or cfg.out_dir.parent) / "editorial" / "identity-decisions.json"


def resolve(doc: dict | None, decisions: dict | None, raws: dict[str, str]) -> dict[str, dict]:
    """Every track's link as it stands: the editor's decision where it was made on the current measurement, else the tool's.
    `person` is set only on a confirmed link."""
    links = {k: dict(v, person=None) for k, v in ((doc or {}).get("links") or {}).items()}
    for k, d in ((decisions or {}).get("tracks") or {}).items():
        sid, tid = k.rsplit("/", 1)
        cur = links.setdefault(k, dict(source=sid, track=tid, status=UNCHECKED, person=None, why="decided, but not in the last import"))
        if raws.get(sid) != d.get("raw"):
            cur["stale_decision"] = dict(decided=d.get("decided"), person=d.get("person"), at=d.get("at"))
            continue
        cur.update(status=d["decided"], person=d.get("person") if d["decided"] == CONFIRMED else None, decided_by=d.get("by"),
                   decided_at=d.get("at"))
        cur.pop("why", None)
    return links


def load_links(cfg) -> dict[str, dict]:
    p, dp = cfg.out_dir / "identity.json", decisions_path(cfg)
    return resolve(json.loads(p.read_text()) if p.exists() else None, json.loads(dp.read_text()) if dp.exists() else None,
                   current_raws(cfg))


def merge_decisions(cur: dict, new: dict, by: str, now: str) -> tuple[dict, dict[str, int]]:
    """New decisions over the recorded ones; a changed decision keeps what it supersedes."""
    out = dict(schema=DECISIONS_SCHEMA, tracks=dict(cur.get("tracks") or {}))
    counts = dict(confirmed=0, rejected=0, unchanged=0, invalid=0)
    for k, d in (new.get("tracks") or {}).items():
        decided, person = d.get("decided"), (d.get("person") or "").strip() or None
        if decided not in (CONFIRMED, REJECTED) or (decided == CONFIRMED and not person) or not d.get("raw") or "/" not in k:
            counts["invalid"] += 1
            continue
        rec = dict(raw=d["raw"], decided=decided, person=person if decided == CONFIRMED else None, proposal=d.get("proposal"),
                   by=d.get("by") or by, at=new.get("made") or now)
        old = out["tracks"].get(k)
        if old and all(old.get(f) == rec[f] for f in ("raw", "decided", "person")):
            counts["unchanged"] += 1
            continue
        if old:
            rec["supersedes"] = {f: old.get(f) for f in ("raw", "decided", "person", "by", "at")}
        out["tracks"][k] = rec
        counts[decided] += 1
    return out, counts


def confirm(cfg, downloaded: Path, by: str) -> dict[str, int]:
    new = json.loads(Path(downloaded).read_text())
    if new.get("schema") != DECISIONS_SCHEMA:
        raise SystemExit(f"not an identity decisions file (schema {DECISIONS_SCHEMA}): {downloaded}")
    path = decisions_path(cfg)
    cur = json.loads(path.read_text()) if path.exists() else {}
    merged, counts = merge_decisions(cur, new, by, dt.datetime.now().astimezone().isoformat(timespec="seconds"))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(merged, indent=1))
    return counts


# ---------------------------------------------------------------------- review
def groups(links: dict[str, dict]) -> list[dict]:
    """What the page shows: confirmed people by the editor's label, then the tool's proposals, largest first."""
    by: dict[tuple[int, str], list[tuple[str, dict]]] = {}
    for k, v in links.items():
        if not v.get("crops"):
            continue
        if v["status"] == CONFIRMED:
            by.setdefault((0, v["person"]), []).append((k, v))
        elif v["status"] == PROPOSED:
            by.setdefault((1, v["proposal"]), []).append((k, v))
    out = []
    for (kind, name), items in by.items():
        items.sort(key=lambda kv: (kv[1]["source"], kv[1]["t0"]))
        out.append(dict(id=f"{'person' if kind == 0 else 'proposal'}:{name}", label=name if kind == 0 else "", proposal=None if kind == 0 else name,
                        clips=len({v["source"] for _, v in items}),
                        tracks=[dict(key=k, raw=v["raw"], crop=v["crops"][0], status=v["status"], conflict=bool(v.get("conflict")),
                                     score=v.get("score"), where=f"{v['source']} {int(v['t0'] // 60):02d}:{v['t0'] % 60:04.1f}") for k, v in items]))
    return sorted(out, key=lambda g: (not g["label"], -len(g["tracks"]), g["id"]))


def review(cfg) -> Path:
    gs = groups(load_links(cfg))
    day = dt.date.today().strftime("%Y%m%d")
    root = cfg.work_dir / "reviews" / f"identity-{day}"
    root.mkdir(parents=True, exist_ok=True)
    crops = Path("..") / ".." / request_dir(cfg).relative_to(cfg.work_dir)
    page = root / "index.html"
    page.write_text(review_html(gs, crops.as_posix(), day))
    return page


def review_html(gs: list[dict], crops: str, day: str) -> str:
    data = json.dumps(dict(groups=gs, crops=crops, key=f"identity-decisions-{day}", schema=DECISIONS_SCHEMA))
    return """<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Identity decisions</title><style>
:root{--bg:#fbfaf7;--fg:#1d1c1a;--mut:#6b675f;--line:#e2ded5;--acc:#2f5d50;--warn:#9a5b00}
@media (prefers-color-scheme:dark){:root{--bg:#161614;--fg:#ecebe6;--mut:#a29e94;--line:#34322d;--acc:#7fb8a6;--warn:#e0a24a}}
body{margin:0;background:var(--bg);color:var(--fg);font:15px/1.45 -apple-system,system-ui,sans-serif}
header{position:sticky;top:0;background:var(--bg);border-bottom:1px solid var(--line);padding:12px 16px;z-index:2;display:flex;gap:12px;align-items:center;flex-wrap:wrap}
h1{font-size:17px;margin:0 auto 0 0}main{max-width:1100px;margin:0 auto;padding:16px}
.group{border:1px solid var(--line);border-radius:10px;padding:12px;margin:0 0 18px}.group.named{border-color:var(--acc)}
.head{display:flex;gap:10px;align-items:center;flex-wrap:wrap}.meta{color:var(--mut);font-size:13px}
input{font:inherit;padding:6px 8px;border:1px solid var(--line);border-radius:6px;background:transparent;color:var(--fg);min-width:220px}
.tiles{display:grid;grid-template-columns:repeat(auto-fill,minmax(120px,1fr));gap:8px;margin-top:10px}
.tile{position:relative;border:2px solid transparent;border-radius:8px;padding:0;background:transparent;cursor:pointer;color:var(--fg);font:inherit;text-align:left}
.tile img{width:100%;aspect-ratio:1;object-fit:cover;border-radius:6px;display:block}
.tile span{display:block;font-size:11px;color:var(--mut);padding:2px 2px 0;overflow-wrap:anywhere}
.tile.drop{opacity:.35;border-color:var(--line)}.tile.drop::after{content:"dropped";position:absolute;top:6px;left:6px;background:var(--bg);font-size:11px;padding:1px 6px;border-radius:999px}
.tile.conflict span{color:var(--warn)}
button.dl{font:inherit;border:1px solid var(--acc);background:var(--acc);color:var(--bg);border-radius:999px;padding:6px 14px;cursor:pointer}
</style></head><body><header><h1>Identity decisions <span id="prog" class="meta"></span></h1><button class="dl" id="dl">Download decisions</button></header><main>
<p>Each box is one person as the identity tool proposed it (or as you already named them). Type a label to confirm the box:
every kept face is linked to that label, every <b>dropped</b> face (click it) is recorded as not this person. Boxes without a label
record nothing. A label can be a name or a code from your own list; it stays on this machine. Faces marked <span style="color:var(--warn)">overlap</span>
share time in one clip with another face in the same box, so at least one of them is wrong. Decisions stay in this browser until you press
<b>Download decisions</b>; then run <code>identity confirm</code> on the file.</p><div id="groups"></div></main><script>
const D=""" + data + """;let S={};try{S=JSON.parse(localStorage.getItem(D.key)||"{}")}catch(e){}
const save=()=>{try{localStorage.setItem(D.key,JSON.stringify(S))}catch(e){};prog()};
const esc=s=>String(s).replace(/[&<>"']/g,c=>({"&":"&amp;","<":"&lt;",">":"&gt;",'"':"&quot;","'":"&#39;"}[c]));
const st=g=>S[g.id]||(S[g.id]={label:g.label,drop:[]});
const prog=()=>{document.getElementById("prog").textContent=`${D.groups.filter(g=>(st(g).label||"").trim()).length} of ${D.groups.length} boxes labelled`};
document.getElementById("groups").innerHTML=D.groups.map((g,i)=>`<section class="group${g.label?" named":""}" data-i="${i}"><div class="head">
<input placeholder="label (name or code)" value="${esc(st(g).label||"")}"><span class="meta">${g.label?"confirmed":"proposed "+esc(g.proposal)} · ${g.tracks.length} faces in ${g.clips} clips</span></div>
<div class="tiles">${g.tracks.map(t=>`<button class="tile${t.conflict?" conflict":""}" data-k="${esc(t.key)}"><img loading="lazy" src="${esc(D.crops+"/"+t.crop)}" alt="face crop">
<span>${esc(t.where)}${t.conflict?" · overlap":""}</span></button>`).join("")}</div></section>`).join("");
document.querySelectorAll(".group").forEach(el=>{const g=D.groups[+el.dataset.i];const inp=el.querySelector("input");
const paint=()=>{const s=st(g);el.querySelectorAll(".tile").forEach(b=>b.classList.toggle("drop",s.drop.includes(b.dataset.k)));el.classList.toggle("named",!!(s.label||"").trim())};
inp.oninput=()=>{st(g).label=inp.value;paint();save()};
el.querySelectorAll(".tile").forEach(b=>b.onclick=()=>{const s=st(g),k=b.dataset.k;s.drop=s.drop.includes(k)?s.drop.filter(x=>x!==k):[...s.drop,k];paint();save()});paint()});
document.getElementById("dl").onclick=()=>{const tracks={};for(const g of D.groups){const s=st(g),label=(s.label||"").trim();if(!label)continue;
for(const t of g.tracks){const drop=s.drop.includes(t.key);tracks[t.key]={raw:t.raw,decided:drop?"rejected":"confirmed",person:drop?null:label,proposal:g.proposal}}}
const b=new Blob([JSON.stringify({schema:D.schema,made:new Date().toISOString(),tracks},null,1)],{type:"application/json"});
const a=document.createElement("a");a.href=URL.createObjectURL(b);a.download=D.key+".json";a.click()};prog();
</script></body></html>"""


# ------------------------------------------------------------------------- cli
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="face identity through an external tool: export crops, import its answer, review, confirm")
    ap.add_argument("cmd", choices=["setup", "run", "export", "import", "review", "confirm"])
    ap.add_argument("--config", help="the project's config (every command but setup)")
    ap.add_argument("--only", help="export: one source id")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--answer", help="import: the identity tool's answer file")
    ap.add_argument("--decisions", help="confirm: the decisions file downloaded from the review page")
    ap.add_argument("--by", default="editor", help="confirm: who made the decisions")
    ap.add_argument("--people", help="run: reference photos, one folder per person named by its label")
    ap.add_argument("--same", type=float, default=SAME, help="run: cosine similarity at or over which faces are proposed as one person")
    a = ap.parse_args(argv)
    if a.cmd == "setup":
        print(f"model ready: {setup()} ({MODEL['licence']})")
        return 0
    if not a.config:
        ap.error(f"{a.cmd} needs --config")
    from .config import load_config
    cfg = load_config(a.config)
    if a.cmd == "export":
        print(export(cfg, a.only, a.workers))
    elif a.cmd in ("import", "run"):
        if a.cmd == "import" and not a.answer:
            ap.error("import needs --answer")
        d = import_answer(cfg, Path(a.answer)) if a.cmd == "import" else run(cfg, Path(a.people) if a.people else None, a.same, a.only, a.workers)
        print(", ".join(f"{k} {v}" for k, v in sorted(d["counts"].items())) + f"; {d['proposed_people']} proposed people; "
              f"{len(d['conflicts'])} overlapping tracks; {len(d['unknown_keys'])} unknown keys; {len(d['malformed'])} malformed entries",
              file=sys.stderr)
        print(cfg.out_dir / "identity.json")
    elif a.cmd == "review":
        print(review(cfg))
    else:
        if not a.decisions:
            ap.error("confirm needs --decisions")
        c = confirm(cfg, Path(a.decisions), a.by)
        print(", ".join(f"{k} {v}" for k, v in c.items()), file=sys.stderr)
        print(decisions_path(cfg))
    return 0


if __name__ == "__main__":
    sys.exit(main())
