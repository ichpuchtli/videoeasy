"""Is the plant on screen the one being named? BioCLIP 2 over the frames under every named species.

The editor (29 Sep 2026): when the film names a species, often by its Latin
name, nobody in the edit can tell whether the tree or plant on screen is that one. The vision
judge (gemma-4) names plants confidently and cannot be trusted to species.
BioCLIP 2 (Imageomics, open weights, trained across the tree of life) runs
here, locally, on graded frames only:

  1. a CLAIM is a picture shown while the words name a species: every cover
     row under a pick whose text matches a spoken form in
     editorial/species-list.json, and the pick's own picture where it has no
     cover and is not the locked-off interview set-up (a walk take pointed at
     the plant; the set-up is a `never_cover` role that is not to camera);
  2. four graded frames across exactly the source interval on screen, three
     square views of each (left, centre, right: BioCLIP centre-crops, and a
     16:9 frame would lose its sides);
  3. the open-ended answer: BioCLIP's tree-of-life species ranking, averaged
     over the twelve views. This is the check, because it is not told what
     to find;
  4. the shortlist answer: the same views ranked over the species list only.
     A shortlist always has a winner (softmax over a short list read 0.99998
     for a single frame), so it orders candidates for a person and never
     settles a claim.

A claim AGREES when the named species (or, for a genus the film names, its
genus) is in the open-ended top five; DISAGREES when it is not and the
open-ended top answer holds OPEN_SURE or more (the model sees something
else); and CANNOT TELL otherwise (a wide shot of mixed bush has no single
answer: on v8, 17 of 23 claims read under 0.1). None
of this is ground truth: the species list is a draft, the model is
uncalibrated on this footage, and a wide shot of mixed bush has no single
answer. A person's review (the last step) turns flags and shortlists into labels.

    uv run --extra species python -m videoeasy.species --config config.<film>.yaml --cut editorial/cut-v8.json
    -> <out_dir>/species.json
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import tempfile
from pathlib import Path

from .evalrun import OK, UNCHECKED, fingerprint, source_hash

FRAMES = 4
MAX_PX = 1280
TOP_K = 5
OPEN_K = 2000              # species ranked per view, so every name on the list gets its own open-ended score
OPEN_SURE = 0.15           # provisional: an open-ended top answer this strong, for another species, is a disagreement


def interview_setup(clip: str, profile=None) -> bool:
    """The sit-down set-up (a never-cover role that is not to camera): its picture is the people, never a plant claim."""
    from .sources import role_of
    r = role_of(clip, profile)
    return r.never_cover and not r.to_camera


def load_list(film: Path) -> dict:
    return json.loads((film / "editorial/species-list.json").read_text(encoding="utf-8"))


def named_in(text: str, species: list[dict]) -> list[dict]:
    """Species whose spoken forms occur in the text (word-start match, case-insensitive)."""
    t = text.lower()
    return [s for s in species if s.get("named") and any(re.search(r"\b" + re.escape(term), t) for term in s["terms"])]


def claims(cut: dict, species: list[dict], unit_source: dict[str, str], profile=None) -> list[dict]:
    """Every picture shown while its pick names a species: (pick, words, named, unit, source, in_s, out_s, kind)."""
    out = []
    for bi, b in enumerate(cut["beats"], 1):
        for j, a in enumerate(b["audio"], 1):
            if a.get("tier", 1) > 1:
                continue
            named = named_in(a["text"], species)
            if not named:
                continue
            pid = f"b{bi}p{j}"
            rows = [v for v in b["video"] if v.get("pick") == pid and v["type"] == "video" and v.get("clip")]
            base = dict(pick=pid, beat=b["title"], words=a["text"], named=[s["sci"] for s in named])
            if rows:
                for v in rows:
                    dur = v["dest_out_s"] - v["dest_in_s"]
                    out.append(dict(base, kind="cover", unit=v["clip"], source=unit_source.get(v["clip"], v["clip"]),
                                    in_s=v["in_s"], out_s=v["in_s"] + dur))
            else:
                src = a["clip"].split("_w")[0]
                if not interview_setup(src, profile):
                    out.append(dict(base, kind="sync", unit=a["clip"], source=src, in_s=a["in_s"], out_s=a["out_s"]))
    return out


def views(img_path: Path, out_dir: Path) -> list[Path]:
    """Left, centre and right squares of a frame."""
    from PIL import Image
    im = Image.open(img_path).convert("RGB")
    w, h = im.size
    side = min(w, h)
    xs = [0, (w - side) // 2, w - side] if w > h else [0]
    paths = []
    for k, x in enumerate(xs):
        p = out_dir / f"{img_path.stem}_{k}.jpg"
        im.crop((x, 0, x + side, side)).save(p, quality=92)
        paths.append(p)
    return paths


def agrees(named: list[str], top: list[dict]) -> bool:
    """The named species is in the open-ended top five; a named genus (several species, e.g. the wattles) counts
    when any of its species or the genus itself is there."""
    genera = {n.split()[0] for n in named}
    return any(t["species"] in named or t["species"].split()[0] in genera for t in top)


def verdict(named: list[str], top: list[dict]) -> str:
    if agrees(named, top):
        return "agrees"
    return "disagrees" if top and top[0]["score"] >= OPEN_SURE else "cannot tell"


def run(cfg, film: Path, cut_path: Path, device: str = "mps") -> dict:
    from bioclip import CustomLabelsClassifier, Rank, TreeOfLifeClassifier
    from . import frames as frames_mod, segments as segments_mod
    lst = load_list(film)
    species = lst["species"]
    units = {u["shot_id"]: u for u in segments_mod.units(cfg)}
    unit_source = {k: u["source_id"] for k, u in units.items()}
    src_path = {u["source_id"]: u["source_path"] for u in units.values()}
    cut = json.loads(cut_path.read_text(encoding="utf-8"))
    cl = claims(cut, species, unit_source, cfg.film)
    tol = TreeOfLifeClassifier(device=device)
    short = CustomLabelsClassifier([s["sci"] for s in species], device=device)
    common = {s["sci"]: s["common"] for s in species}
    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        for n, c in enumerate(cl):
            src = Path(src_path[c["source"]])
            vs = []
            for k, t in enumerate(frames_mod.sample_times(c["in_s"], c["out_s"], FRAMES)):
                f = tdir / f"c{n}_f{k}.jpg"
                frames_mod.extract_frame(src, t, f, cfg.grade_for(src), MAX_PX)
                vs += views(f, tdir)
            if not vs:
                c.update(status=UNCHECKED, why="no frames")
                continue
            acc: dict[str, dict] = {}
            for p in tol.predict([str(v) for v in vs], Rank.SPECIES, k=OPEN_K):
                e = acc.setdefault(p["species"], dict(species=p["species"], common=p.get("common_name") or "", family=p.get("family", ""), score=0.0))
                e["score"] += p["score"] / len(vs)
            top = sorted(acc.values(), key=lambda e: -e["score"])[:TOP_K]
            # each listed species' open-ended probability (out of every species the model knows), for the pick list:
            # absent from a view's top OPEN_K it counts 0 there, so a tiny figure is an upper bound on nothing, not a claim
            list_scores = {x["sci"]: round(acc[x["sci"]]["score"], 4) if x["sci"] in acc else 0.0 for x in species}
            sl: dict[str, float] = {}
            for p in short.predict([str(v) for v in vs], k=len(species)):
                sl[p["classification"]] = sl.get(p["classification"], 0.0) + p["score"] / len(vs)
            ranked = sorted(sl.items(), key=lambda kv: -kv[1])
            rank_of = {sci: i + 1 for i, (sci, _) in enumerate(ranked)}
            c.update(status=OK, views=len(vs), open_top=[dict(e, score=round(e["score"], 3)) for e in top],
                     shortlist_top=[dict(species=s, common=common.get(s, ""), score=round(v, 3)) for s, v in ranked[:TOP_K]],
                     list_scores=list_scores,
                     named_rank=min(rank_of.get(s, 99) for s in c["named"]), agrees=agrees(c["named"], top),
                     verdict=verdict(c["named"], top))
            print(f"  {c['pick']:7} {c['unit']:14} named {', '.join(c['named'])[:48]:48} open-ended {top[0]['species']} {top[0]['score']:.2f}"
                  f" | shortlist rank {c['named_rank']} | {c['verdict']}", file=sys.stderr)
    doc = dict(method=dict(model="BioCLIP 2 (pybioclip)", frames=FRAMES, views_per_frame=3, top_k=TOP_K, device=device, open_sure=OPEN_SURE,
                           species_list=dict(path=str(film / "editorial/species-list.json"), sha=fingerprint(film / "editorial/species-list.json"),
                                             n=len(species)),
                           cut=dict(path=str(cut_path), sha=fingerprint(cut_path)), tool_sha=source_hash(__file__)),
               claims=cl)
    (cfg.out_dir / "species.json").write_text(json.dumps(doc, indent=1))
    return doc


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--cut", required=True, help="relative to the film folder, e.g. editorial/cut-v8.json")
    ap.add_argument("--device", default="mps")
    args = ap.parse_args(argv)
    from .config import load_config
    cfg = load_config(args.config)
    film = cfg.out_dir.parent
    doc = run(cfg, film, film / args.cut, args.device)
    ok = [c for c in doc["claims"] if c.get("status") == OK]
    print(f"{len(doc['claims'])} claims, {len(ok)} measured: " + ", ".join(
        f"{sum(c['verdict'] == v for c in ok)} {v}" for v in ("agrees", "disagrees", "cannot tell")), file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
