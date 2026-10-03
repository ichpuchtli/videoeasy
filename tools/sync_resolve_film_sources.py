"""Mirror explicitly catalogued Resolve bins into a film's isolated ingest inputs.

Resolve is read-only here. The only mutations are the generated manifest and
symlinks below the film directory (`inputs/aroll`, `inputs/broll`,
`resolve-sources.json`); source media and Resolve bins are never modified.
Bin membership is the editor's catalogue, so each bin is named on the command
line with the role its clips take; a clip catalogued under two roles stops
the run.

    uv run python tools/sync_resolve_film_sources.py data/films/<film> \
        --bin "Master/Films/<film>/Aroll=aroll" --bin "Master/Films/<film>/Broll=broll" \
        [--project "<Resolve project>"] [--apply]

The project defaults to the film profile's `layin.project` (film.yaml).
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import date
from pathlib import Path

from videoeasy.config import VIDEO_EXTS
from videoeasy.film import load_film
from videoeasy.resolve import find_folder, get_project

ROLES = ("aroll", "broll")


def parse_bins(specs: list[str]) -> dict[str, str]:
    bins = {}
    for spec in specs:
        path, sep, role = spec.rpartition("=")
        if not sep or role not in ROLES:
            raise SystemExit(f"--bin wants <bin path>=aroll|broll, got {spec!r}")
        bins[path] = role
    return bins


def catalog(film_name: str, project_name: str, bins: dict[str, str]) -> dict:
    project = get_project(expected_name=project_name)
    media_pool = project.GetMediaPool()
    clips_by_path: dict[str, dict] = {}
    skipped: list[dict] = []

    for bin_path, role in bins.items():
        folder = find_folder(media_pool, bin_path)
        for item in folder.GetClipList() or []:
            props = item.GetClipProperty() or {}
            raw_path = props.get("File Path", "")
            path = Path(raw_path) if raw_path else None
            if path is None or path.suffix.lower() not in VIDEO_EXTS:
                skipped.append({"name": item.GetName(), "bin": bin_path})
                continue
            resolved = str(path.resolve())
            if resolved in clips_by_path:
                existing = clips_by_path[resolved]
                if existing["role"] != role:
                    raise RuntimeError(
                        f"{resolved} is catalogued as both {existing['role']} and {role}"
                    )
                evidence = f"bin:{bin_path}"
                if evidence not in existing["evidence"]:
                    existing["evidence"].append(evidence)
                continue
            if not path.is_file():
                raise FileNotFoundError(f"Resolve source is unavailable: {path}")
            clips_by_path[resolved] = {
                "name": path.name,
                "path": resolved,
                "role": role,
                "evidence": [f"bin:{bin_path}"],
                "resolve_path": raw_path,
            }

    clips = sorted(clips_by_path.values(), key=lambda c: (c["role"], c["name"]))
    return {
        "project": project.GetName(),
        "film": film_name,
        "source": "Resolve bins catalogued by the editor",
        "bins": bins,
        "clips": clips,
        "captured_date": date.today().isoformat(),
        "status": "confirmed_by_user_catalog",
        "coverage": "Exact contents of the named bins; duplicate media-pool references are deduplicated by source path.",
        "skipped_non_media_items": skipped,
    }


def sync_inputs(film_root: Path, manifest: dict) -> None:
    desired: dict[Path, str] = {}
    for clip in manifest["clips"]:
        destination = film_root / "inputs" / clip["role"] / clip["name"]
        if destination in desired and desired[destination] != clip["path"]:
            raise RuntimeError(f"input name collision at {destination}")
        desired[destination] = clip["path"]

    for role in ROLES:
        role_dir = film_root / "inputs" / role
        role_dir.mkdir(parents=True, exist_ok=True)
        for existing in role_dir.iterdir():
            if existing in desired:
                continue
            if not existing.is_symlink():
                raise RuntimeError(f"refusing to replace non-symlink input: {existing}")
            existing.unlink()

    for destination, source in desired.items():
        if destination.is_symlink() and os.readlink(destination) == source:
            continue
        if destination.exists() or destination.is_symlink():
            if not destination.is_symlink():
                raise RuntimeError(
                    f"refusing to replace non-symlink input: {destination}"
                )
            destination.unlink()
        destination.symlink_to(source)

    manifest_path = film_root / "resolve-sources.json"
    temporary = manifest_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary.replace(manifest_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("film", help="film directory, e.g. data/films/<film>")
    parser.add_argument("--bin", action="append", required=True, help="<media pool bin path>=aroll|broll (repeatable)")
    parser.add_argument("--project", default=None, help="Resolve project (default: film.yaml layin.project)")
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write the manifest and synchronise managed input symlinks",
    )
    args = parser.parse_args()
    film_root = Path(args.film)
    profile = load_film(film_root)
    project_name = args.project or profile.layin.get("project")
    if not project_name:
        raise SystemExit("no Resolve project: pass --project or set layin.project in film.yaml")
    manifest = catalog(profile.name, project_name, parse_bins(args.bin))
    counts = {
        role: sum(clip["role"] == role for clip in manifest["clips"])
        for role in ROLES
    }
    print(json.dumps({"film": manifest["film"], "counts": counts,
                      "skipped": manifest["skipped_non_media_items"]}, indent=2))
    if args.apply:
        sync_inputs(film_root, manifest)
        print(f"updated {film_root / 'resolve-sources.json'} and inputs")
    else:
        print("dry run; pass --apply to write")


if __name__ == "__main__":
    main()
