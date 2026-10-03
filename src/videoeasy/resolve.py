"""DaVinci Resolve scripting-API bridge (Phase 2 foundation).

Requires Resolve running with external scripting enabled (Preferences →
System → General → External scripting using: Local). Media-pool operations
move clip references only — files on disk are never touched.
"""
from __future__ import annotations

import os
import sys

_API = "/Library/Application Support/Blackmagic Design/DaVinci Resolve/Developer/Scripting"
_LIB = "/Applications/DaVinci Resolve/DaVinci Resolve.app/Contents/Libraries/Fusion/fusionscript.so"


def get_resolve():
    os.environ.setdefault("RESOLVE_SCRIPT_API", _API)
    os.environ.setdefault("RESOLVE_SCRIPT_LIB", _LIB)
    modules = os.path.join(os.environ["RESOLVE_SCRIPT_API"], "Modules")
    if modules not in sys.path:
        sys.path.append(modules)
    import DaVinciResolveScript as dvr

    resolve = dvr.scriptapp("Resolve")
    if resolve is None:
        raise RuntimeError(
            "cannot connect to DaVinci Resolve — is it running with external "
            "scripting enabled (Preferences → System → General)?"
        )
    return resolve


def get_project(expected_name: str | None = None):
    project = get_resolve().GetProjectManager().GetCurrentProject()
    if project is None:
        raise RuntimeError("no project open in Resolve")
    if expected_name and project.GetName() != expected_name:
        raise RuntimeError(
            f"open project is '{project.GetName()}', expected '{expected_name}' — "
            "refusing to modify the wrong project"
        )
    return project


def find_folder(media_pool, path: str):
    """Find a bin by 'Master/Films/<film>/Aroll'-style path."""
    folder = media_pool.GetRootFolder()
    parts = path.split("/")
    if parts and parts[0] == folder.GetName():
        parts = parts[1:]
    for part in parts:
        match = next(
            (s for s in folder.GetSubFolderList() or [] if s.GetName() == part), None
        )
        if match is None:
            raise KeyError(f"bin '{part}' not found under '{folder.GetName()}'")
        folder = match
    return folder


def ensure_subfolder(media_pool, parent, name: str):
    existing = next(
        (s for s in parent.GetSubFolderList() or [] if s.GetName() == name), None
    )
    return existing or media_pool.AddSubFolder(parent, name)


def clips_by_name(folder) -> dict:
    return {c.GetName(): c for c in folder.GetClipList() or []}


def move_clips(media_pool, clips: list, target) -> bool:
    if not clips:
        return True
    return bool(media_pool.MoveClips(clips, target))
