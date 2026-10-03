"""What each recording is, for the tools that must not guess from a filename.

The table is per film and lives in the film's profile (`film.yaml`, see
`film.py`): `roles` names the recording contexts, `clips` maps a clip to its
role exactly, `prefixes` covers families of clips, and anything unmatched
takes `default_role`. A role decides the picture rule the scorer applies to
a bare stretch (`to_camera`), how the story check labels a pick, whether a
unit may ever be cover (`never_cover`), and whether a beat of it is sync
sound on the move (`sync_walk`). A new source must be listed before its picks
are judged; unlisted, it is judged by the default role.
"""
from __future__ import annotations

from .film import Film, Role, load_film

_GENERIC: Film | None = None


def _profile(profile: Film | None) -> Film:
    global _GENERIC
    if profile is not None:
        return profile
    if _GENERIC is None:
        _GENERIC = load_film(None)
    return _GENERIC


def base_clip(clip: str) -> str:
    """`CAM_A001_w04` and `DJI_1001_s01` name windows and sub-shots of `CAM_A001` / `DJI_1001`."""
    return clip.split("_w")[0].split("_s")[0]


def role_of(clip: str, profile: Film | None = None) -> Role:
    return _profile(profile).role_of(base_clip(clip))


def role(clip: str, profile: Film | None = None) -> str:
    return role_of(clip, profile).name


def label(clip: str, profile: Film | None = None) -> str:
    return role_of(clip, profile).label


def to_camera(clip: str, profile: Film | None = None) -> bool:
    return role_of(clip, profile).to_camera


def is_drone(clip: str, profile: Film | None = None) -> bool:
    return role_of(clip, profile).drone


def never_cover(clip: str, profile: Film | None = None) -> bool:
    return role_of(clip, profile).never_cover


def sync_walk(clip: str, profile: Film | None = None) -> bool:
    return role_of(clip, profile).sync_walk


def drone_prefixes(profile: Film | None = None) -> tuple[str, ...]:
    """Name prefixes of the drone sources (prefix rules and exact clips of a drone role), for tools that select by name."""
    p = _profile(profile)
    return tuple([pre for pre, r in p.prefixes if p.roles[r].drone] + [c for c, r in p.clips.items() if p.roles[r].drone])


def placeholder_file(text: str, profile: Film | None = None) -> str:
    """Card file for a placeholder row: the first keyword of the profile's `layin.placeholders`, in listed order,
    that the text contains. Order matters when one card's text names another card's keyword, so list the more
    specific keyword first."""
    table = _profile(profile).layin.get("placeholders") or {}
    t = text.lower()
    for key, card in table.items():
        if key.lower() in t:
            return card
    raise KeyError(f"no placeholder card for {text!r}")
