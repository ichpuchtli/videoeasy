"""One film's specifics, read from `<film_dir>/film.yaml`.

Everything that differs from film to film and is not footage lives here:
what the film is (the register line the prompts carry), which recording
context each clip comes from, the spellings ASR gets wrong, the story
model's film-specific questions, and the Resolve lay-in constants. The file
sits with the film's own data and is never part of the code. A missing file,
or a missing key, falls back to the generic defaults below, so every tool
runs on a new film before anyone has written a profile.

    name: "My film"
    context: >-
      One paragraph: what the film is and its register. Never put example
      metaphor or theme phrases here: the vision model parrots them.
    roles:
      sitdown:   {label: "sit-down interview", never_cover: true}
      presenter: {label: "presenter to camera", to_camera: true, never_cover: true}
      walk:      {label: "walk-and-talk, sync sound", sync_walk: true}
      drone:     {label: "drone", drone: true}
    default_role: walk
    clips: {C0001: sitdown}            # base clip name -> role; exact, overrides prefixes
    prefixes: [[DJI_, drone]]          # first match wins
    aliases: {intended: [asr_variant]} # names, species, jargon the transcriber mishears
    names: [intended]                  # tokens whose edit changes who is meant (a semantic risk)
    story:
      voices: {portrait: "who carries the film", method: "who explains the method"}
      voices_question: "is the balance of voices what the spine asks for"
      questions: {chapter_shape: "does chapter one end where the spine says? at most 40 words"}
    layin:
      project: "Resolve project name"
      bin: [Films, "My film"]
      clip_bins: [Aroll, Broll, Drone, AI Drafts]
      drafts_bin: AI Drafts
      grades:                          # per timeline item, first filename-prefix match wins
        - {name: log, match: C, lut: /path/to/log_to_rec709.cube}
        - {name: drone, match: DJI_, cdl: {NodeIndex: 1, Slope: "1 1 1", Offset: "0 0 0", Power: "1 1 1", Saturation: 1.0}}
      placeholders: {keyword: card_file.mov}   # checked in the listed order
      render_dir: editorial/renders
      grade_trims: editorial/grade-trims.json
      render_preset: "YouTube - 1080p"
      aroll_dir: inputs/aroll

Relative paths under `layin` resolve against the film directory.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import yaml

PROFILE_NAME = "film.yaml"


@dataclass(frozen=True)
class Role:
    """A recording context. `to_camera`: the uncovered picture is a person addressing the audience;
    `never_cover`: no unit from it is ever proposed as cover; `sync_walk`: sync sound of people on the move,
    so a beat of it is never covered by another take of the same kind; `drone`: aerial, measured by moves.py."""
    name: str
    label: str
    to_camera: bool = False
    never_cover: bool = False
    sync_walk: bool = False
    drone: bool = False


DEFAULT_ROLES = {
    "sitdown": Role("sitdown", "sit-down interview", never_cover=True),
    "presenter": Role("presenter", "presenter to camera", to_camera=True, never_cover=True),
    "walk": Role("walk", "walk-and-talk, sync sound", sync_walk=True),
    "drone": Role("drone", "drone", drone=True),
}

DEFAULT_LAYIN = dict(
    project=None,
    bin=[],
    clip_bins=["Aroll", "Broll", "Drone", "AI Drafts"],
    drafts_bin="AI Drafts",
    grades=[],
    placeholders={},
    render_dir="editorial/renders",
    grade_trims="editorial/grade-trims.json",
    render_preset="YouTube - 1080p",
    aroll_dir="inputs/aroll",
)
LAYIN_PATHS = ("render_dir", "grade_trims", "aroll_dir")


@dataclass
class Film:
    root: Path | None
    name: str = "film"
    context: str = ""
    roles: dict[str, Role] = field(default_factory=lambda: dict(DEFAULT_ROLES))
    default_role: str = "walk"
    clips: dict[str, str] = field(default_factory=dict)
    prefixes: list[tuple[str, str]] = field(default_factory=lambda: [("DJI_", "drone")])
    aliases: dict[str, set[str]] = field(default_factory=dict)
    names: set[str] = field(default_factory=set)
    story: dict = field(default_factory=dict)
    layin: dict = field(default_factory=lambda: dict(DEFAULT_LAYIN))

    def role_of(self, base: str) -> Role:
        r = self.clips.get(base)
        if r is None:
            r = next((role for pre, role in self.prefixes if base.startswith(pre)), self.default_role)
        return self.roles[r]


def _role(name: str, spec: dict | str) -> Role:
    if isinstance(spec, str):
        spec = {"label": spec}
    unknown = set(spec) - {"label", "to_camera", "never_cover", "sync_walk", "drone"}
    if unknown:
        raise ValueError(f"film.yaml: role {name!r} has unknown keys {sorted(unknown)}")
    return Role(name, str(spec.get("label", name)), bool(spec.get("to_camera", False)), bool(spec.get("never_cover", False)),
                bool(spec.get("sync_walk", False)), bool(spec.get("drone", False)))


def from_dict(raw: dict, root: Path | None = None) -> Film:
    """A Film from the parsed YAML (or any dict of the same shape)."""
    raw = raw or {}
    roles = {k: _role(k, v) for k, v in raw["roles"].items()} if raw.get("roles") else dict(DEFAULT_ROLES)
    default_role = raw.get("default_role", "walk" if "walk" in roles else next(iter(roles)))
    clips = {str(k): str(v) for k, v in (raw.get("clips") or {}).items()}
    prefixes = [(str(p), str(r)) for p, r in raw["prefixes"]] if "prefixes" in raw else [("DJI_", "drone")]
    for r in [default_role, *clips.values(), *(r for _, r in prefixes)]:
        if r not in roles:
            raise ValueError(f"film.yaml: role {r!r} is used but not defined under roles")
    layin = dict(DEFAULT_LAYIN, **(raw.get("layin") or {}))
    if root is not None:
        for k in LAYIN_PATHS:
            if layin.get(k) and not Path(layin[k]).is_absolute():
                layin[k] = str(root / layin[k])
    return Film(
        root=root,
        name=str(raw.get("name") or (root.name if root else "film")),
        context=" ".join(str(raw.get("context") or "").split()),
        roles=roles,
        default_role=default_role,
        clips=clips,
        prefixes=prefixes,
        aliases={str(k).lower(): {str(v).lower() for v in vs} for k, vs in (raw.get("aliases") or {}).items()},
        names={str(n).lower() for n in (raw.get("names") or [])},
        story=dict(raw.get("story") or {}),
        layin=layin,
    )


def load_film(film_dir: str | Path | None) -> Film:
    """The profile of the film at `film_dir`; generic defaults when there is no film.yaml (or no directory)."""
    if film_dir is None:
        return from_dict({})
    root = Path(film_dir).resolve()
    p = root / PROFILE_NAME
    raw = yaml.safe_load(p.read_text(encoding="utf-8")) if p.exists() else {}
    return from_dict(raw, root)


def film_dir_for(path: str | Path) -> Path | None:
    """The film directory a path inside it belongs to: the nearest ancestor holding a film.yaml."""
    p = Path(path).resolve()
    for d in [p, *p.parents]:
        if (d / PROFILE_NAME).exists():
            return d
    return None
