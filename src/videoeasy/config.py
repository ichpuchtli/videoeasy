from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

VIDEO_EXTS = {".mov", ".mp4", ".mxf", ".m4v", ".avi", ".mkv"}


@dataclass
class GradeProfile:
    name: str
    lut: Path | None = None       # .cube applied via lut3d
    filters: str | None = None    # raw ffmpeg filter string (e.g. eq=...)


@dataclass
class Config:
    root: Path
    aroll_dir: Path
    broll_dir: Path
    grade_default: str
    grade_profiles: dict[str, GradeProfile]
    grade_match: list[dict]       # [{pattern, profile}], first match wins
    selects_docx: Path
    draft_txt: Path | None
    work_dir: Path
    out_dir: Path
    vision_model: str
    vision_url: str
    whisper_model: str
    adaptive_threshold: float
    min_shot_seconds: float
    frames_per_shot: int
    frame_max_px: int
    motion_sample_fps: float
    motion_downscale_width: int
    film_dir: Path | None = None   # where film.yaml lives (see film.py); None = generic profile
    raw: dict = field(repr=False, default_factory=dict)
    _film: object = field(repr=False, default=None)

    @property
    def film(self):
        """The film's profile (film.py): register line, recording roles, ASR aliases."""
        if self._film is None:
            from .film import load_film
            self._film = load_film(self.film_dir)
        return self._film

    @property
    def frames_dir(self) -> Path:
        return self.work_dir / "frames"

    @property
    def sheets_dir(self) -> Path:
        return self.work_dir / "sheets"

    def grade_for(self, source_path: str | Path) -> GradeProfile:
        name = Path(source_path).name.lower()
        for rule in self.grade_match:
            if rule["pattern"].lower() in name:
                return self.grade_profiles[rule["profile"]]
        return self.grade_profiles[self.grade_default]


def load_config(path: str | Path = "config.yaml") -> Config:
    path = Path(path).resolve()
    root = path.parent
    raw = yaml.safe_load(path.read_text())

    def p(key: str) -> Path:
        value = Path(raw["paths"][key])
        return value if value.is_absolute() else root / value

    def resolve(value: str) -> Path:
        path = Path(value)
        return path if path.is_absolute() else root / path

    grade = raw["grade"]
    profiles = {
        name: GradeProfile(
            name=name,
            lut=resolve(spec["lut"]) if "lut" in spec else None,
            filters=spec.get("filters"),
        )
        for name, spec in grade["profiles"].items()
    }

    cfg = Config(
        root=root,
        aroll_dir=p("aroll_dir"),
        broll_dir=p("broll_dir"),
        grade_default=grade["default"],
        grade_profiles=profiles,
        grade_match=grade.get("match", []),
        selects_docx=p("selects_docx"),
        draft_txt=p("draft_txt") if "draft_txt" in raw["paths"] else None,
        work_dir=p("work_dir"),
        out_dir=p("out_dir"),
        vision_model=raw["models"]["vision"],
        vision_url=raw["models"]["vision_url"].rstrip("/"),
        whisper_model=raw["models"]["whisper"],
        adaptive_threshold=float(raw["shots"]["adaptive_threshold"]),
        min_shot_seconds=float(raw["shots"]["min_shot_seconds"]),
        frames_per_shot=int(raw["frames"]["per_shot"]),
        frame_max_px=int(raw["frames"]["max_px"]),
        motion_sample_fps=float(raw["motion"]["sample_fps"]),
        motion_downscale_width=int(raw["motion"]["downscale_width"]),
        film_dir=_film_dir(raw, root, p("out_dir")),
        raw=raw,
    )
    for d in (cfg.work_dir, cfg.out_dir, cfg.frames_dir, cfg.sheets_dir):
        d.mkdir(parents=True, exist_ok=True)
    return cfg


def _film_dir(raw: dict, root: Path, out_dir: Path) -> Path | None:
    """`film_dir:` when the config names it; else the out_dir's parent when a film.yaml sits there
    (the data/films/<film>/{out,work,film.yaml} layout); else None, the generic profile."""
    if raw.get("film_dir"):
        d = Path(raw["film_dir"])
        return d if d.is_absolute() else root / d
    return out_dir.parent if (out_dir.parent / "film.yaml").exists() else None


def list_videos(directory: Path) -> list[Path]:
    """Recursively find video files, following symlinks (footage often lives
    on external drives, linked or pointed at directly)."""
    if not directory.is_dir():
        return []
    found = []
    for dirpath, dirnames, filenames in os.walk(directory, followlinks=True):
        dirnames[:] = [d for d in dirnames if not d.startswith(".")]
        for name in filenames:
            if not name.startswith(".") and Path(name).suffix.lower() in VIDEO_EXTS:
                found.append(Path(dirpath) / name)
    return sorted(found)
