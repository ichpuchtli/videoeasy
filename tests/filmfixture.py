"""A synthetic film profile shared by the tests: no real film, clip or person."""
from videoeasy.film import from_dict

TEST_FILM = from_dict({
    "name": "test film",
    "context": "A short test documentary about a working landscape.",
    "roles": {
        "sitdown": {"label": "sit-down two-shot", "never_cover": True},
        "presenter": {"label": "presenter to camera", "to_camera": True, "never_cover": True},
        "explainer": {"label": "explainer to camera", "to_camera": True},
        "walk": {"label": "walk-and-talk, sync sound", "sync_walk": True},
        "drone": {"label": "drone", "drone": True},
    },
    "default_role": "walk",
    "clips": {"CAMA0124": "sitdown", **{f"CAMB02{n}": "presenter" for n in range(19, 24)}},
    "prefixes": [["CAMB01", "explainer"], ["CAMB02", "explainer"], ["DJI_", "drone"]],
    "aliases": {"rowan": ["rohan"], "mulga": ["mulgar", "mulger"]},
    "names": ["avery", "rowan"],
    "layin": {"placeholders": {"before": "ph01_before_photos.mov", "archive": "ph02_archive_inserts.mov",
                               "trail": "ph04_trail_cam.mov", "sign": "ph03_signs.mov", "aerial": "ph05_timelapse.mov"}},
})
