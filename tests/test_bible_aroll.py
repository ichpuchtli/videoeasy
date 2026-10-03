"""Offline regression: visual A-roll must survive bible assembly."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from videoeasy.bible import run


class BibleArollTest(unittest.TestCase):
    def test_visual_aroll_and_audio_provenance_survive(self):
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)
            shots = [
                {"shot_id": sid, "role": role, "source_path": f"/{sid}.MOV",
                 "in_s": 0, "out_s": 10, "duration_s": 10}
                for sid, role in (("interview", "aroll"), ("trees", "broll"))
            ]
            (out / "shots.json").write_text(json.dumps(shots))
            (out / "annotations.json").write_text(json.dumps({
                "interview": {"subject": "Two people seated on a bench"},
                "trees": {"subject": "Tree canopy"},
            }))
            (out / "transcripts.json").write_text(json.dumps({
                "interview": {"audio_source": "camera", "segments": [
                    {"start": 0, "end": 1, "text": "A spoken sentence."}]},
            }))
            curated = out / "bible-synthesis.md"
            curated.write_text("Human editorial decision")
            cfg = SimpleNamespace(out_dir=out, sheets_dir=out, draft_txt=None)
            text = Path(run(cfg)).read_text()
            for required in ("Two people seated on a bench", "Tree canopy",
                             "role: aroll", "audio source: camera",
                             "A spoken sentence."):
                self.assertIn(required, text)
            self.assertEqual(curated.read_text(), "Human editorial decision")


if __name__ == "__main__":
    unittest.main()
