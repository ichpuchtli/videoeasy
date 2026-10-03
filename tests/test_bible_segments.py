"""Offline regression: windows must reach the bible under their take."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from videoeasy.bible import run


def assemble(tmp, segments, annotations):
    out = Path(tmp)
    (out / "shots.json").write_text(json.dumps([
        {"shot_id": "CAMA0131", "role": "aroll", "source_path": "/CAMA0131.MOV",
         "in_s": 0, "out_s": 358, "duration_s": 358},
    ]))
    (out / "annotations.json").write_text(json.dumps(annotations))
    (out / "segments.json").write_text(json.dumps(segments))
    (out / "transcripts.json").write_text(json.dumps({"CAMA0131": {
        "audio_source": "camera",
        "segments": [{"start": 140, "end": 160, "text": "the shed we built years ago"}]}}))
    curated = out / "bible-synthesis.md"
    curated.write_text("Human editorial decision")
    cfg = SimpleNamespace(out_dir=out, sheets_dir=out, draft_txt=None)
    return Path(run(cfg)).read_text(), curated


def window(index, in_s, out_s, visual_tag=True, speech=True):
    record = {
        "shot_id": f"CAMA0131_w{index:02d}", "parent_shot_id": "CAMA0131",
        "source_id": "CAMA0131", "source_path": "/CAMA0131.MOV", "role": "aroll",
        "in_s": in_s, "out_s": out_s, "duration_s": out_s - in_s,
        "parent_in_s": 0, "parent_out_s": 358, "parent_duration_s": 358,
        "start_boundary": {"method": "speech_pause", "silence_s": 1.4},
        "end_boundary": {"method": "even_split"},
        "visual_tag": visual_tag,
    }
    if speech:
        record["speech"] = {"segment_count": 1, "first_word_s": in_s + 2, "last_word_s": out_s - 3,
                            "text": "the shed we built years ago"}
    return record


class BibleSegmentsTest(unittest.TestCase):
    def test_windows_nest_under_the_whole_take_record(self):
        with tempfile.TemporaryDirectory() as tmp:
            text, curated = assemble(
                tmp,
                [window(0, 0, 47), window(3, 134.2, 178.6)],
                {"CAMA0131": {"subject": "Two people walking a track"},
                 "CAMA0131_w03": {"subject": "Two people seated on a low bench",
                                  "shot_type": "medium", "cut_notes": "settles here"}},
            )
            take = text.index("### CAMA0131")
            transcript = text.index("## A-roll transcript")
            # the whole-take tag survives and the windows sit under it
            self.assertIn("Two people walking a track", text)
            self.assertLess(take, text.index("CAMA0131_w00"))
            self.assertLess(text.index("CAMA0131_w03"), transcript)
            self.assertIn("**CAMA0131_w03** | 00:02:14–00:02:58 (44.4s)", text)
            self.assertIn("Two people seated on a low bench", text)
            # provenance of both edges is legible, and neither is called a cut
            self.assertIn("edges: speech pause 1.4s → even split", text)
            self.assertIn("NOT cuts in the footage", text)
            self.assertEqual(curated.read_text(), "Human editorial decision")

    def test_untagged_window_is_still_searchable(self):
        with tempfile.TemporaryDirectory() as tmp:
            text, _ = assemble(tmp, [window(0, 0, 47, visual_tag=False)], {})
            self.assertIn("CAMA0131_w00", text)
            self.assertIn("the shed we built years ago", text)
            self.assertIn("indexed for search only", text)
            self.assertIn("frames not yet extracted for this window", text)
            # nothing to audit without a tag: no unchecked-audit noise
            self.assertNotIn("tag consistency", text)

    def test_window_speech_quotes_rather_than_duplicating_the_transcript(self):
        with tempfile.TemporaryDirectory() as tmp:
            long_speech = window(0, 0, 47)
            long_speech["speech"]["text"] = " ".join(f"word{i}" for i in range(60))
            text, _ = assemble(tmp, [long_speech], {})
            self.assertIn("word29…", text)
            self.assertNotIn("word31", text)
            self.assertIn("speech [00:00:02–00:00:44]", text)


if __name__ == "__main__":
    unittest.main()
