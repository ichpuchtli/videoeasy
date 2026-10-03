"""Offline tests for windowing long takes.

The failure these windows exist to fix was a tag that described a 358-second
walk-and-talk from four samples ninety seconds apart. So the properties that
matter here are: short takes are untouched, windows cover the take exactly,
a window edge records where it came from, and spoken words stay tied to their
source timestamps.
"""
import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from videoeasy import segments
from videoeasy.frames import sample_count
from videoeasy.prompts import annotation_user_prompt


def shot(shot_id="TAKE", duration=300.0, role="aroll", source_id=None):
    return {"shot_id": shot_id, "source_id": source_id or shot_id,
            "source_path": f"/footage/{source_id or shot_id}.MOV", "role": role,
            "in_s": 0.0, "out_s": duration, "duration_s": duration,
            "fps": 23.976, "start_timecode": "12:47:48:21"}


def speech(*spans):
    """Transcript segments from (start, end, text) triples."""
    return [{"start": s, "end": e, "text": t,
             "words": [{"w": w, "s": s, "e": e} for w in t.split()]}
            for s, e, t in spans]


def config(tmp, shots, transcripts=None, options=None):
    out = Path(tmp)
    (out / "shots.json").write_text(json.dumps(shots))
    if transcripts is not None:
        (out / "transcripts.json").write_text(json.dumps(transcripts))
    return SimpleNamespace(out_dir=out, raw={"segments": options} if options else {})


class RulesTest(unittest.TestCase):
    def test_absent_block_uses_defaults(self):
        rules = segments.rules_from(SimpleNamespace(raw={}))
        self.assertEqual(rules.target_seconds, 45.0)
        self.assertEqual(rules.index_only_sources, ())

    def test_unknown_option_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "unknown segments options: stride"):
            segments.rules_from(SimpleNamespace(raw={"segments": {"stride": 4}}))

    def test_window_cannot_silently_exceed_max(self):
        # A window is placed within half a target of its ideal spot, so a max
        # below 1.5x target would be violated by construction rather than
        # enforced.
        with self.assertRaisesRegex(ValueError, "1.5x"):
            segments.rules_from(SimpleNamespace(raw={"segments": {"target_seconds": 60, "max_seconds": 70}}))

    def test_rejects_nonsense_numbers(self):
        for options in ({"pause_seconds": 0}, {"min_seconds": 90}, {"frames_per_segment": 1},
                        {"index_only_sources": "CAMA0124"}, {"target_seconds": "45"}):
            with self.assertRaises(ValueError):
                segments.rules_from(SimpleNamespace(raw={"segments": options}))


class WindowTest(unittest.TestCase):
    def test_short_takes_are_left_alone(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = config(tmp, [shot("SHORT", 42.0)], {})
            self.assertEqual(segments.build(cfg), [])

    def test_windows_cover_the_take_exactly_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = config(tmp, [shot("TAKE", 358.0)], {})
            windows = segments.build(cfg)
            self.assertGreater(len(windows), 5)
            self.assertEqual(windows[0]["in_s"], 0.0)
            self.assertEqual(windows[-1]["out_s"], 358.0)
            for earlier, later in zip(windows, windows[1:]):
                self.assertEqual(earlier["out_s"], later["in_s"])
            for window in windows:
                self.assertAlmostEqual(window["duration_s"], window["out_s"] - window["in_s"], places=3)
                self.assertGreaterEqual(window["duration_s"], 12.0)
                self.assertLessEqual(window["duration_s"], 90.0)

    def test_sub_shot_parent_times_are_preserved(self):
        # A b-roll shot that is already a slice of its source keeps that
        # offset: windows are source-relative, never shot-relative.
        with tempfile.TemporaryDirectory() as tmp:
            parent = {**shot("DJI_1233_s01", role="broll", source_id="DJI_1233"),
                      "in_s": 51.718, "out_s": 173.365, "duration_s": 121.647}
            windows = segments.build(config(tmp, [parent], {}))
            self.assertEqual(windows[0]["in_s"], 51.718)
            self.assertEqual(windows[-1]["out_s"], 173.365)
            self.assertEqual(windows[0]["parent_in_s"], 51.718)
            self.assertEqual(windows[0]["source_id"], "DJI_1233")
            self.assertTrue(all(w["parent_shot_id"] == "DJI_1233_s01" for w in windows))

    def test_edges_land_on_real_pauses_and_record_them(self):
        with tempfile.TemporaryDirectory() as tmp:
            spoken = speech((0.0, 44.0, "first stretch of talking"),
                            (47.5, 90.0, "second stretch of talking"))
            windows = segments.build(config(tmp, [shot("TALK", 90.0)], {"TALK": {"segments": spoken}}))
            self.assertEqual(len(windows), 2)
            self.assertEqual(windows[0]["out_s"], 45.75)  # midpoint of the 3.5s silence
            self.assertEqual(windows[0]["end_boundary"], {"method": "speech_pause", "silence_s": 3.5})
            self.assertEqual(windows[1]["start_boundary"], {"method": "speech_pause", "silence_s": 3.5})
            self.assertEqual(windows[0]["start_boundary"], {"method": "source_start"})
            self.assertEqual(windows[-1]["end_boundary"], {"method": "source_end"})

    def test_unbroken_speech_is_split_evenly_and_says_so(self):
        with tempfile.TemporaryDirectory() as tmp:
            spoken = speech(*[(t, t + 5.0, "unbroken talking") for t in range(0, 120, 5)])
            windows = segments.build(config(tmp, [shot("SOLID", 120.0)], {"SOLID": {"segments": spoken}}))
            self.assertEqual([w["start_boundary"]["method"] for w in windows],
                             ["source_start", "even_split", "even_split"])
            self.assertNotIn("silence_s", windows[1]["start_boundary"])

    def test_no_transcript_gives_even_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            windows = segments.build(config(tmp, [shot("DRONE", 119.0, role="broll")], {}))
            self.assertEqual(len(windows), 3)
            for window in windows:
                self.assertAlmostEqual(window["duration_s"], 119.0 / 3, places=1)
            self.assertNotIn("speech", windows[0])

    def test_spoken_words_stay_tied_to_source_timestamps(self):
        with tempfile.TemporaryDirectory() as tmp:
            spoken = speech((10.0, 44.0, "stack the boards"), (47.5, 80.0, "under the tarp"))
            windows = segments.build(config(tmp, [shot("TALK", 90.0)], {"TALK": {"segments": spoken}}))
            self.assertEqual(windows[0]["speech"]["text"], "stack the boards")
            self.assertEqual(windows[0]["speech"]["first_word_s"], 10.0)
            self.assertEqual(windows[1]["speech"]["text"], "under the tarp")
            # every spoken segment lands in exactly one window
            self.assertEqual(sum(w["speech"]["segment_count"] for w in windows), len(spoken))

    def test_index_only_sources_are_windowed_but_not_tagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = config(tmp, [shot("CAMA0124", 600.0), shot("OTHER", 600.0)], {},
                         options={"index_only_sources": ["CAMA0124"]})
            windows = segments.build(cfg)
            by_source = {w["source_id"] for w in windows}
            self.assertEqual(by_source, {"CAMA0124", "OTHER"})
            self.assertFalse(any(w["visual_tag"] for w in windows if w["source_id"] == "CAMA0124"))
            self.assertTrue(all(w["visual_tag"] for w in windows if w["source_id"] == "OTHER"))

    def test_ids_are_stable_and_safe_as_directory_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = config(tmp, [shot("DJI_1233_s01", 200.0, role="broll", source_id="DJI_1233")], {})
            first = [w["shot_id"] for w in segments.build(cfg)]
            self.assertEqual(first, [w["shot_id"] for w in segments.build(cfg, force=True)])
            for shot_id in first:
                # the evidence ledger refuses anything else as a directory name
                self.assertRegex(shot_id, re.compile(r"[A-Za-z0-9_.-]+$"))
            self.assertEqual(first[0], "DJI_1233_s01_w00")

    def test_existing_windows_survive_a_later_transcript(self):
        # Re-running must not shift the boundaries of a window whose frames are
        # already extracted and whose tag is already written.
        with tempfile.TemporaryDirectory() as tmp:
            cfg = config(tmp, [shot("TALK", 120.0)], {})
            before = segments.build(cfg)
            (Path(tmp) / "transcripts.json").write_text(json.dumps(
                {"TALK": {"segments": speech((0.0, 30.0, "a"), (40.0, 120.0, "b"))}}))
            self.assertEqual(segments.build(cfg), before)
            after = segments.build(cfg, force=True)
            self.assertNotEqual(after, before)
            self.assertEqual(after[0]["end_boundary"]["method"], "speech_pause")

    def test_units_keeps_whole_takes_alongside_windows(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = config(tmp, [shot("TAKE", 120.0), shot("SHORT", 20.0)], {})
            segments.build(cfg)
            units = segments.units(cfg)
            self.assertIn("TAKE", [u["shot_id"] for u in units])
            self.assertIn("SHORT", [u["shot_id"] for u in units])
            self.assertFalse(segments.is_segment(units[0]))
            self.assertTrue(any(segments.is_segment(u) for u in units))


class SamplingTest(unittest.TestCase):
    def test_windows_are_sampled_densely_and_whole_takes_are_untouched(self):
        cfg = SimpleNamespace(frames_per_shot=8, raw={})
        rules = segments.rules_from(cfg)
        window = {"shot_id": "TAKE_w00", "parent_shot_id": "TAKE", "role": "aroll"}
        self.assertEqual(sample_count(cfg, window, 4, rules), 6)
        self.assertEqual(sample_count(cfg, {"shot_id": "TAKE", "role": "aroll"}, 4, rules), 4)
        self.assertEqual(sample_count(cfg, {"shot_id": "TREE", "role": "broll"}, 4, rules), 8)


class PromptTest(unittest.TestCase):
    def test_window_prompt_refuses_to_call_the_edges_a_cut(self):
        window = {"shot_id": "CAMA0131_w03", "parent_shot_id": "CAMA0131", "role": "aroll",
                  "duration_s": 44.4, "in_s": 134.2, "out_s": 178.6, "parent_duration_s": 358.0,
                  "speech": {"text": "we stack the boards under the tarp"}}
        prompt = annotation_user_prompt(window, None)
        self.assertIn("WINDOW inside the longer continuous take CAMA0131", prompt)
        self.assertIn("134.2s to 178.6s", prompt)
        self.assertIn("NOT an edit point", prompt)
        self.assertIn("do not describe them as a cut", prompt)
        # A spoken topic is not a visible fact: the transcript never reaches
        # the vision model, or the tag starts describing what it heard.
        self.assertNotIn("black plastic", prompt)

    def test_missing_motion_stats_are_stated_not_invented(self):
        # The model claimed "no significant optical flow is detected" on a film
        # with no motion.json. Silence about the stats is what produced that.
        prompt = annotation_user_prompt({"shot_id": "X", "role": "broll", "duration_s": 40.0}, None)
        self.assertIn("No measured motion statistics are supplied", prompt)
        self.assertIn("Do not mention optical flow", prompt)

    def test_whole_take_prompt_is_unchanged(self):
        prompt = annotation_user_prompt(
            {"shot_id": "CAMB0206", "role": "aroll", "duration_s": 42.0}, "slow drift")
        self.assertNotIn("WINDOW", prompt)
        self.assertIn("start to end of the shot", prompt)
        self.assertIn("Measured motion: slow drift", prompt)


if __name__ == "__main__":
    unittest.main()
