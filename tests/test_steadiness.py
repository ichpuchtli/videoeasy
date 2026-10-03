"""Cover rules measured without a model: camera shake (steadiness.py) and talking cover (speech.py). No footage, no models."""
from __future__ import annotations

import unittest

import numpy as np

from videoeasy import brollmatch, cutbuild, speech, steadiness


def pairs_from_path(path_xy: np.ndarray, inliers: int = 200) -> np.ndarray:
    d = np.diff(np.vstack([[0.0, 0.0], path_xy]), axis=0)
    t = np.arange(len(d)) / steadiness.SAMPLE_FPS
    return np.column_stack([t, d, np.full(len(d), inliers, dtype=float)])


class ShakeStats(unittest.TestCase):
    def test_smooth_pan_is_steady(self):
        n = 120  # 10 s
        path = np.column_stack([np.linspace(0, 400, n), np.zeros(n)])  # a 400 px pan over 10 s, perfectly smooth
        st = steadiness.shake_stats(pairs_from_path(path))
        self.assertEqual(st["status"], "ok")
        self.assertLess(st["shake_pct"], 0.2)
        self.assertGreater(st["pan_pct_per_s"], 5.0)

    def test_jitter_is_shaky(self):
        rng = np.random.default_rng(1)
        n = 120
        path = np.cumsum(rng.normal(0, 12.0, size=(n, 2)), axis=0)  # random walk of 12 px per frame pair at 480 wide (2.5% of width)
        st = steadiness.shake_stats(pairs_from_path(path))
        self.assertGreater(st["shake_pct"], steadiness.MAX_SHAKE_PCT)
        self.assertTrue(steadiness.shaky(st))

    def test_a_cut_with_no_inliers_adds_no_motion(self):
        n = 120
        path = np.column_stack([np.linspace(0, 40, n), np.zeros(n)])
        p = pairs_from_path(path)
        p[60, 1:3] = (300.0, -200.0)   # a hard cut: the fit reports a huge jump with no inlier support
        p[60, 3] = 3
        st = steadiness.shake_stats(p)
        self.assertLess(st["shake_pct"], 0.2)
        self.assertGreater(st["unreliable_frac"], 0.0)

    def test_short_unit_is_unchecked_not_steady(self):
        st = steadiness.shake_stats(pairs_from_path(np.zeros((5, 2))))
        self.assertEqual(st["status"], "unchecked")
        self.assertFalse(steadiness.shaky(st))

    def test_profile_reads_shake_along_the_unit(self):
        rng = np.random.default_rng(2)
        n = 240  # 20 s: jitter for 8 s, then a locked-off camera
        steps = np.zeros((n, 2))
        steps[:96] = rng.normal(0, 12.0, size=(96, 2))
        p = pairs_from_path(np.cumsum(steps, axis=0))
        prof = steadiness.profile(p)
        self.assertGreater(prof[0][1], steadiness.MAX_SHAKE_PCT)       # window at 0 s
        self.assertLess(dict(prof)[12.0], 0.1)                          # window at 12 s
        self.assertEqual(prof[1][0] - prof[0][0], steadiness.PROFILE_HOP_S)

    def test_stable_runs_keep_the_steady_part_of_a_shaky_unit(self):
        rng = np.random.default_rng(2)
        n = 240
        steps = np.zeros((n, 2))
        steps[:96] = rng.normal(0, 12.0, size=(96, 2))
        p = pairs_from_path(np.cumsum(steps, axis=0))
        self.assertTrue(steadiness.shaky(steadiness.shake_stats(p)))    # the unit as one number is barred ...
        runs = steadiness.stable_runs(p)
        self.assertEqual(len(runs), 1)                                    # ... but its steady tail is a usable run
        self.assertGreaterEqual(runs[0]["t0"], 7.5)
        self.assertGreater(runs[0]["t1"] - runs[0]["t0"], 10.0)
        self.assertLessEqual(runs[0]["shake_pct"], steadiness.MAX_SHAKE_PCT)
        self.assertEqual(steadiness.stable_runs(p, min_s=15.0), [])      # too short for a longer minimum
        steady = pairs_from_path(np.column_stack([np.linspace(0, 40, n), np.zeros(n)]))
        whole = steadiness.stable_runs(steady)
        self.assertEqual(len(whole), 1)
        self.assertLess(whole[0]["t0"], 0.1)                              # a steady unit is one run from end to end

    def test_a_run_never_bridges_a_window_over_the_limit(self):
        # a half-second jolt mid-take: steady windows either side overlap it, and joining them made one run across
        # the jolt (46 such windows in one film's B-roll, 29 Sep 2026)
        rng = np.random.default_rng(3)
        n = 360
        steps = np.zeros((n, 2))
        steps[180:186] = rng.normal(0, 10.0, size=(6, 2))
        p = pairs_from_path(np.cumsum(steps, axis=0))
        over = [(a, a + steadiness.PROFILE_WIN_S) for a, v in steadiness.profile(p) if v > steadiness.MAX_SHAKE_PCT]
        self.assertTrue(over)
        runs = steadiness.stable_runs(p)
        self.assertEqual(len(runs), 2)
        for r in runs:
            for a, b in over:
                self.assertTrue(b <= r["t0"] + 1e-6 or a >= r["t1"] - 1e-6, (r, a, b))

    def test_ineligible_names_only_measured_units_over_the_limit(self):
        steady = {"A": dict(status="ok", shake_pct=4.1, stable_runs=[], stable_limit=steadiness.MAX_SHAKE_PCT),
                  "B": dict(status="ok", shake_pct=0.3, profile_max=0.5), "C": dict(status="unchecked")}
        out = steadiness.ineligible(steady, ["A", "B", "C", "D"])
        self.assertEqual(list(out), ["A"])
        self.assertIn("4.1", out["A"])
        self.assertEqual(steadiness.ineligible(None, ["A"]), {})

    def test_a_shaky_unit_with_steady_parts_is_restricted_not_barred(self):
        L = steadiness.MAX_SHAKE_PCT
        runs = [dict(t0=32.0, t1=77.5, shake_pct=0.6), dict(t0=87.0, t1=99.0, shake_pct=0.9)]
        steady = {"A": dict(status="ok", shake_pct=6.0, stable_runs=runs, stable_limit=L),           # shaky, steady parts at this limit
                  "B": dict(status="ok", shake_pct=6.0, stable_runs=[], stable_limit=L),             # shaky, nothing usable
                  "D": dict(status="ok", shake_pct=0.3, stable_runs=[dict(t0=0.0, t1=40.0, shake_pct=0.3)], stable_limit=L,
                            profile_max=0.5)}
        cat = {k: dict(in_s=0.0, out_s=100.0) for k in "ABD"}
        out = steadiness.apply_runs(cat, steady)
        self.assertEqual(sorted(out), ["B"])
        self.assertIn("no steady part", out["B"])
        self.assertEqual(cat["A"]["steady"], runs)            # restricted to its steady parts
        self.assertIsNone(cat["D"]["steady"])                 # a steady unit is not restricted
        self.assertEqual(cat["B"]["steady"], [])
        self.assertEqual(steadiness.apply_runs({"A": {}}, None), {})
        stale = {"C": dict(status="ok", shake_pct=6.0, stable_runs=runs, stable_limit=L + 1.0)}   # runs found at another limit
        with self.assertRaises(SystemExit):                    # never a silent whole-unit bar: re-measure first
            steadiness.apply_runs({"C": dict(in_s=0.0, out_s=100.0)}, stale)
        old = {"E": dict(status="ok", shake_pct=6.0)}                                             # measured before runs existed
        with self.assertRaises(SystemExit):
            steadiness.apply_runs({"E": dict(in_s=0.0, out_s=100.0)}, old)

    def test_a_unit_steady_as_a_whole_with_a_shaky_stretch_is_restricted(self):
        # v8 (29 Sep 2026): CAMB0135 measured 1.7 % whole and 2.5 % on the 6.7 s the cut used, across a 1 s shaky
        # window; DJI_1234_s01 1.6 % whole, 3.9 % used. The worst window, not the whole-unit figure, decides.
        L = steadiness.MAX_SHAKE_PCT
        runs = [dict(t0=0.08, t1=4.58, shake_pct=0.54), dict(t0=5.58, t1=26.08, shake_pct=0.37)]
        steady = {"S": dict(status="ok", shake_pct=1.7, stable_runs=runs, stable_limit=L, profile_max=3.1),
                  "SHORT": dict(status="ok", shake_pct=0.6, stable_runs=[], stable_limit=L, profile_max=0.9),   # 3 s steady: no 3 s run
                  "OLD": dict(status="ok", shake_pct=1.7, stable_runs=runs, stable_limit=L)}                     # measured before profile_max
        cat = {k: dict(in_s=0.0, out_s=27.0) for k in ("S", "SHORT")}
        self.assertEqual(steadiness.apply_runs(cat, steady), {})
        self.assertEqual(cat["S"]["steady"], runs)                  # restricted although 1.7 % whole
        self.assertIsNone(cat["SHORT"]["steady"])                   # every window steady: used whole, never barred for being short
        self.assertEqual(steadiness.clamp_to_runs(0.25, 6.66, runs), (5.58, 6.66))   # v8's row moves past the shaky second
        with self.assertRaises(SystemExit):                          # an old file must be re-measured, not read as steady
            steadiness.apply_runs({"OLD": dict(in_s=0.0, out_s=27.0)}, steady)
        bad = {"X": dict(status="ok", shake_pct=1.2, stable_runs=[], stable_limit=L, profile_max=2.6)}
        self.assertIn("up to 2.6%", steadiness.ineligible(bad, ["X"])["X"])

    def test_rows_are_clamped_into_a_steady_part(self):
        runs = [dict(t0=32.0, t1=77.5), dict(t0=87.0, t1=99.0)]
        self.assertEqual(steadiness.clamp_to_runs(0.0, 8.0, runs), (32.0, 8.0))      # before the first part: starts at it
        self.assertEqual(steadiness.clamp_to_runs(70.0, 12.0, runs), (87.0, 12.0))   # 7.5 s left here, 12 s in the next: the next
        self.assertEqual(steadiness.clamp_to_runs(70.0, 20.0, runs), (87.0, 12.0))   # nothing holds 20 s: the part that holds most
        self.assertEqual(steadiness.clamp_to_runs(74.0, 20.0, [dict(t0=32.0, t1=80.0), dict(t0=87.0, t1=90.0)]), (74.0, 6.0))
        self.assertEqual(steadiness.clamp_to_runs(77.0, 8.0, runs), (87.0, 8.0))     # under a second left: the next part
        self.assertIsNone(steadiness.clamp_to_runs(98.5, 8.0, runs))                 # nothing left
        self.assertEqual(steadiness.fmt_runs(runs), "0:32.0-1:17.5, 1:27.0-1:39.0")


TR = {"CAMB0182": {"segments": [{"words": [{"w": "Okay", "s": 0.9, "e": 1.4}, {"w": "so", "s": 1.5, "e": 2.0},
                                            {"w": "fresh", "s": 6.0, "e": 6.6}, {"w": "growth", "s": 6.6, "e": 7.2}]}]}}


class TalkingCover(unittest.TestCase):
    def test_spoken_seconds_and_unknown_source(self):
        self.assertAlmostEqual(speech.spoken_seconds(TR, "CAMB0182", 0.0, 3.0), 1.0)
        self.assertAlmostEqual(speech.spoken_seconds(TR, "CAMB0182_w01", 6.3, 10.0), 0.9)
        self.assertEqual(speech.spoken_seconds(TR, "CAMB0182", 3.0, 5.0), 0.0)
        self.assertIsNone(speech.spoken_seconds(TR, "CAMB0135", 0.0, 5.0))
        self.assertEqual(speech.source_of("DJI_1234_s01"), "DJI_1234")
        self.assertEqual(speech.spoken_seconds(dict(TR, DJI_1234={"segments": [], "role": "broll"}), "DJI_1234_s01", 0.0, 5.0), 0.0)

    def test_talking_cover_lists_talking_and_unknown_rows_not_silent_ones(self):
        cut = {"beats": [{"title": "7", "audio": [dict(clip="CAMA0124", in_s=0, out_s=20, duration_s=20, text="x", tier=1)], "video": [
            dict(type="video", clip="CAMB0182", in_s=0.0, out_s=8.0, pick="b7p3", dest_in_s=346.7),
            dict(type="video", clip="CAMB0182", in_s=3.0, out_s=5.0, pick="b7p4", dest_in_s=354.7),
            dict(type="video", clip="CAMB0135", in_s=0.0, out_s=5.0, pick="b7p5", dest_in_s=356.7),
            dict(type="bare", clip=None, in_s=0.0, out_s=2.0, pick="b7p6")]}]}
        rows = speech.talking_cover(cut, TR)
        self.assertEqual([(r["pick"], r["status"]) for r in rows], [("b7p3", "talking"), ("b7p5", "unknown")])
        self.assertAlmostEqual(rows[0]["speech_s"], 2.2)
        # off camera by the tag, and a beat outside the tier, are not talking cover
        self.assertEqual(speech.talking_cover(cut, TR, people={"CAMB0182": "nobody"}), [dict(rows[1])])
        cut2 = {"beats": [dict(cut["beats"][0], audio=[dict(clip="CAMA0124", in_s=0, out_s=5, duration_s=5, text="x", tier=2)])]}
        self.assertEqual(speech.talking_cover(cut2, TR, tier=1), [])

    def test_sync_picture_is_exempt_but_the_same_take_at_another_time_is_not(self):
        cut = {"beats": [{"title": "16", "audio": [dict(clip="CAMB0182", in_s=0.5, out_s=8.5, duration_s=8.0, text="x", tier=1)], "video": [
            dict(type="video", clip="CAMB0182_w00", in_s=2.5, out_s=6.5, pick="b1p1", dest_in_s=2.0),     # same take, same time: sync
            dict(type="video", clip="CAMB0182", in_s=0.0, out_s=2.0, pick="b1p1", dest_in_s=6.0)]}]}   # same take, 6.5 s earlier
        rows = speech.talking_cover(cut, TR)
        self.assertEqual([(r["dest_in_s"], r["status"]) for r in rows], [(6.0, "talking")])
        self.assertTrue(speech.in_sync("CAMB0182_w00", 2.5, 2.0, ("CAMB0182", 0.5, 0.0)))
        self.assertFalse(speech.in_sync("CAMB0182", 0.0, 6.0, ("CAMB0182", 0.5, 0.0)))
        self.assertFalse(speech.in_sync("CAMB0135", 0.5, 0.0, ("CAMB0182", 0.5, 0.0)))

    def test_talking_units_bars_the_window_with_speech(self):
        units = [dict(id="CAMB0182_w00", in_s=0.0, out_s=8.0), dict(id="CAMB0182_w01", in_s=3.0, out_s=5.5), dict(id="CAMB0135", in_s=0.0, out_s=9.0)]
        out = speech.talking_units(TR, units)
        self.assertEqual(list(out), ["CAMB0182_w00"])

    def test_speech_with_nobody_visible_is_off_camera_and_allowed(self):
        units = [dict(id="CAMB0182_w00", in_s=0.0, out_s=8.0, people="nobody"),
                 dict(id="CAMB0182", in_s=0.0, out_s=8.0, people="one person seen from behind")]
        self.assertEqual(list(speech.talking_units(TR, units)), ["CAMB0182"])

    def test_a_clip_sam_allowed_by_name_is_exempt_and_nothing_else(self):
        from videoeasy import intent
        doc = {"rules": [dict(id="banner", topic="talking-cover-exceptions", kind="talking_cover_exempt", clips=["CAMB0182"],
                              decision="allowed by name", decided="2026-09-29")]}
        exempt = intent.talking_exempt(doc)
        self.assertEqual(exempt, {"CAMB0182"})
        self.assertEqual(intent.talking_exempt(None), set())
        units = [dict(id="CAMB0182_w00", in_s=0.0, out_s=8.0, people="one person"), dict(id="CAMB0199", in_s=0.0, out_s=8.0, people="one person")]
        tr = dict(TR, CAMB0199=TR["CAMB0182"])
        self.assertEqual(list(speech.talking_units(tr, units, exempt=exempt)), ["CAMB0199"])   # a window of the named take is exempt too
        cut = {"beats": [{"audio": [dict(clip="CAMA0124", in_s=100.0, out_s=108.0, duration_s=8.0, text="x", tier=1)],
                          "video": [dict(type="video", clip="CAMB0182", in_s=0.0, out_s=8.0, pick="b1p1", dest_in_s=0.0)]}]}
        self.assertTrue(speech.talking_cover(cut, TR))
        self.assertEqual(speech.talking_cover(cut, TR, exempt=exempt), [])
        f = intent.check_cut(cut, doc)
        self.assertEqual(f[0]["status"], intent.NOT_TESTABLE)
        self.assertIn("CAMB0182", f[0]["evidence"])


class Encoding(unittest.TestCase):
    def test_intent_with_non_ascii_loads_under_an_ascii_locale(self):
        """Resolve's scripting library switches the process locale to C; layin crashed reading intent.json (29 Sep 2026)."""
        import json, locale, tempfile, pathlib
        from videoeasy import intent
        with tempfile.TemporaryDirectory() as d:
            ed = pathlib.Path(d) / "editorial"; ed.mkdir()
            (ed / "intent.json").write_text(json.dumps({"rules": [dict(id="banner", kind="note", decision="relearn · regenerate · reconnect — ok")]},
                                                       ensure_ascii=False), encoding="utf-8")
            old = locale.setlocale(locale.LC_ALL)
            try:
                locale.setlocale(locale.LC_ALL, "C")
                self.assertIn("·", intent.load(d)["rules"][0]["decision"])
            finally:
                locale.setlocale(locale.LC_ALL, old)


class Catalogue(unittest.TestCase):
    def test_heard_in_separates_speech_silence_and_unknown(self):
        tr = dict(TR, CAMB0135={"segments": [], "quality": {"status": "silent"}, "role": "broll"}, CAMB0140={"segments": [], "role": "broll"})
        self.assertEqual(brollmatch.heard_in(tr, "CAMB0182", 0.0, 3.0)["status"], "speech")
        self.assertEqual(brollmatch.heard_in(tr, "CAMB0182", 0.0, 3.0)["words"], ["Okay", "so"])
        self.assertEqual(brollmatch.heard_in(tr, "CAMB0135", 0.0, 3.0)["status"], "no_audio")
        self.assertEqual(brollmatch.heard_in(tr, "CAMB0140", 0.0, 3.0)["status"], "silent")
        self.assertEqual(brollmatch.heard_in(tr, "DJI_1229", 0.0, 3.0)["status"], "not_transcribed")

    def test_catalogue_text_marks_heard_as_said_not_seen(self):
        cat = {"CAMB0182": dict(duration_s=8.0, shot_type="medium", hold=5, shake_pct=0.4, subject="hands and a plant", people="one person",
                                movement="static", mood="quiet", themes=["botany"], notes="sharp",
                                heard=dict(status="speech", words=["Okay", "so"], seconds=1.0, truncated=False), synthesis=["6. Proposed spine"]),
               "DJI_1229": dict(duration_s=45.0, shot_type="landscape", hold=10, shake_pct=None, subject="canopy", people="nobody",
                                movement="slow", mood="still", themes=[], notes="", heard=dict(status="not_transcribed", words=[], seconds=None), synthesis=[])}
        txt = brollmatch.catalogue_text(cat, set())
        self.assertIn('Heard on its audio (said, not necessarily visible): "Okay so"', txt)
        self.assertIn("shake 0.4%", txt)
        self.assertIn("Synthesis cites it under: 6. Proposed spine", txt)
        self.assertIn("Audio: not checked.", txt)

    def test_a_later_synthesis_is_read_only_when_named(self):
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as d:
            out = pathlib.Path(d) / "out"; out.mkdir()
            (out / "bible-synthesis.md").write_text("# First\n## 6. Proposed spine\nuses CAMA0131\n")
            (out / "bible-synthesis-v2.md").write_text("# Second\n## 6. Proposed spine\n#### 14. The walk\nuses CAMA0139\n")
            first = brollmatch.synthesis_links(pathlib.Path(d))
            second = brollmatch.synthesis_links(pathlib.Path(d), "out/bible-synthesis-v2.md")
            self.assertEqual(brollmatch.synthesis_path(pathlib.Path(d)).name, "bible-synthesis.md")
        self.assertIn("CAMA0131", first)
        self.assertNotIn("CAMA0139", first)
        self.assertEqual(second, {"CAMA0139": ["14. The walk"]})

    def test_synthesis_links_map_sources_to_the_sections_that_cite_them(self):
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as d:
            out = pathlib.Path(d) / "out"; out.mkdir()
            (out / "bible-synthesis.md").write_text("# Film\n## 4. Themes\nSeats: CAMA0131 and CAMA0124.\n### 6. Spine\nbeat 2 uses CAMA0131_w04 again\n")
            links = brollmatch.synthesis_links(pathlib.Path(d))
        self.assertEqual(links["CAMA0131"], ["4. Themes", "6. Spine"])
        self.assertEqual(links["CAMA0124"], ["4. Themes"])


class RulesInTheLoop(unittest.TestCase):
    CAT = {"CAMB0182": dict(in_s=0.0, out_s=30.0, duration_s=30.0, source="CAMB0182", role="aroll", frames=[]),
           "CAMB0135": dict(in_s=0.0, out_s=30.0, duration_s=30.0, source="CAMB0135", role="broll", frames=[])}

    def row(self, pick, shot, alts=()):
        v = {s: dict(status="ok", score=3, usable=[1, 2, 3, 4], reason="fine") for s in (shot, *alts) if s}
        return dict(pick=pick, shot=shot, duration_s=6.0, text="words", verdicts=v, alternatives=list(alts), status="ok", need="illustrate")

    def test_builder_drops_an_ineligible_shot_to_bare_with_the_reason(self):
        beat = {"title": "7. What the walk found", "audio": [dict(clip="CAMA0124", in_s=0.0, out_s=6.0, duration_s=6.0, text="x", tier=1)], "video": []}
        bad = {"CAMB0182": "6 s of transcribed speech in the unit: someone on screen is talking"}
        rows = cutbuild.build_beat_video(beat, [self.row("b7p3", "CAMB0182")], self.CAT, 12.0, {}, 0.0, ineligible=bad)
        self.assertEqual([r["type"] for r in rows], ["bare"])
        self.assertIn("not cover", rows[0]["note"])
        self.assertIn("talking", rows[0]["note"])

    def test_builder_does_not_fall_back_to_an_ineligible_alternative(self):
        beat = {"title": "7", "audio": [dict(clip="CAMA0124", in_s=0.0, out_s=6.0, duration_s=6.0, text="x", tier=1)], "video": []}
        bad = {"CAMB0135": "camera shake 4.1% of frame width (limit 2.0)"}
        rows = cutbuild.build_beat_video(beat, [self.row("b7p3", "CAMB0182", ("CAMB0135",))], self.CAT, 12.0, {}, 0.0, ineligible=bad)
        self.assertEqual([r.get("clip") for r in rows], ["CAMB0182"])
        rows = cutbuild.build_beat_video(beat, [self.row("b7p3", "CAMB0135", ("CAMB0182",))], self.CAT, 12.0, {}, 0.0, ineligible=bad)
        self.assertEqual([r["type"] for r in rows], ["bare"])

    def test_builder_keeps_a_shaky_shot_inside_its_steady_part(self):
        cat = {"CAMB0140": dict(in_s=0.0, out_s=105.0, duration_s=105.0, source="CAMB0140", role="broll", frames=[],
                                steady=[dict(t0=32.0, t1=40.0, shake_pct=0.6), dict(t0=87.0, t1=99.0, shake_pct=0.9)])}
        used: dict[str, float] = {}
        row = cutbuild.shot_row(cat, "CAMB0140", 6.0, [1, 2, 3, 4], "fine", used,
                                frames_used=[dict(t=5.0), dict(t=36.0), dict(t=70.0), dict(t=95.0)])
        self.assertEqual((row["in_s"], row["out_s"]), (32.0, 38.0))   # the certified start at 5 s is in the shake: the first steady part
        self.assertIn("steady part", row["note"])
        row2 = cutbuild.shot_row(cat, "CAMB0140", 10.0, [1, 2, 3, 4], "fine", used, frames_used=[dict(t=5.0)])
        self.assertEqual((row2["in_s"], row2["out_s"]), (87.0, 97.0))  # reused: unused footage in the part that holds it, never the shake between
        self.assertNotIn("repeats", row2["note"])
        row3 = cutbuild.shot_row(cat, "CAMB0140", 8.0, [1, 2, 3, 4], "fine", used, frames_used=[dict(t=5.0)])
        self.assertEqual((row3["in_s"], row3["out_s"]), (32.0, 40.0))  # nothing unused holds more: back to the start, said so
        self.assertIn("repeats footage", row3["note"])
        beat = {"title": "x", "audio": [dict(clip="CAMA0124", in_s=0.0, out_s=20.0, duration_s=20.0, text="x", tier=1)], "video": []}
        rows = [dict(self.row("p1", "CAMB0140"), duration_s=6.0), dict(self.row("p2", "CAMB0140"), duration_s=14.0)]
        for r in rows:
            r["verdicts"]["CAMB0140"]["frames_used"] = [dict(t=5.0), dict(t=36.0), dict(t=70.0), dict(t=95.0)]
        out = cutbuild.build_beat_video(beat, rows, cat, 12.0, {})
        vid = [v for v in out if v["type"] == "video"]
        for v in vid:
            self.assertTrue(any(r["t0"] - 0.01 <= v["in_s"] and v["out_s"] <= r["t1"] + 0.01 for r in cat["CAMB0140"]["steady"]), v)
        self.assertAlmostEqual(sum(v["duration_s"] for v in out), 20.0, places=1)   # the pick timing holds; the shake is filled bare

    def test_verifier_sees_only_frames_inside_the_steady_parts(self):
        import tempfile, pathlib
        with tempfile.TemporaryDirectory() as d:
            fdir = pathlib.Path(d)
            for k in range(8):
                (fdir / f"f0{k}.jpg").write_bytes(b"x")
            c = dict(in_s=0.0, out_s=80.0, steady=[dict(t0=40.0, t1=80.0)])
            fu = brollmatch.candidate_frames(c, fdir)
            self.assertTrue(fu)
            self.assertTrue(all(f["t"] >= 40.0 for f in fu), fu)
            c["steady"] = [dict(t0=79.0, t1=80.0)]
            self.assertLess(len(brollmatch.candidate_frames(c, fdir)), 2)   # nothing certifiable: verify reports unchecked

    def test_chooser_treats_an_ineligible_candidate_as_barred(self):
        pick = dict(id="b7p3")
        prop = dict(need="illustrate", candidates=[dict(shot="CAMB0182"), dict(shot="CAMB0135")])
        verdicts = {"CAMB0182": dict(status="ok", score=3), "CAMB0135": dict(status="ok", score=2)}
        row = brollmatch.choose(pick, prop, verdicts, {}, barred={"CAMB0182"})
        self.assertEqual(row["shot"], "CAMB0135")


if __name__ == "__main__":
    unittest.main()
