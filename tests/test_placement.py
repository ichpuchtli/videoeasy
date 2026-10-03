"""Milestone two: every picture row carries its destination, the scorer splits
at pick boundaries, the lay-in plans from destinations, and conform compares
planned rows with read-back items instead of totals. No ffmpeg, no Resolve."""
from __future__ import annotations

import unittest

from videoeasy import cutbuild, cuteval, evalrun, layin, sources
from videoeasy.film import load_film

from filmfixture import TEST_FILM

OK = evalrun.OK
CAT = {"LONG": dict(in_s=10.0, out_s=60.0, duration_s=50.0), "TEN": dict(in_s=100.0, out_s=110.0, duration_s=10.0)}


def _cut(video_rows, picks=((8.0, "CAMA0124", 100.0), (12.0, "CAMA0124", 200.0)), gap=0.5):
    audio = [dict(clip=c, in_s=i, out_s=i + d, duration_s=d, text=f"words of pick {n}", tier=1, source_path=None)
             for n, (d, c, i) in enumerate(picks, 1)]
    return {"version": 9, "film": "t", "beats": [dict(title="beat", notes="", gap_after_s=gap, audio=audio, video=video_rows)]}


class DestinationIntervals(unittest.TestCase):
    def test_builder_stamps_pick_and_destination_on_every_row(self):
        ok = {"status": OK, "score": 3, "usable": [1]}
        rows = [dict(pick="b1p1", text="w", duration_s=8.0, shot="LONG", need="illustrate", score=3, usable=[1], reason="",
                     alternatives=[], verdicts={"LONG": ok}, status=OK),
                dict(pick="b1p2", text="w", duration_s=12.0, shot=None, need="bare", score=None, verdicts={}, status=OK)]
        out = cutbuild.build_beat_video({"video": []}, rows, CAT, 12.0, {}, t0=100.0)
        self.assertEqual([(r["pick"], r["dest_in_s"], r["dest_out_s"]) for r in out], [("b1p1", 100.0, 108.0), ("b1p2", 108.0, 120.0)])

    def test_merged_row_lists_both_picks_and_spans_both(self):
        ok = {"status": OK, "score": 3, "usable": [1]}
        rows = [dict(pick="b1p1", text="w", duration_s=5.0, shot="LONG", score=3, usable=[1], reason="", alternatives=[], verdicts={"LONG": ok}, status=OK),
                dict(pick="b1p2", text="w", duration_s=5.0, shot="LONG", score=3, usable=[1], reason="", alternatives=[], verdicts={"LONG": ok}, status=OK)]
        out = cutbuild.build_beat_video({"video": []}, rows, CAT, 12.0, {}, t0=0.0)
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["picks"], ["b1p1", "b1p2"])
        self.assertEqual((out[0]["dest_in_s"], out[0]["dest_out_s"]), (0.0, 10.0))

    def test_check_cut_accepts_contiguous_rows_and_rejects_gaps_and_legacy(self):
        good = _cut([dict(type="video", clip="LONG", in_s=10.0, out_s=18.0, duration_s=8.0, dest_in_s=0.0, dest_out_s=8.0),
                     dict(type="bare", clip=None, in_s=0.0, out_s=12.0, duration_s=12.0, dest_in_s=8.0, dest_out_s=20.0)])
        self.assertEqual(cutbuild.check_cut(good), [])
        short = _cut([dict(type="video", clip="LONG", in_s=10.0, out_s=18.0, duration_s=8.0, dest_in_s=0.0, dest_out_s=8.0)])
        self.assertTrue(any("video ends 8.00, picks end 20.00" in p for p in cutbuild.check_cut(short)))
        early = _cut([dict(type="video", clip="LONG", in_s=10.0, out_s=18.0, duration_s=8.0, dest_in_s=0.0, dest_out_s=8.0),
                      dict(type="video", clip="TEN", in_s=100.0, out_s=112.0, duration_s=12.0, dest_in_s=6.0, dest_out_s=18.0),
                      dict(type="bare", clip=None, in_s=0.0, out_s=2.0, duration_s=2.0, dest_in_s=18.0, dest_out_s=20.0)])
        self.assertTrue(any("starts 6.0 expected 8.00" in p for p in cutbuild.check_cut(early)))
        legacy = _cut([dict(type="video", clip="LONG", in_s=10.0, out_s=30.0, duration_s=20.0)])
        self.assertTrue(any("no destination interval" in p for p in cutbuild.check_cut(legacy)))


class ScorerSplitsAtPicks(unittest.TestCase):
    def test_one_row_over_two_picks_becomes_two_stretches(self):
        cut = _cut([dict(type="video", clip="LONG", in_s=10.0, out_s=30.0, duration_s=20.0, dest_in_s=0.0, dest_out_s=20.0)])
        segs = cuteval.layout(cut, transcripts={})
        self.assertEqual([(s["pick"], s["t0"], s["t1"], s["src_in"]) for s in segs],
                         [("b1p1", 0.0, 8.0, 10.0), ("b1p2", 8.0, 20.0, 18.0)])
        self.assertTrue(all(s["kind"] == "video" for s in segs))

    def test_legacy_rows_are_laid_consecutively_then_split(self):
        cut = _cut([dict(type="video", clip="LONG", in_s=10.0, out_s=30.0, duration_s=20.0)])
        segs = cuteval.layout(cut, transcripts={})
        self.assertEqual([(s["pick"], s["t0"], s["t1"]) for s in segs], [("b1p1", 0.0, 8.0), ("b1p2", 8.0, 20.0)])

    def test_bare_under_presenter_pick_is_to_camera_by_source_role(self):
        cut = _cut([dict(type="bare", clip=None, in_s=0.0, out_s=20.0, duration_s=20.0, dest_in_s=0.0, dest_out_s=20.0)],
                   picks=((8.0, "CAMB0184", 2.0), (12.0, "CAMA0124", 200.0)))
        segs = cuteval.layout(cut, transcripts={}, profile=TEST_FILM)
        self.assertEqual([s["to_camera"] for s in segs], [True, False])

    def test_pick_spans_and_layout_agree(self):
        cut = _cut([dict(type="bare", clip=None, in_s=0.0, out_s=20.0, duration_s=20.0, dest_in_s=0.0, dest_out_s=20.0)])
        spans = {s["id"]: (s["t0"], s["t1"]) for s in cuteval.pick_spans(cut)}
        for s in cuteval.layout(cut, {}):
            self.assertEqual((s["t0"], s["t1"]), spans[s["pick"]])


class SourceRoles(unittest.TestCase):
    def test_roles(self):
        f = TEST_FILM
        self.assertEqual(sources.role("CAMA0124_w03", f), "sitdown")
        self.assertEqual(sources.role("CAMB0220", f), "presenter")
        self.assertEqual(sources.role("CAMB0184", f), "explainer")
        self.assertEqual(sources.role("CAMA0129_w04", f), "walk")
        self.assertTrue(sources.is_drone("DJI_1230_s01", f))
        self.assertTrue(sources.to_camera("CAMB0184", f))
        self.assertFalse(sources.to_camera("CAMA0124", f))
        self.assertTrue(sources.never_cover("CAMB0221", f))
        self.assertFalse(sources.never_cover("CAMB0184", f))
        self.assertTrue(sources.sync_walk("CAMA0131_w03", f))
        self.assertEqual(sources.drone_prefixes(f), ("DJI_",))

    def test_generic_profile_without_film_yaml(self):
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            g = load_film(d)
        self.assertEqual(sources.role("ANYCLIP01", g), "walk")
        self.assertTrue(sources.is_drone("DJI_1230", g))
        self.assertEqual(g.context, "")
        with self.assertRaises(KeyError):
            sources.placeholder_file("PLACEHOLDER: anything", g)

    def test_profile_is_read_from_film_yaml_and_validated(self):
        import pathlib, tempfile
        from videoeasy.film import film_dir_for
        with tempfile.TemporaryDirectory() as d:
            root = pathlib.Path(d)
            (root / "editorial").mkdir()
            (root / "film.yaml").write_text("name: t\nroles:\n  a: {label: A, to_camera: true}\ndefault_role: a\nprefixes: []\n"
                                            "layin: {render_dir: editorial/renders}\n")
            f = load_film(root)
            self.assertEqual(film_dir_for(root / "editorial" / "cut-v1.json"), root.resolve())
            self.assertTrue(sources.to_camera("X0001", f))
            self.assertEqual(f.layin["render_dir"], str(root.resolve() / "editorial/renders"))
            (root / "film.yaml").write_text("roles:\n  a: {label: A}\ndefault_role: b\n")
            with self.assertRaises(ValueError):
                load_film(root)

    def test_placeholder_keywords_are_checked_in_listed_order(self):
        # the trail-cam card's text names a sign too: 'trail' is listed first, so it wins
        self.assertEqual(sources.placeholder_file("PLACEHOLDER: trail-cam clips: fox, sign", TEST_FILM), "ph04_trail_cam.mov")
        self.assertEqual(sources.placeholder_file("PLACEHOLDER: painted sign", TEST_FILM), "ph03_signs.mov")


def _measure(path, a, b, mono):
    return (-30.0, -6.0)


def _stereo(clip, path):
    return False


class Leveler(unittest.TestCase):
    def test_leveler_covers_only_the_shortfall_past_the_cap(self):
        from videoeasy.layin import leveler_for
        self.assertEqual(leveler_for(-50.0), 0.0)     # 26 dB needed: clip gain suffices
        self.assertEqual(leveler_for(-57.5), 3.5)     # 33.5 needed: 3.5 past the cap
        self.assertEqual(leveler_for(-61.7), 6.0)     # 37.7 needed: leveler maxes at 6
        self.assertEqual(leveler_for(None), 0.0)

    def test_conform_reports_a_missing_leveler(self):
        from videoeasy.layin import conform
        pl = {"audio": [dict(pick="b1p1", name="CAMB0184.MOV", track=1, head=0, tail=0, rec=0, dur=100, sf=10, ef=110, gain=30.0,
                             mono=False, fade_in=0, fade_out=0, leveler=5.0, gap_after=0)], "video": [], "markers": []}
        item = dict(type="audio", track=1, name="CAMB0184.MOV", rec=0, dur=100, src_in=10, src_start=10, src_end=110,
                    fades={"FadeIn": 0, "FadeOut": 0}, gain=30.0, mono_left=False, leveler=0.0)
        vitem = dict(type="video", track=1, name="CAMB0184.MOV", rec=0, dur=100, src_in=10, src_start=10, src_end=110)
        cf = conform(pl, dict(fps=24.0, items=[item, vitem], markers={}))
        self.assertEqual([d["kind"] for d in cf["diffs"]], ["leveler"])
        item["leveler"] = 5.0
        cf = conform(pl, dict(fps=24.0, items=[item, vitem], markers={}))
        self.assertEqual(cf["diffs"], [])


class LayPlan(unittest.TestCase):
    def _plan(self, transcripts=None, video=None, picks=None):
        video = video if video is not None else [dict(type="video", clip="DJI_1230", in_s=4.0, out_s=12.0, duration_s=8.0, dest_in_s=0.0, dest_out_s=8.0),
                                                  dict(type="bare", clip=None, in_s=0.0, out_s=12.0, duration_s=12.0, dest_in_s=8.0, dest_out_s=20.0)]
        cut = _cut(video, picks=picks or ((8.0, "CAMA0124", 100.0), (12.0, "CAMA0124", 200.0)))
        return layin.plan(cut, transcripts or {}, measure=_measure, mono_of=_stereo)

    def test_gain_is_bounded_by_peak_ceiling_and_cap(self):
        pl = self._plan()
        self.assertEqual([r["gain"] for r in pl["audio"]], [4.5, 4.5])  # min(30, -24 - (-30) = 6, -1.5 - (-6) = 4.5)
        self.assertEqual(layin.gain_for(-60.0, -40.0), 30.0)
        self.assertIsNone(layin.gain_for(None, None))
        self.assertEqual(pl["unmeasured_gains"], [])

    def test_crossfade_extension_and_alternating_tracks(self):
        pl = self._plan()
        a, b = pl["audio"]
        self.assertEqual((a["track"], b["track"]), (1, 2))
        self.assertEqual(a["head"], 0)
        self.assertEqual(b["head"], 6)          # 3 s of room, no words in the (empty) transcript
        self.assertEqual(a["tail"], 6)          # want = 0 + 12 - 6
        self.assertEqual((a["fade_out"], b["fade_in"]), (12, 12))
        self.assertEqual(a["rec"], 0)
        self.assertEqual(b["rec"], 192)

    def test_measured_hard_in_takes_no_handle_and_no_fade_over_its_words(self):
        # a pick that starts right after a one-word reply the stored transcript does not hold
        cut = _cut([dict(type="bare", clip=None, in_s=0.0, out_s=20.0, duration_s=20.0, dest_in_s=0.0, dest_out_s=20.0)])
        cut["beats"][0]["audio"][1]["max_head_s"] = 0.0
        a, b = layin.plan(cut, {}, measure=_measure, mono_of=_stereo)["audio"]
        self.assertEqual(b["head"], 0)
        self.assertEqual(a["tail"], 12)          # the outgoing room tone carries the whole overlap
        self.assertEqual((a["fade_out"], b["fade_in"]), (12, 2))
        cut["beats"][0]["audio"][1]["max_head_s"] = 0.1   # measured room of 0.1 s: a 2-frame handle, the usual crossfade
        a, b = layin.plan(cut, {}, measure=_measure, mono_of=_stereo)["audio"]
        self.assertEqual(b["head"], 2)
        self.assertEqual(b["fade_in"], 12)
        self.assertEqual(layin.grade_trim_for([dict(clip="CAMA0138", in_s=240.0, out_s=252.0, cdl={})], "CAMA0138.MOV", 244.4, 251.4)["clip"], "CAMA0138")
        self.assertIsNone(layin.grade_trim_for([dict(clip="CAMA0138", in_s=240.0, out_s=252.0, cdl={})], "CAMA0138.MOV", 322.9, 333.3))
        self.assertIsNotNone(layin.grade_trim_for([dict(clip="CAMB0186", cdl={})], "CAMB0186.MOV", None, None))
        # the span an item shows comes from its plan row (media seconds), never from Resolve's source timecode
        pl = self._plan()
        v = pl["video"][0]
        self.assertEqual(layin.item_source_span(pl, v["name"], v["rec"], 3), (v["sf"] / v["fps"], v["ef"] / v["fps"]))
        r = pl["audio"][1]
        s0, s1 = layin.item_source_span(pl, r["name"], r["rec"] - r["head"], 1)
        self.assertAlmostEqual(s0, 200.0 - r["head"] / 24, places=3)
        self.assertEqual(layin.item_source_span(pl, "CAMA7299.MOV", 0, 1), (None, None))

    def test_word_caps_tail_and_incoming_head_grows(self):
        # a word at 208.05 s in the source sits right after pick 1 (200-208): its tail cannot extend
        tr = {"CAMA0124": {"segments": [{"words": [{"s": 108.05, "e": 108.5, "w": "next"}]}]}}
        pl = self._plan(transcripts=tr)
        a, b = pl["audio"]
        self.assertEqual(a["tail"], 0)
        self.assertEqual(b["head"], 12)         # incoming head absorbs the shortfall
        self.assertEqual(a["fade_out"], 12)

    def _two_beats(self, cover_to):
        tr = {"CAMA0124": {"segments": [{"words": [{"s": 108.05, "e": 108.5, "w": "a"}, {"s": 199.0, "e": 199.95, "w": "b"}]}]}}
        def beat(title, clip, in_s, d, t0, video):
            return dict(title=title, notes="", gap_after_s=0.5, video=video,
                        audio=[dict(clip=clip, in_s=in_s, out_s=in_s + d, duration_s=d, text="w", tier=1, source_path=None)])
        v1 = [dict(type="video", clip="DJI_1230", in_s=4.0, out_s=4.0 + cover_to, duration_s=cover_to, dest_in_s=0.0, dest_out_s=cover_to)]
        if cover_to < 8.0:
            v1.append(dict(type="bare", clip=None, in_s=0.0, out_s=8.0 - cover_to, duration_s=8.0 - cover_to, dest_in_s=cover_to, dest_out_s=8.0))
        v2 = [dict(type="bare", clip=None, in_s=0.0, out_s=12.0, duration_s=12.0, dest_in_s=8.5, dest_out_s=20.5)]
        cut = {"version": 9, "film": "t", "beats": [beat("one", "CAMA0124", 100.0, 8.0, 0.0, v1), beat("two", "CAMA0124", 200.0, 12.0, 8.5, v2)]}
        return layin.plan(cut, tr, measure=_measure, mono_of=_stereo)

    def test_residual_hole_gets_video_fill_unless_cover_spans_it(self):
        # words on both sides of the beat gap: neither take may extend, so the 12-frame gap is a hole
        pl = self._two_beats(cover_to=6.0)
        self.assertEqual(pl["residual_gaps"], [("b1p1", 192, 12)])
        self.assertEqual(pl["fills"], 1)
        fill = pl["video"][-1]
        self.assertEqual((fill["name"], fill["rec"], fill["dur"], fill["sf"]), ("CAMA0124.MOV", 192, 12, 192 + 2400))
        # cover that reaches the beat end is extended across the gap, so no fill is needed (and none would fit)
        pl = self._two_beats(cover_to=8.0)
        self.assertEqual(len(pl["residual_gaps"]), 1)
        self.assertEqual(pl["fills"], 0)
        self.assertEqual(pl["video"][0]["dur"], 192 + 12)

    def test_picks_rounded_a_frame_apart_leave_no_empty_frame(self):
        # 8.02 s rounds to 192 frames twice, but the third pick starts at round(16.04 * 24) = 385: one frame between
        # picks two and three. With words on both sides neither take may extend, so the hole must get picture
        # (v8 rendered a black frame at each such join, 29 Sep 2026)
        tr = {"CAMA0124": {"segments": [{"words": [{"s": 208.03, "e": 208.4, "w": "a"}, {"s": 299.5, "e": 299.99, "w": "b"}]}]}}
        cut = _cut([dict(type="bare", clip=None, in_s=0.0, out_s=20.04, duration_s=20.04, dest_in_s=0.0, dest_out_s=20.04)],
                   picks=((8.02, "CAMA0124", 100.0), (8.02, "CAMA0124", 200.0), (4.0, "CAMA0124", 300.0)))
        pl = layin.plan(cut, tr, measure=_measure, mono_of=_stereo)
        a1, a2, a3 = pl["audio"]
        self.assertEqual((a2["rec"] + a2["dur"], a3["rec"]), (384, 385))
        self.assertIn(("b1p2", 384, 1), pl["residual_gaps"])
        shown = set()
        for r in pl["audio"]:
            shown.update(range(r["rec"] - r["head"], r["rec"] + r["dur"] + r["tail"]))
        for v in pl["video"]:
            shown.update(range(v["rec"], v["rec"] + v["dur"]))
        self.assertEqual(sorted(set(range(a3["rec"] + a3["dur"])) - shown), [])

    def test_drone_source_frames_use_its_own_rate(self):
        pl = self._plan()
        v = pl["video"][0]
        self.assertEqual(v["name"], "DJI_1230.MOV")
        self.assertEqual(v["dur"], 192)                 # 8 s at 24 fps
        self.assertEqual(v["ef"] - v["sf"], 192)        # 8 s at 23.976 rounds to the same count here

    def test_legacy_cut_without_destinations_is_refused(self):
        with self.assertRaises(ValueError):
            self._plan(video=[dict(type="video", clip="DJI_1230", in_s=4.0, out_s=12.0, duration_s=8.0)])


class Conform(unittest.TestCase):
    def _pl(self):
        return layin.plan(_cut([dict(type="video", clip="DJI_1230", in_s=4.0, out_s=12.0, duration_s=8.0, dest_in_s=0.0, dest_out_s=8.0),
                                dict(type="bare", clip=None, in_s=0.0, out_s=12.0, duration_s=12.0, dest_in_s=8.0, dest_out_s=20.0)]),
                          {}, measure=_measure, mono_of=_stereo)

    def _items_from(self, pl, drift=0):
        items = []
        for r in pl["audio"]:
            for typ in ("audio", "video"):
                it = dict(type=typ, track=r["track"], name=r["name"], rec=r["rec"] - r["head"] + drift, dur=r["dur"] + r["head"] + r["tail"],
                          src_in=r["sf"] - r["head"], src_start=0, src_end=0)
                if typ == "audio":
                    it.update(fades={"FadeIn": r["fade_in"], "FadeOut": r["fade_out"]}, gain=r["gain"], mono_left=r["mono"])
                items.append(it)
        for v in pl["video"]:
            items.append(dict(type="video", track=3, name=v["name"], rec=v["rec"] + drift, dur=v["dur"], src_in=v["sf"], src_start=0, src_end=0))
        return items

    def test_exact_readback_conforms(self):
        pl = self._pl()
        cf = layin.conform(pl, {"items": self._items_from(pl)})
        self.assertEqual(cf["status"], OK)
        self.assertEqual(cf["matched"], cf["planned"])

    def test_one_frame_drift_is_within_tolerance_three_is_not(self):
        pl = self._pl()
        self.assertEqual(layin.conform(pl, {"items": self._items_from(pl, drift=1)})["status"], OK)
        cf = layin.conform(pl, {"items": self._items_from(pl, drift=3)})
        self.assertEqual(cf["status"], "mismatch")
        self.assertTrue(all(d["kind"] == "missing" for d in cf["diffs"]))

    def test_missing_cover_and_wrong_gain_are_named_not_inferred(self):
        pl = self._pl()
        items = self._items_from(pl)
        items = [it for it in items if it["track"] != 3]
        items[0]["gain"] = 0.0
        cf = layin.conform(pl, {"items": items})
        kinds = sorted(d["kind"] for d in cf["diffs"])
        self.assertEqual(kinds, ["gain", "missing"])
        self.assertEqual(cf["diffs"][-1]["note"], "")

    def test_drone_in_point_is_compared_in_timeline_frames(self):
        pl = self._pl()
        items = self._items_from(pl)
        dji = next(it for it in items if it["name"].startswith("DJI"))
        v = pl["video"][0]
        dji["src_in"] = int(v["sf"] * 24 / layin.FPS_DJI)   # what Resolve reports for a 23.976 source
        self.assertEqual(layin.conform(pl, {"items": items})["status"], OK)
        dji["src_in"] += 5
        self.assertEqual(layin.conform(pl, {"items": items})["status"], "mismatch")

    def test_unexpected_item_is_listed(self):
        pl = self._pl()
        items = self._items_from(pl) + [dict(type="video", track=3, name="STRAY.MOV", rec=5, dur=10, src_in=0, src_start=0, src_end=0)]
        cf = layin.conform(pl, {"items": items})
        self.assertEqual([u["name"] for u in cf["unexpected"]], ["STRAY.MOV"])


if __name__ == "__main__":
    unittest.main()


class Channels(unittest.TestCase):
    PL = {"audio": [dict(pick="b1p1", rec=0, dur=240, mono=False), dict(pick="b1p2", rec=240, dur=240, mono=True), dict(pick="b1p3", rec=480, dur=5, mono=True)]}

    def test_mono_pick_on_one_channel_breaks_the_render(self):
        from videoeasy import layin
        rep = layin.verify_channels(self.PL, "r.mp4", measure=lambda a, b: (-26.9, -38.5))
        self.assertEqual(rep["status"], "broken")
        self.assertEqual([x["pick"] for x in rep["one_channel"]], ["b1p2"])
        self.assertEqual(rep["checked"], 1)   # the 5-frame pick is too short to integrate and is skipped
        ok = layin.verify_channels(self.PL, "r.mp4", measure=lambda a, b: (-26.9, -26.9))
        self.assertEqual(ok["status"], "ok")
        unc = layin.verify_channels(self.PL, "r.mp4", measure=lambda a, b: (None, None))
        self.assertEqual(unc["status"], "unchecked")

    def test_scorer_lists_one_channel_picks(self):
        from videoeasy import cuteval
        spans = [dict(id="b1p2", beat=1, clip="CAMB0220", t0=10.0, t1=20.0)]
        rows = cuteval.pick_loudness(None, spans, measure=lambda a, b: (-28.6, -1.5), channels=lambda a, b: (-26.9, -38.5))
        summ = cuteval.loudness_summary(rows, mono={"b1p2": True})
        self.assertEqual([x["pick"] for x in summ["one_channel"]], ["b1p2"])
        self.assertEqual(cuteval.loudness_summary(rows, mono={"b1p2": False})["one_channel"], [])   # a stereo sit-down pick differs L/R by design
        self.assertEqual(rows[0]["lr_diff"], 11.6)
