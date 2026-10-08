"""Facial register moments: tracks, cue runs, sound events, reading validation and fusion. No footage, no models."""
from __future__ import annotations

import unittest

from videoeasy import faces, film, register, registerpage, sounds

FPS = faces.SAMPLE_FPS
QUIET = {c: 0.05 for c in faces.CUES}


def det(t, x=100.0, y=100.0, size=200.0, yaw=0.0, cues=None, px=None):
    d = dict(t=round(t, 3), box=(x, y, size, size), px=px if px is not None else size, det=0.9, pose=[yaw, 0.0, 0.0], sharp=50.0,
             cues=dict(QUIET, **(cues or {})) if cues is not False else None)
    d["why"] = faces.unreadable_why(d)
    return d


def track(cue_at, seconds=10.0, size=200.0, yaw=0.0):
    """One readable track; cue_at(t) -> {cue: value} overrides the quiet face."""
    dets = [det(i / FPS, size=size, yaw=yaw, cues=cue_at(i / FPS)) for i in range(int(seconds * FPS))]
    return faces.track_record("t00", dets, FPS)


def rec_of(*tracks, seconds=10.0):
    return dict(fps=FPS, duration_s=seconds, path="/footage/clip.mp4", frame_w=1080, frame_h=1920, tracks=list(tracks))


VOCAB = register.vocabulary(film.from_dict({}))


class Tracks(unittest.TestCase):
    def test_two_faces_side_by_side_stay_two_tracks(self):
        dets = []
        for i in range(20):
            dets += [det(i / FPS, x=100 + i), det(i / FPS, x=600 - i)]
        trs = faces.link(dets)
        self.assertEqual(len(trs), 2)
        self.assertTrue(all(len(tr) == 20 for tr in trs))

    def test_a_short_absence_rejoins_a_long_one_splits(self):
        short = [det(i / FPS) for i in range(10)] + [det(i / FPS) for i in range(14, 24)]   # 0.8 s gap
        self.assertEqual(len(faces.link(short)), 1)
        long_ = [det(i / FPS) for i in range(10)] + [det(i / FPS) for i in range(17, 27)]   # 1.4 s gap
        self.assertEqual(len(faces.link(long_)), 2)

    def test_a_moving_camera_keeps_the_track_by_centre(self):
        dets = [det(i / FPS, x=100 + 90 * i) for i in range(10)]   # 90 px a frame on a 200 px face: no overlap at IoU 0.3
        self.assertEqual(len(faces.link(dets)), 1)

    def test_unreadable_is_named_never_neutral(self):
        self.assertEqual(det(0, size=40).get("why"), "small")
        self.assertEqual(det(0, yaw=60).get("why"), "side-on")
        self.assertEqual(det(0, cues=False).get("why"), "no landmarks")
        self.assertIsNone(det(0).get("why"))

    def test_baseline_needs_enough_readable_time(self):
        self.assertIsNotNone(track(lambda t: {}, seconds=4)["baseline"])
        self.assertIsNone(track(lambda t: {}, seconds=2)["baseline"])


class Moments(unittest.TestCase):
    def moments(self, cue_at, seconds=10.0, events=None):
        tr = track(cue_at, seconds)
        pop = register.population_baseline([tr])
        return register.source_moments("clip", rec_of(tr, seconds=seconds), dict(events=events or []), pop, VOCAB)

    def test_a_held_smile_is_a_face_moment(self):
        ms = [m for m in self.moments(lambda t: {"smile": 0.8} if 4 <= t <= 6 else {}) if m["kind"] == "face"]
        self.assertEqual(len(ms), 1)
        self.assertEqual([c["cue"] for c in ms[0]["cues"]], ["smile"])
        self.assertLessEqual(ms[0]["t0"], 4.0)
        self.assertGreaterEqual(ms[0]["t1"], 6.0)

    def test_a_resting_frown_is_the_baseline_not_a_moment(self):
        ms = [m for m in self.moments(lambda t: {"frown": 0.45}) if m["kind"] == "face"]
        self.assertEqual(ms, [])

    def test_a_blink_is_not_a_closure_a_long_closure_is(self):
        blink = [m for m in self.moments(lambda t: {"eyes_closed": 0.9} if 3 <= t < 4 else {}) if m["kind"] == "face"]
        self.assertEqual(blink, [])
        closed = [m for m in self.moments(lambda t: {"eyes_closed": 0.9} if 3 <= t <= 6 else {}) if m["kind"] == "face"]
        self.assertEqual([c["cue"] for c in closed[0]["cues"]], ["eyes_closed"])

    def test_moving_lips_are_measured(self):
        talk = [m for m in self.moments(lambda t: {"smile": 0.8, "jaw_open": 0.1 + 0.4 * (int(t * FPS) % 2)} if 3 <= t <= 7 else {})
                if m["kind"] == "face"]
        self.assertTrue(talk[0]["lips_moving"])
        still = [m for m in self.moments(lambda t: {"smile": 0.8} if 3 <= t <= 7 else {}) if m["kind"] == "face"]
        self.assertFalse(still[0]["lips_moving"])

    def test_a_laugh_heard_with_no_face_moment_is_its_own_moment(self):
        ev = [dict(family="laughter", t0=8.0, t1=9.0, peak=0.6, classes=["Laughter"])]
        ms = self.moments(lambda t: {}, events=ev)
        heard = [m for m in ms if m["kind"] == "heard"]
        self.assertEqual(len(heard), 1)
        self.assertIsNone(heard[0]["track"])

    def test_moments_are_deterministic(self):
        f = lambda t: {"smile": 0.8} if 4 <= t <= 6 else {}  # noqa: E731
        self.assertEqual([m["id"] for m in self.moments(f, 30)], [m["id"] for m in self.moments(f, 30)])


class Sounds(unittest.TestCase):
    def test_consecutive_windows_join_and_quiet_ones_do_not_count(self):
        ws = [[i * 0.96, {"Laughter": 0.5 if 2 <= i <= 4 else 0.05, "Speech": 0.6}] for i in range(10)]
        ev = sounds.events(ws)
        self.assertEqual(len(ev), 1)
        self.assertEqual(ev[0]["family"], "laughter")
        self.assertAlmostEqual(ev[0]["t0"], 1.92, places=2)

    def test_speech_and_music_are_context_not_events(self):
        ws = [[i * 0.96, {"Speech": 0.9, "Music": 0.9}] for i in range(5)]
        self.assertEqual(sounds.events(ws), [])


class Readings(unittest.TestCase):
    def test_schema(self):
        ok = register.validate_reading(dict(register="grief", strength=2, visible="x", tears="yes", speaking="nope"), VOCAB)
        self.assertEqual(ok["status"], "ok")
        self.assertEqual(ok["speaking"], "cannot tell")
        for bad in (dict(register="melancholy", strength=2), dict(register="joy", strength=0), dict(register="none", strength=3),
                    dict(register="joy", strength="lots"), dict(register="joy", strength=5), ["joy"]):
            self.assertEqual(register.validate_reading(bad, VOCAB)["status"], "invalid", bad)

    def moment(self, cues=(), heard=(), lips=False):
        return dict(cues=[dict(cue=c, peak=0.8, baseline=0.1) for c in cues], heard=[dict(family=h) for h in heard], lips_moving=lips)

    def reading(self, reg, strength=2, **kw):
        return dict(dict(status="ok", register=reg, strength=strength, visible="", tears="no", speaking="no", same_face="yes"), **kw)

    def test_fusion(self):
        f = register.fuse(self.moment(["smile"]), self.reading("laughter"), VOCAB)
        self.assertEqual((f["status"], f["label"], f["support"]), ("agreed", "laughter", ["smile"]))
        f = register.fuse(self.moment([], ["crying"]), self.reading("grief"), VOCAB)
        self.assertEqual((f["status"], f["support"]), ("agreed", ["heard:crying"]))
        self.assertEqual(register.fuse(self.moment(["smile"]), self.reading("grief"), VOCAB)["status"], "seen_only")
        self.assertIsNone(register.fuse(self.moment(["smile"]), self.reading("grief"), VOCAB)["label"])
        self.assertEqual(register.fuse(self.moment(["brow_down"]), self.reading("none", 0), VOCAB)["status"], "measured_only")
        self.assertEqual(register.fuse(self.moment([]), self.reading("none", 0), VOCAB)["status"], "none")
        self.assertEqual(register.fuse(self.moment(["smile"]), self.reading("joy", same_face="no"), VOCAB)["status"], "unreadable")
        self.assertEqual(register.fuse(self.moment(["smile"]), None, VOCAB)["status"], "unchecked")
        self.assertEqual(register.fuse(self.moment(["smile"]), dict(status="invalid"), VOCAB)["label"], None)

    def test_speaking_from_either_modality(self):
        self.assertTrue(register.fuse(self.moment(["smile"], lips=True), self.reading("joy"), VOCAB)["speaking"])
        self.assertTrue(register.fuse(self.moment(["smile"]), self.reading("joy", speaking="yes"), VOCAB)["speaking"])

    def test_parroting_counts_definition_copies(self):
        rs = [dict(status="ok", visible="mouth open in a smile, cheeks raised"), dict(status="ok", visible="looks down at his hands")]
        self.assertEqual(register.parroting(rs, VOCAB), 0.5)


class Vocabulary(unittest.TestCase):
    def test_film_profile_replaces_the_default(self):
        p = film.from_dict({"face_registers": {"awe": {"definition": "eyes wide, mouth open, looking up", "cues": ["eye_wide"]},
                                                "grief": "crying"}})
        v = register.vocabulary(p)
        self.assertEqual(set(v), {"awe", "grief"})
        self.assertEqual(v["awe"]["cues"], {"eye_wide"})
        self.assertIn("frown", v["grief"]["cues"])   # a known term keeps its default support

    def test_bad_profiles_are_refused(self):
        with self.assertRaises(ValueError):
            film.from_dict({"face_registers": {"awe": {"cues": ["eye_wide"]}}})
        with self.assertRaises(ValueError):
            register.vocabulary(film.from_dict({"face_registers": {"awe": {"definition": "x", "cues": ["tail_wag"]}}}))
        with self.assertRaises(ValueError):   # a gloss-led definition makes the term unreachable (lessons 2.12)
            register.vocabulary(film.from_dict({"face_registers": {"effort": "straining: teeth gritted"}}))


class Sample(unittest.TestCase):
    def test_stratified_sample_is_stable_and_spans_strata(self):
        ms = []
        for i in range(40):
            st = ["agreed", "seen_only", "none", "measured_only"][i % 4]
            ms.append(dict(id=f"m{i}", kind="control" if i % 5 == 0 else "face", fused=dict(status=st), reading=dict(register="joy")))
        a, b = register.stratified(ms, 12), register.stratified(ms, 12)
        self.assertEqual([m["id"] for m in a], [m["id"] for m in b])
        self.assertEqual({m["fused"]["status"] for m in a}, {"agreed", "seen_only", "none", "measured_only"})
        self.assertTrue(any(m["kind"] == "control" for m in a))


class Find(unittest.TestCase):
    def test_query_by_register_strength_and_silence(self):
        def m(i, status, label, seen, strength, speaking=False):
            return dict(id=f"m{i}", source="clip", t0=float(i), fused=dict(status=status, label=label, speaking=speaking),
                        reading=dict(register=seen, strength=strength))
        doc = dict(moments=[m(1, "agreed", "grief", "grief", 3), m(2, "agreed", "grief", "grief", 1),
                            m(3, "agreed", "grief", "grief", 3, speaking=True), m(4, "seen_only", None, "grief", 2),
                            m(5, "agreed", "joy", "joy", 3)])
        self.assertEqual([x["id"] for x in register.find(doc, "grief", min_strength=2, silent=True)], ["m1"])
        self.assertEqual([x["id"] for x in register.find(doc, "grief", ("agreed", "seen_only"), 2)], ["m1", "m3", "m4"])


class Showcase(unittest.TestCase):
    def test_confident_examples_are_silent_strong_visible_and_one_per_clip(self):
        def m(i, src, strength=3, speaking=False, px=200.0, label="joy"):
            return dict(id=f"m{i}", source=src, track="t00", px=px, baseline="own", cues=[dict(cue="smile", peak=0.9, baseline=0.1)],
                        heard=[], fused=dict(status="agreed", label=label, speaking=speaking, support=["smile"]),
                        reading=dict(register=label, strength=strength))
        ms = [m(1, "a"), m(2, "a"), m(3, "b", strength=1), m(4, "c", speaking=True), m(5, "d", px=40.0), m(6, "e", strength=2)]
        doc = dict(vocabulary={"joy": "smiling"}, moments=ms)
        got = registerpage.select(doc, per=6)["by_register"]["joy"]
        self.assertEqual([x["id"] for x in got], ["m1", "m6"])


class Calibration(unittest.TestCase):
    def test_review_page_is_blind(self):
        page = register.review_html([dict(id="clip:t00:4.0", n=1, clip="media/a.mp4", strip=None, where="clip 00:04.0", track=True)],
                                    ["grief", "joy"], {"grief": "crying", "joy": "smiling"}, "20261004")
        self.assertIn("clip:t00:4.0", page)
        for leak in ('"fused"', '"reading"', '"strength"', '"visible"', '"register":', "agreed", "seen_only"):
            self.assertNotIn(leak, page)

    def test_precision_and_recall_per_term(self):
        import json
        import tempfile
        from pathlib import Path
        from types import SimpleNamespace

        def m(i, label, seen, kind="face"):
            return dict(id=f"m{i}", kind=kind, fused=dict(label=label, status="agreed" if label else "seen_only"),
                        reading=dict(register=seen))
        moments = [m(1, "grief", "grief"), m(2, "grief", "grief"), m(3, None, "grief"), m(4, None, "none", "control")]
        labels = {"m1": {"labels": ["grief"]}, "m2": {"labels": ["sadness"]}, "m3": {"labels": ["grief"]},
                  "m4": {"labels": ["joy"]}}
        with tempfile.TemporaryDirectory() as d:
            out = Path(d)
            (out / "register.json").write_text(json.dumps(dict(created="x", vocabulary={"grief": "", "sadness": "", "joy": ""},
                                                               moments=moments)))
            (out / "labels.json").write_text(json.dumps(dict(labels=labels)))
            text = register.calibrate(SimpleNamespace(out_dir=out), out / "labels.json")
        self.assertIn("| grief | 2 | 1 / 2 (50%) | 2 / 3 (67%) | 50% | 100% |", text)
        self.assertIn("1 showed a register to the editor", text)


if __name__ == "__main__":
    unittest.main()
