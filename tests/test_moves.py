"""Drone moves (moves.py): the per-pair measurement on synthetic frames with known motion, the segmentation and
labels on synthetic takes, and the catalogue rule. No footage, no models."""
from __future__ import annotations

import math
import unittest

import numpy as np

from videoeasy import moves, steadiness

FPS = moves.SAMPLE_FPS
PX = moves.WIDTH / 100 / FPS          # pixels per pair for 1 % of frame width per second


def texture(h=270, w=480, seed=0):
    import cv2
    rng = np.random.default_rng(seed)
    big = rng.integers(0, 255, size=(h // 6, w // 6)).astype(np.uint8)
    img = cv2.resize(big, (w, h), interpolation=cv2.INTER_CUBIC)
    return cv2.GaussianBlur(img, (3, 3), 0)


def warp(img, dx=0.0, dy=0.0, scale=1.0, deg=0.0):
    import cv2
    h, w = img.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), -deg, scale)   # cv2 angles are anticlockwise; ours clockwise
    M[0, 2] += dx
    M[1, 2] += dy
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


class PairMotion(unittest.TestCase):
    def setUp(self):
        self.img = texture()

    def test_shift_of_the_centre(self):
        r = moves.pair_motion(self.img, warp(self.img, dx=5.0, dy=-2.0))
        self.assertAlmostEqual(r[0], 5.0, delta=0.3)
        self.assertAlmostEqual(r[1], -2.0, delta=0.3)
        self.assertAlmostEqual(r[2], 0.0, delta=0.003)

    def test_zoom_is_not_read_as_a_pan(self):
        # the fit's translation is about the top-left corner: a zoom would read as a pan without the centre shift
        r = moves.pair_motion(self.img, warp(self.img, scale=1.02))
        self.assertAlmostEqual(r[2], math.log(1.02), delta=0.003)
        self.assertLess(abs(r[0]) + abs(r[1]), 0.5)

    def test_rotation_clockwise_on_screen(self):
        r = moves.pair_motion(self.img, warp(self.img, deg=1.0))
        self.assertAlmostEqual(r[3], math.radians(1.0), delta=0.003)
        self.assertLess(abs(r[0]) + abs(r[1]), 0.5)

    def test_orbit_holds_the_centre_while_the_edges_travel(self):
        cur = warp(self.img, dx=6.0)
        h, w = self.img.shape
        cur[h // 3:2 * h // 3, w // 3:2 * w // 3] = self.img[h // 3:2 * h // 3, w // 3:2 * w // 3]
        r = dict(zip(moves.COLS[1:], moves.pair_motion(self.img, cur)))
        self.assertLess(abs(r["vx_c"]), 1.0)
        self.assertAlmostEqual(r["vx_e"], 6.0, delta=0.5)

    def test_featureless_pair_is_unreliable(self):
        flat = np.full((270, 480), 128, np.uint8)
        self.assertEqual(moves.pair_motion(flat, flat)[4], 0)


def take(*parts, region=None):
    """Synthetic motion rows: parts are (seconds, lat, vert, zoom, rot) in % of frame width per second."""
    rows, t = [], 0.0
    for sec, lat, vert, zoom, rot in parts:
        for _ in range(int(round(sec * FPS))):
            t += 1 / FPS
            dx, dy = lat * PX, vert * PX
            reg = dict(vx_c=dx, vx_e=dx, vx_t=dx, vx_b=dx, vy_t=dy, vy_b=dy)
            reg.update((region or {}).get("fn", lambda dx, dy: {})(dx, dy))
            rows.append((t, dx, dy, zoom / 50 / FPS, math.radians(0) + rot / 50 / FPS, 200,
                         reg["vx_c"], reg["vx_e"], reg["vx_t"], reg["vx_b"], reg["vy_t"], reg["vy_b"]))
    return np.array(rows, dtype=float)


class Segments(unittest.TestCase):
    def test_pan_hover_push_in(self):
        segs = moves.segments(take((6, 3.0, 0, 0, 0), (4, 0, 0, 0, 0), (6, 0, 0, 3.0, 0)))
        clean = [s for s in segs if s["clean"]]
        self.assertEqual([s["kind"] for s in clean], ["pan", "hover", "push-in"])
        self.assertIn("pan left", clean[0]["label"])            # the scene moving right: the camera pans left
        self.assertGreater(clean[0]["t1"] - clean[0]["t0"], 4.5)
        self.assertGreater(clean[2]["t1"] - clean[2]["t0"], 4.5)
        for a, b in zip(segs, segs[1:]):
            self.assertAlmostEqual(a["t1"], b["t0"], places=2)   # segments tile the take

    def test_a_hesitating_pan_is_not_clean(self):  # noqa: D102
        parts = [(1.5, 8.0 if k % 2 else 2.0, 0, 0, 0) for k in range(6)]   # speed coming and going, one direction
        segs = moves.segments(take(*parts))
        self.assertTrue(segs)
        self.assertFalse(any(s["clean"] and s["t1"] - s["t0"] > 4 for s in segs))   # never one long clean move
        self.assertGreaterEqual(len([s for s in segs if s["kind"] == "pan"]), 3)        # cut at its slowdowns

    def test_an_eased_move_is_clean(self):
        # rise, cruise, settle over 4 s: the shape of a good move, which the first measure called a wobble
        ramp = [(0.5, v, 0, 0, 0) for v in (1.0, 2.0, 3.5, 5.0)] + [(1.0, 5.0, 0, 0, 0)] + [(0.5, v, 0, 0, 0) for v in (3.5, 2.0, 1.0)]
        segs = moves.segments(take((2, 0, 0, 0, 0), *ramp, (3, 0, 0, 0, 0)))
        pans = [s for s in segs if s["kind"] == "pan"]
        self.assertEqual(len(pans), 1)
        self.assertLess(pans[0]["dip"], 0.05)
        self.assertTrue(pans[0]["clean"], pans[0]["why"])

    def test_a_long_move_with_one_slowdown_becomes_two_clean_moves(self):
        # DJI_1235: a 62 s pull-out with one slowdown was rejected whole by the first dip rule
        segs = moves.segments(take((2, 0, 0, 0, 0), (10, 0, 0, -5.0, 0), (1.5, 0, 0, -1.2, 0), (10, 0, 0, -5.0, 0), (2, 0, 0, 0, 0)))
        pulls = [s for s in segs if s["kind"] == "pull-out"]
        self.assertEqual(len(pulls), 2, [(s["t0"], s["t1"], s["why"]) for s in segs])
        self.assertTrue(all(s["clean"] for s in pulls), [s["why"] for s in pulls])
        self.assertGreater(sum(s["t1"] - s["t0"] for s in pulls), 19.0)

    def test_dip_is_zero_for_one_peak_and_measures_a_hesitation(self):
        self.assertEqual(moves.dip(np.array([1, 3, 5, 5, 4, 2, 1.0])), 0.0)
        self.assertAlmostEqual(moves.dip(np.array([1, 5, 2, 5, 1.0])), 0.6)

    def test_a_reversal_splits_the_move(self):
        segs = moves.segments(take((5, 4.0, 0, 0, 0), (5, -4.0, 0, 0, 0)))
        clean = [s for s in segs if s["clean"]]
        self.assertEqual([s["label"].split()[-1] for s in clean], ["left", "right"])
        self.assertTrue(all(s["t1"] - s["t0"] < 5.5 for s in segs))

    def test_orbit_and_rise_labels(self):
        orbit = take((6, 4.0, 0, 0, 0), region={"fn": lambda dx, dy: dict(vx_c=0.1 * dx)})
        s = [x for x in moves.segments(orbit) if x["clean"]][0]
        self.assertEqual(s["kind"], "orbit")
        self.assertIn("orbit right", s["label"])                 # the background runs the way the camera travels
        rise = take((6, 0, 4.0, 0, 0), region={"fn": lambda dx, dy: dict(vy_t=dy, vy_b=2.5 * dy)})
        self.assertEqual([x for x in moves.segments(rise) if x["clean"]][0]["kind"], "rise")
        tilt = take((6, 0, -4.0, 0, 0))
        self.assertEqual([x for x in moves.segments(tilt) if x["clean"]][0]["label"], "steady tilt down")

    def test_a_shaky_stretch_is_not_clean(self):
        rows = take((8, 3.0, 0, 0, 0))
        rng = np.random.default_rng(1)
        rows[:, 1] += rng.normal(0, 12.0, len(rows))           # jitter on the path, the move underneath unchanged
        segs = moves.segments(rows)
        self.assertFalse(any(s["clean"] for s in segs))
        self.assertTrue(any("shake" in " ".join(s["why"]) for s in segs))


class CatalogueRule(unittest.TestCase):
    def test_drone_units_are_narrowed_to_their_clean_parts(self):
        cat = {"DJI_1239_w03": dict(in_s=120.0, out_s=160.0, steady=None),
               "DJI_1241_w01": dict(in_s=37.8, out_s=75.6, steady=[dict(t0=53.4, t1=66.9)]),
               "DJI_1240": dict(in_s=0.0, out_s=69.4, steady=None),
               "CAMB0150": dict(in_s=0.0, out_s=20.0, steady=None),
               "DJI_1229": dict(in_s=0.0, out_s=45.0, steady=None)}
        mv = {"DJI_1239_w03": dict(status="ok", clean=[dict(t0=122.0, t1=131.0, kind="pan", label="slow pan left"),
                                                       dict(t0=140.0, t1=141.5, kind="hover", label="hover")]),
              "DJI_1241_w01": dict(status="ok", clean=[dict(t0=50.0, t1=60.0, kind="orbit", label="steady orbit left")]),
              "DJI_1240": dict(status="ok", clean=[]),
              "CAMB0150": dict(status="ok", clean=[])}
        out = moves.apply_clean(cat, mv)
        self.assertEqual(cat["DJI_1239_w03"]["steady"], [dict(t0=122.0, t1=131.0, kind="pan", label="slow pan left")])
        self.assertEqual(cat["DJI_1241_w01"]["steady"][0]["t0"], 53.4)   # inside the steady run as well as the move
        self.assertEqual(cat["DJI_1241_w01"]["steady"][0]["t1"], 60.0)
        self.assertEqual(list(out), ["DJI_1240"])                          # no clean part: not cover
        self.assertIsNone(cat["CAMB0150"]["steady"])                       # not a drone unit
        self.assertIsNone(cat["DJI_1229"]["steady"])                       # unmeasured: the scorer measures the render
        self.assertEqual(steadiness.clamp_to_runs(120.0, 8.0, cat["DJI_1239_w03"]["steady"]), (122.0, 8.0))


    def test_builder_puts_a_drone_row_on_a_clean_move_and_names_it(self):
        from videoeasy import cutbuild
        cat = {"DJI_1239_w03": dict(in_s=120.0, out_s=160.0, duration_s=40.0, steady=None)}
        moves.apply_clean(cat, {"DJI_1239_w03": dict(status="ok", clean=[dict(t0=126.0, t1=137.0, kind="orbit", label="slow orbit left")])})
        row = cutbuild.shot_row(cat, "DJI_1239_w03", 8.0, [1], "ok", used_in={})
        self.assertEqual((row["in_s"], row["out_s"]), (126.0, 134.0))   # not the head of the clip, where the pilot settles
        self.assertIn("clean drone move: slow orbit left", row["note"])


    def test_fresh_footage_beats_a_longer_repeat(self):
        from videoeasy import cutbuild
        cat = {"DJI_1234_s01": dict(in_s=83.25, out_s=131.9, duration_s=48.7, steady=None)}
        moves.apply_clean(cat, {"DJI_1234_s01": dict(status="ok", clean=[
            dict(t0=98.93, t1=104.6, kind="pull-out", label="fast pull-out"),
            dict(t0=110.36, t1=114.2, kind="push-in", label="steady push-in")])})
        used = {}
        first = cutbuild.shot_row(cat, "DJI_1234_s01", 8.5, [1], "", used_in=used)
        again = cutbuild.shot_row(cat, "DJI_1234_s01", 6.1, [1], "", used_in=used,
                                  frames_used=[dict(t=98.93)])
        self.assertEqual(first["in_s"], 98.93)
        self.assertEqual(again["in_s"], 110.36)                 # the next clean move, not the same pull-out again
        self.assertNotIn("repeats footage", again["note"])

class RenderCheck(unittest.TestCase):
    def test_a_drone_stretch_across_a_turn_is_listed(self):
        from unittest import mock
        from videoeasy import cuteval
        rows = take((6, 4.0, 0, 0, 0), (6, -4.0, 0, 0, 0))            # a pan that reverses halfway through the stretch
        rows[:, 0] += 100.0
        steady_rows = rows[:, [0, 1, 2, 5]]
        segs = [dict(kind="video", clip="DJI_1239_w03", src_in=130.0, t0=100.0, t1=112.0, dur=12.0, pick="b1p1", beat=1),
                dict(kind="video", clip="DJI_1232_w01", src_in=40.0, t0=100.0, t1=105.5, dur=5.5, pick="b1p1", beat=1)]
        with mock.patch.object(moves, "cached_motion", return_value=rows), \
                mock.patch.object(steadiness, "cached_path", return_value=steady_rows):
            out = cuteval.cover_rules(segs, {}, "render.mp4", "cache", 2.0)
        self.assertEqual([s["clip"] for s in out["drone_unclean"]], ["DJI_1239_w03"])
        self.assertEqual([s["clip"] for s in out["drone_clean"]], ["DJI_1232_w01"])   # 5.5 s on the first pan alone
        self.assertLess(out["drone_unclean"][0]["move"]["share"], 0.9)

if __name__ == "__main__":
    unittest.main()
