"""Milestone one of the eval loop: failed evidence never becomes acceptance,
caches are keyed on content, reports complete for every finding shape, and
the builder conserves each pick's timing.

Fixtures follow the regression list of the external eval review (25 Sep
2026; its findings are folded into docs/eval-design.md). No model, ffmpeg or
Resolve calls."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from videoeasy import brollmatch, cutbuild, cuteval, evalrun, storycheck


def _seg(i, t0, dur, kind="video", clip="X", vlm=None):
    s = dict(id=i, beat=1, title="b", kind=kind, clip=clip, src_in=0.0, t0=t0, t1=t0 + dur, dur=dur, words="w",
             frames=[], stats=None, exposure_flags=None)
    if vlm is not None:
        s["vlm"] = vlm
    return s


class Fingerprints(unittest.TestCase):
    def test_same_path_changed_content_changes_fingerprint(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "render.mp4"
            p.write_bytes(b"red" * 100)
            a = evalrun.fingerprint(p)
            p.write_bytes(b"blue" * 100)
            b = evalrun.fingerprint(p)
        self.assertNotEqual(a, b)
        self.assertEqual(len(a), 16)

    def test_frames_dir_is_keyed_on_render_hash(self):
        with tempfile.TemporaryDirectory() as td:
            d1 = cuteval.frames_dir_for(Path(td), "aaaa")
            d2 = cuteval.frames_dir_for(Path(td), "bbbb")
        self.assertNotEqual(d1, d2)
        self.assertTrue(str(d1).endswith("aaaa"))

    def test_transcript_cache_key_changes_with_render_and_spans(self):
        k1 = storycheck.transcript_cache_key("r1", [(1, 0.0, 10.0)])
        k2 = storycheck.transcript_cache_key("r2", [(1, 0.0, 10.0)])
        k3 = storycheck.transcript_cache_key("r1", [(1, 0.0, 11.0)])
        self.assertEqual(len({k1, k2, k3}), 3)

    def test_stale_transcript_cache_is_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            c = Path(td) / "t.json"
            c.write_text(json.dumps({"key": "old", "words": []}))
            self.assertFalse(storycheck.cache_valid(c, "new"))
            self.assertTrue(storycheck.cache_valid(c, "old"))
            c.write_text("not json")
            self.assertFalse(storycheck.cache_valid(c, "old"))

    def test_refuse_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "report.json"
            p.write_text("{}")
            with self.assertRaises(SystemExit):
                evalrun.refuse_overwrite([p], overwrite=False)
            evalrun.refuse_overwrite([p], overwrite=True)
            evalrun.refuse_overwrite([Path(td) / "absent.json"], overwrite=False)


class MeasurementsFailHonestly(unittest.TestCase):
    def test_joins_on_missing_render_is_unchecked_not_clean(self):
        j = cuteval.joins(Path("/nonexistent/render.mp4"))
        self.assertEqual(j["status"], evalrun.UNCHECKED)
        self.assertIsNone(j["black"])
        self.assertIsNone(j["silence"])
        self.assertTrue(j["errors"])

    def test_signalstats_on_missing_frame_raises(self):
        with self.assertRaises(RuntimeError):
            cuteval.signalstats(Path("/nonexistent/frame.jpg"))

    def test_exposure_flags_without_stats_is_none(self):
        self.assertIsNone(cuteval.exposure_flags(None))
        self.assertIsNone(cuteval.exposure_flags({}))
        self.assertEqual(cuteval.exposure_flags({"YAVG": 100, "YLOW": 20, "YHIGH": 200, "SATAVG": 40}), [])

    def test_ffprobe_duration_missing_file(self):
        self.assertIsNone(evalrun.ffprobe_duration("/nonexistent.mp4"))


class JudgementValidation(unittest.TestCase):
    def test_bare_stretch_scores_by_rule_not_by_relation(self):
        j = cuteval.validate_judgement({"relation": "supports", "score": 1, "technical": ["none"]}, by_rule=True)
        self.assertEqual(j["status"], "ok")
        self.assertEqual(j["score"], 1)
        j = cuteval.validate_judgement({"relation": "supports", "score": 1, "technical": ["none"]})
        self.assertEqual(j["status"], "invalid")
        j = cuteval.validate_judgement({"relation": "supports", "score": 9, "technical": ["none"]}, by_rule=True)
        self.assertEqual(j["status"], "invalid")

    def test_score_99_is_invalid(self):
        v = cuteval.validate_judgement({"relation": "illustrates", "score": 99, "technical": ["none"]})
        self.assertEqual(v["status"], evalrun.INVALID)
        self.assertIsNone(v["score"])

    def test_relation_and_score_must_agree(self):
        v = cuteval.validate_judgement({"relation": "generic", "score": 3, "technical": []})
        self.assertEqual(v["status"], evalrun.INVALID)
        self.assertIsNone(v["score"])

    def test_technical_string_is_coerced_and_unknown_kept_aside(self):
        v = cuteval.validate_judgement({"relation": "supports", "score": 2, "technical": "none"})
        self.assertEqual(v["status"], evalrun.OK)
        self.assertEqual(v["technical"], ["none"])
        v = cuteval.validate_judgement({"relation": "supports", "score": 2, "technical": ["grainy", "shaky"]})
        self.assertEqual(v["technical"], ["shaky"])
        self.assertEqual(v["technical_unknown"], ["grainy"])

    def test_good_judgement_is_ok(self):
        v = cuteval.validate_judgement({"relation": "Illustrates", "score": "3", "technical": []})
        self.assertEqual((v["status"], v["score"], v["relation"]), (evalrun.OK, 3, "illustrates"))

    def test_non_object_is_invalid(self):
        self.assertEqual(cuteval.validate_judgement("3")["status"], evalrun.INVALID)

    def test_verify_verdict_usable_must_name_sent_frames(self):
        v = brollmatch.validate_verdict({"score": 2, "usable": [1, 7]}, n_frames=3)
        self.assertEqual(v["status"], evalrun.OK)
        self.assertEqual(v["usable"], [1])
        v = brollmatch.validate_verdict({"score": 2, "usable": [9]}, n_frames=3)
        self.assertEqual(v["status"], evalrun.INVALID)
        v = brollmatch.validate_verdict({"score": "high"}, n_frames=3)
        self.assertEqual(v["status"], evalrun.INVALID)
        self.assertIsNone(v["score"])


class CandidateEligibility(unittest.TestCase):
    pick = {"id": "b1p1"}

    def test_error_verdict_is_not_eligible(self):
        prop = {"need": "illustrate", "candidates": [{"shot": "A"}]}
        row = brollmatch.choose(self.pick, prop, {"A": {"error": "simulated timeout", "status": evalrun.UNCHECKED}})
        self.assertEqual(row["status"], evalrun.UNCHECKED)
        self.assertIsNone(row["shot"])
        self.assertIsNone(row["score"])

    def test_unverified_candidate_is_unchecked_not_1_5(self):
        prop = {"need": "support", "candidates": [{"shot": "A"}, {"shot": "B"}]}
        row = brollmatch.choose(self.pick, prop, {})
        self.assertEqual(row["status"], evalrun.UNCHECKED)
        self.assertIsNone(row["shot"])

    def test_invalid_verdict_skipped_ok_verdict_chosen(self):
        prop = {"need": "support", "candidates": [{"shot": "A"}, {"shot": "B"}]}
        verdicts = {"A": {"status": evalrun.INVALID, "score": None}, "B": {"status": evalrun.OK, "score": 2, "usable": [1]}}
        row = brollmatch.choose(self.pick, prop, verdicts)
        self.assertEqual((row["status"], row["shot"], row["score"]), (evalrun.OK, "B", 2))
        self.assertEqual(row["unverified"], ["A"])

    def test_legacy_verdict_without_status_but_integer_score_is_ok(self):
        self.assertTrue(brollmatch.verdict_ok({"score": 3}))
        self.assertFalse(brollmatch.verdict_ok({"score": None}))
        self.assertFalse(brollmatch.verdict_ok({"error": "x"}))
        self.assertFalse(brollmatch.verdict_ok(None))

    def test_all_verified_zero_falls_to_bare_with_ok_status(self):
        prop = {"need": "support", "candidates": [{"shot": "A"}]}
        row = brollmatch.choose(self.pick, prop, {"A": {"status": evalrun.OK, "score": 0}})
        self.assertEqual((row["status"], row["need"], row["shot"]), (evalrun.OK, "bare", None))

    def test_placeholder_numbering_matches_builder(self):
        beat_video = [{"type": "video"}, {"type": "placeholder", "text": "first card"}, {"type": "placeholder", "text": "second card"}]
        ph = {f"ph{k}": v["text"] for k, v in enumerate([v for v in beat_video if v["type"] == "placeholder"], 1)}
        self.assertEqual(ph, {"ph1": "first card", "ph2": "second card"})


class ReportShapes(unittest.TestCase):
    def _report(self, joins):
        segs = [_seg(0, 0.0, 5.0, vlm={"status": evalrun.OK, "score": 2, "relation": "supports", "technical": ["none"]}),
                _seg(1, 5.0, 5.0, vlm={"status": evalrun.UNCHECKED, "score": None, "error": "timeout"})]
        rh = dict(total_spoken_s=10.0, bare_s=0.0, bare_pct=0, jump_cuts=0, joins=joins,
                  beats=[dict(beat=1, title="b", spoken_s=10.0, bare_s=0.0, bare_pct=0, shots=2, mean_hold_s=5.0,
                              longest_bare_s=0.0, jump_cuts_at=[], t0=0.0)])
        cut = {"version": 9, "beats": []}
        with tempfile.TemporaryDirectory() as td:
            out = Path(td) / "r"
            cuteval.write_report(out, cut, segs, rh, None, Path("x.mp4"))
            return out.with_suffix(".md").read_text()

    def test_silence_only_black_only_both_neither_and_unchecked(self):
        for joins in ({"status": "ok", "black": [], "silence": [(1.0, 1.3)], "errors": []},
                      {"status": "ok", "black": [(2.0, 2.1)], "silence": [], "errors": []},
                      {"status": "ok", "black": [(2.0, 2.1)], "silence": [(1.0, 1.3)], "errors": []},
                      {"status": "ok", "black": [], "silence": [], "errors": []},
                      {"status": "unchecked", "black": None, "silence": [(1.0, 1.3)], "errors": ["blackdetect exit 1"]}):
            md = self._report(joins)
            self.assertIn("Joins:", md)
        self.assertIn("unchecked", md)

    def test_unchecked_stretch_is_reported_and_not_in_mean(self):
        md = self._report({"status": "ok", "black": [], "silence": [], "errors": []})
        self.assertIn("1 of 2 stretches", md)
        self.assertIn("Unchecked: **1 stretches", md)
        self.assertIn("## Unchecked stretches", md)

    def test_coverage_counts(self):
        segs = [_seg(0, 0.0, 5.0, vlm={"status": evalrun.OK, "score": 2}), _seg(1, 5.0, 3.0)]
        cov = cuteval.coverage(segs, skipped=False)
        self.assertEqual((cov["judged"], cov["unchecked"], cov["unchecked_s"]), (1, 1, 3.0))


CAT = {
    "SHORT": dict(in_s=0.0, out_s=3.0, duration_s=3.0),
    "LONG": dict(in_s=10.0, out_s=60.0, duration_s=50.0),
    "TEN": dict(in_s=100.0, out_s=110.0, duration_s=10.0),
}


def _row(pick, dur, shot, verdicts=None, alternatives=(), status=evalrun.OK, usable=(1,), **extra):
    return dict(pick=pick, text="words", duration_s=dur, shot=shot, need="illustrate", score=3, usable=list(usable),
                reason="", alternatives=list(alternatives), verdicts=verdicts or {}, status=status, **extra)


class BuilderTiming(unittest.TestCase):
    def total(self, rows):
        return round(sum(r["duration_s"] for r in rows), 2)

    def test_short_source_in_split_pick_keeps_next_pick_at_20s(self):
        ok = {"status": evalrun.OK, "score": 3, "usable": [1]}
        rows = [_row("p1", 20.0, "SHORT", {"SHORT": ok, "LONG": ok}, alternatives=["LONG"]),
                _row("p2", 5.0, "TEN", {"TEN": ok})]
        out = cutbuild.build_beat_video({"video": []}, rows, CAT, max_hold=12.0, used_in={})
        self.assertEqual(self.total(out), 25.0)
        # the second pick's picture starts at 20 s
        t = 0.0
        for r in out:
            if r["type"] == "video" and r["clip"] == "TEN":
                break
            t += r["duration_s"]
        self.assertAlmostEqual(t, 20.0, places=2)
        # the short source's deficit went to the next piece of the same pick
        self.assertEqual(out[0]["clip"], "SHORT")
        self.assertEqual(out[1]["clip"], "LONG")

    def test_short_last_piece_runs_the_first_on_instead_of_flashing_the_speaker(self):
        ok = {"status": evalrun.OK, "score": 3, "usable": [1]}
        cat = dict(CAT, NINE=dict(in_s=0.0, out_s=9.85, duration_s=9.85))
        rows = [_row("p1", 20.0, "LONG", {"LONG": ok, "NINE": ok}, alternatives=["NINE"])]
        out = cutbuild.build_beat_video({"video": []}, rows, cat, 12.0, {})
        self.assertEqual([r["type"] for r in out], ["video", "video"])  # no four-frame bare row
        self.assertEqual(self.total(out), 20.0)
        self.assertEqual((out[0]["clip"], out[0]["duration_s"]), ("LONG", 10.15))
        self.assertAlmostEqual(out[1]["dest_in_s"], 10.15, places=2)
        # with no room anywhere the speaker still fills the rest
        cat["TEN9"] = dict(in_s=0.0, out_s=9.85, duration_s=9.85)
        rows = [_row("p1", 20.0, "TEN", {"TEN": ok, "TEN9": ok}, alternatives=["TEN9"])]
        out = cutbuild.build_beat_video({"video": []}, rows, cat, 12.0, {})
        self.assertEqual(out[-1]["type"], "bare")
        self.assertEqual(self.total(out), 20.0)

    def test_word_timed_segments_alternate_speaker_and_cover(self):
        # a welcome: the speaker, the land under the words that name it, the speaker, the land again; every segment on its word
        ok = {"status": evalrun.OK, "score": 3, "usable": [1]}
        weak = {"status": evalrun.OK, "score": 1, "usable": [1]}
        cat = dict(CAT, NINE=dict(in_s=0.0, out_s=9.85, duration_s=9.85))
        segs = [dict(until_s=2.0), dict(until_s=7.0, shot="LONG"), dict(until_s=13.0), dict(until_s=20.0, shot="NINE")]
        rows = [_row("p1", 20.0, "LONG", {"LONG": ok, "NINE": ok}, segments=segs)]
        out = cutbuild.build_beat_video({"video": []}, rows, cat, 12.0, {})
        self.assertEqual([(r["type"], r.get("clip")) for r in out], [("bare", None), ("video", "LONG"), ("bare", None), ("video", "NINE")])
        self.assertEqual([r["dest_in_s"] for r in out], [0.0, 2.0, 7.0, 13.0])
        self.assertEqual(self.total(out), 20.0)
        # a segment whose shot has no verdict of 2 goes to the speaker, and the next segment still starts on its word
        rows = [_row("p1", 20.0, "LONG", {"LONG": weak, "NINE": ok}, segments=segs)]
        out = cutbuild.build_beat_video({"video": []}, rows, cat, 12.0, {})
        self.assertEqual([r["type"] for r in out], ["bare", "bare", "bare", "video"])
        self.assertIn("no accepted verdict", out[1]["note"])
        self.assertEqual(out[3]["dest_in_s"], 13.0)
        # a measured-ineligible shot is never placed through a segment either
        out = cutbuild.build_beat_video({"video": []}, [_row("p1", 20.0, "LONG", {"LONG": ok, "NINE": ok}, segments=segs)], cat, 12.0, {},
                                        ineligible={"NINE": "shake over the limit"})
        self.assertEqual(out[-1]["type"], "bare")
        self.assertEqual(self.total(out), 20.0)

    def test_subsecond_remainder_is_preserved(self):
        ok = {"status": evalrun.OK, "score": 3, "usable": [1]}
        rows = [_row("p1", 0.6, "TEN", {"TEN": ok}), _row("p2", 4.0, "LONG", {"LONG": ok})]
        out = cutbuild.build_beat_video({"video": []}, rows, CAT, 12.0, {})
        self.assertEqual(self.total(out), 4.6)
        self.assertEqual(out[0]["type"], "bare")

    def test_only_final_sample_usable_never_backs_up(self):
        frames_used = [dict(t=100.25 + i * (9.5 / 3), pos=i / 3) for i in range(4)]  # last at 109.75
        row = cutbuild.shot_row(CAT, "TEN", 6.0, usable=[4], note="", used_in={}, frames_used=frames_used)
        self.assertIsNone(row)  # 0.25 s certified from the last sample: not a cut
        row = cutbuild.shot_row(CAT, "TEN", 6.0, usable=[3], note="", used_in={}, frames_used=frames_used)
        self.assertIsNotNone(row)
        self.assertGreaterEqual(row["in_s"], frames_used[2]["t"] - 0.01)
        self.assertLessEqual(row["out_s"], 110.0)

    def test_reused_shot_repeats_certified_footage_rather_than_going_bare(self):
        used_in = {"TEN": 108.5}  # 1.5 s left from the last use
        row = cutbuild.shot_row(CAT, "TEN", 6.0, usable=[1], note="", used_in=used_in)
        self.assertIsNotNone(row)
        self.assertEqual(row["duration_s"], 6.0)
        self.assertGreaterEqual(row["in_s"], 100.0)
        self.assertIn("repeats footage", row["note"])
        used_in = {"TEN": 103.0}  # 7 s left: continue, no repeat
        row = cutbuild.shot_row(CAT, "TEN", 6.0, usable=[1], note="", used_in=used_in)
        self.assertEqual((row["in_s"], row["duration_s"]), (103.0, 6.0))
        self.assertNotIn("repeats", row["note"])

    def test_short_piece_hands_deficit_to_next_piece(self):
        ok = {"status": evalrun.OK, "score": 3, "usable": [1]}
        rows = [_row("p1", 20.0, "SHORT", {"SHORT": ok, "LONG": ok}, alternatives=["LONG"])]
        out = cutbuild.build_beat_video({"video": []}, rows, CAT, max_hold=12.0, used_in={})
        self.assertEqual(self.total(out), 20.0)
        self.assertEqual([r["clip"] for r in out], ["SHORT", "LONG"])
        self.assertEqual(out[1]["duration_s"], 17.0)

    def test_legacy_verdict_without_frames_used_does_not_back_up_either(self):
        row = cutbuild.shot_row(CAT, "TEN", 6.0, usable=[4], note="", used_in={})
        self.assertIsNone(row)

    def test_merge_blocked_when_next_verdict_rejects_shot(self):
        ok = {"status": evalrun.OK, "score": 3, "usable": [1]}
        bad = {"status": evalrun.OK, "score": 0, "usable": [], "reason": "unwatchable blur"}
        rows = [_row("p1", 5.0, "LONG", {"LONG": ok}), _row("p2", 5.0, "LONG", {"LONG": bad})]
        out = cutbuild.build_beat_video({"video": []}, rows, CAT, 12.0, {})
        self.assertEqual(self.total(out), 10.0)
        self.assertEqual(out[0]["duration_s"], 5.0)  # not extended across the rejected pick
        self.assertEqual(out[1]["type"], "bare")

    def test_merge_allowed_when_next_verdict_accepts(self):
        ok = {"status": evalrun.OK, "score": 3, "usable": [1]}
        rows = [_row("p1", 5.0, "LONG", {"LONG": ok}), _row("p2", 5.0, "LONG", {"LONG": ok})]
        out = cutbuild.build_beat_video({"video": []}, rows, CAT, 12.0, {})
        self.assertEqual(len(out), 1)
        self.assertEqual(out[0]["duration_s"], 10.0)

    def test_watchable_requires_accepted_verdict(self):
        self.assertFalse(cutbuild.watchable(None))
        self.assertFalse(cutbuild.watchable({}))
        self.assertFalse(cutbuild.watchable({"error": "timeout", "status": evalrun.UNCHECKED}))
        self.assertFalse(cutbuild.watchable({"status": evalrun.OK, "score": 0, "usable": [1], "reason": "unwatchable blur"}))
        self.assertTrue(cutbuild.watchable({"status": evalrun.OK, "score": 2, "usable": [1, 2], "reason": "blurry at the end"}))

    def test_unchecked_plan_row_becomes_provisional_bare(self):
        rows = [_row("p1", 8.0, None, status=evalrun.UNCHECKED)]
        out = cutbuild.build_beat_video({"video": []}, rows, CAT, 12.0, {})
        self.assertEqual(self.total(out), 8.0)
        self.assertTrue(out[0].get("provisional"))
        self.assertEqual(out[0]["type"], "bare")

    def test_error_verdict_on_chosen_shot_falls_to_bare_not_cover(self):
        rows = [_row("p1", 5.0, "LONG", {"LONG": {"status": evalrun.UNCHECKED, "error": "timeout"}})]
        out = cutbuild.build_beat_video({"video": []}, rows, CAT, 12.0, {})
        self.assertEqual(self.total(out), 5.0)
        self.assertEqual(out[0]["type"], "bare")

    def test_placeholder_remainder_kept(self):
        ph = {"type": "placeholder", "text": "PLACEHOLDER: card", "in_s": 0.0, "out_s": 8.0, "duration_s": 8.0}
        rows = [_row("p1", 13.0, "PH:ph1", {})]
        out = cutbuild.build_beat_video({"video": [ph]}, rows, CAT, 12.0, {})
        self.assertEqual(self.total(out), 13.0)
        self.assertEqual(out[0]["duration_s"], 12.0)
        self.assertEqual(out[0]["type"], "placeholder")
        self.assertEqual(out[1]["type"], "bare")


class CandidateFrames(unittest.TestCase):
    def test_frames_spread_over_whatever_the_cache_holds(self):
        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            for i in range(6):
                (d / f"f0{i}.jpg").write_bytes(b"x")
            fu = brollmatch.candidate_frames({"in_s": 0.0, "out_s": 10.0}, d)
            self.assertEqual(len(fu), 4)
            self.assertEqual([Path(f["file"]).name for f in fu], ["f00.jpg", "f02.jpg", "f03.jpg", "f05.jpg"])
            self.assertTrue(all(0.0 <= f["pos"] <= 1.0 for f in fu))
            for i in range(4, 6):
                (d / f"f0{i}.jpg").unlink()
            fu = brollmatch.candidate_frames({"in_s": 0.0, "out_s": 10.0}, d)
            self.assertEqual(len(fu), 4)
            self.assertEqual(brollmatch.candidate_frames({"in_s": 0.0, "out_s": 10.0}, d / "none"), [])


if __name__ == "__main__":
    unittest.main()
