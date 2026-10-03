"""The resolved-intent manifest: precedence, cut checks, read-back checks, and the builder honouring it. No models."""
from __future__ import annotations

import copy
import unittest

from videoeasy import cutbuild, intent

DOC = {
    "film": "t",
    "rules": [
        {"id": "fence-out", "topic": "fence", "kind": "words_forbidden", "words": ["fence"], "decision": "no fence talk",
         "origin": "subject", "by": "subject A", "source": "cam", "decided": "2025-01-28"},
        {"id": "fence-history", "topic": "fence", "kind": "words_allowed_history", "words": ["fence"],
         "forbidden_phrases": ["the dispute"], "decision": "history ok", "origin": "editor", "by": "editor", "source": "syn",
         "decided": "2026-09-14", "supersedes": ["fence-out"]},
        {"id": "pause", "topic": "pause", "kind": "no_cover", "scope": {"beat_title_contains": "PAUSE"},
         "decision": "hold", "origin": "agent", "by": "room", "source": "syn", "decided": "2026-09-14"},
        {"id": "explainer-face", "topic": "explainer-pic", "kind": "no_cover_pick", "scope": {"clip": "CAMB0184", "text_contains": "I'm Rowan"},
         "decision": "face", "origin": "editor", "by": "editor", "source": "syn", "decided": "2026-09-25"},
        {"id": "explainer-present", "topic": "explainer", "kind": "must_be_present", "clips": ["CAMB0184"], "story": "explainer inside",
         "decision": "explainer in", "origin": "editor", "by": "editor", "source": "syn", "decided": "2026-09-25"},
        {"id": "walk", "topic": "walk", "kind": "cover_only_named_detail", "scope": {"beat_notes_contains": "Walking interlude"},
         "never_from_prefix": "CAMA01", "decision": "bare unless named", "origin": "agent", "by": "room", "source": "cut", "decided": "2026-09-25"},
        {"id": "lead-in", "topic": "lead", "kind": "cover_required", "scope": {"beat_title_contains": "tools", "pick": "last"},
         "clips_any": ["CAMB0182"], "decision": "lead-in cover", "origin": "editor", "by": "editor", "source": "syn", "decided": "2026-09-25"},
        {"id": "plane", "topic": "plane", "kind": "clip_span_excluded", "clip": "CAMA0131", "in_s": 228.0, "out_s": 240.0,
         "decision": "no plane", "origin": "subject", "by": "subject B", "source": "cam", "decided": "2025-01-28"},
        {"id": "old-length", "topic": "length", "kind": "open", "decision": "12 min", "origin": "editor", "by": "editor", "source": "x", "decided": "2026-09-01"},
        {"id": "new-length", "topic": "length", "kind": "open", "decision": "13 min", "origin": "editor", "by": "editor", "source": "x", "decided": "2026-09-25"},
    ],
}


def pick(clip, in_s, dur, text):
    return dict(type="audio", clip=clip, in_s=in_s, out_s=round(in_s + dur, 2), duration_s=dur, text=text, tier=1)


def vrow(kind, dest_in, dur, clip=None, text=None):
    r = dict(type=kind, clip=clip, in_s=0.0, out_s=dur, duration_s=dur, dest_in_s=dest_in, dest_out_s=round(dest_in + dur, 2))
    if text:
        r["text"] = text
    return r


def cut_ok():
    return {"beats": [
        {"title": "7. tools", "notes": "", "gap_after_s": 0.0,
         "audio": [pick("CAMA0124", 100.0, 10.0, "so much mulga"), pick("CAMA0124", 120.0, 10.0, "cutting the old wire")],
         "video": [vrow("video", 0.0, 10.0, "CAMA0117_w00"), vrow("video", 10.0, 10.0, "CAMB0182_w00")]},
        {"title": "7b. Explainer", "notes": "the explainer's own sync", "gap_after_s": 0.0,
         "audio": [pick("CAMB0184", 2.8, 3.0, "Hi, I'm Rowan from the crew."), pick("CAMB0177", 18.0, 5.0, "the tool")],
         "video": [vrow("bare", 20.0, 3.0), vrow("video", 23.0, 5.0, "CAMA0117_w01")]},
        {"title": "9b. Walk", "notes": "Walking interlude: sync", "gap_after_s": 0.0,
         "audio": [pick("CAMA0125", 0.6, 8.0, "what's that one")], "video": [vrow("bare", 28.0, 8.0)]},
        {"title": "12. THE PAUSE", "notes": "no cover", "gap_after_s": 0.0,
         "audio": [pick("CAMA0124", 2825.0, 6.0, "the fence used to run here")], "video": [vrow("bare", 36.0, 6.0)]},
    ]}


class Resolve(unittest.TestCase):
    def test_supersedes_and_latest_on_topic_win(self):
        res = intent.resolve(DOC)
        ids = [r["id"] for r in res["active"]]
        self.assertNotIn("fence-out", ids)
        self.assertIn("fence-history", ids)
        self.assertNotIn("old-length", ids)
        self.assertIn("new-length", ids)
        why = {x["id"]: x["why"] for x in res["retired"]}
        self.assertIn("fence-history", why["fence-out"])
        self.assertIn("new-length", why["old-length"])

    def test_brief_names_sources_and_what_no_longer_applies(self):
        b = intent.brief(DOC)
        self.assertIn("[fence-history]", b)
        self.assertIn("No longer applies", b)
        self.assertIn("[fence-out]", b)
        self.assertIn("explainer inside", intent.story_additions(DOC))


class CheckCut(unittest.TestCase):
    def status(self, findings):
        return {f["rule"]: f["status"] for f in findings}

    def test_clean_cut(self):
        st = self.status(intent.check_cut(cut_ok(), DOC))
        self.assertEqual(st["pause"], "honoured")
        self.assertEqual(st["explainer-face"], "honoured")
        self.assertEqual(st["explainer-present"], "honoured")
        self.assertEqual(st["walk"], "honoured")
        self.assertEqual(st["lead-in"], "honoured")
        self.assertEqual(st["plane"], "honoured")
        self.assertEqual(st["fence-history"], "review")  # the mention is allowed, listed for a look
        self.assertEqual(st["new-length"], "open")

    def test_cover_on_the_pause_is_broken_whatever_its_score(self):
        c = cut_ok()
        c["beats"][3]["video"] = [vrow("video", 36.0, 6.0, "CAMA0140_w00")]
        f = next(x for x in intent.check_cut(c, DOC) if x["rule"] == "pause")
        self.assertEqual(f["status"], "broken")
        self.assertIn("CAMA0140_w00", f["evidence"])

    def test_cover_on_the_explainers_greeting_is_broken(self):
        c = cut_ok()
        c["beats"][1]["video"][0] = vrow("video", 20.0, 3.0, "CAMA0117_w02")
        self.assertEqual(self.status(intent.check_cut(c, DOC))["explainer-face"], "broken")

    def test_explainer_missing_is_broken(self):
        c = cut_ok()
        c["beats"][1]["audio"][0]["tier"] = 2
        self.assertEqual(self.status(intent.check_cut(c, DOC))["explainer-present"], "broken")

    def test_walk_covered_by_another_walk_take_is_broken_by_a_detail_shot_is_review(self):
        c = cut_ok()
        c["beats"][2]["video"] = [vrow("video", 28.0, 8.0, "CAMA0131_w03")]
        self.assertEqual(self.status(intent.check_cut(c, DOC))["walk"], "broken")
        c["beats"][2]["video"] = [vrow("video", 28.0, 8.0, "DJI_1230_s01")]
        self.assertEqual(self.status(intent.check_cut(c, DOC))["walk"], "review")

    def test_lead_in_missing_is_broken(self):
        c = cut_ok()
        c["beats"][0]["video"][1] = vrow("video", 10.0, 10.0, "CAMA0117_w03")
        self.assertEqual(self.status(intent.check_cut(c, DOC))["lead-in"], "broken")

    def test_forbidden_framing_on_the_heard_transcript(self):
        st = self.status(intent.check_cut(cut_ok(), DOC, heard_text="it became the dispute over the fence"))
        self.assertEqual(st["fence-history"], "broken")

    def test_pick_over_the_aeroplane(self):
        c = cut_ok()
        c["beats"][2]["audio"] = [pick("CAMA0131", 230.0, 8.0, "listen")]
        self.assertEqual(self.status(intent.check_cut(c, DOC))["plane"], "broken")

    def test_dropped_lines_stay_dropped_and_their_neighbours_stay(self):
        doc = copy.deepcopy(DOC)
        doc["rules"].append({"id": "drops", "topic": "drops", "kind": "clip_span_excluded", "decision": "dropped for length",
                             "spans": [dict(clip="CAMA0124", in_s=110.0, out_s=120.0, text="a dropped line"),
                                       dict(clip="CAMA0125", in_s=20.0, out_s=30.0, text="another")],
                             "origin": "editor", "by": "editor", "source": "page", "decided": "2026-09-29"})
        self.assertEqual(self.status(intent.check_cut(cut_ok(), doc))["drops"], "honoured")  # 100-110 and 120-130 only meet its edges
        c = cut_ok()
        c["beats"][0]["audio"][1] = pick("CAMA0124", 115.0, 10.0, "brought back")
        f = next(x for x in intent.check_cut(c, doc) if x["rule"] == "drops")
        self.assertEqual(f["status"], "broken")
        self.assertIn("b1p2", f["evidence"])

    def test_rows_without_destinations_are_not_testable_never_honoured(self):
        c = cut_ok()
        for v in c["beats"][3]["video"]:
            v.pop("dest_in_s"); v.pop("dest_out_s")
        self.assertEqual(self.status(intent.check_cut(c, DOC))["pause"], "not_testable")

    def test_protected_picks_and_walk_rules(self):
        c = cut_ok()
        self.assertEqual(intent.protected_picks(c, DOC), {"b2p1": "explainer-face", "b4p1": "pause"})
        self.assertEqual(intent.walk_rules(c, DOC), {"b3p1": "CAMA01"})


class CheckReadback(unittest.TestCase):
    def plan(self):
        return {"audio": [dict(pick="b1p1", name="CAMA0124.MOV", track=1, head=0, rec=0, dur=240),
                          dict(pick="b1p2", name="CAMA0124.MOV", track=1, head=0, rec=240, dur=240),
                          dict(pick="b2p1", name="CAMB0184.MOV", track=2, head=6, rec=480, dur=72),
                          dict(pick="b2p2", name="CAMB0177.MOV", track=1, head=6, rec=552, dur=120),
                          dict(pick="b3p1", name="CAMA0125.MOV", track=2, head=6, rec=672, dur=192),
                          dict(pick="b4p1", name="CAMA0124.MOV", track=1, head=6, rec=864, dur=144)]}

    def rb(self, cover):
        # the A-roll's own picture and sound sit on V1/A1 and V2/A2 (alternating, heads extended); cover on V3
        items = []
        for a in self.plan()["audio"]:
            for typ in ("video", "audio"):
                items.append(dict(type=typ, track=a["track"], name=a["name"], rec=a["rec"] - a["head"], dur=a["dur"] + a["head"]))
        items += [dict(type="video", track=3, name=n, rec=r, dur=d) for n, r, d in cover]
        return dict(fps=24.0, items=items, markers={})

    def test_clean_timeline(self):
        rb = self.rb([("CAMA0117.MOV", 0, 240), ("CAMB0182.MOV", 240, 240), ("CAMA0117.MOV", 552, 120)])
        st = {f["rule"]: f["status"] for f in intent.check_readback(rb, self.plan(), cut_ok(), DOC)}
        self.assertEqual(st["pause"], "honoured")
        self.assertEqual(st["explainer-face"], "honoured")
        self.assertEqual(st["lead-in"], "honoured")
        self.assertEqual(st["walk"], "honoured")
        self.assertEqual(st["explainer-present"], "honoured")

    def test_cover_item_over_the_pause_is_broken_on_the_timeline(self):
        rb = self.rb([("CAMB0182.MOV", 240, 240), ("CAMA0140.MOV", 900, 48)])
        st = {f["rule"]: f["status"] for f in intent.check_readback(rb, self.plan(), cut_ok(), DOC)}
        self.assertEqual(st["pause"], "broken")

    def test_walk_covered_by_walk_take_on_the_timeline(self):
        rb = self.rb([("CAMB0182.MOV", 240, 240), ("CAMA0131.MOV", 700, 100)])
        st = {f["rule"]: f["status"] for f in intent.check_readback(rb, self.plan(), cut_ok(), DOC)}
        self.assertEqual(st["walk"], "broken")

    def test_lead_in_missing_on_the_timeline(self):
        rb = self.rb([("CAMA0117.MOV", 240, 240)])
        st = {f["rule"]: f["status"] for f in intent.check_readback(rb, self.plan(), cut_ok(), DOC)}
        self.assertEqual(st["lead-in"], "broken")


class BuilderHonoursIntent(unittest.TestCase):
    CAT = {"CAMA0140_w00": dict(in_s=0.0, out_s=30.0, duration_s=30.0, source="CAMA0140", role="broll", frames=[]),
           "CAMA0131_w03": dict(in_s=0.0, out_s=30.0, duration_s=30.0, source="CAMA0131", role="broll", frames=[]),
           "DJI_1230_s01": dict(in_s=0.0, out_s=30.0, duration_s=30.0, source="DJI_1230", role="broll", frames=[])}

    def row(self, pick, shot, alts=()):
        v = {s: dict(status="ok", score=3, usable=[1, 2, 3, 4], reason="fine") for s in (shot, *alts) if s}
        return dict(pick=pick, shot=shot, duration_s=6.0, text="words", verdicts=v, alternatives=list(alts), status="ok", need="illustrate")

    def test_protected_pick_stays_bare_whatever_the_plan_says(self):
        beat = {"title": "12. THE PAUSE", "audio": [pick("CAMA0124", 0, 6.0, "x")], "video": []}
        rows = cutbuild.build_beat_video(beat, [self.row("b4p1", "CAMA0140_w00")], self.CAT, 12.0, {}, 0.0, protected={"b4p1": "pause"})
        self.assertEqual([r["type"] for r in rows], ["bare"])
        self.assertIn("pause", rows[0]["note"])
        self.assertEqual(rows[0]["dest_out_s"], 6.0)

    def test_walk_take_dropped_alternative_kept(self):
        beat = {"title": "9b. Walk", "audio": [pick("CAMA0125", 0, 6.0, "x")], "video": []}
        rows = cutbuild.build_beat_video(beat, [self.row("b3p1", "CAMA0131_w03", ("DJI_1230_s01",))], self.CAT, 12.0, {}, 0.0,
                                         never_from={"b3p1": "CAMA01"})
        self.assertEqual([r["type"] for r in rows], ["bare"])
        rows = cutbuild.build_beat_video(beat, [self.row("b3p1", "DJI_1230_s01", ("CAMA0131_w03",))], self.CAT, 12.0, {}, 0.0,
                                         never_from={"b3p1": "CAMA01"})
        self.assertEqual([r.get("clip") for r in rows], ["DJI_1230_s01"])


if __name__ == "__main__":
    unittest.main()
