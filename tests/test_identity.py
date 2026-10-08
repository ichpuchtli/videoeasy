"""Face identity: the crops chosen, the grouping, the answer checked, decisions bound to a measurement.
No footage, no models; and recognition only ever in its own worker environment."""
from __future__ import annotations

import json
import re
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from videoeasy import identity, register

ROOT = Path(__file__).resolve().parents[1]


def sample(t, px=120, yaw=0.0, readable=True):
    return [t, 100, 100, px, px, readable, yaw, 0.0, None]


def req_track(source, track, t0=0.0, t1=5.0):
    return dict(key=f"{source}/{track}", source=source, track=track, t0=t0, t1=t1, px=120,
                crops=[dict(file=f"crops/{source}/{track}-x.jpg", t=t0, box=[0, 0, 1, 1])])


REQ = dict(schema=identity.REQUEST_SCHEMA, request_id="r1", sources={"a": "a.f1.raw.json", "b": "b.f1.raw.json"},
           tracks=[req_track("a", "t00"), req_track("a", "t01"), req_track("b", "t00"), req_track("b", "t01"), req_track("b", "t02")])
RAWS = {"a": "a.f1.raw.json", "b": "b.f1.raw.json"}


class Crops(unittest.TestCase):
    def test_frontal_large_readable_and_spread(self):
        tr = dict(samples=[sample(0.0, yaw=30), sample(0.2, yaw=2), sample(0.4, yaw=1), sample(3.0, yaw=5),
                           sample(5.0, px=60), sample(6.0, readable=False), sample(8.0, yaw=10)])
        got = [s[0] for s in identity.pick_samples(tr)]
        self.assertEqual(got, [0.4, 3.0, 8.0])   # 0.2 is within a second of 0.4; small and unreadable never qualify

    def test_no_recognisable_view_no_crops(self):
        self.assertEqual(identity.pick_samples(dict(samples=[sample(0.0, px=50), sample(1.0, readable=False)])), [])


class Grouping(unittest.TestCase):
    def test_similar_tracks_group_a_lone_track_is_no_proposal(self):
        rng = np.random.default_rng(0)
        a, b = (identity._unit(rng.normal(size=512)) for _ in range(2))
        near = lambda v: identity._unit(v + 0.15 * identity._unit(rng.normal(size=512)))  # noqa: E731
        emb = {"crops/a/t00-x.jpg": near(a), "crops/a/t01-x.jpg": near(b), "crops/b/t00-x.jpg": near(a), "crops/b/t01-x.jpg": None,
               "crops/b/t02-x.jpg": identity._unit(rng.normal(size=512))}
        ans = identity.propose(REQ, emb, {})
        got = {t["key"]: t["person"] for t in ans["tracks"]}
        self.assertEqual(got["a/t00"], got["b/t00"])
        self.assertIsNotNone(got["a/t00"])
        self.assertEqual([got[k] for k in ("a/t01", "b/t01", "b/t02")], [None, None, None])   # one track only; no face; no match
        self.assertIn("non-commercial", ans["tool"]["model_licence"])
        with_ref = {t["key"]: t["person"] for t in identity.propose(REQ, emb, {"P07": b})["tracks"]}
        self.assertEqual(with_ref["a/t01"], "ref:P07")


class Answer(unittest.TestCase):
    def answer(self, tracks, **kw):
        return dict(dict(schema=identity.ANSWER_SCHEMA, request_id="r1", tool=dict(name="test"), tracks=tracks), **kw)

    def test_statuses(self):
        ans = self.answer([dict(key="a/t00", person="g1", score=0.8), dict(key="b/t00", person="g1", score=0.7),
                           dict(key="a/t01", person=None, why="no other track matched"), dict(key="b/t01", person=["x"]),
                           dict(key="zz/t09", person="g2"), dict(person="g3")])
        doc = identity.resolve_answer(REQ, ans, RAWS)
        st = {k: v["status"] for k, v in doc["links"].items()}
        self.assertEqual(st, {"a/t00": "proposed", "b/t00": "proposed", "a/t01": "none", "b/t01": "invalid", "b/t02": "unchecked"})
        self.assertEqual(doc["links"]["a/t00"]["proposal"], "g1")
        self.assertNotIn("person", doc["links"]["a/t00"])   # a proposal is never a name
        self.assertEqual(doc["unknown_keys"], ["zz/t09"])
        self.assertEqual(len(doc["malformed"]), 1)
        self.assertEqual(doc["proposed_people"], 1)

    def test_a_source_measured_again_is_unchecked(self):
        doc = identity.resolve_answer(REQ, self.answer([dict(key="a/t00", person="g1")]), dict(RAWS, a="a.f2.raw.json"))
        self.assertEqual(doc["links"]["a/t00"]["status"], "unchecked")

    def test_one_person_cannot_be_two_faces_at_once(self):
        req = dict(REQ, tracks=[req_track("a", "t00", 0, 5), req_track("a", "t01", 3, 9), req_track("a", "t02", 20, 25)])
        doc = identity.resolve_answer(req, self.answer([dict(key=f"a/t0{i}", person="g1") for i in range(3)]), RAWS)
        self.assertEqual(doc["conflicts"], ["a/t00", "a/t01"])

    def test_import_refuses_an_answer_for_another_export(self):
        with tempfile.TemporaryDirectory() as d:
            cfg = SimpleNamespace(work_dir=Path(d) / "work", out_dir=Path(d) / "out", film_dir=Path(d))
            (cfg.work_dir / "identity").mkdir(parents=True)
            cfg.out_dir.mkdir()
            (cfg.work_dir / "identity" / "request.json").write_text(json.dumps(REQ))
            (Path(d) / "ans.json").write_text(json.dumps(self.answer([], request_id="other")))
            with self.assertRaises(SystemExit):
                identity.import_answer(cfg, Path(d) / "ans.json")


class Decisions(unittest.TestCase):
    DOC = dict(links={"a/t00": dict(source="a", track="t00", status="proposed", proposal="g1", raw="a.f1.raw.json"),
                      "b/t00": dict(source="b", track="t00", status="proposed", proposal="g1", raw="b.f1.raw.json")})

    def test_only_a_confirmed_link_names_a_person(self):
        dec = dict(tracks={"a/t00": dict(raw="a.f1.raw.json", decided="confirmed", person="P07"),
                           "b/t00": dict(raw="b.f1.raw.json", decided="rejected", person=None)})
        links = identity.resolve(self.DOC, dec, RAWS)
        self.assertEqual((links["a/t00"]["status"], links["a/t00"]["person"]), ("confirmed", "P07"))
        self.assertEqual((links["b/t00"]["status"], links["b/t00"]["person"]), ("rejected", None))
        self.assertIsNone(identity.resolve(self.DOC, None, RAWS)["a/t00"]["person"])

    def test_a_decision_on_an_old_measurement_does_not_apply(self):
        dec = dict(tracks={"a/t00": dict(raw="a.f0.raw.json", decided="confirmed", person="P07")})
        link = identity.resolve(self.DOC, dec, RAWS)["a/t00"]
        self.assertEqual((link["status"], link["person"]), ("proposed", None))
        self.assertEqual(link["stale_decision"]["person"], "P07")

    def test_merge_keeps_what_it_supersedes(self):
        cur = dict(tracks={"a/t00": dict(raw="a.f1.raw.json", decided="confirmed", person="P07", by="Sam", at="t1")})
        new = dict(made="t2", tracks={"a/t00": dict(raw="a.f1.raw.json", decided="confirmed", person="P08"),
                                      "b/t00": dict(raw="b.f1.raw.json", decided="confirmed", person="P07"),
                                      "b/t01": dict(raw="b.f1.raw.json", decided="confirmed", person=""),
                                      "b/t02": dict(raw="b.f1.raw.json", decided="maybe", person="P07")})
        out, counts = identity.merge_decisions(cur, new, "editor", "now")
        self.assertEqual(counts, dict(confirmed=2, rejected=0, unchanged=0, invalid=2))
        self.assertEqual(out["tracks"]["a/t00"]["supersedes"]["person"], "P07")
        _, again = identity.merge_decisions(out, new, "editor", "now")
        self.assertEqual(again["unchanged"], 2)

    def test_review_groups_named_first_and_page_is_local(self):
        links = identity.resolve(dict(links={k: dict(v, crops=["crops/x.jpg"], t0=1.0, t1=2.0) for k, v in self.DOC["links"].items()}),
                                 dict(tracks={"a/t00": dict(raw="a.f1.raw.json", decided="confirmed", person="P07")}), RAWS)
        gs = identity.groups(links)
        self.assertEqual([g["id"] for g in gs], ["person:P07", "proposal:g1"])
        page = identity.review_html(gs, "../../identity", "20261008")
        self.assertNotIn("http", page.replace("http-equiv", ""))   # nothing leaves the machine


class Find(unittest.TestCase):
    def test_find_one_persons_faces(self):
        def m(i, source, track):
            return dict(id=f"m{i}", source=source, track=track, t0=float(i), fused=dict(status="agreed", label="joy", speaking=False),
                        reading=dict(register="joy", strength=2))
        doc = dict(moments=[m(1, "a", "t00"), m(2, "a", "t01"), m(3, "b", "t00"), m(4, "b", None)])
        got = register.find(doc, "joy", tracks={("a", "t00"), ("b", "t00")})
        self.assertEqual([x["id"] for x in got], ["m1", "m3"])


class Licence(unittest.TestCase):
    def test_recognition_only_in_its_own_worker(self):
        """The repository is MIT and recognition models carry their own licences: the model is downloaded at setup,
        never shipped, and only the PEP 723 worker's own environment ever holds a recognition library."""
        banned = re.compile(r"^\s*(import|from)\s+(insightface|face_recognition|deepface|facenet_pytorch|dlib|arcface)\b", re.M)
        for p in (ROOT / "src" / "videoeasy").glob("*.py"):
            if p.name != identity.WORKER.name:
                self.assertIsNone(banned.search(p.read_text()), p.name)
        self.assertTrue(identity.WORKER.read_text().startswith("# /// script"))
        deps = (ROOT / "pyproject.toml").read_text().lower()
        for name in ("insightface", "face_recognition", "face-recognition", "deepface", "facenet", "dlib"):
            self.assertNotIn(name, deps)
        weights = [p for d in ("src", "tests", "docs") for p in (ROOT / d).rglob("*") if p.suffix in (".onnx", ".pth", ".task", ".tflite", ".zip")]
        self.assertEqual(weights, [])
        self.assertTrue(identity.MODEL["sha256"] and identity.MODEL["url"].startswith("https://github.com/deepinsight/"))


if __name__ == "__main__":
    unittest.main()
