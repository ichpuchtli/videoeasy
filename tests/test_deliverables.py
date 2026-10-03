"""Event projects: schedule validation, clock reading, clip assignment, items, status evidence, Resolve bins.
No footage, no models, no Resolve (the media pool is a fake)."""
import copy
import datetime as dt
import json
import tempfile
import unittest
from pathlib import Path
from zoneinfo import ZoneInfo

import yaml

from videoeasy import deliverables as d

REPO = Path(__file__).resolve().parents[1]
SYD = ZoneInfo("Australia/Sydney")


def base_doc() -> dict:
    return {
        "project": "Retreat", "timezone": "Australia/Sydney",
        "days": {"day1": "2026-03-13", "day2": "2026-03-14"},
        "cameras": [
            {"name": "A cam", "prefix": "MVI_", "path_contains": "/op1/", "operator": "Op 1", "clock": "creation_time_utc"},
            {"name": "B cam", "prefix": "MVI_", "path_contains": "/op2/", "operator": "Op 2", "clock": "creation_time_local"},
            {"name": "Drone", "prefix": "DJI_", "operator": "Op 2", "clock": "creation_time_local", "clock_offset_s": -3600},
        ],
        "people": [{"id": "fac-a", "name": "Facilitator A", "role": "facilitator"}],
        "registers": ["focus", "laughter", "stillness"],
        "sessions": [
            {"id": "s-morning", "day": "day1", "start": "09:00", "title": "Morning"},   # no end: runs to s-workshop
            {"id": "s-workshop", "day": "day1", "start": "13:00", "end": "15:00", "title": "Workshop",
             "capture": [{"operator": "Op 1", "text": "feature", "feeds": ["feature"], "registers": ["focus"]},
                         {"operator": "Op 2", "text": "interviews", "feeds": ["before-after"]},
                         "both: register b-roll"],
             "registers": ["laughter"], "feeds": ["broll"]},
            {"id": "s-track-a", "day": "day2", "start": "08:00", "end": "10:00", "title": "Track A", "cameras": ["A cam"]},
            {"id": "s-track-b", "day": "day2", "start": "08:00", "end": "10:00", "title": "Track B", "cameras": ["B cam", "Drone"],
             "registers": ["stillness"]},
        ],
        "deliverable_types": [
            {"id": "feature", "title": "Features", "count": 2},
            {"id": "before-after", "title": "Before/after", "count": 1},
            {"id": "broll", "title": "B-roll package", "count": 1},
        ],
    }


def clip(cid, path, created, dur=60.0, role="broll", tc=None, fps=25.0):
    return {"id": cid, "path": path, "role": role, "duration_s": dur, "creation_time": created, "start_timecode": tc, "fps": fps}


class Validate(unittest.TestCase):
    def test_example_validates(self):
        doc = yaml.safe_load((REPO / "deliverables.example.yaml").read_text())
        self.assertEqual(d.validate(doc), ([], []))

    def test_base_doc_validates_with_an_allowed_parallel_pair(self):
        self.assertEqual(d.validate(base_doc()), ([], []))

    def test_overlap_rejected_when_a_session_lists_no_cameras(self):
        doc = base_doc()
        del doc["sessions"][3]["cameras"]
        errors, conflicts = d.validate(doc)
        self.assertEqual(errors, [])
        self.assertEqual(len(conflicts), 1)
        self.assertIn("s-track-a", conflicts[0])
        self.assertIn("s-track-b", conflicts[0])

    def test_overlap_rejected_when_cameras_are_shared(self):
        doc = base_doc()
        doc["sessions"][3]["cameras"] = ["A cam", "Drone"]
        self.assertEqual(len(d.validate(doc)[1]), 1)

    def test_errors_name_their_path(self):
        doc = base_doc()
        doc["timezone"] = "Mars/Olympus"
        doc["typo_key"] = 1
        doc["sessions"][0]["day"] = "day9"
        doc["sessions"][1]["end"] = "12:00"                       # before its start
        doc["sessions"][1]["capture"][0]["operator"] = "Op 9"     # no such operator
        doc["sessions"][1]["feeds"] = ["nope"]
        doc["sessions"][2]["cameras"] = ["Z cam"]
        doc["sessions"].append(dict(doc["sessions"][0]))          # duplicate id
        doc["cameras"].append({"name": "Bad", "clock": "sundial"})
        doc["deliverable_types"][0]["count"] = "two"
        errors, _ = d.validate(doc)
        text = "\n".join(errors)
        for needle in ("typo_key: unknown top-level key", "timezone:", "day: 'day9'", "end: 12:00 is not after start",
                       "operator: 'Op 9'", "feeds: 'nope'", "cameras: 'Z cam'", "id: duplicate",
                       "cameras[3]: needs prefix, path_contains, or both", "clock: 'sundial'", "count: required whole number"):
            self.assertIn(needle, text)

    def test_items_are_checked_against_types_and_counts(self):
        items = [{"id": "x", "type": "broll", "status": "planned"}, {"id": "y", "type": "broll", "status": "done"},
                 {"id": "x", "type": "ghost", "people": ["nobody"]}]
        text = "\n".join(d.validate(base_doc(), items)[0])
        for needle in ("id: duplicate", "type: 'ghost'", "status: 'done'", "people: 'nobody'", "items of type 'broll', which counts 1"):
            self.assertIn(needle, text)

    def test_unquoted_yaml_times_read_as_minutes(self):
        self.assertEqual(d.minutes(yaml.safe_load("t: 07:30")["t"]), 450)
        self.assertEqual(d.minutes("07:30"), 450)
        self.assertIsNone(d.minutes("7.30"))
        self.assertIsNone(d.minutes("25:00"))

    def test_missing_end_runs_to_next_conflicting_start(self):
        w = d.windows(base_doc())
        self.assertEqual(w["s-morning"]["end"], 13 * 60)
        doc = base_doc()
        doc["sessions"][2].pop("end")
        doc["sessions"][3].pop("end")   # parallel sessions on disjoint cameras do not end each other
        w = d.windows(doc)
        self.assertEqual(w["s-track-a"]["end"], d.DAY_END)


class Clocks(unittest.TestCase):
    cam_utc = {"name": "A", "prefix": "A", "clock": "creation_time_utc"}
    cam_local = {"name": "B", "prefix": "B", "clock": "creation_time_local"}

    def test_utc_and_local_readings(self):
        t, how = d.true_start(clip("A1", "", "2026-03-13T22:30:00.000000Z"), self.cam_utc, SYD)
        self.assertEqual(t.replace(tzinfo=None), dt.datetime(2026, 3, 14, 9, 30))   # AEDT, UTC+11
        self.assertEqual(how, "creation_time_utc+0s")
        t, _ = d.true_start(clip("B1", "", "2026-03-14T09:30:00Z"), self.cam_local, SYD)
        self.assertEqual(t.replace(tzinfo=None), dt.datetime(2026, 3, 14, 9, 30))   # the label Z is ignored

    def test_daylight_saving_ends_between_two_readings(self):
        before, _ = d.true_start(clip("A1", "", "2026-04-04T00:00:00Z"), self.cam_utc, SYD)
        after, _ = d.true_start(clip("A2", "", "2026-04-05T00:00:00Z"), self.cam_utc, SYD)
        self.assertEqual(before.hour, 11)   # AEDT
        self.assertEqual(after.hour, 10)    # AEST from 3 am on 5 April 2026

    def test_offset_and_end_stamp(self):
        cam = dict(self.cam_utc, clock_offset_s=-90, stamp="end")
        t, how = d.true_start(clip("A1", "", "2026-03-13T23:00:00Z", dur=600), cam, SYD)
        self.assertEqual(t.replace(tzinfo=None), dt.datetime(2026, 3, 14, 9, 48, 30))
        self.assertIn("stamp=end", how)

    def test_timecode_takes_the_date_nearest_the_creation_time(self):
        cam = {"name": "R", "prefix": "R", "clock": "timecode"}
        t, _ = d.true_start(clip("R1", "", "2026-03-13T22:31:05Z", tc="09:30:00:12", fps=25), cam, SYD)
        self.assertEqual(t.replace(tzinfo=None), dt.datetime(2026, 3, 14, 9, 30, 0, 480000))
        # creation time just after midnight local, timecode just before: the previous day
        t, _ = d.true_start(clip("R2", "", "2026-03-13T13:05:00Z", tc="23:50:00:00"), cam, SYD)
        self.assertEqual(t.replace(tzinfo=None), dt.datetime(2026, 3, 13, 23, 50))

    def test_no_time_is_a_reason_not_a_guess(self):
        self.assertIsNone(d.true_start(clip("A1", "", None), self.cam_utc, SYD)[0])
        self.assertIn("no creation_time", d.true_start(clip("A1", "", None), self.cam_utc, SYD)[1])
        self.assertIn("does not parse", d.true_start(clip("A1", "", "yesterday"), self.cam_utc, SYD)[1])
        cam = {"name": "R", "prefix": "R", "clock": "timecode"}
        self.assertIn("no date", d.true_start(clip("R1", "", None, tc="09:00:00:00"), cam, SYD)[1])

    def test_colliding_prefixes_resolved_by_path(self):
        cams = base_doc()["cameras"]
        self.assertEqual(d.camera_for(clip("MVI_0001", "/cards/OP1/MVI_0001.MP4", None), cams)["name"], "A cam")
        self.assertEqual(d.camera_for(clip("mvi_0001", "/cards/op2/mvi_0001.mp4", None), cams)["name"], "B cam")
        self.assertIsNone(d.camera_for(clip("MVI_0001", "/cards/op3/MVI_0001.MP4", None), cams))
        self.assertEqual(d.camera_for(clip("DJI_1001", "/anywhere/DJI_1001.MP4", None), cams)["name"], "Drone")


class Assign(unittest.TestCase):
    def run_assign(self, clips, doc=None):
        return d.assign_clips(doc or base_doc(), {"aroll": [c for c in clips if c["role"] == "aroll"],
                                                  "broll": [c for c in clips if c["role"] == "broll"]})

    def test_assignment_and_every_unassigned_reason(self):
        r = self.run_assign([
            clip("MVI_1001", "/op1/MVI_1001.MP4", "2026-03-12T23:00:00Z"),            # 10:00 local day1 -> morning
            clip("MVI_1002", "/op1/MVI_1002.MP4", "2026-03-13T05:00:00Z"),            # 16:00 local: after the workshop
            clip("MVI_1003", "/op1/MVI_1003.MP4", "2026-03-20T00:00:00Z"),            # not a project day
            clip("XYZ_1004", "/elsewhere/XYZ_1004.MP4", "2026-03-12T23:00:00Z"),      # no camera
            clip("MVI_1005", "/op1/MVI_1005.MP4", None),                              # no time
            clip("MVI_1006", "/op1/MVI_1006.MP4", "2026-03-13T01:50:00Z", dur=1200),  # 12:50, runs past 13:00
        ])
        self.assertEqual(r["MVI_1001"]["session"], "s-morning")
        self.assertEqual(r["MVI_1001"]["start_local"], "2026-03-13T10:00:00+11:00")
        self.assertTrue(r["MVI_1002"]["reason"].startswith("outside every session for A cam"))
        self.assertTrue(r["MVI_1003"]["reason"].startswith("day not in days (2026-03-20)"))
        self.assertTrue(r["XYZ_1004"]["reason"].startswith("no camera matches"))
        self.assertTrue(r["MVI_1005"]["reason"].startswith("no time"))
        self.assertIsNone(r["MVI_1005"]["session"])
        self.assertEqual(r["MVI_1006"]["session"], "s-morning")
        self.assertTrue(r["MVI_1006"]["spans_boundary"])
        self.assertEqual(r["MVI_1006"]["ends_in"], "s-workshop")

    def test_parallel_sessions_take_only_their_own_cameras(self):
        r = self.run_assign([
            clip("MVI_2001", "/op1/MVI_2001.MP4", "2026-03-13T22:00:00Z"),   # A cam 09:00 day2 -> Track A
            clip("MVI_2002", "/op2/MVI_2002.MP4", "2026-03-14T09:00:00Z"),   # B cam, local clock -> Track B
            clip("DJI_2003", "/drone/DJI_2003.MP4", "2026-03-14T10:00:00Z"), # drone an hour fast -> 09:00 -> Track B
        ])
        self.assertEqual(r["MVI_2001"]["session"], "s-track-a")
        self.assertEqual(r["MVI_2002"]["session"], "s-track-b")
        self.assertEqual(r["DJI_2003"]["session"], "s-track-b")
        self.assertEqual(r["DJI_2003"]["time_source"], "creation_time_local-3600s")
        self.assertEqual(r["MVI_2002"]["register_candidates"], ["stillness"])

    def test_a_clip_in_two_sessions_is_ambiguous_never_picked(self):
        doc = base_doc()
        del doc["sessions"][3]["cameras"]   # a conflict validate rejects; assign must still not choose
        r = self.run_assign([clip("MVI_2001", "/op1/MVI_2001.MP4", "2026-03-13T22:00:00Z")], doc)
        self.assertIsNone(r["MVI_2001"]["session"])
        self.assertEqual(r["MVI_2001"]["reason"], "ambiguous: s-track-a, s-track-b")

    def test_feeds_and_registers_follow_the_operator_line(self):
        r = self.run_assign([
            clip("MVI_3001", "/op1/MVI_3001.MP4", "2026-03-13T02:30:00Z"),                  # Op 1 in the workshop
            clip("MVI_3002", "/op2/MVI_3002.MP4", "2026-03-13T13:30:00Z"),                  # Op 2, local clock
            clip("DJI_3003", "/d/DJI_3003.MP4", "2026-03-13T14:30:00Z"),                    # Op 2's drone
            clip("MVI_3004", "/op1/MVI_3004.MP4", "2026-03-13T02:30:00Z", role="aroll"),    # interview: no registers
        ])
        self.assertEqual(r["MVI_3001"]["session"], "s-workshop")
        self.assertEqual(r["MVI_3001"]["feeds_candidates"], ["feature"])
        self.assertEqual(r["MVI_3001"]["register_candidates"], ["focus"])
        self.assertEqual(r["MVI_3002"]["feeds_candidates"], ["before-after"])
        self.assertEqual(r["MVI_3002"]["register_candidates"], ["laughter"])   # Op 2's line names none: the session's
        self.assertEqual(r["DJI_3003"]["feeds_candidates"], ["before-after"])
        self.assertEqual(r["MVI_3004"]["register_candidates"], [])
        doc = base_doc()
        doc["cameras"][0]["operator"] = "Op 3"   # no capture line for this operator: the session's feeds
        r = self.run_assign([clip("MVI_3001", "/op1/MVI_3001.MP4", "2026-03-13T02:30:00Z")], doc)
        self.assertEqual(r["MVI_3001"]["feeds_candidates"], ["broll"])

    def test_clock_table_shows_early_clips(self):
        inv = {"broll": [clip("DJI_4001", "/d/DJI_4001.MP4", "2026-03-13T08:00:00Z"),   # 07:00 after offset: early
                         clip("DJI_4002", "/d/DJI_4002.MP4", "2026-03-13T11:00:00Z")]}
        rows = d.clock_table(base_doc(), inv)
        self.assertEqual(rows[0]["camera"], "Drone")
        self.assertEqual((rows[0]["clips"], rows[0]["before_first_session"], rows[0]["first"]), (2, 1, "07:00:00"))


class Items(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_plan_creates_then_appends_without_touching_existing_bytes(self):
        doc = base_doc()
        first = d.stubs(doc, [])
        self.assertEqual([i["id"] for i in first], ["feature-01", "feature-02", "before-after-01", "broll-01"])
        self.assertEqual(first[2]["sessions"], ["s-workshop"])   # its only feeder, through a capture line
        self.assertEqual(first[3]["sessions"], ["s-workshop"])
        self.assertEqual(first[0]["sessions"], ["s-workshop"])
        text = "# hand notes\nitems:\n  - id: feature-01   # kept\n    type: feature\n    status: cut\n    notes: mine\n"
        items = yaml.safe_load(text)["items"]
        doc["deliverable_types"][0]["count"] = 3
        new = d.stubs(doc, items)
        self.assertEqual([i["id"] for i in new], ["feature-02", "feature-03", "before-after-01", "broll-01"])
        out = d.append_items(text, items, new)
        self.assertTrue(out.startswith(text))
        self.assertEqual(yaml.safe_load(out)["items"], items + new)

    def test_plan_refuses_unparsable_or_unsafe_files(self):
        (self.tmp / "items.yaml").write_text("items: [\n  broken")
        with self.assertRaises(SystemExit):
            d.load_items(self.tmp)
        text = "items:\n  - id: a\n    type: feature\nlater: 1\n"
        with self.assertRaises(SystemExit):
            d.append_items(text, yaml.safe_load(text)["items"], [{"id": "b", "type": "feature"}])

    def test_set_changes_one_entry_and_no_other_byte(self):
        text = ("items:\n  - id: a\n    type: feature\n    status: planned   # comment kept\n    notes: ''\n\n"
                "  - id: b\n    type: feature\n    status: planned\n    notes: ''\n  # trailing comment\n")
        items = yaml.safe_load(text)["items"]
        upd = d.parse_set(["status=cut", "people=fac-a", "notes=first pass"], base_doc())
        out = d.set_item(text, items, "b", upd)
        self.assertTrue(out.startswith(text.split("  - id: b")[0]))
        self.assertTrue(out.endswith("  # trailing comment\n"))
        b = yaml.safe_load(out)["items"][1]
        self.assertEqual((b["status"], b["people"], b["notes"]), ("cut", ["fac-a"], "first pass"))
        with self.assertRaises(SystemExit):
            d.parse_set(["status=done"], base_doc())
        with self.assertRaises(SystemExit):
            d.parse_set(["people=nobody"], base_doc())
        with self.assertRaises(SystemExit):
            d.set_item(text, items, "zzz", upd)

    def test_planned_stubs_sharing_a_session_can_each_be_set(self):
        # Regression: stubs fed by one session shared a list, were dumped as &id001/*id001, and `set` on the
        # anchoring entry left the later aliases dangling (a real 65-item plan).
        doc = base_doc()
        doc["deliverable_types"][0]["count"] = 4
        text = d.append_items(None, [], d.stubs(doc, []))
        self.assertNotIn("&id", text)
        self.assertNotIn("*id", text)
        for iid in ("feature-01", "feature-02", "feature-03"):
            items = yaml.safe_load(text)["items"]
            text = d.set_item(text, items, iid, d.parse_set(["status=footage", "people=fac-a"], doc))
        got = {i["id"]: i["status"] for i in yaml.safe_load(text)["items"]}
        self.assertEqual([got[f"feature-0{n}"] for n in (1, 2, 3, 4)], ["footage", "footage", "footage", "planned"])

    def test_set_refuses_cleanly_when_an_anchor_ties_entries_together(self):
        text = ("items:\n  - id: a\n    type: feature\n    sessions: &s [s-workshop]\n"
                "  - id: b\n    type: feature\n    sessions: *s\n")
        items = yaml.safe_load(text)["items"]
        with self.assertRaises(SystemExit) as caught:
            d.set_item(text, items, "a", d.parse_set(["status=cut"], base_doc()))
        self.assertIn("anchor", str(caught.exception))

    def test_status_evidence(self):
        items = [{"id": "feature-01", "type": "feature", "status": "cut"},
                 {"id": "feature-02", "type": "feature", "status": "delivered"},
                 {"id": "broll-01", "type": "broll", "status": "footage"}]
        rows, claims = d.status_rows(base_doc(), items, self.tmp)
        self.assertEqual({c["item"] for c in claims}, {"feature-01", "feature-02"})
        f1 = self.tmp / "items/feature-01/editorial"
        f1.mkdir(parents=True)
        (f1 / "cut-v1.json").write_text("{}")
        f2 = self.tmp / "items/feature-02/editorial/renders"
        f2.mkdir(parents=True)
        (f2.parent / "cut-v3.json").write_text("{}")
        (f2 / "f2-v3.mp4").write_bytes(b"")
        (f2 / "f2-v3.master.mp4").write_bytes(b"")
        (f2 / "f2-v3.master.manifest.json").write_text(json.dumps({"rule": {"status": "broken"}}))
        rows, claims = d.status_rows(base_doc(), items, self.tmp)
        self.assertEqual([c["problem"] for c in claims], ["delivered: delivery loudness rule broken on every master"])
        (f2 / "f2-v3.master.manifest.json").write_text(json.dumps({"rule": {"status": "honoured"}}))
        rows, claims = d.status_rows(base_doc(), items, self.tmp)
        self.assertEqual(claims, [])
        self.assertEqual(rows[0]["delivered"], 1)
        md = d.status_md(base_doc(), items, rows, claims, None)
        self.assertIn("- [x] feature-02 · delivered", md)
        self.assertIn("1 more not yet listed", md)   # before-after lists 0 of 1
        self.assertNotIn(str(self.tmp), md)

    def test_scope_filters_types(self):
        doc = base_doc()
        doc["deliverable_types"][0]["scope"] = ["editor"]
        rows, _ = d.status_rows(doc, [], self.tmp, scope="editor")
        self.assertEqual([r["type"] for r in rows], ["feature"])

    def test_item_dir_links_shared_dirs_and_replaces_nothing(self):
        (self.tmp / "out").mkdir()
        (self.tmp / "work").mkdir()
        made = d.item_dir(self.tmp, "feature-01")
        self.assertEqual(len(made), 2)
        self.assertTrue((self.tmp / "items/feature-01/out").is_symlink())
        self.assertTrue((self.tmp / "items/feature-01/editorial").is_dir())
        self.assertEqual(d.item_dir(self.tmp, "feature-01"), [])

    def test_cli_validate_exit_codes(self):
        (self.tmp / "deliverables.yaml").write_text((REPO / "deliverables.example.yaml").read_text())
        self.assertEqual(d.main(["validate", "--project", str(self.tmp)]), 0)
        bad = yaml.safe_load((REPO / "deliverables.example.yaml").read_text())
        bad["sessions"][4].pop("cameras")   # Track A loses its camera list: overlaps Track B
        (self.tmp / "deliverables.yaml").write_text(yaml.safe_dump(bad))
        self.assertEqual(d.main(["validate", "--project", str(self.tmp)]), 2)


class FakeClip:
    def __init__(self, name):
        self.name = name

    def GetName(self):
        return self.name


class FakeFolder:
    def __init__(self, name):
        self.name, self.subs, self.clips = name, [], []

    def GetName(self):
        return self.name

    def GetSubFolderList(self):
        return list(self.subs)

    def GetClipList(self):
        return list(self.clips)


class FakePool:
    """MoveClips reports success and silently skips the clips named in `drop`, as the real one can."""

    def __init__(self, drop=()):
        self.root = FakeFolder("Master")
        self.drop = set(drop)

    def GetRootFolder(self):
        return self.root

    def AddSubFolder(self, parent, name):
        f = FakeFolder(name)
        parent.subs.append(f)
        return f

    def MoveClips(self, clips, target):
        for c in clips:
            if c.name in self.drop:
                continue
            for f in self._all(self.root):
                if c in f.clips:
                    f.clips.remove(c)
            target.clips.append(c)
        return True

    def _all(self, f):
        yield f
        for s in f.subs:
            yield from self._all(s)


class Bins(unittest.TestCase):
    def setUp(self):
        self.assign = {"MVI_1001": {"session": "s-morning"}, "MVI_1002": {"session": "s-workshop"},
                       "MVI_1003": {"session": "s-workshop"}, "MVI_1004": {"session": None, "reason": "no time"}}
        self.names = ["MVI_1001.MP4", "MVI_1002.MP4", "MVI_1003.MP4", "MVI_1004.MP4", "Other.mov"]

    def test_plan(self):
        plan = d.plan_bins(base_doc(), self.assign, self.names)
        self.assertIn("Retreat/day1/s-morning Morning", plan["folders"])
        self.assertIn("Retreat/B-roll package/focus", plan["folders"])
        self.assertEqual(plan["moves"]["Retreat/day1/s-workshop Workshop"], ["MVI_1002.MP4", "MVI_1003.MP4"])
        self.assertEqual(plan["unassigned"], ["MVI_1004.MP4"])
        self.assertEqual(plan["not_in_inventory"], ["Other.mov"])

    def test_apply_then_verify_catches_a_move_that_reported_success(self):
        pool = FakePool(drop={"MVI_1003.MP4"})
        source = pool.AddSubFolder(pool.root, "Footage")
        source.clips = [FakeClip(n) for n in self.names]
        plan = d.plan_bins(base_doc(), self.assign, self.names)
        d.apply_bins(pool, source, plan)
        rep = d.verify_bins(pool, plan)
        self.assertFalse(rep["ok"])
        self.assertEqual(rep["bins"]["Retreat/day1/s-workshop Workshop"], dict(planned=2, verified=1, missing=["MVI_1003.MP4"]))
        self.assertEqual(rep["bins"]["Retreat/day1/s-morning Morning"]["verified"], 1)
        self.assertEqual(rep["folders_missing"], [])
        self.assertEqual({c.name for c in source.clips}, {"MVI_1003.MP4", "MVI_1004.MP4", "Other.mov"})
        pool.drop.clear()
        d.apply_bins(pool, source, plan)   # re-running creates no duplicate bins and finishes the move
        self.assertTrue(d.verify_bins(pool, plan)["ok"])
        day1 = d._walk(pool, "Retreat/day1", create=False)
        self.assertEqual(len(day1.GetSubFolderList()), 2)


if __name__ == "__main__":
    unittest.main()
