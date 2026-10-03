"""Talk analysis invariants: chunk grouping, float gain, sentences, measured room, gates, honest verdicts,
ranking, reports, the cut export and the Resolve plan/verify. No network, no models, no Resolve."""
import csv
import datetime as dt
import json
import random
import shutil
import subprocess
import tempfile
import unittest
import wave
from pathlib import Path

import numpy as np

from videoeasy import talks, talks_resolve
from videoeasy.evalrun import INVALID, OK, UNCHECKED

FFMPEG = shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None
T = dt.datetime(2026, 3, 6, 9, 0, 0)


def chunk(name, start, dur, session="06"):
    return talks.Chunk(path=f"/x/{name}", name=name, session=session, start=start, duration_s=dur)


class ChunkGrouping(unittest.TestCase):
    def test_default_pattern_parses_dji_names(self):
        self.assertEqual(talks.parse_name("DJI_06_20260306_090000.WAV"), ("06", T))
        self.assertEqual(talks.parse_name("dji_06_20260306_090000.wav"), ("06", T))
        self.assertIsNone(talks.parse_name("TX01_0001.WAV"))

    def test_custom_pattern(self):
        pat = r"^MIC(?P<session>\d+)-(?P<date>\d{8})-(?P<time>\d{6})\.wav$"
        self.assertEqual(talks.parse_name("MIC2-20260306-090000.wav", pat), ("2", T))

    def test_contiguous_chunks_are_ordered(self):
        a = chunk("DJI_06_20260306_090000.WAV", T, 1846.0)
        b = chunk("DJI_06_20260306_093046.WAV", T + dt.timedelta(seconds=1846), 1846.0)
        c = chunk("DJI_06_20260306_100132.WAV", T + dt.timedelta(seconds=3693), 600.0)   # 1 s jitter: whole-second names
        g = talks.group_contiguous([c, a, b])
        self.assertEqual([x.name for x in g["chunks"]], [a.name, b.name, c.name])
        self.assertEqual([x["gap_s"] for x in g["gaps"]], [0.0, 1.0])

    def test_gap_is_refused_and_named(self):
        a = chunk("DJI_06_20260306_090000.WAV", T, 1846.0)
        b = chunk("DJI_06_20260306_094000.WAV", T + dt.timedelta(seconds=2400), 100.0)
        with self.assertRaises(talks.TalkError) as e:
            talks.group_contiguous([a, b])
        self.assertIn("not contiguous", str(e.exception))
        self.assertIn("DJI_06_20260306_094000.WAV", str(e.exception))

    def test_two_sessions_refused(self):
        with self.assertRaises(talks.TalkError):
            talks.group_contiguous([chunk("a", T, 10, "06"), chunk("b", T + dt.timedelta(seconds=10), 10, "07")])

    def test_unparsed_names_need_trust(self):
        a = talks.Chunk(path="/x/a.wav", name="a.wav", duration_s=10)
        b = talks.Chunk(path="/x/b.wav", name="b.wav", duration_s=10)
        with self.assertRaises(talks.TalkError):
            talks.group_contiguous([a, b])
        self.assertEqual(talks.group_contiguous([a, b], assume_contiguous=True)["method"], "given order (unchecked)")
        self.assertEqual(talks.group_contiguous([a])["method"], "single chunk")

    def test_find_session_by_number_and_by_name(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            (d / "Left").mkdir()
            (d / "Right").mkdir()
            for n in ("DJI_06_20260306_090000.WAV", "DJI_06_20260306_093046.WAV", "DJI_07_20260306_120000.WAV"):
                (d / "Left" / n).touch()
            (d / "Right" / "DJI_12_20260306_090010.WAV").touch()
            self.assertEqual([c.name for c in talks.find_session(d, "6")], ["DJI_06_20260306_090000.WAV", "DJI_06_20260306_093046.WAV"])
            self.assertEqual([c.name for c in talks.find_session(d, "DJI_06_20260306_093046.WAV")], ["DJI_06_20260306_093046.WAV"])
            (d / "Right" / "DJI_06_20260306_140000.WAV").touch()
            with self.assertRaises(talks.TalkError):
                talks.find_session(d, "06")       # the same number in two folders: name the folder


class GainMath(unittest.TestCase):
    def test_gain_targets_the_higher_peak(self):
        g, why = talks.plan_gain(2.0, 1.5)
        self.assertAlmostEqual(g, -1.0 - 20 * np.log10(2.0), places=6)
        self.assertEqual(why, "talk sample peak")
        g, why = talks.plan_gain(1.0, 1.2)
        self.assertAlmostEqual(g, -1.0 - 20 * np.log10(1.2), places=6)
        self.assertIn("resampled", why)

    def test_silence_refused(self):
        with self.assertRaises(talks.TalkError):
            talks.plan_gain(0.0, 0.0)

    @unittest.skipUnless(FFMPEG, "ffmpeg not installed")
    def test_float_overs_survive_and_working_file_does_not_clip(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            src = d / "DJI_01_20260306_090000.WAV"
            # ffmpeg's sine is 1/8 full scale; +24 dB puts the float peak near +5.9 dBFS
            subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=48000:duration=2",
                            "-af", "volume=24dB", "-c:a", "pcm_f32le", str(src)], check=True)
            before = src.stat().st_mtime, src.stat().st_size
            doc = talks.prepare([talks.Chunk(path=str(src), name=src.name, session="01", start=T)], d / "out")
            self.assertEqual(doc["status"], OK)
            c = doc["chunks"][0]
            self.assertGreater(c["peak_dbfs"], 5.0)
            self.assertGreater(c["overs_samples"], 0)
            self.assertAlmostEqual(doc["gain_db"], -1.0 - doc["talk_peak_dbfs"], places=2)
            self.assertEqual(doc["working"]["clipped_samples"], 0)
            with wave.open(doc["working"]["path"]) as w:
                self.assertEqual((w.getframerate(), w.getnchannels(), w.getsampwidth()), (16000, 1, 2))
                x = np.frombuffer(w.readframes(w.getnframes()), dtype="<i2")
            self.assertLess(int(np.abs(x.astype(int)).max()), 32767)
            self.assertAlmostEqual(20 * np.log10(np.abs(x).max() / 32768), -1.0, delta=0.1)
            self.assertEqual((src.stat().st_mtime, src.stat().st_size), before)   # the original is never written
            self.assertFalse((d / (src.name + ".sha256")).exists())
            again = talks.prepare([talks.Chunk(path=str(src), name=src.name, session="01", start=T)], d / "out")
            self.assertEqual(again["key"], doc["key"])                            # unchanged input: cached


def words(spec):
    """[(text, start, end), ...] -> word dicts."""
    return [dict(w=t, s=s, e=e) for t, s, e in spec]


class Sentences(unittest.TestCase):
    def test_punctuation_and_pauses_close_sentences(self):
        w = words([("We", 0.0, 0.2), ("began.", 0.25, 0.6), ("Then", 0.7, 0.9), ("nothing", 0.95, 1.3),
                   ("grew", 2.2, 2.5), ("back", 2.55, 2.9)])
        s = talks.build_sentences(w)
        self.assertEqual([x["text"] for x in s], ["We began.", "Then nothing", "grew back"])
        self.assertEqual([x["id"] for x in s], ["S0001", "S0002", "S0003"])
        self.assertAlmostEqual(s[1]["pause_after"], 0.9)
        self.assertIsNone(s[2]["pause_after"])

    def test_long_run_split_at_longest_pause(self):
        w = [dict(w=f"w{i}", s=i * 1.0, e=i * 1.0 + 0.5) for i in range(20)]
        w += [dict(w=f"x{i}", s=20.05 + i, e=20.55 + i) for i in range(20)]   # one 0.55 s pause, 19.5-20.05: under the pause rule
        s = talks.build_sentences(w, pause_s=0.6, max_s=30.0)
        self.assertEqual(len(s), 2)
        self.assertEqual(s[0]["words"][-1]["w"], "w19")
        self.assertTrue(all(x["t1"] - x["t0"] <= 30.0 for x in s))


def env_from(segments, n, floor=-60.0):
    e = np.full(n, floor, dtype=np.float32)
    for a, b, lvl in segments:
        e[a:b] = lvl
    return e


class RoomAndOffMic(unittest.TestCase):
    def test_edge_room_from_energy(self):
        env = env_from([(100, 200, -20.0), (230, 300, -20.0)], 400)
        self.assertEqual(talks.edge_room(env, 1.0, 2.0, -54.0), (1.0, 0.3))   # 1 s before, 0.3 s to the next speech
        self.assertEqual(talks.edge_room(env, 1.0, 2.0, -54.0, cap_s=0.5), (0.5, 0.3))
        self.assertEqual(talks.edge_room(env, 1.05, 2.0, -54.0)[0], 0.0)     # a loud frame right at the edge: no room

    def test_off_mic_and_noise_floor(self):
        # floor -60 everywhere else; the wearer at -20, a second voice at -38
        env = env_from([(0, 1000, -60.0), (100, 300, -20.0), (400, 600, -38.0), (700, 900, -20.0)], 1000)
        s = [dict(id="S0001", t0=1.0, t1=3.0, words=words([("a", 1.0, 3.0)])),
             dict(id="S0002", t0=4.0, t1=6.0, words=words([("b", 4.0, 6.0)])),
             dict(id="S0003", t0=7.0, t1=9.0, words=words([("c", 7.0, 9.0)]))]
        lv = talks.measure_sentences(s, env)
        self.assertEqual(lv["status"], OK)
        self.assertEqual(lv["noise_floor_db"], -60.0)
        self.assertEqual([x["off_mic"] for x in s], [False, True, False])
        self.assertEqual(s[0]["head_room"], talks.ROOM_CAP_S if 1.0 >= talks.ROOM_CAP_S else 1.0)

    def test_room_unmeasurable_when_speech_sits_on_the_floor(self):
        env = env_from([(0, 500, -50.0), (100, 300, -45.0)], 500)
        s = [dict(id="S0001", t0=1.0, t1=3.0, words=words([("a", 1.0, 3.0)]))]
        lv = talks.measure_sentences(s, env)
        self.assertEqual(lv["room_status"], UNCHECKED)
        self.assertIsNone(s[0]["head_room"])          # None, never 0: unmeasured is not tight


class Beats(unittest.TestCase):
    ids = [f"S{i:04d}" for i in range(1, 11)]

    def test_valid(self):
        b, r = talks.validate_beats([dict(title="A", first="S0001", last="S0004"), dict(title="B", first="S0005", last="S0010")], self.ids)
        self.assertEqual(r, [])
        self.assertEqual([x["n"] for x in b], [1, 2])

    def test_gap_overlap_unknown_and_incomplete_are_invalid(self):
        for beats in ([dict(title="A", first="S0001", last="S0004"), dict(title="B", first="S0006", last="S0010")],
                      [dict(title="A", first="S0001", last="S0005"), dict(title="B", first="S0005", last="S0010")],
                      [dict(title="A", first="S0001", last="S0099")],
                      [dict(title="A", first="S0001", last="S0008")],
                      [dict(title="", first="S0001", last="S0010")], [], "nope"):
            b, r = talks.validate_beats(beats, self.ids)
            self.assertIsNone(b, beats)
            self.assertTrue(r)

    def test_windows_and_merge(self):
        sents = [dict(id=f"S{i:04d}", t0=i * 5.0, t1=i * 5.0 + 4, text="word " * 40, off_mic=False) for i in range(1, 61)]
        ids = [s["id"] for s in sents]
        wins = talks.windows_for(sents, budget_tokens=1500, overlap=5)
        self.assertGreater(len(wins), 1)
        self.assertEqual(wins[0][0], 0)
        self.assertEqual(wins[-1][1], len(sents))
        parts = []
        for lo, hi in wins:                       # each window answers with beats of 7 sentences covering exactly it
            sub = ids[lo:hi]
            parts.append(((lo, hi), [dict(title="t", first=sub[k], last=sub[min(k + 6, len(sub) - 1)]) for k in range(0, len(sub), 7)]))
        merged = talks.merge_windows(parts, ids)
        b, r = talks.validate_beats(merged, ids)
        self.assertEqual(r, [])


# ------------------------------------------------------------------ a synthetic analysed talk
def synthetic_talk(d: Path) -> None:
    """Two 30 s chunks; twelve sentences, one off-mic; one sentence crosses the join."""
    chunks = []
    for i, name in enumerate(("DJI_06_20260306_090000.WAV", "DJI_06_20260306_090030.WAV")):
        chunks.append(dict(path=f"/media/mic/{name}", name=name, session="06", start=(T + dt.timedelta(seconds=30 * i)).isoformat(), duration_s=30.0,
                           channels=1, channel_used=1, sample_rate=48000, sha=f"sha{i}", peak_dbfs=-3.0, peak_linear=0.7, overs_samples=0, overs_s=0.0,
                           loudness=dict(lufs=-30.0, true_peak_dbtp=-2.5, status=OK), talk_t0=30.0 * i, talk_t1=30.0 * (i + 1)))
    prep = dict(key="prep1", status=OK, notes=[], meta=dict(talk_id="talk-01", speaker="Speaker A", title="Opening talk"),
                contiguity=dict(method="start times in names", gaps=[]), chunks=chunks, talk_peak_dbfs=-3.0, resampled_peak_dbfs=-3.1, overs_samples=0,
                gain_db=2.0, gain_reason="talk sample peak", target_peak_dbfs=-1.0,
                working=dict(path=str(d / "work.wav"), sample_rate=16000, duration_s=60.0, peak_dbfs=-1.0, clipped_samples=0, sha="w1"))
    spec = [  # id, t0, t1, text, head, tail, off_mic
        ("S0001", 0.5, 4.0, "Nobody tells you how heavy silence gets.", 0.5, 0.4, False),
        ("S0002", 4.4, 9.0, "I sat with that for years before I said it out loud.", 0.4, 0.3, False),
        ("S0003", 9.3, 12.0, "And then the group changed it.", 0.3, 0.05, False),
        ("S0004", 12.05, 15.0, "Can you say more about that?", 0.05, 0.5, True),
        ("S0005", 15.5, 22.0, "The first night we sat around the fire and nobody spoke.", 0.5, 0.4, False),
        ("S0006", 22.4, 27.0, "By the third night every man had said something true.", 0.4, 0.5, False),
        ("S0007", 27.5, 33.0, "That is what I came back for, every single year.", 0.5, 0.6, False),
        ("S0008", 33.6, 38.0, "It is not about fixing anything.", 0.6, 0.4, False),
        ("S0009", 38.4, 44.0, "It is about being witnessed by people who will not look away.", 0.4, 0.5, False),
        ("S0010", 44.5, 48.0, "So that is the work.", 0.5, 0.5, False),
        ("S0011", 48.5, 54.0, "Come as you are and leave a little lighter.", 0.5, 0.6, False),
        ("S0012", 54.6, 58.0, "Thank you.", 0.6, 1.0, False),
    ]
    sents = []
    for sid, t0, t1, text, head, tail, off in spec:
        toks = text.split()
        step = (t1 - t0) / len(toks)
        ws = [dict(w=tok, s=round(t0 + k * step, 3), e=round(t0 + (k + 1) * step - 0.02, 3)) for k, tok in enumerate(toks)]
        sents.append(dict(id=sid, t0=t0, t1=t1, text=text, words=ws, head_room=head, tail_room=tail, level_db=-40.0 if off else -25.0, off_mic=off))
    (d / "prepare.json").write_text(json.dumps(prep))
    (d / "transcript.json").write_text(json.dumps(dict(key="tr1", status=OK, model="whisper", language="en", segments=[{}] * 12, dropped=[])))
    (d / "sentences.json").write_text(json.dumps(dict(key="sent1", status=OK, sentences=sents, counts=dict(sentences=12, off_mic=1),
                                                      levels=dict(noise_floor_db=-60.0, speech_db=-22.0, wearer_db=-25.0, quiet_db=-54.0, reason=None))))


BEATS_ANSWER = {"beats": [dict(title="The weight of silence", first="S0001", last="S0004", summary="carrying it alone", energy="low", turn="opens cold"),
                          dict(title="Around the fire", first="S0005", last="S0009", summary="what the nights did", energy="high", turn="the group"),
                          dict(title="The invitation", first="S0010", last="S0012", summary="the ask", energy="medium", turn="to the audience")]}

BITES_ANSWERS = {
    "The weight of silence": {"candidates": [
        dict(kind="hook", first="S0001", last="S0001", why="cold open", register="grief", cover_brief="face"),
        dict(kind="bite", first="S0001", last="S0002", why="admission", register="grief", cover_brief="a man alone at dawn"),
        dict(kind="bite", first="S0003", last="S0004", why="turn", register="connection", cover_brief="the circle"),       # off-mic inside
        dict(kind="hook", first="S0001", last="S0001", why="again", register="grief", cover_brief="face"),                # duplicate span
        dict(kind="bite", first="S0002", last="S0002", why="short", register="nonsense", cover_brief="x")]},             # bad register
    "Around the fire": {"candidates": [
        dict(kind="bite", first="S0005", last="S0007", why="the nights", register="connection", cover_brief="fire at night, faces in firelight"),
        dict(kind="bite", first="S0006", last="S0007", why="overlaps", register="connection", cover_brief="fire"),
        dict(kind="bite", first="S0008", last="S0009", why="witnessed", register="connection", cover_brief="men listening"),
        dict(kind="hook", first="S0003", last="S0003", why="outside beat", register="none", cover_brief="x")]},
    "The invitation": {"candidates": [
        dict(kind="closer", first="S0010", last="S0011", why="lands", register="joy", cover_brief="the group walking out"),
        dict(kind="hook", first="S0010", last="S0010", why="dangling", register="none", cover_brief="face")]},
}


def verdict_for(text):
    if "witnessed" in text:
        raise TimeoutError("model did not answer")
    if text.startswith("Nobody"):
        return dict(standalone=3, refers_to_unsaid=[], strength=3 if text.endswith("gets.") else 2, reason="clear and felt")
    if "fire" in text:
        return dict(standalone=3, refers_to_unsaid=[], strength=3, reason="vivid")
    if text.startswith("By the third"):
        return dict(standalone=2, refers_to_unsaid=["the nights"], strength=3, reason="needs the first night")
    if text.startswith("So that"):
        return dict(standalone=1, refers_to_unsaid=["the work"], strength=2, reason="refers back")
    return dict(standalone=9, strength=1, reason="out of range")     # invalid


def fake_chat(prompt):
    if "Divide " in prompt:
        return BEATS_ANSWER
    if "Propose the candidates" in prompt:
        title = prompt.split('"')[1]
        return BITES_ANSWERS[title]
    if "meeting them cold" in prompt:
        return verdict_for(prompt.split('\n\n"')[1].split('"\n\n')[0])
    raise AssertionError("unexpected prompt")


VOCAB = ["grief", "connection", "joy"]


class Gate(unittest.TestCase):
    def setUp(self):
        self.s = [dict(id="S0001", t0=0.0, t1=4.0, text="This changed everything.", head_room=0.1, tail_room=0.4, off_mic=False),
                  dict(id="S0002", t0=4.2, t1=9.0, text="We stayed.", head_room=0.2, tail_room=0.05, off_mic=False),
                  dict(id="S0003", t0=9.1, t1=60.0, text="Long.", head_room=None, tail_room=None, off_mic=False)]
        self.by = {x["id"]: x for x in self.s}
        self.order = {x["id"]: i for i, x in enumerate(self.s)}
        self.beat = dict(n=1, first="S0001", last="S0003")

    def g(self, **kw):
        c = dict(kind="bite", first="S0001", last="S0002", why="w", register="none", cover_brief="trees")
        c.update(kw)
        return talks.gate(c, self.by, self.order, self.beat, ["grief"])

    def test_flags(self):
        c = self.g()
        self.assertEqual(c["status"], OK)
        self.assertEqual(c["flags"], ["dangling_start:this", "tight_in", "tight_out"])
        self.assertEqual(c["id"], "b-S0001-S0002")

    def test_durations_by_kind(self):
        self.assertEqual(self.g(kind="hook")["status"], OK)                      # 9 s: inside the hook limits (3-12 s)
        self.assertEqual(self.g(kind="hook", last="S0003")["status"], talks.EXCLUDED)   # 60 s
        self.assertEqual(self.g(kind="hook", last="S0001")["status"], OK)        # 4 s hook
        c = self.g(first="S0003", last="S0003")
        self.assertEqual(c["status"], talks.EXCLUDED)
        self.assertIn("over the bite maximum", c["reasons"][0])
        self.assertIn("room_unmeasured", c["flags"])

    def test_invalid(self):
        self.assertEqual(self.g(kind="montage")["status"], INVALID)
        self.assertEqual(self.g(first="S0002", last="S0001")["status"], INVALID)
        self.assertEqual(self.g(cover_brief="")["status"], INVALID)
        self.assertEqual(self.g(register="joy")["status"], INVALID)
        self.assertEqual(talks.gate(dict(kind="bite", first="S0001", last="S0002", register="joy", cover_brief="x"),
                                    self.by, self.order, self.beat, [])["register"], "none")   # no vocabulary: nothing to read against


class Pipeline(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        synthetic_talk(self.d)

    def tearDown(self):
        self.tmp.cleanup()

    def run_all(self, chat=fake_chat):
        b = talks.beats_stage(self.d, chat=chat, model="fake")
        bt = talks.bites_stage(self.d, VOCAB, chat=chat, model="fake")
        v = talks.verify_stage(self.d, chat=chat, model="fake")
        a = talks.report_stage(self.d)
        return b, bt, v, a

    def test_gates(self):
        b, bt, _, _ = self.run_all()
        self.assertEqual(b["status"], OK)
        c = {(x["kind"], x.get("first"), x.get("last")): x for x in bt["candidates"]}
        self.assertEqual(c[("bite", "S0003", "S0004")]["status"], talks.EXCLUDED)                 # off-mic sentence inside
        self.assertIn("off-mic", c[("bite", "S0003", "S0004")]["reasons"][0])
        self.assertIn("dangling_start:and", c[("bite", "S0003", "S0004")]["flags"])
        self.assertEqual(c[("bite", "S0002", "S0002")]["status"], INVALID)                        # register outside the vocabulary
        self.assertEqual(c[("hook", "S0003", "S0003")]["status"], INVALID)                        # outside its beat
        self.assertEqual(c[("hook", "S0010", "S0010")]["status"], OK)                             # a dangling start is a flag, not a gate
        self.assertIn("dangling_start:so", c[("hook", "S0010", "S0010")]["flags"])
        hooks = [x for x in bt["candidates"] if x["kind"] == "hook" and x.get("first") == "S0001"]
        self.assertEqual(len(hooks), 1)                                                            # the duplicate span is one candidate
        self.assertEqual(hooks[0]["also_proposed_as"], ["hook"])
        self.assertEqual(c[("bite", "S0005", "S0007")]["status"], OK)                             # 17.5 s bite across the chunk join

    def test_unverified_candidates_carry_no_score_and_no_rank(self):
        _, _, v, a = self.run_all()
        self.assertEqual(v["status"], UNCHECKED)
        by = {x["id"]: x for x in a["candidates"] if x.get("id")}
        w = by["b-S0008-S0009"]
        self.assertEqual(w["verification"]["status"], UNCHECKED)
        self.assertNotIn("score", w)
        self.assertNotIn("rank", w)
        self.assertIn("b-S0008-S0009", a["unchecked"])
        self.assertNotIn("b-S0008-S0009", a["ranked"]["bite"])
        self.assertEqual(a["status"], UNCHECKED)
        md = (self.d / "analysis.md").read_text()
        self.assertIn("Not verified (no score, not ranked)", md)

    def test_ranking_overlap_and_determinism(self):
        _, _, _, a = self.run_all()
        self.assertEqual(a["ranked"]["bite"][0], "b-S0005-S0007")     # 6/6
        by = {x["id"]: x for x in a["candidates"] if x.get("id")}
        self.assertEqual(by["b-S0006-S0007"]["superseded_by"], "b-S0005-S0007")
        self.assertEqual(a["ranked"]["hook"], ["h-S0001-S0001", "h-S0010-S0010"])   # 6/6, then the flagged 3/6
        cands = [dict(x) for x in a["candidates"]]
        for x in cands:
            x.pop("rank", None)
            x.pop("superseded_by", None)
            x.pop("score", None)
        orders = set()
        for seed in range(5):
            shuffled = [dict(x) for x in cands]
            random.Random(seed).shuffle(shuffled)
            orders.add(json.dumps(talks.rank(shuffled)))
        self.assertEqual(len(orders), 1)

    def test_rerun_does_no_model_work(self):
        self.run_all()
        calls = []

        def counting(p):
            calls.append(p)
            return fake_chat(p)
        talks.beats_stage(self.d, chat=counting, model="fake")
        talks.bites_stage(self.d, VOCAB, chat=counting, model="fake")
        v = talks.verify_stage(self.d, chat=counting, model="fake")
        self.assertEqual(len(calls), 1)            # only the verdict that failed (unchecked) is asked again
        self.assertIn("witnessed", calls[0])
        self.assertEqual(v["model_calls"], 1)

    def test_invalid_beats_stop_the_bites(self):
        bad = lambda p: {"beats": [dict(title="A", first="S0001", last="S0005")]} if "Divide " in p else fake_chat(p)  # noqa: E731
        b = talks.beats_stage(self.d, chat=bad, model="fake")
        self.assertEqual(b["status"], INVALID)
        self.assertIsNone(b["beats"])
        bt = talks.bites_stage(self.d, VOCAB, chat=bad, model="fake")
        self.assertEqual(bt["status"], UNCHECKED)
        self.assertEqual(bt["candidates"], [])

    def test_transport_failure_is_unchecked_and_retried(self):
        def down(p):
            raise ConnectionError("refused")
        b = talks.beats_stage(self.d, chat=down, model="fake")
        self.assertEqual(b["status"], UNCHECKED)
        b = talks.beats_stage(self.d, chat=fake_chat, model="fake")           # no --overwrite needed: nothing was established
        self.assertEqual(b["status"], OK)

    def test_reports_carry_source_chunk_times(self):
        _, _, _, a = self.run_all()
        with open(self.d / "selects.csv") as f:
            rows = list(csv.DictReader(f))
        fire = [r for r in rows if r["id"] == "b-S0005-S0007"]
        self.assertEqual([r["part"] for r in fire], ["1/2", "2/2"])           # across the join: one row per chunk
        self.assertEqual(fire[0]["source_file"], "DJI_06_20260306_090000.WAV")
        self.assertEqual(fire[1]["source_file"], "DJI_06_20260306_090030.WAV")
        self.assertAlmostEqual(float(fire[0]["source_in_s"]), 15.5 - 0.3, places=3)   # handle = min(room 0.5, 0.3)
        self.assertAlmostEqual(float(fire[1]["source_in_s"]), 0.0, places=3)
        self.assertAlmostEqual(float(fire[1]["source_out_s"]), 33.0 + 0.3 - 30.0, places=3)
        md = (self.d / "analysis.md").read_text()
        self.assertIn("DJI_06_20260306_090030.WAV 00:00-00:03", md)
        self.assertIn("## Beats", md)
        self.assertIn("Cover: fire at night, faces in firelight", md)
        self.assertIn("spans_chunks", md)

    def test_export_cut(self):
        self.run_all()
        cut, by_chunk = talks.export_cut(self.d)
        titles = [b["title"] for b in cut["beats"]]
        self.assertTrue(titles[0].startswith("hook #1"))
        hook = cut["beats"][0]
        self.assertTrue(hook["notes"].startswith("no cover"))                 # a face brief stays bare in brollmatch
        fire = next(b for b in cut["beats"] if "Cover brief: fire" in b["notes"])
        self.assertEqual([p["clip"] for p in fire["audio"]], ["DJI_06_20260306_090000", "DJI_06_20260306_090030"])
        self.assertTrue(all(abs(p["duration_s"] - (p["out_s"] - p["in_s"])) < 1e-6 for p in fire["audio"]))
        self.assertEqual(fire["video"], [])
        seg = by_chunk["DJI_06_20260306_090030"]["segments"][0]
        self.assertGreaterEqual(seg["words"][0]["s"], 0.0)                     # chunk-local times
        self.assertLess(seg["words"][-1]["e"], 30.0)

    def test_cli_export_and_resolve_dry_run(self):
        self.run_all()
        self.assertEqual(talks.main(["export-cut", "--out", str(self.d)]), 0)
        cut = json.loads((self.d / "cut-selects.json").read_text())
        self.assertTrue(cut["beats"])
        self.assertTrue((self.d / "cut-selects.manifest.json").exists())
        with self.assertRaises(SystemExit):                                   # a second export is refused without --overwrite
            talks.main(["export-cut", "--out", str(self.d)])
        self.assertEqual(talks.main(["resolve", "--out", str(self.d), "--fps", "25"]), 0)
        pl = json.loads((self.d / "resolve-plan.json").read_text())
        self.assertEqual(pl["fps"], 25.0)
        self.assertEqual(pl["timeline"], "talk-01 selects")
        with self.assertRaises(SystemExit):                                   # --apply needs the project named
            talks.main(["resolve", "--out", str(self.d), "--apply", "--overwrite"])

    def test_report_refuses_to_overwrite_a_different_run(self):
        self.run_all(chat=lambda p: fake_chat(p) if "meeting them cold" not in p or "witnessed" not in p else verdict_for("fire"))
        prev = json.loads((self.d / "analysis.json").read_text())
        self.assertEqual(prev["status"], OK)
        sd = json.loads((self.d / "sentences.json").read_text())
        sd["key"] = "sent2"
        (self.d / "sentences.json").write_text(json.dumps(sd))
        with self.assertRaises(SystemExit):
            talks.beats_stage(self.d, chat=fake_chat, model="fake")


# ------------------------------------------------------------------ Resolve, with fakes
class FakeClip:
    def __init__(self, path):
        self.path = path

    def GetClipProperty(self, k):
        return self.path if k == "File Path" else None

    def GetName(self):
        return Path(self.path).name


class FakeFolder:
    def __init__(self, name):
        self.name, self.subs, self.clips = name, [], []

    def GetName(self):
        return self.name

    def GetSubFolderList(self):
        return list(self.subs)

    def GetClipList(self):
        return list(self.clips)


class FakeItem:
    def __init__(self, name, start, dur):
        self.name, self.start, self.dur = name, start, dur

    def GetName(self):
        return self.name

    def GetStart(self):
        return self.start

    def GetDuration(self):
        return self.dur


class FakeTimeline:
    def __init__(self, name, fps=25.0):
        self.name, self.fps, self.items, self.markers = name, fps, [], {}

    def GetName(self):
        return self.name

    def GetStartFrame(self):
        return 90000

    def GetSetting(self, k):
        return str(self.fps) if k == "timelineFrameRate" else None

    def GetTrackCount(self, kind):
        return 1 if kind == "audio" else 1

    def GetItemListInTrack(self, kind, i):
        return list(self.items) if kind == "audio" and i == 1 else []

    def AddMarker(self, frame, color, name, note, dur, custom):
        if frame in self.markers:
            return False
        self.markers[frame] = dict(color=color, name=name, note=note, duration=dur)
        return True

    def GetMarkers(self):
        return dict(self.markers)


class FakeMediaPool:
    def __init__(self, drop_last=False):
        self.root, self.current, self.timelines, self.drop_last = FakeFolder("Master"), None, [], drop_last

    def GetRootFolder(self):
        return self.root

    def AddSubFolder(self, parent, name):
        f = FakeFolder(name)
        parent.subs.append(f)
        return f

    def SetCurrentFolder(self, f):
        self.current = f
        return True

    def ImportMedia(self, paths):
        clips = [FakeClip(p) for p in paths]
        self.current.clips.extend(clips)
        return clips

    def CreateEmptyTimeline(self, name):
        tl = FakeTimeline(name)
        self.timelines.append(tl)
        return tl

    def AppendToTimeline(self, infos):
        tl = self.timelines[-1]
        if self.drop_last:
            infos = infos[:-1]           # a silent partial failure the read-back must catch
        for i in infos:
            tl.items.append(FakeItem(i["mediaPoolItem"].GetName(), i["recordFrame"], i["endFrame"] - i["startFrame"]))
        return [object()] * len(infos)


class FakeProject:
    def __init__(self, mp, name="Event 2026"):
        self.mp, self.name = mp, name

    def GetName(self):
        return self.name

    def GetMediaPool(self):
        return self.mp

    def GetTimelineCount(self):
        return len(self.mp.timelines)

    def GetTimelineByIndex(self, i):
        return self.mp.timelines[i - 1]

    def SetCurrentTimeline(self, tl):
        return True


class FakeResolve:
    def __init__(self, project, page="edit"):
        self.project, self.page, self.saved = project, page, 0

    def GetProjectManager(self):
        return self

    def GetCurrentProject(self):
        return self.project

    def SaveProject(self):
        self.saved += 1
        return True

    def GetCurrentPage(self):
        return self.page


class ResolvePlan(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        synthetic_talk(self.d)
        Pipeline.run_all(self)
        self.a = json.loads((self.d / "analysis.json").read_text())

    def tearDown(self):
        self.tmp.cleanup()

    def test_plan(self):
        pl = talks_resolve.plan(self.a, fps=24.0)
        self.assertEqual(pl["candidates"][0], "h-S0001-S0001")
        fire = [it for it in pl["items"] if it["cand"] == "b-S0005-S0007"]
        self.assertEqual(len(fire), 2)
        self.assertEqual(fire[1]["rec"], fire[0]["rec"] + fire[0]["ef"] - fire[0]["sf"])     # butted across the join
        recs = [m["frame"] for m in pl["markers"]]
        self.assertEqual(recs, sorted(set(recs)))                                               # one marker per frame, in order
        self.assertEqual(pl["markers"][0]["color"], "Red")
        self.assertIn("cover:", pl["markers"][0]["note"])

    def test_apply_verifies_by_readback(self):
        mp = FakeMediaPool()
        res = FakeResolve(FakeProject(mp))
        rep = talks_resolve.apply(self.a, "Event 2026", "talk-01 selects", "Talks/talk-01", resolve=res)
        self.assertEqual(rep["fps"], 25.0)                                # the live timeline's rate wins
        self.assertEqual(rep["verify"]["status"], OK, rep["verify"])
        self.assertEqual(rep["verify"]["items_verified"], rep["verify"]["items_planned"])
        self.assertEqual(sorted(rep["imported"]), ["DJI_06_20260306_090000.WAV", "DJI_06_20260306_090030.WAV"])
        self.assertEqual(res.saved, 1)

    def test_partial_append_is_caught(self):
        mp = FakeMediaPool(drop_last=True)
        rep = talks_resolve.apply(self.a, "Event 2026", "talk-01 selects", "Talks/talk-01", resolve=FakeResolve(FakeProject(mp)))
        self.assertEqual(rep["verify"]["status"], "broken")
        self.assertEqual(rep["verify"]["failures"][0]["what"], "item missing")

    def test_guards(self):
        mp = FakeMediaPool()
        with self.assertRaises(RuntimeError):
            talks_resolve.apply(self.a, "Other project", "x", "Talks/x", resolve=FakeResolve(FakeProject(mp)))
        with self.assertRaises(RuntimeError):
            talks_resolve.apply(self.a, "Event 2026", "x", "Talks/x", resolve=FakeResolve(FakeProject(mp), page=None))
        talks_resolve.apply(self.a, "Event 2026", "x", "Talks/x", resolve=FakeResolve(FakeProject(mp)))
        with self.assertRaises(RuntimeError):                              # an existing timeline is never replaced
            talks_resolve.apply(self.a, "Event 2026", "x", "Talks/x", resolve=FakeResolve(FakeProject(mp)))

    def test_verify_duration_mismatch(self):
        pl = talks_resolve.plan(self.a, fps=24.0)
        rb = dict(items=[dict(track=1, name=it["name"], rec=it["rec"], dur=(it["ef"] - it["sf"]) * 2) for it in pl["items"]],
                  markers=[dict(frame=m["frame"], name=m["name"]) for m in pl["markers"]])
        v = talks_resolve.verify(pl, rb)
        self.assertEqual(v["status"], "broken")
        self.assertTrue(all(f["what"] == "duration" for f in v["failures"]))


if __name__ == "__main__":
    unittest.main()
