"""Speech alignment as token edits, and the padded recheck states. No models, no ffmpeg."""
from __future__ import annotations

import unittest
from pathlib import Path

from videoeasy import speech

from filmfixture import TEST_FILM

LEX = speech.lexicon_for(TEST_FILM)


class Analyse(unittest.TestCase):
    def test_deleted_interior_negation_is_a_risk_despite_high_coverage(self):
        a = speech.analyse("we do not want to move these old fence posts today",
                           "we do want to move these old fence posts today".split())
        self.assertGreaterEqual(a["coverage"], 0.85)
        self.assertTrue(a["flag"])
        self.assertEqual([(r["cls"], r["kind"], r["token"]) for r in a["risks"]], [("polarity", "dropped", "not")])

    def test_inserted_interior_negation_is_a_risk_at_full_coverage(self):
        a = speech.analyse("we want to move these old fence posts today",
                           "we want to not move these old fence posts today".split())
        self.assertEqual(a["coverage"], 1.0)
        self.assertEqual(a["insertions"], ["not"])
        self.assertTrue(a["flag"])
        self.assertEqual(a["risks"][0]["kind"], "inserted")

    def test_contraction_polarity(self):
        a = speech.analyse("I can't see it from here", "I can see it from here".split())
        self.assertTrue(a["flag"])
        self.assertEqual([(r["cls"], r["kind"], r["token"]) for r in a["risks"]], [("polarity", "dropped", "not")])
        b = speech.analyse("we do not want", "we dont want".split())
        self.assertFalse(b["flag"])  # contractions and their apostrophe-less spellings agree with the expanded form

    def test_numbers_and_expanded_contractions_agree(self):
        self.assertFalse(speech.analyse("So today we're at the depot, five years", "so today we are at the depot 5 years".split())["flag"])

    def test_recheck_hears_an_edge_word_and_discounts_the_neighbour(self):
        pick = dict(id="b6p2", t0=203.2, t1=209.9, intended="Yeah, we opened the gate")
        first = speech.analyse(pick["intended"], "opened the gate".split())
        def transcribe(media, t0, t1, pad):  # 'which is actually' is the previous pick's tail, 'yeah we' sits on the edge
            ws = ["which", "is", "actually", "yeah", "we", "opened", "the", "gate"]
            return [{"w": w, "s": t0 - 1.0 + i * 0.35, "e": t0 - 1.0 + i * 0.35 + 0.3} for i, w in enumerate(ws)]
        r = speech.recheck(Path("r.mp4"), pick, first, None, None, None, transcribe=transcribe,
                           prev_words={"which", "is", "actually"})
        self.assertEqual(r["state"], "cleared_on_recheck")

    def test_modality_and_quantity(self):
        a = speech.analyse("I would like to try it again", "I like to try it again".split())
        self.assertEqual([r["cls"] for r in a["risks"]], ["modality"])
        b = speech.analyse("two months later", "ten months later".split())
        self.assertEqual([(r["cls"], r["token"], r["heard"]) for r in b["risks"]], [("quantity", "two", "ten")])

    def test_interjected_pronoun_is_not_escalated_but_an_inserted_number_is(self):
        a = speech.analyse("we found old bottles", "we found you know old bottles".split())
        self.assertEqual(a["insertions"], ["you", "know"])
        self.assertEqual(a["risks"], [])
        b = speech.analyse("it took months", "it took two months".split())
        self.assertEqual([r["cls"] for r in b["risks"]], ["quantity"])

    def test_pronoun_swap_is_an_actor_change(self):
        a = speech.analyse("we painted the shed", "they painted the shed".split())
        self.assertEqual([(r["cls"], r["token"], r["heard"]) for r in a["risks"]], [("actor", "we", "they")])

    def test_garbled_block_is_coverage_not_a_word_swap(self):
        a = speech.analyse("we work mostly in the hills across the valley",
                           "love practising slow building and sharing time with the valley wherever we go".split())
        self.assertTrue(a["flag"])
        self.assertEqual(a["risks"], [])
        self.assertLess(a["coverage"], 0.8)

    def test_name_alias(self):
        # a name from the film's lexicon is a critical token: fuzzy matching never touches it, only its alias does
        a = speech.analyse("you can see Rowan here", "you can see Rohan here".split(), lex=LEX)
        self.assertFalse(a["flag"])
        b = speech.analyse("you can see Rowan here", "you can see Avery here".split(), lex=LEX)
        self.assertEqual([(r["cls"], r["token"]) for r in b["risks"]], [("name", "rowan")])

    def test_alias_and_fuzzy_spelling_are_not_lost_words(self):
        a = speech.analyse("so much mulga and saltbush", "so much mulger and saltbsh".split(), lex=LEX)
        self.assertEqual(a["coverage"], 1.0)
        self.assertFalse(a["flag"])
        # without the film's lexicon the alias is a lost word (the fuzzy fallback still catches the near spelling)
        self.assertEqual(speech.analyse("so much mulga and saltbush", "so much mulger and saltbsh".split())["coverage"], 0.8)
        # the cut text may carry the transcript's spelling; the heard side the real one
        b = speech.analyse("so much Mulger here", "so much mulga here".split(), lex=LEX)
        self.assertEqual(b["coverage"], 1.0)

    def test_fuzzy_never_maps_onto_or_from_a_critical_token(self):
        a = speech.analyse("it was not there", "it was nut there".split())
        # 'nut' must not become 'not'
        self.assertTrue(a["flag"])
        self.assertEqual(a["risks"][0]["token"], "not")

    def test_boundary_leaks_are_not_internal_insertions(self):
        a = speech.analyse("we had more time to look around", "yeah we had more time to look around and so".split(),
                           prev_words={"yeah"}, next_words=set())
        self.assertEqual(a["insertions"], [])
        self.assertEqual(a["leak_head"], [])
        self.assertEqual(a["leak_tail"], ["and", "so"])
        self.assertTrue(a["flag"])  # two leaked words

    def test_nothing_heard(self):
        a = speech.analyse("Look at that colour.", [])
        self.assertEqual(a["coverage"], 0.0)
        self.assertEqual(a["omissions"], ["look", "at", "that", "colour"])
        self.assertTrue(a["flag"])

    def test_clean(self):
        a = speech.analyse("Look at that colour.", "look at that colour".split())
        self.assertFalse(a["flag"])
        self.assertEqual(a["ops"], [])


class Recheck(unittest.TestCase):
    PICK = dict(id="b15p1", t0=517.0, t1=518.5, intended="Look at that colour.")

    def _fake(self, render_words, source_words):
        def transcribe(media, t0, t1, pad):
            words = source_words if "source" in str(media) else render_words
            n = len(words)
            return [{"w": w, "s": t0 + 0.3 + i * (t1 - t0 - 0.6) / max(1, n), "e": t0 + 0.3 + (i + 0.5) * (t1 - t0 - 0.6) / max(1, n)}
                    for i, w in enumerate(words)]
        return transcribe

    def _first(self, heard):
        return speech.analyse(self.PICK["intended"], heard)

    def test_cleared_when_the_padded_pass_hears_it(self):
        r = speech.recheck(Path("r.mp4"), self.PICK, self._first([]), Path("source.MOV"), 76.0, 77.5,
                           transcribe=self._fake(["look", "at", "that", "colour"], ["look", "at", "that", "colour"]))
        self.assertEqual(r["state"], "cleared_on_recheck")
        self.assertEqual(r["first"]["reasons"], ["coverage 0.0"])
        self.assertEqual(r["second"]["reasons"], [])

    def test_confirmed_when_two_render_passes_agree_and_the_source_has_the_words(self):
        pick = dict(id="b11p3", t0=400.0, t1=405.0, intended="I'm not lifting the boards")
        first = speech.analyse(pick["intended"], "I'm lifting the boards".split())
        fake = self._fake("I'm lifting the boards".split(), "I'm not lifting the boards".split())
        lost = lambda *a: dict(status="ok", speech_windows=10, min_corr=0.02, median_corr=0.9, lost=[(402.5, 0.02)])
        r = speech.recheck(Path("r.mp4"), pick, first, Path("source.MOV"), 55.6, 71.2, transcribe=fake, waveform=lost)
        self.assertEqual(r["state"], "confirmed")
        self.assertIn("lost these words", r["action"])

    def test_waveform_overrules_two_asr_passes_when_every_speech_window_matches(self):
        # v8 (29 Sep 2026): a pick with two words missing on both render passes, clean on the source, and every
        # half-second of the source's speech in the render (min corr 0.78)
        pick = dict(id="b11p3", t0=400.0, t1=405.0, intended="I'm not lifting the boards")
        first = speech.analyse(pick["intended"], "I'm lifting the boards".split())
        fake = self._fake("I'm lifting the boards".split(), "I'm not lifting the boards".split())
        intact = lambda *a: dict(status="ok", speech_windows=10, min_corr=0.78, median_corr=0.9, lost=[])
        r = speech.recheck(Path("r.mp4"), pick, first, Path("source.MOV"), 55.6, 71.2, transcribe=fake, waveform=intact)
        self.assertEqual(r["state"], "waveform_intact")
        self.assertTrue(r["action"].startswith("none"))
        def broken(*a):
            raise RuntimeError("ffmpeg died")
        r = speech.recheck(Path("r.mp4"), pick, first, Path("source.MOV"), 55.6, 71.2, transcribe=fake, waveform=broken)
        self.assertEqual(r["state"], "confirmed")                  # no measurement, no downgrade
        self.assertEqual(r["waveform"]["status"], "unchecked")

    def test_intent_mismatch_when_the_source_says_something_else(self):
        pick = dict(id="b11p3", t0=400.0, t1=405.0, intended="I'm not lifting the boards")
        first = speech.analyse(pick["intended"], "I'm lifting the boards".split())
        r = speech.recheck(Path("r.mp4"), pick, first, Path("source.MOV"), 55.6, 71.2,
                           transcribe=self._fake("I'm lifting the boards".split(), "I'm lifting the boards".split()))
        self.assertEqual(r["state"], "intent_mismatch")

    def test_unresolved_when_the_passes_disagree(self):
        pick = dict(id="b11p3", t0=400.0, t1=405.0, intended="I'm not lifting the boards")
        first = speech.analyse(pick["intended"], "I'm lifting the boards".split())
        r = speech.recheck(Path("r.mp4"), pick, first, Path("source.MOV"), 55.6, 71.2,
                           transcribe=self._fake("I'm the boards".split(), "I'm not lifting the boards".split()))
        self.assertEqual(r["state"], "unresolved")
        self.assertIn("no trim", r["action"])

    def test_failed_recheck_is_unchecked_and_unresolved(self):
        def boom(*a):
            raise RuntimeError("ffmpeg died")
        r = speech.recheck(Path("r.mp4"), self.PICK, self._first([]), None, None, None, transcribe=boom)
        self.assertEqual(r["status"], "unchecked")
        self.assertEqual(r["state"], "unresolved")

    def test_key_changes_with_render_and_pad(self):
        k1 = speech.recheck_key("aaaa", self.PICK, 2.0)
        self.assertNotEqual(k1, speech.recheck_key("bbbb", self.PICK, 2.0))
        self.assertNotEqual(k1, speech.recheck_key("aaaa", self.PICK, 3.0))


if __name__ == "__main__":
    unittest.main()
