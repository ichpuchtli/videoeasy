"""Species claims (species.py): the pure parts. No model, no footage."""
from __future__ import annotations

import unittest

from videoeasy import species

from filmfixture import TEST_FILM

LIST = [dict(sci="Quercus robur", common="English oak", named=True, terms=["oak"]),
        dict(sci="Salix alba", common="white willow", named=True, terms=["willow"]),
        dict(sci="Salix fragilis", common="crack willow", named=True, terms=["willow"]),
        dict(sci="Fraxinus excelsior", common="ash", named=False, terms=[]),
        dict(sci="Turdus merula", common="blackbird", named=True, terms=["blackbird"])]


class Claims(unittest.TestCase):
    def test_spoken_forms_name_species(self):
        self.assertEqual([s["sci"] for s in species.named_in("Here's the oak that I planted", LIST)], ["Quercus robur"])
        self.assertEqual(len(species.named_in("the little willows that are coming back", LIST)), 2)
        self.assertEqual(species.named_in("an old ash", LIST), [])                    # a species not named in the film is never 'named'
        self.assertEqual(species.named_in("a dark cloak", LIST), [])                    # word-start only

    def test_cover_rows_and_uncovered_walk_picks_are_claims_the_sitdown_is_not(self):
        cut = {"beats": [
            dict(title="walk", audio=[dict(clip="CAMA0131", in_s=10.0, out_s=16.0, text="my oak", tier=1),
                                      dict(clip="CAMA0124", in_s=100.0, out_s=104.0, text="the blackbird", tier=1)],
                 video=[dict(type="bare", pick="b1p1", clip=None, dest_in_s=0, dest_out_s=6),
                        dict(type="bare", pick="b1p2", clip=None, dest_in_s=6, dest_out_s=10)]),
            dict(title="cover", audio=[dict(clip="CAMA0124", in_s=200.0, out_s=205.0, text="the little willows", tier=1)],
                 video=[dict(type="video", pick="b2p1", clip="CAMB0214", in_s=0.25, dest_in_s=10.0, dest_out_s=14.0)])]}
        cl = species.claims(cut, LIST, {"CAMB0214": "CAMB0214"}, TEST_FILM)
        self.assertEqual([(c["kind"], c["unit"], c["in_s"], c["out_s"]) for c in cl],
                         [("sync", "CAMA0131", 10.0, 16.0), ("cover", "CAMB0214", 0.25, 4.25)])

    def test_agreement_is_the_named_species_or_genus_in_the_open_top_five(self):
        top = [dict(species="Hedera helix"), dict(species="Salix caprea")]
        self.assertTrue(species.agrees(["Salix alba", "Salix fragilis"], top))    # the genus the film names
        self.assertFalse(species.agrees(["Quercus robur"], [dict(species="Fraxinus excelsior")]))

    def test_a_strong_other_answer_disagrees_a_weak_one_cannot_tell(self):
        self.assertEqual(species.verdict(["Quercus robur"], [dict(species="Fraxinus excelsior", score=0.23)]), "disagrees")
        self.assertEqual(species.verdict(["Rubus fruticosus"], [dict(species="Babyrousa babyrussa", score=0.01)]), "cannot tell")
        self.assertEqual(species.verdict(["Turdus merula"], [dict(species="Turdus merula", score=0.48)]), "agrees")


if __name__ == "__main__":
    unittest.main()
