"""BirdNET wrapper (birds.py): the pure parts. No model, no audio."""
from __future__ import annotations

import unittest

from videoeasy import birds


class Birds(unittest.TestCase):
    def test_birdnet_week(self):
        self.assertEqual(birds.birdnet_week("2024-03-15"), 11)   # four BirdNET weeks a month: the third of March
        self.assertEqual(birds.birdnet_week("2025-01-01"), 1)
        self.assertEqual(birds.birdnet_week("2025-12-31"), 48)

    def test_species_name(self):
        self.assertEqual(birds.split_name("Turdus merula_Eurasian Blackbird"), ("Turdus merula", "Eurasian Blackbird"))

    def test_summary_counts_confident_detections_per_species(self):
        src = {"A": dict(detections=[dict(sci="Turdus merula", common="Eurasian Blackbird", confidence=0.81),
                                     dict(sci="Turdus merula", common="Eurasian Blackbird", confidence=0.2)]),
               "B": dict(detections=[dict(sci="Turdus merula", common="Eurasian Blackbird", confidence=0.6),
                                     dict(sci="Erithacus rubecula", common="European Robin", confidence=0.3)])}
        s = birds.summarise(src)
        self.assertEqual(s[0]["common"], "Eurasian Blackbird")
        self.assertEqual((s[0]["detections"], s[0]["confident"], s[0]["best"], s[0]["sources"]), (3, 2, 0.81, ["A", "B"]))
        self.assertEqual(s[1]["confident"], 0)


if __name__ == "__main__":
    unittest.main()
