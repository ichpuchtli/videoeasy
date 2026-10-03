"""Delivery master: the loudnorm parse and the rule verdict. No ffmpeg."""
import json
import re
import unittest

from videoeasy import deliver


class Deliver(unittest.TestCase):
    def test_verdict(self):
        self.assertEqual(deliver.verdict(dict(status="ok", lufs=-16.3, true_peak=-1.2), -16.0, -1.0, 1.0), "honoured")
        self.assertEqual(deliver.verdict(dict(status="ok", lufs=-18.0, true_peak=-1.2), -16.0, -1.0, 1.0), "broken")
        self.assertEqual(deliver.verdict(dict(status="ok", lufs=-16.0, true_peak=-0.2), -16.0, -1.0, 1.0), "broken")
        self.assertEqual(deliver.verdict(dict(status="unchecked"), -16.0, -1.0, 1.0), "unchecked")

    def test_pass1_json_is_found_in_ffmpeg_output(self):
        stderr = 'noise\n[Parsed_loudnorm_0 @ 0x1] \n{\n\t"input_i" : "-24.80",\n\t"input_tp" : "-1.50",\n\t"input_lra" : "9.80",\n\t"input_thresh" : "-35.90",\n\t"target_offset" : "0.10"\n}\n'
        m = re.search(r"\{[^{}]*\"input_i\"[^{}]*\}", stderr, re.S)
        self.assertEqual(json.loads(m.group(0))["input_i"], "-24.80")


if __name__ == "__main__":
    unittest.main()
