"""Bible retains frame evidence rather than only a lossy final annotation."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from videoeasy.bible import run


class BibleEvidenceTests(unittest.TestCase):
    def render(self, view):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shot = {"shot_id": "interview", "role": "aroll", "source_path": "/interview.MOV",
                    "in_s": 0, "out_s": 1200, "duration_s": 1200}
            (root / "shots.json").write_text(json.dumps([shot]))
            annotations = {"interview": {"subject": "Two people seated throughout."}}
            encoded = json.dumps(annotations)
            (root / "annotations.json").write_text(encoded)
            (root / "bible-synthesis.md").write_text("the editor's editorial intent")
            cfg = SimpleNamespace(out_dir=root, work_dir=root, sheets_dir=root, draft_txt=None)
            with patch("videoeasy.evidence.read_evidence", return_value=view) as reader:
                rendered = Path(run(cfg)).read_text()
            reader.assert_called_once_with(cfg, shot, annotations["interview"])
            self.assertEqual((root / "annotations.json").read_text(), encoded)
            self.assertEqual((root / "bible-synthesis.md").read_text(), "the editor's editorial intent")
            return rendered

    def ledger(self):
        return {"frames": [{"evidence_id": "interview:f00", "source_time_s": 0.25,
                            "timestamp_provenance": "inferred_from_pipeline_sampling",
                            "observation": {"visible_facts": ["Empty chairs beside a table."],
                                            "uncertainties": ["Small dark object is indistinct."]}}]}

    def test_findings_and_all_frame_facts_survive(self):
        rendered = self.render({"status": "needs_review", "errors": [], "ledger": self.ledger(),
                                "findings": [{"kind": "contradiction", "field": "subject",
                                              "claim": "Two people seated throughout.",
                                              "evidence_ids": ["interview:f00"],
                                              "reason": "Opening observation describes empty chairs."}]})
        for text in ("tag consistency: needs_review", "not footage verification", "interview:f00",
                     "source 0.250s", "inferred_from_pipeline_sampling", "Empty chairs beside a table.",
                     "uncertain: Small dark object is indistinct.", "contradiction in subject",
                     "Two people seated throughout."):
            self.assertIn(text, rendered)

    def test_annotation_staleness_does_not_erase_current_frame_evidence(self):
        rendered = self.render({"status": "unchecked", "findings": [], "ledger": self.ledger(),
                                "errors": ["missing or stale annotation audit"]})
        self.assertIn("tag consistency: unchecked", rendered)
        self.assertIn("stale annotation audit", rendered)
        self.assertIn("Empty chairs beside a table.", rendered)

    def test_stale_frame_ledger_is_not_rendered(self):
        rendered = self.render({"status": "unchecked", "findings": [], "ledger": None,
                                "errors": ["missing or stale frame evidence"]})
        self.assertIn("stale frame evidence", rendered)
        self.assertNotIn("sampled-frame observations", rendered)

    def test_no_issue_is_not_footage_verification(self):
        rendered = self.render({"status": "no_issue_detected", "findings": [], "errors": [],
                                "ledger": self.ledger()})
        self.assertIn("no_issue_detected (local-model check; not footage verification)", rendered)
        self.assertIn("unverified local-model descriptions", rendered)


if __name__ == "__main__":
    unittest.main()
