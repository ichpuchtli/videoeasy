"""CLI audit outcomes must stay distinguishable without modifying annotations."""
import unittest
from unittest.mock import patch

from typer.testing import CliRunner

from videoeasy.cli import app


class AuditCliTests(unittest.TestCase):
    def test_explicit_single_shot_is_required(self):
        with patch("videoeasy.evidence.run") as audit:
            result = CliRunner().invoke(app, ["audit"])
        self.assertNotEqual(result.exit_code, 0)
        audit.assert_not_called()

    def test_status_exit_codes_and_force_passthrough(self):
        cfg = object()
        for status, expected in (("no_issue_detected", 0), ("unchecked", 1), ("needs_review", 2)):
            with self.subTest(status=status), patch("videoeasy.cli.load_config", return_value=cfg), \
                    patch("videoeasy.evidence.run", return_value={"status": status}) as audit:
                result = CliRunner().invoke(app, ["audit", "--only", "clip", "--force"])
            self.assertEqual(result.exit_code, expected, result.output)
            audit.assert_called_once_with(cfg, only="clip", force=True)

    def test_invalid_selection_is_not_a_clean_check(self):
        with patch("videoeasy.cli.load_config", return_value=object()), \
                patch("videoeasy.evidence.run", side_effect=ValueError("unknown shot")):
            result = CliRunner().invoke(app, ["audit", "--only", "missing"])
        self.assertEqual(result.exit_code, 1)
        self.assertIn("unknown shot", result.output)


if __name__ == "__main__":
    unittest.main()
