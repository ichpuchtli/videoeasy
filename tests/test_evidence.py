"""Offline contracts, not a claim that the local checker detects every mistake."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from videoeasy import evidence


class EvidenceTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        root = Path(self.temp.name)
        self.cfg = SimpleNamespace(work_dir=root / "work", frames_dir=root / "work/frames",
                                   out_dir=root / "out", vision_url="http://localhost:1234/v1",
                                   vision_model="local-gemma", frames_per_shot=8,
                                   grade_for=lambda _: SimpleNamespace(name="graded", lut=None, filters="eq=saturation=1.1"))
        self.shot = {"shot_id": "clip", "source_path": "/original/clip.MOV", "in_s": 0, "out_s": 30, "role": "broll"}
        self.annotation = {"subject": "Two people seated", "movement": "The camera is static throughout the take",
                           "cut_notes": "Use for conversation coverage"}
        self.frames = self.cfg.frames_dir / "clip"
        self.frames.mkdir(parents=True)
        for i in range(4):
            (self.frames / f"f{i:02d}.jpg").write_bytes(f"frame{i}".encode())
        self.sidecars = self.cfg.work_dir / "evidence/clip"
        self.findings = []
        self.calls = []
        self.mock = patch.object(evidence, "_request", side_effect=self.request).start()
        self.addCleanup(patch.stopall)

    def request(self, cfg, system, user, image_path=None):
        self.calls.append((system, user, image_path))
        if image_path:
            facts = ["Empty chairs; no people visible"] if image_path.stem == "f00" else ["Two people seated"]
            parsed = {"visible_facts": facts, "uncertainties": []}
        else:
            parsed = {"reviewed_evidence_ids": [f"clip:f{i:02d}" for i in range(4)], "findings": self.findings}
        return {"raw": json.dumps(parsed), "parsed": parsed, "errors": []}

    def run_audit(self, **kwargs):
        return evidence.audit_shot(self.cfg, self.shot, self.annotation, **kwargs)

    def finding(self, kind="omission", field="subject", ids=None):
        return {"kind": kind, "field": field, "claim": "Empty opening chairs omitted",
                "evidence_ids": ids or ["clip:f00"], "reason": "Opening differs from seated middle samples"}

    def test_empty_opening_omission_retained_with_reference_and_no_tag_rewrite(self):
        original = json.dumps(self.annotation)
        self.findings = [self.finding()]
        result = self.run_audit()
        self.assertEqual(result["status"], "needs_review")
        self.assertEqual(result["findings"], self.findings)
        self.assertEqual(result["ledger"]["frames"][0]["observation"]["visible_facts"], ["Empty chairs; no people visible"])
        self.assertEqual(json.dumps(self.annotation), original)
        self.assertFalse(self.cfg.out_dir.exists())
        self.assertIn("omitted salient visible facts", evidence.AUDITOR_PROMPT)
        self.assertEqual(len(list((self.sidecars / "attempts").glob("*.json"))), 5)

    def test_all_take_static_claim_is_represented_as_unsupported_not_verified(self):
        self.findings = [self.finding("unsupported_claim", "movement", ["clip:f00", "clip:f03"])]
        self.assertEqual(self.run_audit()["status"], "needs_review")
        self.assertIn("whole-take stability", evidence.AUDITOR_PROMPT)
        self.assertIn("sparse stills cannot", evidence.AUDITOR_PROMPT)

    def test_clean_check_is_only_no_issue_detected_and_resume_uses_cache(self):
        result = self.run_audit()
        self.assertEqual(result["status"], "no_issue_detected")
        self.assertEqual(result["provenance"], "unverified_local_model")
        self.assertEqual(len(self.calls), 5)
        self.run_audit()
        self.assertEqual(len(self.calls), 5)
        self.run_audit(force=True)
        self.assertEqual(len(self.calls), 10)
        self.assertEqual(len(list((self.sidecars / "attempts").glob("*.json"))), 10)

    def test_unknown_evidence_id_fails_closed_and_raw_is_preserved(self):
        self.findings = [self.finding(ids=["invented:f99"])]
        self.assertEqual(self.run_audit()["status"], "unchecked")
        audit = json.loads((self.sidecars / "audit.json").read_text())
        self.assertIn("invented:f99", audit["raw"])
        self.assertTrue(audit["errors"])
        self.findings = []
        self.assertEqual(self.run_audit()["status"], "no_issue_detected")
        self.assertEqual(len(self.calls), 6)  # checker only retried

    def test_failed_checker_is_unchecked_and_retries_without_losing_observations(self):
        normal = self.request
        def fail(cfg, system, user, image_path=None):
            return normal(cfg, system, user, image_path) if image_path else {"raw": "truncated", "parsed": None, "errors": ["timeout"]}
        self.mock.side_effect = fail
        self.assertEqual(self.run_audit()["status"], "unchecked")
        self.assertEqual(json.loads((self.sidecars / "audit.json").read_text())["raw"], "truncated")
        self.mock.side_effect = normal
        self.assertEqual(self.run_audit()["status"], "no_issue_detected")
        self.assertEqual(len(self.calls), 5)

    def test_failed_frame_never_silently_dropped_and_only_failed_frame_retried(self):
        normal = self.request
        def fail(cfg, system, user, image_path=None):
            if image_path and image_path.stem == "f02":
                return {"raw": "bad observation", "parsed": {"wrong": True}, "errors": []}
            return normal(cfg, system, user, image_path)
        self.mock.side_effect = fail
        self.assertEqual(self.run_audit()["status"], "unchecked")
        ledger = json.loads((self.sidecars / "ledger.json").read_text())
        self.assertEqual(len(ledger["frames"]), 4)
        self.assertTrue(ledger["frames"][2]["errors"])
        self.assertTrue(all(call[2] for call in self.calls))  # no checker call
        self.mock.side_effect = normal
        self.assertEqual(self.run_audit()["status"], "no_issue_detected")
        self.assertEqual(len(self.calls), 5)

    def test_annotation_change_invalidates_only_checker_and_preserves_current_ledger(self):
        self.run_audit()
        self.annotation["subject"] = "Changed annotation"
        stale = evidence.read_evidence(self.cfg, self.shot, self.annotation)
        self.assertEqual(stale["status"], "unchecked")
        self.assertIsNotNone(stale["ledger"])
        self.run_audit()
        self.assertEqual(len(self.calls), 6)

    def test_frame_hash_change_invalidates_ledger_and_audit(self):
        self.run_audit()
        (self.frames / "f00.jpg").write_bytes(b"changed")
        stale = evidence.read_evidence(self.cfg, self.shot, self.annotation)
        self.assertEqual(stale["status"], "unchecked")
        self.assertIsNone(stale["ledger"])
        self.run_audit()
        self.assertEqual(len(self.calls), 10)

    def test_model_observer_and_auditor_prompt_changes_invalidate_correct_stage(self):
        self.run_audit()
        self.cfg.vision_model = "new-local-model"
        self.run_audit()
        self.assertEqual(len(self.calls), 10)
        with patch.object(evidence, "OBSERVER_PROMPT", evidence.OBSERVER_PROMPT + " changed"):
            self.run_audit()
            self.assertEqual(len(self.calls), 15)
            with patch.object(evidence, "AUDITOR_PROMPT", evidence.AUDITOR_PROMPT + " changed"):
                stale = evidence.read_evidence(self.cfg, self.shot, self.annotation)
                self.assertEqual(stale["status"], "unchecked")
                self.assertIsNotNone(stale["ledger"])
                self.run_audit()
                self.assertEqual(len(self.calls), 16)

    def test_grade_and_time_changes_invalidate_evidence(self):
        self.run_audit()
        self.cfg.grade_for = lambda _: SimpleNamespace(name="changed", lut=None, filters="eq=saturation=1.2")
        self.assertIsNone(evidence.read_evidence(self.cfg, self.shot, self.annotation)["ledger"])
        self.run_audit()
        self.shot["out_s"] = 40
        self.assertIsNone(evidence.read_evidence(self.cfg, self.shot, self.annotation)["ledger"])

    def test_missing_middle_or_previously_known_last_frame_fails_closed(self):
        self.run_audit()
        (self.frames / "f03.jpg").unlink()
        self.assertEqual(self.run_audit()["status"], "unchecked")
        self.assertEqual(len(self.calls), 5)
        (self.frames / "f02.jpg").rename(self.frames / "f03.jpg")
        self.assertEqual(self.run_audit()["status"], "unchecked")
        self.assertEqual(len(self.calls), 5)

    def test_malformed_and_partial_ledgers_are_not_current(self):
        self.run_audit()
        path = self.sidecars / "ledger.json"
        good = json.loads(path.read_text())
        for frames in (None, "bad", {}, [None] * 4, ["bad"] * 4, [[]] * 4, good["frames"][:2]):
            with self.subTest(frames=frames):
                path.write_text(json.dumps({**good, "frames": frames}))
                result = evidence.read_evidence(self.cfg, self.shot, self.annotation)
                self.assertEqual(result["status"], "unchecked")
                self.assertIsNone(result["ledger"])
        path.write_text(json.dumps({**good, "frames": good["frames"][:2]}))
        self.run_audit()
        self.assertEqual(len(self.calls), 8)  # two resumed frames, then checker

    def test_malformed_ledger_can_resume(self):
        self.run_audit()
        path = self.sidecars / "ledger.json"
        good = json.loads(path.read_text())
        for frames in (None, "bad", [None] * 4):
            path.write_text(json.dumps({**good, "frames": frames}))
            self.assertEqual(self.run_audit()["status"], "no_issue_detected")

    def test_strict_checker_and_observer_schemas(self):
        ids = ["a", "b"]
        for value in (None, [], {}, {"reviewed_evidence_ids": ids, "findings": [], "verified": True},
                      {"reviewed_evidence_ids": ["a", "a"], "findings": []},
                      {"reviewed_evidence_ids": ["a"], "findings": []},
                      {"reviewed_evidence_ids": ids, "findings": [None]}):
            self.assertTrue(evidence.audit_errors(value, ids, self.annotation))
        for value in (None, {}, {"visible_facts": [], "uncertainties": []},
                      {"visible_facts": [""], "uncertainties": []},
                      {"visible_facts": ["a"], "uncertainties": "unclear"}):
            self.assertTrue(evidence.observation_errors(value))

    def test_remote_or_ollama_backend_rejected_before_any_call(self):
        for url in ("https://remote.example/v1", "http://localhost:11434/v1"):
            self.cfg.vision_url = url
            self.assertEqual(self.run_audit()["status"], "unchecked")
        self.assertFalse(self.calls)

    def test_run_requires_exact_one_shot_and_existing_annotation(self):
        self.cfg.out_dir.mkdir()
        (self.cfg.out_dir / "shots.json").write_text(json.dumps([self.shot]))
        (self.cfg.out_dir / "annotations.json").write_text("{}")
        for only in ("", " ", "unknown"):
            with self.assertRaises(ValueError):
                evidence.run(self.cfg, only)
        self.assertEqual(evidence.run(self.cfg, "clip")["status"], "unchecked")
        self.assertFalse(self.calls)

    def test_nonfinite_time_records_unchecked_instead_of_crashing(self):
        for time in (float("nan"), float("inf")):
            self.shot["out_s"] = time
            self.assertEqual(self.run_audit()["status"], "unchecked")
        self.assertFalse(self.calls)

    def test_actual_request_image_hash_mismatch_fails_closed(self):
        normal = self.request
        def changed(cfg, system, user, image_path=None):
            record = normal(cfg, system, user, image_path)
            if image_path:
                record["request"] = {"image_sha256": "different-pixels"}
            return record
        self.mock.side_effect = changed
        result = self.run_audit()
        self.assertEqual(result["status"], "unchecked")
        self.assertTrue(all(call[2] for call in self.calls))

    def test_stage_token_budgets_have_separate_cache_identity(self):
        self.run_audit()
        self.cfg.raw = {"evidence_audit": {"auditor_max_tokens": 2048}}
        current = evidence.read_evidence(self.cfg, self.shot, self.annotation)
        self.assertEqual(current["status"], "unchecked")
        self.assertIsNotNone(current["ledger"])
        self.run_audit()
        self.assertEqual(len(self.calls), 6)
        audit = json.loads((self.sidecars / "audit.json").read_text())
        self.assertEqual(audit["settings"]["max_output_tokens"], 2048)
        self.cfg.raw["evidence_audit"]["observer_max_tokens"] = 2048
        self.assertIsNone(evidence.read_evidence(self.cfg, self.shot, self.annotation)["ledger"])
        self.run_audit()
        self.assertEqual(len(self.calls), 11)
        ledger = json.loads((self.sidecars / "ledger.json").read_text())
        self.assertEqual(ledger["identity"]["settings"]["max_output_tokens"], 2048)

    def test_invalid_token_budget_configuration_fails_before_any_call(self):
        for options in (None, [], {"max_tokens": 1024}, {"observer_max_tokens": True},
                        {"observer_max_tokens": 1536.0}, {"observer_max_tokens": "1536"},
                        {"auditor_max_tokens": 0}, {"auditor_max_tokens": -1},
                        {"auditor_max_tokens": 32769}):
            with self.subTest(options=options):
                self.cfg.raw = {"evidence_audit": options}
                self.assertEqual(self.run_audit()["status"], "unchecked")
        self.assertFalse(self.calls)

    def test_transport_reasoning_store_and_effective_endpoint_invalidate_cache(self):
        self.run_audit()
        with patch.object(evidence, "TRANSPORT", "changed_transport"):
            self.assertIsNone(evidence.read_evidence(self.cfg, self.shot, self.annotation)["ledger"])
        original = evidence.request_settings
        for key, value in (("reasoning", "on"), ("store", True)):
            with patch.object(evidence, "request_settings", side_effect=lambda cfg, stage: {**original(cfg, stage), key: value}):
                self.assertIsNone(evidence.read_evidence(self.cfg, self.shot, self.annotation)["ledger"])
        self.cfg.vision_url = "http://localhost:4321/v1"
        self.assertIsNone(evidence.read_evidence(self.cfg, self.shot, self.annotation)["ledger"])
        ledger = json.loads((self.sidecars / "ledger.json").read_text())
        self.assertEqual(ledger["identity"]["endpoint"], "http://localhost:1234/api/v1/chat")
        self.assertEqual(ledger["identity"]["settings"]["reasoning"], "off")
        self.assertFalse(ledger["identity"]["settings"]["store"])

    def test_prompt_guards_compatible_details_and_gender_without_weakening_omissions(self):
        self.assertIn("Extra compatible detail", evidence.AUDITOR_PROMPT)
        self.assertIn("omission, not a contradiction", evidence.AUDITOR_PROMPT)
        self.assertIn("Do not infer or validate gender", evidence.AUDITOR_PROMPT)
        self.assertIn("not infer gender", evidence.OBSERVER_PROMPT)

    def test_prompt_defines_hold_and_requires_whole_annotation_state_checks(self):
        self.assertIn("hold_seconds is a RECOMMENDED EDIT HOLD, not source duration", evidence.AUDITOR_PROMPT)
        self.assertIn("Read the WHOLE annotation", evidence.AUDITOR_PROMPT)
        self.assertIn("Do not assume cross-frame person matching", evidence.AUDITOR_PROMPT)
        self.assertIn("Separately check salient objects", evidence.AUDITOR_PROMPT)
        self.assertIn("the first sampled state, and the last sampled state", evidence.AUDITOR_PROMPT)
        self.assertIn("cannot establish an exact usable", evidence.AUDITOR_PROMPT)

    def test_observer_requires_visible_count_not_generic_unknowns(self):
        self.assertIn("visible person count or no visible people as a fact", evidence.OBSERVER_PROMPT)
        self.assertIn("only when visibility is genuinely ambiguous", evidence.OBSERVER_PROMPT)
        self.assertIn("Uncertainties are only ambiguous VISIBLE details", evidence.OBSERVER_PROMPT)
        self.assertIn("not generic unknown", evidence.OBSERVER_PROMPT)


class TransportTests(unittest.TestCase):
    def setUp(self):
        self.cfg = SimpleNamespace(vision_url="http://localhost:1234/v1", vision_model="gemma")

    def body(self):
        return {"model_instance_id": "gemma", "output": [{"type": "message", "content":
                '{"visible_facts":["Blue square and yellow circle"],"uncertainties":[]}'}],
                "stats": {"input_tokens": 474, "total_output_tokens": 113,
                          "reasoning_output_tokens": 0}}

    def call(self, body, image_path=None):
        import httpx
        response = httpx.Response(200, json=body, request=httpx.Request("POST", "http://localhost:1234/api/v1/chat"))
        with patch.object(evidence.httpx, "post", return_value=response) as post:
            result = evidence._request(self.cfg, "system", "user", image_path)
        return result, post

    def test_native_text_payload_and_provenance(self):
        result, post = self.call(self.body())
        self.assertFalse(result["errors"])
        self.assertEqual(post.call_args.args[0], "http://localhost:1234/api/v1/chat")
        self.assertEqual(post.call_args.kwargs["json"], {"model": "gemma", "system_prompt": "system", "input": "user",
                         "temperature": 0.0, "max_output_tokens": 1536, "reasoning": "off", "store": False})
        self.assertEqual(result["model_instance_id"], "gemma")
        self.assertEqual(result["request"]["transport"], evidence.TRANSPORT)
        self.assertNotIn("finish_reason", result)

    def test_cap_hit_even_valid_json_and_malformed_json_fail_closed(self):
        for tokens in (1536, 1600):
            body = self.body()
            body["stats"]["total_output_tokens"] = tokens
            result, _ = self.call(body)
            self.assertTrue(result["errors"])
            self.assertIsNone(result["parsed"])
            self.assertEqual(result["raw"], body["output"][0]["content"])
        body = self.body()
        body["output"][0]["content"] = '{"partial":'
        result, _ = self.call(body)
        self.assertTrue(result["errors"])
        self.assertEqual(result["raw"], '{"partial":')

    def test_ignored_reasoning_and_unexpected_output_fail_closed(self):
        for kind in ("reasoning", "tool_call", "invalid_tool_call", "unknown", "message"):
            body = self.body()
            body["output"].append({"type": kind, "content": "extra"})
            result, _ = self.call(body)
            self.assertTrue(result["errors"])
            self.assertEqual(json.loads(result["response_body"]), body)
        body = self.body()
        body["stats"]["reasoning_output_tokens"] = 1
        self.assertTrue(self.call(body)[0]["errors"])

    def test_invalid_counts_model_and_output_are_rejected(self):
        for key in ("input_tokens", "total_output_tokens", "reasoning_output_tokens"):
            for value in (None, -1, True, "2", 2.5):
                body = self.body()
                body["stats"][key] = value
                self.assertTrue(self.call(body)[0]["errors"])
            body = self.body()
            del body["stats"][key]
            self.assertTrue(self.call(body)[0]["errors"])
        for body in ([], {}, {**self.body(), "model_instance_id": None},
                     {**self.body(), "output": None}, {**self.body(), "output": ["invalid"]},
                     {**self.body(), "response_id": "unexpectedly-stored"}):
            self.assertTrue(self.call(body)[0]["errors"])

    def test_native_endpoint_derivation_rejects_ambiguous_urls(self):
        for url in ("http://localhost:1234/evil", "http://user@localhost:1234/v1", "http://localhost:1234/v1?extra=yes",
                    "http://localhost:1234/v1#frag", "http://remote.example/v1", "http://localhost:11434/v1"):
            self.cfg.vision_url = url
            with self.assertRaises(ValueError):
                evidence._local_endpoint(self.cfg)
        self.cfg.vision_url = "http://[::1]:1234/v1"
        self.assertEqual(evidence._local_endpoint(self.cfg), "http://[::1]:1234/api/v1/chat")

    def test_missing_frame_preserves_error_without_network_call(self):
        cfg = SimpleNamespace(vision_url="http://localhost:1234/v1", vision_model="gemma")
        with patch.object(evidence.httpx, "post") as post:
            result = evidence._request(cfg, "system", "user", Path("/nonexistent/frame.jpg"))
        self.assertTrue(result["errors"])
        post.assert_not_called()

    def test_real_shaped_context_error_keeps_body_and_effective_budget(self):
        import httpx
        cfg = SimpleNamespace(vision_url="http://localhost:1234/v1", vision_model="gemma")
        response = httpx.Response(400, json={"error": "Context size has been exceeded."},
                                  request=httpx.Request("POST", cfg.vision_url + "/chat/completions"))
        with patch.object(evidence.httpx, "post", return_value=response) as post:
            result = evidence._request(cfg, evidence.AUDITOR_PROMPT, "unaltered ledger" * 300)
        self.assertTrue(result["errors"])
        self.assertIn("Context size has been exceeded", result["response_body"])
        self.assertEqual(result["request"]["settings"]["max_output_tokens"], 1536)
        self.assertEqual(post.call_args.kwargs["json"]["max_output_tokens"], 1536)
        self.assertEqual(post.call_args.kwargs["json"]["input"], "unaltered ledger" * 300)
        self.assertEqual(post.call_count, 1)  # no hidden larger-budget retry

    def test_success_retains_stats_and_native_image_payload(self):
        self.cfg.raw = {"evidence_audit": {"observer_max_tokens": 2048}}
        body = self.body()
        with tempfile.TemporaryDirectory() as temp:
            frame = Path(temp) / "frame.jpg"
            frame.write_bytes(b"test-frame")
            result, post = self.call(body, frame)
        self.assertFalse(result["errors"])
        self.assertEqual(result["usage"], body["stats"])
        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["max_output_tokens"], 2048)
        self.assertEqual(payload["input"], [{"type": "text", "content": "user"},
                         {"type": "image", "data_url": "data:image/jpeg;base64,dGVzdC1mcmFtZQ=="}])
        self.assertEqual(result["request"]["image_sha256"], evidence.digest(b"test-frame"))

    def test_native_preflight_requires_colors_in_valid_facts(self):
        for parsed, errors, succeeds in (({"visible_facts": ["yellow circle blue square"], "uncertainties": []}, [], True),
                ({"visible_facts": ["grey view"], "uncertainties": ["yellow blue"]}, [], False),
                ({"visible_facts": "yellow blue", "uncertainties": []}, [], False),
                (None, ["server failed"], False)):
            with patch.object(evidence, "_request", return_value={"parsed": parsed, "errors": errors}) as request:
                result = evidence.check_native(self.cfg, b"synthetic")
            self.assertEqual(not result["errors"], succeeds)
            self.assertIsInstance(request.call_args.args[3], Path)


class NativePreflightCliTests(unittest.TestCase):
    def test_optional_native_preflight_uses_synthetic_jpeg_and_propagates_failures(self):
        from typer.testing import CliRunner
        from videoeasy.cli import app
        runner = CliRunner()
        cfg = SimpleNamespace(grade_profiles={}, vision_url="http://localhost:1234/v1", vision_model="gemma")
        native_result = {"errors": [], "parsed": {"visible_facts": ["yellow circle blue square"], "uncertainties": []}}
        with patch("videoeasy.cli.load_config", return_value=cfg), \
                patch("subprocess.run", return_value=SimpleNamespace(returncode=0)), \
                patch("videoeasy.vlm.chat_vision", return_value='{"colors":["blue","yellow"]}'), \
                patch.dict("sys.modules", {"mlx_whisper": SimpleNamespace()}), \
                patch.object(evidence, "check_native", return_value=native_result) as native:
            self.assertEqual(runner.invoke(app, ["check"]).exit_code, 0)
            native.assert_not_called()
            result = runner.invoke(app, ["check", "--audit"])
            self.assertEqual(result.exit_code, 0, result.output)
            self.assertEqual(native.call_args.args[0], cfg)
            self.assertTrue(native.call_args.args[1].startswith(b"\xff\xd8"))
            native.return_value = {"errors": ["SUSPECT: native images missing"], "parsed": None}
            self.assertEqual(runner.invoke(app, ["check", "--audit"]).exit_code, 1)


if __name__ == "__main__":
    unittest.main()
