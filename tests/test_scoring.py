"""Offline protocol tests. Every HTTP call uses an in-memory fake session."""
import contextlib
import copy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

from optiatlas import scoring


def question():
    return {"id": "sample_q1", "type": "tf_unknown", "question": "Is the optical relation visible?",
        "check": "Visible optical relation", "rubric": {"A": "Target exists, evidence is unavailable.", "B": "Target absent or visibly wrong.", "C": "Target and relation are visible."},
        "options": [{"key": "A", "text": "Cannot determine"}, {"key": "B", "text": "False"}, {"key": "C", "text": "True"}],
        "answer": "C", "target_missing_answer": "B", "judge_hint": "Inspect the actual image.",
        "option_permutation": {"version": "6.1", "seed": 20261026,
            "display_to_canonical": {"A": "C", "B": "B", "C": "A"},
            "canonical_to_display": {"C": "A", "B": "B", "A": "C"}},
        "revision_provenance": {"private": "must never enter the wire payload"}}


def sample():
    return {"sample_id": "sample", "category_id": "optical_relation", "benchmark_version": "6.1",
        "generation_prompt": "A visible optical relation.", "questions": [question()]}


class FakeResponse:
    def __init__(self, text, status=200, model="fixture-judge"):
        self.status_code = status
        self.body = {"model": model, "choices": [{"finish_reason": "stop", "message": {"content": text}}]}

    def json(self): return self.body


class FakeSession:
    def __init__(self, responses): self.responses = iter(responses); self.calls = []
    def post(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return next(self.responses)
    def close(self): pass


class ScoringProtocolTests(unittest.TestCase):
    def setUp(self):
        from PIL import Image
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.image = self.root / "sample.png"
        Image.new("RGB", (4, 4), (20, 40, 60)).save(self.image)

    def tearDown(self): self.temporary.cleanup()

    def judge(self, responses):
        judge = scoring.make_judge(model="fixture-judge", api_key="not-a-secret", base_url="https://example.invalid/v1", max_attempts=2, retry_sleep=0)
        session = FakeSession(responses); judge._thread_local.session = session
        return judge, session

    def test_frozen_instruction_hashes_and_wire_allowlist(self):
        scoring.instructions(Path(scoring.__file__).with_name("judge_instructions.md"))
        judge, _ = self.judge([])
        payload = judge._build_payload(self.image, question(), "A visible optical relation.")
        current = json.loads(payload["messages"][1]["content"][0]["text"])["current_question"]
        self.assertEqual(set(current), {"question_id", "question", "check", "rubric", "options", "judge_hint"})
        self.assertEqual(payload["temperature"], 0.0)
        self.assertEqual(payload["max_tokens"], 4096)
        self.assertFalse(any(key in current for key in ["answer", "option_permutation", "revision_provenance", "target_missing_answer"]))

    def test_bijection_seed_and_option_semantics_are_validated(self):
        scoring.normalize_question(question())
        for mutation in ["seed", "inverse", "text", "answer"]:
            q = question()
            if mutation == "seed": q["option_permutation"]["seed"] += 1
            elif mutation == "inverse": q["option_permutation"]["canonical_to_display"]["A"] = "A"
            elif mutation == "text": q["options"][0]["text"] = "True"
            else: q["answer"] = "A"
            with self.assertRaises(ValueError): scoring.normalize_question(q)

    def test_displayed_choice_is_mapped_back_before_metrics(self):
        response = json.dumps({"question_id": "sample_q1", "choice": "C", "evidence": ["Visible relation."]})
        judge, session = self.judge([FakeResponse(response)])
        result = judge.judge_one(self.image, question(), "A visible optical relation.")
        self.assertEqual((result["displayed_choice"], result["choice"], result["canonical_choice"]), ("C", "A", "A"))
        self.assertNotIn("raw_response", result)
        self.assertEqual(session.calls[0][1]["allow_redirects"], False)
        self.assertNotIn("params", session.calls[0][1])

    def test_duplicate_json_keys_retry_and_then_parse_strictly(self):
        malformed = '{"question_id":"sample_q1","choice":"C","choice":"B","evidence":["Visible."]}'
        valid = json.dumps({"question_id": "sample_q1", "choice": "B", "evidence": ["Visible contradiction."]})
        judge, session = self.judge([FakeResponse(malformed), FakeResponse(valid)])
        result = judge.judge_one(self.image, question(), "A visible optical relation.")
        self.assertEqual(result["attempts"], 2)
        self.assertEqual(result["choice"], "B")
        self.assertEqual(len(session.calls), 2)

    def test_http_failure_is_unresolved_and_never_cannot_determine(self):
        judge, session = self.judge([FakeResponse("", status=401)])
        result = judge.judge_one(self.image, question(), "A visible optical relation.")
        self.assertEqual(result["status"], "technical_failure")
        self.assertIsNone(result["choice"])
        self.assertEqual(len(session.calls), 1)
        result["category_id"] = "optical_relation"
        summary = scoring.aggregate([result])
        self.assertEqual((summary["C"], summary["unresolved"]), (0, 1))
        self.assertIsNone(summary["P_percent"])

    def test_wrong_served_model_is_not_silently_substituted(self):
        response = json.dumps({"question_id": "sample_q1", "choice": "A", "evidence": ["Target exists, detail unavailable."]})
        judge, session = self.judge([FakeResponse(response, model="different-model")])
        result = judge.judge_one(self.image, question(), "A visible optical relation.")
        self.assertEqual(result["error_type"], "served_model_mismatch")
        self.assertEqual(len(session.calls), 1)
        self.assertIsNone(result["choice"])

    def test_corrupt_image_does_not_call_transport(self):
        bad = self.root / "bad.png"; bad.write_bytes(b"not an image")
        judge, session = self.judge([])
        result = judge.judge_one(bad, question(), "A visible optical relation.")
        self.assertEqual(result["error_type"], "image_error")
        self.assertEqual(session.calls, [])

    def test_question_micro_metrics_and_invalid_mapping(self):
        mapping = question()["option_permutation"]
        rows = [{"category_id": "category", "status": "success", "choice": canonical,
            "canonical_choice": canonical, "displayed_choice": mapping["canonical_to_display"][canonical],
            "option_permutation": mapping} for canonical in ["A", "A", "B", "C"]]
        summary = scoring.aggregate(rows)
        self.assertAlmostEqual(summary["precision"], 2 / 3)
        self.assertEqual(summary["judgability"], 3 / 4)
        self.assertEqual(summary["overall"], 1 / 2)
        rows[0]["displayed_choice"] = "B"
        with self.assertRaises(ValueError): scoring.aggregate(rows)

    def test_manifest_binds_prompt_hash_and_keeps_paths_inside_root(self):
        manifest = {"records": [{"sample_id": "sample", "generation_prompt": sample()["generation_prompt"], "image_path": "sample.png", "image_sha256": scoring.digest(self.image), "status": "submitted"}]}
        (self.root / "manifest.json").write_text(json.dumps(manifest))
        found, identity = scoring.image_inputs([sample()], generation_dir=self.root)
        self.assertEqual(found["sample"]["expected_sha256"], scoring.digest(self.image))
        self.assertIsNotNone(identity)
        manifest["records"][0]["image_path"] = "../outside.png"
        (self.root / "manifest.json").write_text(json.dumps(manifest))
        with self.assertRaises(ValueError): scoring.image_inputs([sample()], generation_dir=self.root)

    def test_cli_default_plan_creates_no_output_and_never_constructs_client(self):
        benchmark = self.root / "benchmark.json"; benchmark.write_text(json.dumps([sample()]))
        out = self.root / "scores"
        argv = ["--benchmark", str(benchmark), "--images", str(self.root), "--base-url", "https://example.invalid/v1", "--model", "fixture-judge", "--output", str(out)]
        with mock.patch.object(scoring, "make_judge", side_effect=AssertionError("must not construct transport")):
            with contextlib.redirect_stdout(io.StringIO()): self.assertEqual(scoring.main(argv), 0)
        self.assertFalse(out.exists())

    def test_key_echo_is_redacted_from_structured_evidence(self):
        response = json.dumps({"question_id": "sample_q1", "choice": "C", "evidence": ["not-a-secret"]})
        judge, _ = self.judge([FakeResponse(response)])
        result = judge.judge_one(self.image, question(), "A visible optical relation.")
        self.assertEqual(result["evidence"], ["[REDACTED]"])


if __name__ == "__main__":
    unittest.main()
