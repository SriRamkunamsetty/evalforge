"""Comprehensive tests for EvalForge AI/ML engine: agreement metrics, bootstrap, LLM judge, and datasets."""
import os
import tempfile
import unittest
from unittest import mock

from evalforge.ml import (
    cohens_kappa,
    gwet_ac1,
    krippendorff_alpha,
    pairwise_agreement_matrix,
    bradley_terry_bootstrap,
    compute_win_probability_matrix,
    HeuristicJudge,
    LLMJudge,
    calibrate_judge_against_human,
    uncertainty_sampling,
    active_queue_prioritization,
    export_preference_dataset,
    DatasetFormat,
    validate_preference_pair,
)
from evalforge.server import create_app
from evalforge.worker import EvaluationWorker


class MLAgreementTests(unittest.TestCase):
    def test_gwet_ac1_handles_prevalence_paradox(self):
        # High observed agreement (90%), but skewed distribution
        # Cohen's kappa drops heavily, while Gwet's AC1 reflects high agreement
        rater_a = ["a"] * 90 + ["b"] * 10
        rater_b = ["a"] * 85 + ["b"] * 5 + ["a"] * 5 + ["b"] * 5
        kappa = cohens_kappa(rater_a, rater_b)
        ac1 = gwet_ac1(rater_a, rater_b)
        self.assertGreater(ac1, 0.70)
        self.assertGreater(ac1, kappa)

    def test_krippendorff_alpha_nominal(self):
        # Perfect agreement across 3 raters
        units = {
            1: {"r1": "a", "r2": "a", "r3": "a"},
            2: {"r1": "b", "r2": "b", "r3": "b"},
            3: {"r1": "a", "r2": "a", "r3": "a"},
        }
        alpha = krippendorff_alpha(units, level_of_measurement="nominal")
        self.assertAlmostEqual(alpha, 1.0)

    def test_krippendorff_alpha_with_missing_data(self):
        # Raters do not evaluate every single item
        units = {
            1: {"r1": "a", "r2": "a"},
            2: {"r2": "b", "r3": "b"},
            3: {"r1": "a", "r3": "a"},
            4: {"r1": "b", "r2": "a"},  # 1 disagreement
        }
        alpha = krippendorff_alpha(units, level_of_measurement="nominal")
        self.assertGreater(alpha, 0.50)
        self.assertLessEqual(alpha, 1.0)

    def test_pairwise_agreement_matrix(self):
        by_rater = {
            "r1": {1: "a", 2: "b", 3: "a", 4: "b", 5: "a"},
            "r2": {1: "a", 2: "b", 3: "a", 4: "b", 5: "a"},
        }
        res = pairwise_agreement_matrix(by_rater, min_overlap=5)
        self.assertEqual(len(res), 1)
        self.assertEqual(res[0]["score"], 1.0)


class MLBootstrapTests(unittest.TestCase):
    def test_bradley_terry_bootstrap_confidence_intervals(self):
        battles = [("m1", "m2")] * 20 + [("m2", "m3")] * 15 + [("m1", "m3")] * 10
        res = bradley_terry_bootstrap(battles, n_bootstrap=50, ci=0.95, seed=42)
        self.assertIn("m1", res)
        self.assertIn("m2", res)
        self.assertIn("m3", res)
        # Verify ordering: m1 > m2 > m3
        self.assertGreater(res["m1"]["elo"], res["m2"]["elo"])
        self.assertGreater(res["m2"]["elo"], res["m3"]["elo"])
        # Verify CI bounds
        for m in ("m1", "m2", "m3"):
            self.assertLessEqual(res[m]["elo_low"], res[m]["elo"])
            self.assertGreaterEqual(res[m]["elo_high"], res[m]["elo"])

    def test_win_probability_matrix(self):
        strengths = {"m1": 2.0, "m2": 1.0}
        mat = compute_win_probability_matrix(strengths)
        self.assertAlmostEqual(mat["m1"]["m2"], 2.0 / 3.0, places=3)
        self.assertAlmostEqual(mat["m2"]["m1"], 1.0 / 3.0, places=3)
        self.assertEqual(mat["m1"]["m1"], 0.5)


class MLJudgeTests(unittest.TestCase):
    def setUp(self):
        self.judge = HeuristicJudge()

    def test_heuristic_judge_favors_complete_accurate_response(self):
        prompt = "Write a python function to add two numbers."
        good_resp = "Here is the code:\n```python\ndef add(a, b):\n    return a + b\n```\nIt returns the sum."
        weak_resp = "I don't know."
        result = self.judge.evaluate(prompt, good_resp, weak_resp)
        self.assertEqual(result.winner, "a")
        self.assertGreater(result.confidence, 0.6)
        self.assertTrue(len(result.scores) >= 3)

    def test_llm_judge_fallback_and_position_bias(self):
        # Mock LLM that is biased to always vote for [[A]] regardless of content
        def biased_llm(_prompt: str) -> str:
            return "Winner: [[A]]\nRationale: Always A."

        judge = LLMJudge(completion_fn=biased_llm)
        result = judge.evaluate("Test prompt", "Response 1", "Response 2", check_position_bias=True)
        # Position bias check should catch inconsistency when reversed and downgrade to tie
        self.assertFalse(result.position_consistent)
        self.assertEqual(result.winner, "tie")

    def test_judge_calibration(self):
        votes = [
            ("t1", "a", "a"),
            ("t2", "b", "b"),
            ("t3", "a", "b"),
        ]
        cal = calibrate_judge_against_human(votes)
        self.assertAlmostEqual(cal["accuracy"], 2.0 / 3.0, places=3)


class MLActiveSamplingTests(unittest.TestCase):
    def test_uncertainty_sampling_scores(self):
        # Equal strengths -> maximum entropy / highest priority
        score_equal = uncertainty_sampling(1.0, 1.0)
        # One-sided strength -> low entropy
        score_skewed = uncertainty_sampling(10.0, 0.1)
        self.assertGreater(score_equal, score_skewed)

    def test_active_queue_prioritization(self):
        tasks = [
            {"id": 1, "model_a": "m1", "model_b": "m2"},  # equal strength
            {"id": 2, "model_a": "m1", "model_b": "m3"},  # unequal strength
        ]
        strengths = {"m1": 1.0, "m2": 1.0, "m3": 10.0}
        prioritized = active_queue_prioritization(tasks, strengths, {})
        self.assertEqual(prioritized[0]["id"], 1)


class MLExportTests(unittest.TestCase):
    def test_validate_preference_pair(self):
        ok, _ = validate_preference_pair("Prompt?", "A valid good response", "Another valid response")
        self.assertTrue(ok)
        bad, msg = validate_preference_pair("?", "X", "X")
        self.assertFalse(bad)

    def test_export_formats(self):
        records = [{
            "prompt": "Test prompt",
            "model_a": "alpha",
            "response_a": "Alpha response",
            "model_b": "beta",
            "response_b": "Beta response",
            "winner": "a",
        }]
        dpo = export_preference_dataset(records, DatasetFormat.DPO)
        self.assertEqual(len(dpo), 1)
        self.assertEqual(dpo[0]["chosen"], "Alpha response")
        self.assertEqual(dpo[0]["rejected"], "Beta response")

        kto = export_preference_dataset(records, DatasetFormat.KTO)
        self.assertEqual(len(kto), 2)
        self.assertTrue(kto[0]["label"])
        self.assertFalse(kto[1]["label"])


class MLServerEndpointsTests(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(os.environ, {"API_KEY": "test-key"})
        self.env.start()
        self.app = create_app(os.path.join(tempfile.mkdtemp(), "test_ml.db"), {"RATE_LIMIT": 10000})
        self.c = self.app.test_client()

    def tearDown(self):
        self.env.stop()

    def test_judge_evaluate_endpoint(self):
        payload = {
            "prompt": "Write a greeting.",
            "response_a": "Hello there! How are you doing today?",
            "response_b": "Hi.",
        }
        resp = self.c.post("/api/v1/judge/evaluate", json=payload, headers={"X-API-Key": "test-key"})
        self.assertEqual(resp.status_code, 200)
        data = resp.get_json()
        self.assertIn(data["winner"], ("a", "b", "tie"))
        self.assertIn("confidence", data)

    def test_prioritized_queue_and_openapi(self):
        # OpenAPI spec
        spec = self.c.get("/api/openapi.json")
        self.assertEqual(spec.status_code, 200)
        self.assertEqual(spec.get_json()["openapi"], "3.0.3")

        # Queue
        q = self.c.get("/api/v1/queue/prioritized")
        self.assertEqual(q.status_code, 200)
        self.assertIn("prioritized_tasks", q.get_json())

    def test_background_worker_job_processing(self):
        worker = EvaluationWorker()
        res = worker.process_job("bootstrap_leaderboard", {"battles": [("a", "b"), ("b", "c")], "n_bootstrap": 10})
        self.assertEqual(res["status"], "completed")
        self.assertIn("result", res)


if __name__ == "__main__":
    unittest.main()
