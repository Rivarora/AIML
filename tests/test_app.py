"""Black-box tests for the Flask product API and page shell.

Run with:
    python -m unittest discover -s tests -v

The tests use Flask's test client, so no server or network access is required.
Prediction tests are skipped only when the saved model artifacts have not been
generated yet.
"""
from __future__ import annotations

import unittest
from pathlib import Path

from app import TASK_META, app
from scripts.model_config import fields_for


ROOT = Path(__file__).resolve().parent.parent
EXAM_ARTIFACT = ROOT / "models" / TASK_META["exam"]["artifact"]
PLACEMENT_ARTIFACT = ROOT / "models" / TASK_META["placement"]["artifact"]


def sample_payload(track: str) -> dict:
    """Build a valid payload from the shared schema example values."""
    payload = {}
    for field in fields_for(track):
        payload[field["name"]] = field["example"]
    return payload


class PageTests(unittest.TestCase):
    def setUp(self) -> None:
        app.config.update(TESTING=True)
        self.client = app.test_client()

    def test_index_renders_product_shell(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        body = response.get_data(as_text=True)
        self.assertIn("Student Performance Analytics", body)
        self.assertIn("Placement Readiness", body)
        self.assertIn("Model Insights", body)
        # Assignment/rubric language should not lead the product UI.
        lowered = body.lower()
        for forbidden in ("rubric", "evaluator", "assignment submission"):
            self.assertNotIn(forbidden, lowered)

    def test_static_assets_exist(self) -> None:
        for path in ("/assets/styles.css", "/assets/app.js", "/assets/favicon.svg"):
            with self.subTest(path=path):
                response = self.client.get(path)
                try:
                    self.assertEqual(response.status_code, 200)
                finally:
                    response.close()

    def test_llms_txt_is_served(self) -> None:
        response = self.client.get("/llms.txt")
        try:
            self.assertEqual(response.status_code, 200)
            self.assertIn(
                "Student Performance Analytics", response.get_data(as_text=True)
            )
        finally:
            response.close()

    def test_redirects_and_security_headers(self) -> None:
        response = self.client.get("/")
        self.assertEqual(response.headers.get("X-Content-Type-Options"), "nosniff")
        self.assertEqual(response.headers.get("X-Frame-Options"), "DENY")


class ApiContractTests(unittest.TestCase):
    def setUp(self) -> None:
        app.config.update(TESTING=True)
        self.client = app.test_client()

    def test_health_reports_core_status(self) -> None:
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIn(data["status"], {"ok", "degraded"})
        self.assertIsInstance(data["core_ready"], bool)
        self.assertIn("exam", data["models"])
        self.assertIn("placement", data["models"])
        self.assertIsInstance(data["artifacts"], dict)

    def test_prediction_schema_matches_shared_config(self) -> None:
        response = self.client.get("/api/prediction-schema")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        for track in ("exam", "placement"):
            with self.subTest(track=track):
                schema_names = [field["name"] for field in data["tracks"][track]["fields"]]
                config_names = [field["name"] for field in fields_for(track)]
                self.assertEqual(schema_names, config_names)
                self.assertEqual(
                    data["tracks"][track]["feature_order"],
                    [field["column"] for field in fields_for(track)],
                )

    def test_summary_shape(self) -> None:
        response = self.client.get("/api/summary")
        if response.status_code == 503:
            self.skipTest("Processed datasets are not generated yet.")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertGreater(data["exam_rows"], 0)
        self.assertGreater(data["placement_rows"], 0)
        self.assertIs(data["datasets_are_independent"], True)
        self.assertIs(data["source_records_genuinely_linked"], False)

    def test_advisory_rules_are_marked_advisory(self) -> None:
        response = self.client.get("/api/advisory-rules")
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIs(data["advisory_only"], True)
        self.assertTrue(data["rules"])
        for rule in data["rules"]:
            self.assertIn("threshold", rule)
            self.assertIn("message", rule)

    def test_unknown_api_route_returns_json_404(self) -> None:
        response = self.client.get("/api/does-not-exist")
        self.assertEqual(response.status_code, 404)
        self.assertEqual(response.get_json()["error"], "Endpoint not found.")

    def test_wrong_method_returns_json_405(self) -> None:
        response = self.client.get("/api/predict/exam")
        self.assertEqual(response.status_code, 405)
        self.assertIn("error", response.get_json())


class ValidationTests(unittest.TestCase):
    def setUp(self) -> None:
        app.config.update(TESTING=True)
        self.client = app.test_client()

    def post(self, track: str, payload):
        return self.client.post(
            f"/api/predict/{track}",
            json=payload,
            content_type="application/json",
        )

    def test_non_object_body_is_rejected(self) -> None:
        response = self.client.post("/api/predict/exam", json=[1, 2, 3])
        self.assertEqual(response.status_code, 400)
        self.assertIn("JSON object", response.get_json()["error"])

    def test_missing_field_is_rejected(self) -> None:
        payload = sample_payload("exam")
        payload.pop("attendance")
        response = self.post("exam", payload)
        self.assertEqual(response.status_code, 400)
        data = response.get_json()
        self.assertEqual(data["field"], "attendance")
        self.assertIn("required", data["error"])

    def test_out_of_range_field_is_rejected(self) -> None:
        payload = sample_payload("exam")
        payload["attendance"] = 250
        response = self.post("exam", payload)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["field"], "attendance")

    def test_non_numeric_field_is_rejected(self) -> None:
        payload = sample_payload("exam")
        payload["hours_studied"] = "many"
        response = self.post("exam", payload)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["field"], "hours_studied")

    def test_boolean_is_not_accepted_as_number(self) -> None:
        payload = sample_payload("exam")
        payload["tutoring_sessions"] = True
        response = self.post("exam", payload)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["field"], "tutoring_sessions")

    def test_invalid_category_is_rejected(self) -> None:
        payload = sample_payload("placement")
        payload["internship_experience"] = "Sometimes"
        response = self.post("placement", payload)
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["field"], "internship_experience")

    def test_category_is_case_insensitive(self) -> None:
        if not PLACEMENT_ARTIFACT.exists():
            self.skipTest("Placement artifact is not generated yet.")
        payload = sample_payload("placement")
        payload["internship_experience"] = "yes"
        response = self.post("placement", payload)
        self.assertEqual(response.status_code, 200)

    def test_interventions_accept_partial_input(self) -> None:
        response = self.client.post(
            "/api/interventions",
            json={"attendance": 50, "communication_skills": 3},
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertIs(data["advisory_only"], True)
        self.assertEqual(sorted(data["provided_fields"]), ["attendance", "communication_skills"])
        self.assertGreaterEqual(len(data["suggestions"]), 2)

    def test_interventions_reject_out_of_range_value(self) -> None:
        response = self.client.post("/api/interventions", json={"attendance": 1000})
        self.assertEqual(response.status_code, 400)
        self.assertEqual(response.get_json()["field"], "attendance")


class PredictionTests(unittest.TestCase):
    def setUp(self) -> None:
        app.config.update(TESTING=True)
        self.client = app.test_client()

    @unittest.skipUnless(EXAM_ARTIFACT.exists(), "Exam artifact is not generated yet.")
    def test_exam_prediction_returns_bounded_result(self) -> None:
        response = self.client.post("/api/predict/exam", json=sample_payload("exam"))
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["task"], "regression")
        self.assertGreaterEqual(data["score"], 0)
        self.assertLessEqual(data["score"], 100)
        if data["interval"] is not None:
            low, high = data["interval"]
            self.assertLessEqual(low, data["score"])
            self.assertGreaterEqual(high, data["score"])
        self.assertTrue(data["limitations"])
        self.assertIn("advisory_suggestions", data)

    @unittest.skipUnless(PLACEMENT_ARTIFACT.exists(), "Placement artifact is not generated yet.")
    def test_placement_prediction_returns_probability(self) -> None:
        response = self.client.post(
            "/api/predict/placement", json=sample_payload("placement")
        )
        self.assertEqual(response.status_code, 200)
        data = response.get_json()
        self.assertEqual(data["task"], "classification")
        self.assertGreaterEqual(data["probability"], 0)
        self.assertLessEqual(data["probability"], 1)
        self.assertIs(data["exam_records_linked"], False)
        self.assertTrue(data["warning"])
        self.assertTrue(data["limitations"])

    @unittest.skipUnless(EXAM_ARTIFACT.exists(), "Exam artifact is not generated yet.")
    def test_exam_prediction_is_deterministic(self) -> None:
        payload = sample_payload("exam")
        first = self.client.post("/api/predict/exam", json=payload).get_json()
        second = self.client.post("/api/predict/exam", json=payload).get_json()
        self.assertEqual(first["score"], second["score"])


if __name__ == "__main__":
    unittest.main()
