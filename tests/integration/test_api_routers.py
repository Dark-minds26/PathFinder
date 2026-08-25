"""FastAPI router tests against real trained artifacts. Needs
`pip install fastapi httpx` (not available in the sandbox this was
built in - see the Phase 3 writeup) and `python main.py` run first.
"""
import unittest
from pathlib import Path

ARTIFACTS_PRESENT = Path("artifacts/model/model.pkl").exists()

try:
    from fastapi.testclient import TestClient
    FASTAPI_AVAILABLE = True
except ImportError:
    FASTAPI_AVAILABLE = False


@unittest.skipUnless(FASTAPI_AVAILABLE, "pip install fastapi httpx")
@unittest.skipUnless(ARTIFACTS_PRESENT, "run `python main.py` to produce artifacts first")
class TestAPIRouters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from api.main import app
        cls.client = TestClient(app)

    def test_health_check(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_full_flow_new_user_chat_to_path_to_explain(self):
        user_id = "api_test_user_1"

        chat_resp = self.client.post(
            "/profile/chat",
            json={"user_id": user_id, "message": "I want to become a data scientist"},
        )
        self.assertEqual(chat_resp.status_code, 200)
        self.assertIn("reply", chat_resp.json())

        self.client.post(
            "/profile/chat",
            json={"user_id": user_id, "message": "I know python basics, beginner level, visual learner"},
        )

        path_resp = self.client.post("/path/generate", json={"user_id": user_id})
        self.assertEqual(path_resp.status_code, 200)
        body = path_resp.json()
        self.assertEqual(body["source"], "live_profile")
        self.assertGreater(len(body["path"]), 0)

        top_course = body["path"][0]["course_id"]
        explain_resp = self.client.get(f"/explain/{top_course}/{user_id}")
        self.assertEqual(explain_resp.status_code, 200)
        explain_body = explain_resp.json()
        self.assertIn("explanation", explain_body)
        self.assertEqual(len(explain_body["feature_attributions"]), 6)

    def test_explain_unknown_course_returns_404(self):
        response = self.client.get("/explain/not_a_real_course/some_user")
        self.assertEqual(response.status_code, 404)

    def test_assessment_submit_reroutes_on_failing_score(self):
        user_id = "api_test_user_2"
        self.client.post(
            "/profile/chat",
            json={"user_id": user_id, "message": "backend engineer, I know python basics, intermediate"},
        )
        response = self.client.post(
            "/assessment/submit",
            params={"user_id": user_id, "skill_id": "python_basics", "score": 40.0},
        )
        self.assertEqual(response.status_code, 200)
        skills_in_path = {step["skill_id"] for step in response.json()["path"]}
        self.assertIn("python_basics", skills_in_path)


if __name__ == "__main__":
    unittest.main()
