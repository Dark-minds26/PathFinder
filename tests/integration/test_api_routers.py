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
@unittest.skipUnless(
    ARTIFACTS_PRESENT, "run `python main.py` to produce artifacts first"
)
class TestAPIRouters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from api.main import app

        cls.client = TestClient(app)

    def test_health_check(self):
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        body = response.json()
        self.assertEqual(body["status"], "ok")
        self.assertTrue(body["artifacts_loaded"])
        self.assertEqual(body["recommender_backend"], "lightgbm-lambdarank")
        self.assertIn(
            body["explainer_backend"], {"ConfiguredExplainer", "ShapTreeExplainer"}
        )
        self.assertIn(
            body["llm_backend"],
            {
                "GroqClient",
                "OpenAIClient",
                "MistralClient",
                "LocalStubLLMClient",
            },
        )

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
            json={
                "user_id": user_id,
                "message": "I know python basics, beginner level, visual learner, 8 hours per week",
            },
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
        self.assertEqual(len(explain_body["feature_attributions"]), 5)
        self.assertIn("learning_style_fit", explain_body["feature_attributions"])
        self.assertIn("time_fit", explain_body["feature_attributions"])

    def test_chat_profile_change_returns_updated_path(self):
        user_id = "api_profile_change_user"
        from api.dependencies import get_profile_store
        import json

        store = get_profile_store()
        data = (
            json.loads(store.path.read_text())
            if store.path.exists() and store.path.read_text().strip()
            else {}
        )
        data.pop(user_id, None)
        store.path.write_text(json.dumps(data, indent=2, sort_keys=True))
        r1 = self.client.post(
            "/profile/chat",
            json={"user_id": user_id, "message": "I want to become an AI engineer"},
        )
        self.assertEqual(r1.status_code, 200)
        self.assertTrue(r1.json()["roadmap_updated"])
        r2 = self.client.post(
            "/profile/chat",
            json={
                "user_id": user_id,
                "message": "I learn through practical projects and can study 7 hours a day",
            },
        )
        self.assertEqual(r2.status_code, 200)
        body = r2.json()
        self.assertEqual(body["learning_style"], "practice")
        self.assertEqual(body["weekly_hours"], 49.0)
        self.assertTrue(body["roadmap_updated"])
        self.assertGreater(len(body["path"]), 0)

    def test_explain_unknown_course_returns_404(self):
        response = self.client.get("/explain/not_a_real_course/some_user")
        self.assertEqual(response.status_code, 404)

    def test_assessment_submit_reroutes_on_failing_score(self):
        user_id = "api_test_user_2"
        self.client.post(
            "/profile/chat",
            json={
                "user_id": user_id,
                "message": "backend engineer, I know python basics, intermediate",
            },
        )
        response = self.client.post(
            "/assessment/submit",
            json={"user_id": user_id, "skill_id": "python_basics", "score": 40.0},
        )
        self.assertEqual(response.status_code, 200)
        skills_in_path = {step["skill_id"] for step in response.json()["path"]}
        self.assertIn("python_basics", skills_in_path)

    def test_profile_hydration_endpoint(self):
        user_id = "profile_hydration_user"
        from api.dependencies import get_profile_store
        import json

        store = get_profile_store()
        data = (
            json.loads(store.path.read_text())
            if store.path.exists() and store.path.read_text().strip()
            else {}
        )
        data.pop(user_id, None)
        store.path.write_text(json.dumps(data, indent=2, sort_keys=True))
        try:
            self.client.post(
                "/profile/chat",
                json={"user_id": user_id, "message": "I want to become an AI engineer"},
            )
            r = self.client.get(f"/profile/{user_id}")
            self.assertEqual(r.status_code, 200)
            self.assertEqual(r.json()["profile"]["goal_id"], "goal_ai_engineer")
            self.assertGreater(r.json()["profile_completeness"], 0)
        finally:
            data = (
                json.loads(store.path.read_text())
                if store.path.exists() and store.path.read_text().strip()
                else {}
            )
            data.pop(user_id, None)
            store.path.write_text(json.dumps(data, indent=2, sort_keys=True))


if __name__ == "__main__":
    unittest.main()


def _cleanup_live_profile(user_id):
    from api.dependencies import get_profile_store
    import json

    store = get_profile_store()
    path = store.path
    data = (
        json.loads(path.read_text())
        if path.exists() and path.read_text().strip()
        else {}
    )
    data.pop(user_id, None)
    path.write_text(json.dumps(data, indent=2, sort_keys=True))


class TestStage3APIContracts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from api.main import app

        cls.client = TestClient(app)

    def test_generate_requires_goal(self):
        uid = "stage3_no_goal_user"
        _cleanup_live_profile(uid)
        r = self.client.post("/path/generate", json={"user_id": uid})
        self.assertEqual(r.status_code, 400)
        self.assertEqual(
            r.json()["detail"],
            "Tell me what role you're targeting before generating a path.",
        )

    def test_assessment_score_is_validated(self):
        r = self.client.post(
            "/assessment/submit",
            json={"user_id": "u", "skill_id": "python_basics", "score": 101},
        )
        self.assertEqual(r.status_code, 422)

    def test_mastered_failed_reintroduced_transition(self):
        uid = "stage3_reroute_user"
        _cleanup_live_profile(uid)
        try:
            self.client.post(
                "/profile/chat",
                json={"user_id": uid, "message": "I want to become a backend engineer"},
            )
            self.client.post(
                "/profile/chat",
                json={"user_id": uid, "message": "I already know Python basics"},
            )
            # Self-report is stored as evidence, not mastery. Validate it before
            # asserting prerequisite-boundary behavior.
            from api.dependencies import get_profile_store

            get_profile_store().set_mastery(uid, "python_basics", 0.90, "assessment")
            before = self.client.post("/path/generate", json={"user_id": uid})
            self.assertEqual(before.status_code, 200)
            self.assertNotIn(
                "python_basics", {x["skill_id"] for x in before.json()["path"]}
            )
            failed = self.client.post(
                "/assessment/submit",
                json={"user_id": uid, "skill_id": "python_basics", "score": 40},
            )
            self.assertEqual(failed.status_code, 200)
            after = failed.json()
            self.assertIn("python_basics", {x["skill_id"] for x in after["path"]})
            from api.dependencies import get_profile_store

            self.assertNotIn("python_basics", get_profile_store().get(uid)["skill_ids"])
        finally:
            _cleanup_live_profile(uid)

    def test_frontend_contracts(self):
        html = Path("api/static/index.html").read_text(encoding="utf-8")
        self.assertIn('id="generate-btn"', html)
        self.assertNotIn(
            "score.textContent = 'predicted score ' + step.predicted_score", html
        )
        self.assertIn(
            "Tell me what role you're targeting before generating a path.", html
        )
        self.assertIn("checkpoint", html.lower())
        self.assertIn("Adaptive trail", html)
        self.assertIn("Profile completeness", html)
        self.assertIn("Learning progress", html)
        self.assertIn("Take checkpoint", html)
        self.assertIn("/profile/${encodeURIComponent(USER_ID)}", html)
        self.assertNotIn("prompt(q.question", html)


class TestPhase3AssessmentLifecycle(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from api.main import app

        cls.client = TestClient(app)

    def test_api_mastered_failed_reintroduced_and_revalidated(self):
        uid = "phase3_assessment_lifecycle_user"
        _cleanup_live_profile(uid)
        try:
            self.client.post(
                "/profile/chat",
                json={"user_id": uid, "message": "I want to become a backend engineer"},
            )
            from api.dependencies import get_profile_store

            store = get_profile_store()
            store.set_mastery(uid, "python_basics", 1.0, "assessment")

            # Mastered skill is outside the normal path.
            before = self.client.post("/path/generate", json={"user_id": uid})
            self.assertEqual(before.status_code, 200)
            self.assertNotIn(
                "python_basics", {x["skill_id"] for x in before.json()["path"]}
            )

            # Real checkpoint failure transitions validated -> failed and reintroduces the skill.
            failed = self.client.post(
                "/assessment/submit",
                json={
                    "user_id": uid,
                    "skill_id": "python_basics",
                    "answers": {
                        "python_basics_1": 0,
                        "python_basics_2": 0,
                        "python_basics_3": 0,
                    },
                },
            )
            self.assertEqual(failed.status_code, 200)
            p = store.get(uid)
            self.assertEqual(p["mastery_state"]["python_basics"]["status"], "failed")
            self.assertIn(
                "python_basics", {x["skill_id"] for x in failed.json()["path"]}
            )
            self.assertTrue(
                any(
                    x.get("type") == "assessment_failed"
                    and x.get("previous_status") == "validated"
                    for x in p["learning_history"]
                )
            )

            # Passing a real checkpoint validates it and removes it from the path again.
            passed = self.client.post(
                "/assessment/submit",
                json={
                    "user_id": uid,
                    "skill_id": "python_basics",
                    "answers": {
                        "python_basics_1": 1,
                        "python_basics_2": 1,
                        "python_basics_3": 1,
                    },
                },
            )
            self.assertEqual(passed.status_code, 200)
            p = store.get(uid)
            self.assertEqual(p["mastery_state"]["python_basics"]["status"], "validated")
            self.assertTrue(
                any(
                    x.get("type") == "assessment_validated"
                    and x.get("previous_status") == "failed"
                    for x in p["learning_history"]
                )
            )
            self.assertNotIn(
                "python_basics", {x["skill_id"] for x in passed.json()["path"]}
            )
        finally:
            _cleanup_live_profile(uid)
