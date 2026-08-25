import os
import unittest
from unittest.mock import patch

from src.recommender.llm.llm_client import LocalStubLLMClient, get_llm_client

GOALS = [{"goal_id": "goal_backend_eng", "title": "Backend engineer"}]
SKILLS = [
    {"skill_id": "python_basics", "skill_name": "Python basics"},
    {"skill_id": "sql_basics", "skill_name": "SQL basics"},
]


class TestGetLLMClient(unittest.TestCase):
    def test_falls_back_when_no_api_key_set(self):
        with patch.dict(os.environ, {}, clear=True):
            client = get_llm_client()
        self.assertIsInstance(client, LocalStubLLMClient)

    def test_falls_back_when_key_set_but_package_missing(self):
        # groq/openai aren't installed in this environment either way,
        # so this also covers the ImportError branch of the factory.
        with patch.dict(os.environ, {"GROQ_API_KEY": "fake", "LLM_PROVIDER": "groq"}):
            client = get_llm_client()
        self.assertIsInstance(client, LocalStubLLMClient)


class TestLocalStubLLMClient(unittest.TestCase):
    def setUp(self):
        self.client = LocalStubLLMClient()

    def test_extracts_goal_by_title_substring(self):
        result = self.client.profile_turn([], "I want to be a backend engineer", GOALS, SKILLS)
        self.assertEqual(result["goal_id"], "goal_backend_eng")

    def test_extracts_skill_by_significant_word_not_just_full_name(self):
        result = self.client.profile_turn([], "I already know some Python", [], SKILLS)
        self.assertIn("python_basics", result["skill_ids"])

    def test_does_not_match_unrelated_skill(self):
        result = self.client.profile_turn([], "I already know some Python", [], SKILLS)
        self.assertNotIn("sql_basics", result["skill_ids"])

    def test_extracts_experience_and_style_keywords(self):
        result = self.client.profile_turn(
            [], "I'm a beginner and I learn best by reading", GOALS, SKILLS
        )
        self.assertEqual(result["experience_level"], "beginner")
        self.assertEqual(result["learning_style"], "reading")

    def test_no_match_returns_none_and_prompts_for_more(self):
        result = self.client.profile_turn([], "hello there", GOALS, SKILLS)
        self.assertIsNone(result["goal_id"])
        self.assertEqual(result["skill_ids"], [])
        self.assertIn("tell me", result["reply"].lower())

    def test_explain_returns_readable_fallback_paragraph(self):
        text = self.client.explain(
            "REST APIs deep dive", "Backend engineer",
            {"goal_alignment": 0.6, "skill_gap_match": -0.3, "popularity": 0.1},
        )
        self.assertIn("REST APIs deep dive", text)
        self.assertGreater(len(text), 20)


if __name__ == "__main__":
    unittest.main()
