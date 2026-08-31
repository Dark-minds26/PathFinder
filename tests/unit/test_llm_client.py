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
        # Explicitly simulate the optional dependency being unavailable;
        # this test must not depend on the machine's installed packages.
        import builtins

        real_import = builtins.__import__

        def missing_groq(name, *args, **kwargs):
            if name == "groq":
                raise ImportError("simulated missing groq package")
            return real_import(name, *args, **kwargs)

        with (
            patch.dict(os.environ, {"GROQ_API_KEY": "fake", "LLM_PROVIDER": "groq"}),
            patch("builtins.__import__", side_effect=missing_groq),
        ):
            client = get_llm_client()
        self.assertIsInstance(client, LocalStubLLMClient)


class TestLocalStubLLMClient(unittest.TestCase):
    def setUp(self):
        self.client = LocalStubLLMClient()

    def test_extracts_goal_by_title_substring(self):
        result = self.client.profile_turn(
            [], "I want to be a backend engineer", GOALS, SKILLS
        )
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

    def test_real_llm_labels_are_canonicalized_safely(self):
        from src.recommender.llm.llm_client import _validate_profile_result

        result = _validate_profile_result(
            {
                "reply": "ok",
                "goal_id": "AI engineer",
                "skill_ids": ["Python"],
                "unmastered_skill_ids": [],
                "experience_level": "intermediate",
                "learning_style": "hands-on",
                "weekly_hours": "7 hours/day",
                "interests": ["genai"],
                "roadmap_preferences": {},
            },
            [{"goal_id": "goal_ai_engineer", "title": "AI engineer"}],
            [{"skill_id": "python_basics", "skill_name": "Python basics"}],
        )
        self.assertEqual(result["goal_id"], "goal_ai_engineer")
        self.assertEqual(result["skill_ids"], ["python_basics"])
        self.assertEqual(result["learning_style"], "practice")
        self.assertEqual(result["weekly_hours"], 49.0)
        self.assertEqual(result["interests"], ["generative_ai"])

    def test_valid_profile_json_is_parsed(self):
        from src.recommender.llm.llm_client import _parse_profile_json

        result = _parse_profile_json(
            '{"reply":"ok","extracted":{"goal_id":"goal_backend_eng","skill_ids":["python_basics"],'
            '"unmastered_skill_ids":[],"experience_level":"intermediate",'
            '"learning_style":"practice","weekly_hours":8}}'
        )
        self.assertEqual(result["goal_id"], "goal_backend_eng")
        self.assertEqual(result["skill_ids"], ["python_basics"])
        self.assertEqual(result["weekly_hours"], 8)

    def test_prompt_contains_profile_contract(self):
        from src.recommender.llm.prompt_templates import SYSTEM_PROMPT_PROFILING

        self.assertIn("MASTERED skills only", SYSTEM_PROMPT_PROFILING)
        self.assertIn("unmastered_skill_ids", SYSTEM_PROMPT_PROFILING)
        self.assertIn("weekly_hours", SYSTEM_PROMPT_PROFILING)
        self.assertIn("canonical ids", SYSTEM_PROMPT_PROFILING)

    def test_negative_skill_is_not_marked_mastered(self):
        result = self.client.profile_turn(
            [], "I am weak at Python and need to learn it", [], SKILLS
        )
        self.assertNotIn("python_basics", result["skill_ids"])
        self.assertIn("python_basics", result["unmastered_skill_ids"])

    def test_extracts_weekly_hours(self):
        result = self.client.profile_turn(
            [], "I can study 8 hours per week", [], SKILLS
        )
        self.assertEqual(result["weekly_hours"], 8.0)

    def test_unknown_skill_id_is_rejected(self):
        from src.recommender.llm.llm_client import _validate_profile_result

        bad = {
            "reply": "ok",
            "goal_id": None,
            "skill_ids": ["fake"],
            "unmastered_skill_ids": [],
            "experience_level": None,
            "learning_style": None,
            "weekly_hours": None,
        }
        with self.assertRaises(Exception):
            _validate_profile_result(bad, [], SKILLS)

    def test_malformed_json_is_rejected(self):
        from src.recommender.llm.llm_client import _parse_profile_json

        with self.assertRaises(Exception):
            _parse_profile_json("not json")

    def test_missing_required_json_field_is_rejected(self):
        from src.recommender.llm.llm_client import _parse_profile_json

        with self.assertRaises(Exception):
            _parse_profile_json('{"reply":"ok"}')

    def test_no_match_returns_none_and_prompts_for_more(self):
        result = self.client.profile_turn([], "hello there", GOALS, SKILLS)
        self.assertIsNone(result["goal_id"])
        self.assertEqual(result["skill_ids"], [])
        self.assertIn("tell me", result["reply"].lower())

    def test_groq_api_failure_is_not_silently_swallowed(self):
        from unittest.mock import Mock
        from src.recommender.llm.llm_client import GroqClient

        client = object.__new__(GroqClient)
        client.model = "test-model"
        client.client = Mock()
        client.client.chat.completions.create.side_effect = RuntimeError(
            "API unavailable"
        )
        with self.assertRaises(Exception):
            client.profile_turn([], "hello", GOALS, SKILLS)

    def test_explain_returns_readable_fallback_paragraph(self):
        text = self.client.explain(
            "REST APIs deep dive",
            "Backend engineer",
            {"goal_alignment": 0.6, "skill_gap_match": -0.3, "popularity": 0.1},
        )
        self.assertIn("REST APIs deep dive", text)
        self.assertGreater(len(text), 20)


if __name__ == "__main__":
    unittest.main()

    def test_profile_intent_is_explicit_for_roadmap_questions(self):
        result = self.client.profile_turn(
            [], "What are the supported roadmap paths?", GOALS, SKILLS
        )
        self.assertEqual(result["intent"], "roadmap_question")

    def test_profile_intent_is_explicit_for_unsupported_goals(self):
        result = self.client.profile_turn(
            [], "I want to become an IAS officer", GOALS, SKILLS
        )
        self.assertEqual(result["intent"], "unsupported_goal")
