"""Integration tests against real trained artifacts (run `python
main.py` first) - exercises RAG retrieval and the full profiling
conversation with LocalStubLLMClient, no network or API key needed."""
import shutil
import tempfile
import unittest
from pathlib import Path

from src.recommender.config.configuration import ConfigurationManager
from src.recommender.llm.conversation_manager import ConversationManager
from src.recommender.llm.llm_client import LocalStubLLMClient
from src.recommender.llm.profile_store import ProfileStore
from src.recommender.llm.rag_engine import RAGEngine

ARTIFACTS_PRESENT = Path("artifacts/model/model.pkl").exists()


@unittest.skipUnless(ARTIFACTS_PRESENT, "run `python main.py` to produce artifacts first")
class TestRAGRetrieval(unittest.TestCase):
    def setUp(self):
        cfg = ConfigurationManager()
        ingestion_dir = cfg.get_data_ingestion_config().ingested_data_dir
        self.rag = RAGEngine(
            preprocessor_path=cfg.get_data_transformation_config().preprocessor_object_path,
            skills_path=f"{ingestion_dir}/skills.csv",
            goals_path=f"{ingestion_dir}/career_goals.csv",
        )

    def test_retrieve_skills_returns_requested_count(self):
        results = self.rag.retrieve_skills("I know some python and pandas", top_k=5)
        self.assertEqual(len(results), 5)
        self.assertTrue(all("skill_id" in r and "skill_name" in r for r in results))

    def test_retrieve_skills_ranks_relevant_skill_first(self):
        results = self.rag.retrieve_skills("python programming basics", top_k=3)
        self.assertEqual(results[0]["skill_id"], "python_basics")

    def test_retrieve_goals_ranks_relevant_goal_first(self):
        results = self.rag.retrieve_goals("I want to become a data scientist", top_k=2)
        self.assertEqual(results[0]["goal_id"], "goal_data_scientist")


@unittest.skipUnless(ARTIFACTS_PRESENT, "run `python main.py` to produce artifacts first")
class TestConversationalProfiling(unittest.TestCase):
    def setUp(self):
        cfg = ConfigurationManager()
        ingestion_dir = cfg.get_data_ingestion_config().ingested_data_dir
        self.rag = RAGEngine(
            preprocessor_path=cfg.get_data_transformation_config().preprocessor_object_path,
            skills_path=f"{ingestion_dir}/skills.csv",
            goals_path=f"{ingestion_dir}/career_goals.csv",
        )
        self.tmp_dir = Path(tempfile.mkdtemp())
        self.store = ProfileStore(str(self.tmp_dir / "profiles.json"))
        self.llm = LocalStubLLMClient()

    def tearDown(self):
        shutil.rmtree(self.tmp_dir, ignore_errors=True)

    def test_full_conversation_builds_a_complete_profile(self):
        manager = ConversationManager("new_user_1", self.rag, self.llm, self.store)

        reply1, c1 = manager.handle_turn(
            "Hi, I want to become a data scientist eventually."
        )
        self.assertIsInstance(reply1, str)
        self.assertGreater(c1, 0.0)
        self.assertEqual(self.store.get("new_user_1").get("goal_id"), "goal_data_scientist")

        reply2, c2 = manager.handle_turn(
            "I already know python basics and sql basics. I'm intermediate level."
        )
        self.assertGreater(c2, c1)
        profile = self.store.get("new_user_1")
        self.assertIn("python_basics", profile["skill_ids"])
        self.assertIn("sql_basics", profile["skill_ids"])
        self.assertEqual(profile["experience_level"], "intermediate")

        reply3, c3 = manager.handle_turn("I like hands-on practice the most.")
        self.assertEqual(c3, 1.0)
        self.assertEqual(self.store.get("new_user_1")["learning_style"], "practice")

    def test_conversation_history_accumulates(self):
        manager = ConversationManager("new_user_2", self.rag, self.llm, self.store)
        manager.handle_turn("hello")
        manager.handle_turn("I know javascript basics")
        self.assertEqual(len(manager.history), 4)  # 2 user + 2 assistant turns


if __name__ == "__main__":
    unittest.main()
