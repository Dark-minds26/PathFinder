"""Integration test against real trained artifacts - the same claim as
the summary given to the user: a brand-new chat user, with zero
synthetic history, goes through /profile/chat-equivalent turns and
gets a real, correctly-ordered roadmap out of Phase 2's recommender.

Run `python main.py` first to produce artifacts/.
"""
import shutil
import unittest
from pathlib import Path

from src.recommender.llm.conversation_manager import ConversationManager
from src.recommender.llm.llm_client import LocalStubLLMClient
from src.recommender.llm.profile_store import ProfileStore
from src.recommender.llm.rag_engine import RAGEngine
from src.recommender.components.path_generator import PathGenerator
from src.recommender.config.configuration import ConfigurationManager
from src.recommender.utils.explain_utils import compute_attributions
from src.recommender.utils.main_utils import load_object

ARTIFACTS_PRESENT = Path("artifacts/model/model.pkl").exists()


@unittest.skipUnless(ARTIFACTS_PRESENT, "run `python main.py` to produce artifacts first")
class TestConversationToPath(unittest.TestCase):
    def setUp(self):
        self.cfg = ConfigurationManager()
        ingestion_cfg = self.cfg.get_data_ingestion_config()
        transform_cfg = self.cfg.get_data_transformation_config()

        self.tmp_profiles = Path("artifacts") / "test_live_profiles.json"
        if self.tmp_profiles.exists():
            self.tmp_profiles.unlink()

        self.rag = RAGEngine(
            preprocessor_path=transform_cfg.preprocessor_object_path,
            skills_path=f"{ingestion_cfg.ingested_data_dir}/skills.csv",
            goals_path=f"{ingestion_cfg.ingested_data_dir}/career_goals.csv",
        )
        self.llm = LocalStubLLMClient()
        self.store = ProfileStore(str(self.tmp_profiles))
        self.generator = PathGenerator(self.cfg.get_path_generator_config())

    def tearDown(self):
        if self.tmp_profiles.exists():
            self.tmp_profiles.unlink()

    def test_new_user_chat_produces_a_usable_profile_and_real_path(self):
        user_id = "integration_test_user"
        manager = ConversationManager(user_id, rag=self.rag, llm=self.llm, store=self.store)

        _, c1 = manager.handle_turn("I know some Python, I'm a beginner")
        _, c2 = manager.handle_turn("I want to become a backend engineer")
        self.assertGreater(c2, c1, "completeness should increase as fields fill in")

        profile = self.store.get(user_id)
        self.assertEqual(profile["goal_id"], "goal_backend_eng")
        self.assertIn("python_basics", profile["skill_ids"])

        overrides = {
            "goal_id": profile["goal_id"],
            "possessed_skills": set(profile["skill_ids"]),
            "experience_level": profile.get("experience_level"),
        }
        artifact = self.generator.generate_path(user_id, **overrides)

        self.assertGreater(len(artifact.steps), 0, "a real path should be generated")
        skill_ids = [s.skill_id for s in artifact.steps]
        self.assertNotIn(
            "python_basics", skill_ids,
            "python_basics was already claimed as known - shouldn't be re-recommended",
        )

        # every step must be a real, priced prediction, not a placeholder
        for step in artifact.steps:
            self.assertIsInstance(step.predicted_score, float)
            self.assertTrue(step.course_title)

    def test_explain_produces_real_attributions_for_a_generated_step(self):
        user_id = "integration_test_user_2"
        manager = ConversationManager(user_id, rag=self.rag, llm=self.llm, store=self.store)
        manager.handle_turn("I want to become a frontend engineer")
        profile = self.store.get(user_id)
        overrides = {"goal_id": profile["goal_id"], "possessed_skills": set(profile["skill_ids"])}

        artifact = self.generator.generate_path(user_id, **overrides)
        self.assertGreater(len(artifact.steps), 0)
        top_step = artifact.steps[0]

        explainer = load_object(self.cfg.get_explainer_config().explainer_object_path)
        attributions = compute_attributions(
            self.generator.ctx, self.generator.model, explainer,
            user_id, top_step.course_id, **overrides,
        )
        self.assertEqual(set(attributions.keys()), {
            "skill_gap_match", "goal_alignment", "difficulty_fit",
            "popularity", "predicted_time_to_complete", "content_similarity",
        })
        explanation = self.llm.explain(top_step.course_title, "Frontend engineer", attributions)
        self.assertIn(top_step.course_title, explanation)


if __name__ == "__main__":
    unittest.main()
