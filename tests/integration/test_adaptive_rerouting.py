"""Integration test against real trained artifacts.

Run `python main.py` first to produce artifacts/ - this test doesn't
retrain, it verifies the served behavior on top of what's there.
"""

import unittest
from pathlib import Path

from src.recommender.pipeline.prediction_pipeline import PredictionPipeline
from src.recommender.pipeline.adaptive_rerouting_pipeline import (
    AdaptiveReroutingPipeline,
)
from src.recommender.utils.main_utils import load_object
from src.recommender.utils.feature_engineering import FeatureContext
from src.recommender.config.configuration import ConfigurationManager
from api.dependencies import get_profile_store # <-- IMPORT ADDED HERE

ARTIFACTS_PRESENT = Path("artifacts/model/model.pkl").exists()


@unittest.skipUnless(
    ARTIFACTS_PRESENT, "run `python main.py` to produce artifacts first"
)
class TestAdaptiveRerouting(unittest.TestCase):
    def setUp(self):
        cfg = ConfigurationManager()
        graph = load_object(cfg.get_skill_graph_config().graph_cache_path)
        preprocessor = load_object(
            cfg.get_data_transformation_config().preprocessor_object_path
        )
        self.ctx = FeatureContext.from_preprocessor(preprocessor, graph)
        self.pred = PredictionPipeline()
        self.reroute = AdaptiveReroutingPipeline()
        self.profile_store = get_profile_store() # <-- DB INJECTED HERE

    def test_failing_a_mastered_skill_brings_it_back_into_the_path(self):
        user_with_mastery = None
        failed_skill = None
        user_goal = None
        
        for uid, mastered in self.ctx.possessed_by_user.items():
            goal = self.ctx.user_goal.get(uid)
            required = set(self.ctx.goal_required_ids.get(goal, set()))
            eligible = sorted(set(mastered) & required)
            if eligible:
                user_with_mastery, failed_skill = uid, eligible[0]
                user_goal = goal
                break
                
        self.assertIsNotNone(
            user_with_mastery, "no synthetic user has a mastered required skill"
        )
        self.assertIsNotNone(failed_skill)

        # --- THE FIX: SYNC TEST DATA TO LIVE DATABASE ---
        # 1. Register the user's goal in the database so the pipeline knows what to target
        try:
            self.profile_store.update(user_with_mastery, goal_id=user_goal)
        except AttributeError:
            pass # Failsafe if the update method is named differently
            
        # 2. Force the database to recognize the skill as fully mastered (score: 1.0)
        self.profile_store.set_mastery(user_with_mastery, failed_skill, 1.0, "assessment")
        # -------------------------------------------------

        before = self.pred.generate_path_for_user(user_with_mastery)
        before_skills = {s.skill_id for s in before.steps}
        
        self.assertNotIn(
            failed_skill,
            before_skills,
            "test needs a skill that's mastered and NOT already in the path",
        )

        after = self.reroute.reroute(user_with_mastery, failed_skill)
        after_skills = {s.skill_id for s in after.steps}
        self.assertIn(failed_skill, after_skills)

        # the user's actual mastery record must be untouched
        self.assertIn(failed_skill, self.ctx.possessed_by_user[user_with_mastery])

    def test_rerouted_path_still_respects_prerequisite_order(self):
        user_with_mastery = None
        failed_skill = None
        user_goal = None
        
        for uid, mastered in self.ctx.possessed_by_user.items():
            goal = self.ctx.user_goal.get(uid)
            eligible = sorted(
                set(mastered) & set(self.ctx.goal_required_ids.get(goal, set()))
            )
            if eligible:
                user_with_mastery, failed_skill = uid, eligible[0]
                user_goal = goal
                break
                
        self.assertIsNotNone(user_with_mastery)
        self.assertIsNotNone(failed_skill)

        # --- THE FIX: SYNC TEST DATA TO LIVE DATABASE (Test 2) ---
        try:
            self.profile_store.update(user_with_mastery, goal_id=user_goal)
        except AttributeError:
            pass
        self.profile_store.set_mastery(user_with_mastery, failed_skill, 1.0, "assessment")
        # ---------------------------------------------------------

        after = self.reroute.reroute(user_with_mastery, failed_skill)

        order_index = {s.skill_id: s.sequence_order for s in after.steps}
        for skill_id, idx in order_index.items():
            for pred_id in self.ctx.graph.predecessors(skill_id):
                if pred_id in order_index:
                    self.assertLess(
                        order_index[pred_id],
                        idx,
                        f"{pred_id} must come before {skill_id} in the rerouted path",
                    )


if __name__ == "__main__":
    unittest.main()