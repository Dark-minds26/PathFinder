import shutil
import tempfile
import unittest
from pathlib import Path

from src.recommender.llm.profile_store import ProfileStore
from src.recommender.assessment_engine import score_answers


class TestAssessmentMasteryContract(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.store = ProfileStore(str(self.tmp / "profiles.json"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_assessment_scores_map_to_deterministic_states(self):
        uid = "u"
        self.store.update(uid, goal_id="goal_ai_engineer")
        self.store.set_mastery(uid, "docker_basics", 1.0, "assessment")
        self.assertEqual(
            self.store.get(uid)["mastery_state"]["docker_basics"]["status"], "validated"
        )
        self.assertIn("docker_basics", self.store.get(uid)["skill_ids"])

        self.store.set_mastery(uid, "docker_basics", 0.75, "assessment")
        p = self.store.get(uid)
        self.assertEqual(p["mastery_state"]["docker_basics"]["status"], "needs_review")
        self.assertNotIn("docker_basics", p["skill_ids"])
        self.assertIn("docker_basics", p["unmastered_skill_ids"])

        self.store.set_mastery(uid, "docker_basics", 0.40, "assessment")
        p = self.store.get(uid)
        self.assertEqual(p["mastery_state"]["docker_basics"]["status"], "failed")
        self.assertNotIn("docker_basics", p["skill_ids"])

        self.store.set_mastery(uid, "docker_basics", 1.0, "assessment")
        p = self.store.get(uid)
        self.assertEqual(p["mastery_state"]["docker_basics"]["status"], "validated")
        self.assertIn("docker_basics", p["skill_ids"])
        self.assertNotIn("docker_basics", p["unmastered_skill_ids"])

    def test_checkpoint_answer_scoring_is_deterministic(self):
        answers = {"docker_basics_1": 1, "docker_basics_2": 1, "docker_basics_3": 1}
        self.assertEqual(score_answers("docker_basics", answers), 100.0)
        self.assertEqual(
            score_answers("docker_basics", {**answers, "docker_basics_1": 0}), 66.7
        )


if __name__ == "__main__":
    unittest.main()
