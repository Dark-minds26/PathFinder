import shutil
import tempfile
import unittest
from pathlib import Path

from src.recommender.llm.profile_store import ProfileStore


class TestProfileStore(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.store = ProfileStore(str(self.tmp / "profiles.json"))

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_new_user_has_empty_profile(self):
        self.assertEqual(self.store.get("nobody"), {"skill_ids": []})
        self.assertEqual(self.store.completeness("nobody"), 0.0)

    def test_completeness_increases_as_fields_fill_in(self):
        self.store.update("u1", goal_id="goal_x")
        self.assertAlmostEqual(self.store.completeness("u1"), 0.25)
        self.store.update("u1", experience_level="beginner")
        self.assertAlmostEqual(self.store.completeness("u1"), 0.5)
        self.store.update("u1", new_skill_ids=["a", "b"])
        self.assertAlmostEqual(self.store.completeness("u1"), 0.75)
        self.store.update("u1", learning_style="reading")
        self.assertAlmostEqual(self.store.completeness("u1"), 1.0)

    def test_skills_accumulate_across_updates_without_duplicates(self):
        self.store.update("u1", new_skill_ids=["a", "b"])
        self.store.update("u1", new_skill_ids=["b", "c"])
        self.assertEqual(sorted(self.store.get("u1")["skill_ids"]), ["a", "b", "c"])

    def test_mark_unmastered_removes_a_previously_recorded_skill(self):
        self.store.update("u1", new_skill_ids=["a", "b"])
        self.store.mark_unmastered("u1", "a")
        self.assertEqual(self.store.get("u1")["skill_ids"], ["b"])

    def test_persists_across_new_store_instances_same_path(self):
        self.store.update("u1", goal_id="goal_x")
        reopened = ProfileStore(str(self.tmp / "profiles.json"))
        self.assertEqual(reopened.get("u1")["goal_id"], "goal_x")


if __name__ == "__main__":
    unittest.main()
