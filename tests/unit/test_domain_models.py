import unittest
from pydantic import ValidationError

from src.recommender.entity.domain_models import Goal, Mastery, Resource, Skill, User


class TestDomainModels(unittest.TestCase):
    def test_user_exposes_explicit_mastery_states(self):
        user = User(
            user_id="u1",
            mastery={
                "python": Mastery(skill_id="python", score=0.92, status="validated"),
                "docker": Mastery(skill_id="docker", score=0.35, status="failed"),
            },
        )
        self.assertEqual(user.validated_skill_ids, {"python"})
        self.assertEqual(user.failed_skill_ids, {"docker"})

    def test_resource_supports_course_project_and_assessment_contract(self):
        resource = Resource(
            resource_id="r1",
            title="Build a model API",
            resource_type="project",
            skill_ids=["model_serving"],
            duration_hours=8,
            difficulty="intermediate",
            format="interactive",
        )
        self.assertEqual(resource.resource_type, "project")

    def test_resource_rejects_duplicate_skill_ids(self):
        with self.assertRaises(ValidationError):
            Resource(resource_id="r1", title="Bad", skill_ids=["python", "python"])

    def test_goal_and_skill_have_explicit_prerequisites(self):
        skill = Skill(skill_id="ml", name="Machine learning", category="ai", prerequisite_skill_ids=["stats"])
        goal = Goal(goal_id="g1", title="AI engineer", domain="ai", required_skill_ids=[skill.skill_id])
        self.assertEqual(skill.prerequisite_skill_ids, ["stats"])
        self.assertEqual(goal.required_skill_ids, ["ml"])
