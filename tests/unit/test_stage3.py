import unittest
from pathlib import Path
from unittest.mock import Mock
from src.recommender.components.path_generator import PathGenerator
from src.recommender.entity.artifact_entity import PathGeneratorArtifact

class TestStage3PathStates(unittest.TestCase):
    def test_artifact_state(self):
        a=PathGeneratorArtifact('u',[],state='mastered',message='done'); self.assertEqual(a.state,'mastered'); self.assertEqual(a.message,'done')
    def test_no_candidates(self):
        g=object.__new__(PathGenerator); g.config=Mock(max_path_length=20,candidate_courses_per_skill=5,min_confidence=0.0); g._ctx=Mock(); g._model=Mock(); g._course_skill_map={}; g._ctx.missing_skills_for_user.return_value=['s1']
        a=g.generate_path('u',goal_id='g',possessed_skills=set()); self.assertEqual(a.state,'no_candidates')

class TestStage3Docs(unittest.TestCase):
    def test_deployment_storage_honesty(self):
        t=Path('DEPLOYMENT.md').read_text(); self.assertIn('local/demo storage only',t); self.assertIn('not a production database',t)
    def test_setup_metadata(self):
        t=Path('setup.py').read_text(); self.assertIn('"api*"',t); self.assertIn('"static/*"',t)
