import json, shutil, tempfile
import unittest
from pathlib import Path
from src.recommender.llm.profile_store import ProfileStore
from src.recommender.goal_intelligence import normalize_goal
from src.recommender.assessment_engine import score_answers
from src.recommender.project_catalog import project_for

class TestStage4LearningLoop(unittest.TestCase):
    def test_dynamic_goal_is_understood_without_being_fabricated(self):
        curated=[{"goal_id":"goal_ai_engineer","title":"AI engineer"}]
        spec=normalize_goal("I want to become a cybersecurity analyst",curated,[])
        self.assertIsNotNone(spec); self.assertEqual(spec.source,"dynamic"); self.assertFalse(spec.resource_available)
        self.assertGreater(len(spec.competencies),2)

    def test_mastery_and_history_are_persistent(self):
        d=Path(tempfile.mkdtemp())
        try:
            store=ProfileStore(str(d/'p.json')); store.update('u',goal_id='goal_ai_engineer')
            store.set_mastery('u','docker_basics',.45,'assessment')
            self.assertEqual(store.get('u')['mastery']['docker_basics'],.45)
            self.assertIn('docker_basics',store.get('u')['unmastered_skill_ids'])
            store.set_mastery('u','docker_basics',.92,'assessment')
            self.assertIn('docker_basics',store.get('u')['skill_ids'])
        finally: shutil.rmtree(d,ignore_errors=True)

    def test_real_checkpoint_scoring(self):
        questions={'docker_basics_1':1,'docker_basics_2':1,'docker_basics_3':1}
        self.assertEqual(score_answers('docker_basics',questions),100.0)
        self.assertEqual(score_answers('docker_basics',{'docker_basics_1':0,'docker_basics_2':1,'docker_basics_3':1}),66.7)

    def test_projects_are_first_class(self):
        p=project_for('rag_systems')
        self.assertEqual(p['project_id'],'project_rag_systems'); self.assertIn('rag_systems',p['skills'])

if __name__=='__main__': unittest.main()
