import unittest
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

from src.recommender.components.data_transformation import _temporal_rows, split_user_ids
from src.recommender.components.path_generator import PathGenerator
from src.recommender.entity.artifact_entity import DataValidationArtifact, ModelEvaluatorArtifact
from src.recommender.pipeline.training_pipeline import TrainingPipeline
from src.recommender.utils.explain_utils import compute_attributions


class FakeContext:
    def missing_skills_for_user(self, user_id, possessed_skills=None, **kwargs):
        possessed_skills = set(possessed_skills or set())
        return [] if "s1" in possessed_skills else ["s1"]

    def build_features(self, user_id, course_id, missing_set, **kwargs):
        return {
            "skill_gap_match": float(len(missing_set)),
            "goal_alignment": 0.0,
            "difficulty_fit": 0.0,
            "popularity": float(kwargs.get("popularity_override", 0.5)),
            "normalized_course_duration": 0.0,
            "learning_style_fit": 0.5,
            "time_fit": 0.5,
            "content_similarity": 0.0,
        }


class TestLeakageFixes(unittest.TestCase):
    def test_user_split_is_deterministic_and_disjoint(self):
        users = [f"u{i:03d}" for i in range(100)]
        train_a, eval_a = split_user_ids(users, seed=42)
        train_b, eval_b = split_user_ids(users, seed=42)
        self.assertEqual(train_a, train_b)
        self.assertEqual(eval_a, eval_b)
        self.assertEqual(len(eval_a), 20)
        self.assertTrue(train_a.isdisjoint(eval_a))
        self.assertEqual(train_a | eval_a, set(users))

    def test_future_mastery_does_not_change_earlier_features(self):
        ctx = FakeContext()
        first = pd.DataFrame([
            {"event_id": "e1", "user_id": "u1", "course_id": "c1", "skill_id": "s1", "event_type": "completed", "score": 50, "occurred_at": "2026-01-01"},
        ])
        with_future = pd.concat([
            first,
            pd.DataFrame([{
                "event_id": "e2", "user_id": "u1", "course_id": "c2", "skill_id": "s1", "event_type": "completed", "score": 100, "occurred_at": "2026-02-01",
            }]),
        ], ignore_index=True)

        before, _ = _temporal_rows(first, {"u1"}, ctx, negative_samples_per_user=0, course_ids=["c1", "c2"])
        after, _ = _temporal_rows(with_future, {"u1"}, ctx, negative_samples_per_user=0, course_ids=["c1", "c2"])
        self.assertEqual(before[0]["skill_gap_match"], after[0]["skill_gap_match"])


class TestPipelineGates(unittest.TestCase):
    def test_validation_failure_hard_stops_before_graph_build(self):
        pipeline = TrainingPipeline()
        ingestion = Mock()
        validation = DataValidationArtifact(False, "validation.yaml")
        pipeline.config_manager = Mock()
        pipeline.config_manager.get_data_ingestion_config.return_value = Mock()
        pipeline.config_manager.get_data_validation_config.return_value = Mock()

        with patch("src.recommender.pipeline.training_pipeline.DataIngestion") as ingestion_cls, \
             patch("src.recommender.pipeline.training_pipeline.DataValidation") as validation_cls, \
             patch("src.recommender.pipeline.training_pipeline.SkillGraphBuilder") as graph_cls:
            ingestion_cls.return_value.initiate_data_ingestion.return_value = ingestion
            validation_cls.return_value.initiate_data_validation.return_value = validation
            with self.assertRaises(ValueError):
                pipeline.run()
            graph_cls.assert_not_called()

    def test_rejected_model_stops_before_explainer(self):
        pipeline = TrainingPipeline()
        pipeline.config_manager = Mock()
        pipeline.config_manager.get_data_ingestion_config.return_value = Mock()
        pipeline.config_manager.get_data_validation_config.return_value = Mock()
        pipeline.config_manager.get_skill_graph_config.return_value = Mock()
        pipeline.config_manager.get_data_transformation_config.return_value = Mock()
        pipeline.config_manager.get_model_trainer_config.return_value = Mock()
        pipeline.config_manager.get_model_evaluator_config.return_value = Mock()
        pipeline.config_manager.get_explainer_config.return_value = Mock()

        with patch("src.recommender.pipeline.training_pipeline.DataIngestion") as ingestion_cls, \
             patch("src.recommender.pipeline.training_pipeline.DataValidation") as validation_cls, \
             patch("src.recommender.pipeline.training_pipeline.SkillGraphBuilder") as graph_cls, \
             patch("src.recommender.pipeline.training_pipeline.DataTransformation") as transform_cls, \
             patch("src.recommender.pipeline.training_pipeline.ModelTrainer") as trainer_cls, \
             patch("src.recommender.pipeline.training_pipeline.ModelEvaluator") as evaluator_cls, \
             patch("src.recommender.pipeline.training_pipeline.Explainer") as explainer_cls:
            ingestion_cls.return_value.initiate_data_ingestion.return_value = Mock()
            validation_cls.return_value.initiate_data_validation.return_value = DataValidationArtifact(True, "report.yaml")
            transform_cls.return_value.initiate_data_transformation.return_value = Mock()
            trainer_cls.return_value.initiate_model_training.return_value = Mock()
            evaluator_cls.return_value.initiate_model_evaluation.return_value = ModelEvaluatorArtifact(False, "candidate.pkl", {"ndcg_at_k": 0.2})
            with self.assertRaises(ValueError):
                pipeline.run()
            explainer_cls.assert_not_called()


class TestConfigBehavior(unittest.TestCase):
    def test_top_k_attributions_is_enforced(self):
        class Explainer:
            top_k_features = 2
            def shap_values(self, X):
                return np.array([[0.1, -0.9, 0.3, 0.2, 0.05, -0.4, 0.15, 0.25]])

        class Ctx(FakeContext):
            pass

        result = compute_attributions(Ctx(), Mock(), Explainer(), "u1", "c1")
        self.assertEqual(len(result), 2)
        self.assertIn("learning_style_fit", result)
        self.assertIn("time_fit", result)

    def test_min_confidence_is_relative_to_candidate_scores(self):
        scores = PathGenerator._confidence(np.array([1.0, 1.0]))
        self.assertAlmostEqual(float(scores[0]), 0.5, places=6)
        scores = PathGenerator._confidence(np.array([10.0, 0.0]))
        self.assertGreater(float(scores[0]), 0.99)


if __name__ == "__main__":
    unittest.main()
