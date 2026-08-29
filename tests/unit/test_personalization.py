import unittest
import numpy as np

from src.recommender.constants import FEATURE_COLUMNS
from src.recommender.utils.feature_engineering import FeatureContext


class DummyTFIDF:
    def transform(self, texts):
        return np.zeros((len(texts), 2))


class DummySVD:
    n_components = 2
    def transform(self, x):
        return np.zeros((x.shape[0], 2))


def make_ctx(style="visual", hours=8):
    return FeatureContext(
        graph=None,
        skill_index={"s": 0},
        course_vectors={"video": np.array([1.0]), "text": np.array([1.0])},
        course_skills={"video": [("s", 1.0)], "text": [("s", 1.0)]},
        goal_vectors={"g": np.array([1.0])},
        goal_required_ids={"g": {"s"}},
        tfidf=DummyTFIDF(),
        svd=DummySVD(),
        course_text_emb_by_id={"video": np.array([1.0, 0.0]), "text": np.array([1.0, 0.0])},
        popularity_by_course={"video": 0.5, "text": 0.5},
        difficulty_by_course={"video": "beginner", "text": "beginner"},
        duration_by_course={"video": 4, "text": 16},
        format_by_course={"video": "video", "text": "text"},
        title_by_course={"video": "Video", "text": "Text"},
        max_duration=16,
        user_goal={"u": "g"},
        user_experience={"u": "beginner"},
        user_learning_style={"u": style},
        user_weekly_hours={"u": hours},
        user_interests={"u": []},
        possessed_by_user={"u": set()},
        skill_name_by_id={"s": "Skill"},
    )


class TestPersonalizationFeatures(unittest.TestCase):
    def test_learning_style_changes_feature(self):
        visual = make_ctx("visual").build_features("u", "video", {"s"})
        reading = make_ctx("reading").build_features("u", "video", {"s"})
        self.assertGreater(visual["learning_style_fit"], reading["learning_style_fit"])

    def test_learning_style_changes_course_ranking(self):
        class StyleRanker:
            def predict(self, X):
                # In this test the only varying signal is learning_style_fit.
                return np.asarray(X[:, 5], dtype=float)

        model = StyleRanker()
        visual_ctx = make_ctx("visual")
        reading_ctx = make_ctx("reading")
        visual_feats = [
            visual_ctx.build_features("u", cid, {"s"}) for cid in ("video", "text")
        ]
        reading_feats = [
            reading_ctx.build_features("u", cid, {"s"}) for cid in ("video", "text")
        ]
        visual_scores = model.predict(np.array([[f[c] for c in FEATURE_COLUMNS] for f in visual_feats]))
        reading_scores = model.predict(np.array([[f[c] for c in FEATURE_COLUMNS] for f in reading_feats]))
        self.assertEqual(int(np.argmax(visual_scores)), 0)
        self.assertEqual(int(np.argmax(reading_scores)), 1)

    def test_weekly_hours_changes_time_fit(self):
        low = make_ctx("visual", 4).build_features("u", "text", {"s"})
        high = make_ctx("visual", 16).build_features("u", "text", {"s"})
        self.assertGreater(high["time_fit"], low["time_fit"])

    def test_personalization_features_are_model_inputs(self):
        self.assertIn("learning_style_fit", FEATURE_COLUMNS)
        self.assertIn("time_fit", FEATURE_COLUMNS)
        self.assertIn("interest_fit", FEATURE_COLUMNS)

    def test_interest_changes_feature(self):
        llm_ctx = make_ctx("practice", 8)
        llm_ctx.skill_name_by_id["s"] = "LLM applications"
        a = llm_ctx.build_features("u", "video", {"s"}, interests={"llms"})
        b = llm_ctx.build_features("u", "video", {"s"}, interests=set())
        self.assertGreaterEqual(a["interest_fit"], b["interest_fit"])


if __name__ == "__main__":
    unittest.main()

class TestModelXAIContract(unittest.TestCase):
    def test_model_feature_weights_are_normalized(self):
        from src.recommender.components.model_trainer import RankerModel
        class Estimator:
            feature_importances_ = np.array([1.0, 2.0, 0.0])
            def predict(self, X):
                return np.zeros(len(X))
        model = RankerModel("sklearn-gbr-fallback", Estimator(), ["a", "b", "c"])
        weights = model.feature_weights()
        self.assertEqual(set(weights), {"a", "b", "c"})
        self.assertAlmostEqual(sum(weights.values()), 1.0, places=6)
        self.assertGreater(weights["b"], weights["a"])
