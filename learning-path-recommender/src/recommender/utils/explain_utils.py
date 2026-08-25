"""Ties the trained model, its explainer, and a FeatureContext together
to answer 'why was this course recommended' for one (user, course)
pair. Used by the /explain router and safe to unit test directly
without going through FastAPI."""
import sys

import numpy as np

from src.recommender.constants import FEATURE_COLUMNS
from src.recommender.exception import RecommenderException
from src.recommender.logger import logging


def compute_attributions(
    ctx,
    model,
    explainer,
    user_id: str,
    course_id: str,
    goal_id: str | None = None,
    possessed_skills: set | None = None,
    experience_level: str | None = None,
) -> dict:
    """Returns {feature_name: contribution} for this exact prediction -
    real Shapley values when the explainer is shap.TreeExplainer,
    baseline-perturbation contributions when it's the fallback (see
    Explainer in components/explainer.py) - same shape either way."""
    try:
        missing = set(
            ctx.missing_skills_for_user(user_id, goal_id=goal_id, possessed_skills=possessed_skills)
        )
        feats = ctx.build_features(
            user_id, course_id, missing, goal_id=goal_id, experience_level=experience_level
        )
        X = np.array([[feats[c] for c in FEATURE_COLUMNS]])
        raw = explainer.shap_values(X)
        contributions = raw[0] if getattr(raw, "ndim", 1) == 2 else raw
        return {name: round(float(c), 4) for name, c in zip(FEATURE_COLUMNS, contributions)}
    except Exception as e:
        raise RecommenderException(e, sys) from e


def predicted_score(model, ctx, user_id, course_id, goal_id=None, possessed_skills=None, experience_level=None) -> float:
    try:
        missing = set(
            ctx.missing_skills_for_user(user_id, goal_id=goal_id, possessed_skills=possessed_skills)
        )
        feats = ctx.build_features(
            user_id, course_id, missing, goal_id=goal_id, experience_level=experience_level
        )
        X = np.array([[feats[c] for c in FEATURE_COLUMNS]])
        return round(float(model.predict(X)[0]), 4)
    except Exception as e:
        raise RecommenderException(e, sys) from e
