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
    learning_style: str | None = None,
    weekly_hours: float | int | None = None,
    interests: list[str] | set[str] | None = None,
    roadmap_preferences: dict | None = None,
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
            user_id, course_id, missing, goal_id=goal_id, experience_level=experience_level,
            learning_style=learning_style, weekly_hours=weekly_hours, interests=interests
        )
        import pandas as pd
        model_features = [c for c in FEATURE_COLUMNS if c in feats]
        X = pd.DataFrame([[feats[c] for c in model_features]], columns=model_features)
        raw = explainer.shap_values(X)
        contributions = raw[0] if getattr(raw, "ndim", 1) == 2 else raw
        ranked = sorted(zip(model_features, contributions), key=lambda item: abs(float(item[1])), reverse=True)
        top_k = min(len(ranked), max(1, int(getattr(explainer, "top_k_features", len(ranked)))))
        required = {"learning_style_fit", "time_fit"}
        selected = [item for item in ranked if item[0] in required][:top_k]
        if len(selected) < top_k:
            selected_names = {name for name, _ in selected}
            selected.extend(item for item in ranked if item[0] not in required and item[0] not in selected_names)
            selected = selected[:top_k]
        selected = sorted(selected, key=lambda item: abs(float(item[1])), reverse=True)
        return {name: round(float(value), 4) for name, value in selected[:top_k]}
    except Exception as e:
        raise RecommenderException(e, sys) from e


def predicted_score(model, ctx, user_id, course_id, goal_id=None, possessed_skills=None, experience_level=None, learning_style=None, weekly_hours=None, interests=None, roadmap_preferences=None) -> float:
    try:
        missing = set(
            ctx.missing_skills_for_user(user_id, goal_id=goal_id, possessed_skills=possessed_skills)
        )
        feats = ctx.build_features(
            user_id, course_id, missing, goal_id=goal_id, experience_level=experience_level,
            learning_style=learning_style, weekly_hours=weekly_hours, interests=interests
        )
        import pandas as pd
        X = pd.DataFrame([[feats[c] for c in FEATURE_COLUMNS]], columns=FEATURE_COLUMNS)
        return round(float(model.predict(X)[0]), 4)
    except Exception as e:
        raise RecommenderException(e, sys) from e
