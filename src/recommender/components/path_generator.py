import sys

import numpy as np
import pandas as pd

from src.recommender.constants import FEATURE_COLUMNS
from src.recommender.entity.config_entity import PathGeneratorConfig
from src.recommender.entity.artifact_entity import PathGeneratorArtifact, PathStep
from src.recommender.exception import RecommenderException
from src.recommender.logger import logging
from src.recommender.utils.main_utils import load_object
from src.recommender.utils.feature_engineering import FeatureContext


class PathGenerator:
    """Combines the skill graph's topological ordering with the trained
    ranker's per-skill course scores to produce one ranked, sequenced
    roadmap for a user. Model, graph, and preprocessor are loaded once
    and cached on the instance - cheap to reuse across many requests."""

    def __init__(self, config: PathGeneratorConfig) -> None:
        self.config = config
        self._ctx: FeatureContext | None = None
        self._model = None
        self._course_skill_map: dict | None = None

    def _ensure_loaded(self) -> None:
        if self._ctx is not None:
            return
        graph = load_object(self.config.graph_path)
        self._model = load_object(self.config.model_path)
        preprocessor = load_object(self.config.preprocessor_path)
        self._ctx = FeatureContext.from_preprocessor(preprocessor, graph)

        bridge = pd.read_csv(self.config.bridge_course_skill_path)
        course_skill_map: dict = {}
        for _, row in bridge.sort_values("skill_weight", ascending=False).iterrows():
            course_skill_map.setdefault(row["skill_id"], []).append(row["course_id"])
        self._course_skill_map = course_skill_map

    def generate_path(
        self, user_id: str, exclude_mastered_skills: set | None = None
    ) -> PathGeneratorArtifact:
        try:
            self._ensure_loaded()
            logging.info(f"Generating path for user {user_id}")
            ctx = self._ctx

            ordered_skills = ctx.missing_skills_for_user(user_id, exclude_mastered_skills)
            missing_set = set(ordered_skills)
            used_courses: set = set()
            steps = []

            for order_idx, skill_id in enumerate(
                ordered_skills[: self.config.max_path_length], start=1
            ):
                candidates = [
                    c
                    for c in self._course_skill_map.get(skill_id, [])
                    if c not in used_courses
                ][: self.config.candidate_courses_per_skill]
                if not candidates:
                    continue

                feats = [ctx.build_features(user_id, cid, missing_set) for cid in candidates]
                X = np.array([[f[c] for c in FEATURE_COLUMNS] for f in feats])
                scores = self._model.predict(X)
                best_i = int(np.argmax(scores))
                best_course = candidates[best_i]
                used_courses.add(best_course)

                steps.append(
                    PathStep(
                        skill_id=skill_id,
                        course_id=best_course,
                        course_title=ctx.title_by_course.get(best_course, best_course),
                        sequence_order=order_idx,
                        predicted_score=round(float(scores[best_i]), 4),
                    )
                )

            logging.info(f"Path for {user_id}: {len(steps)} steps across {len(ordered_skills)} missing skills")
            return PathGeneratorArtifact(user_id=user_id, steps=steps)
        except Exception as e:
            raise RecommenderException(e, sys) from e
