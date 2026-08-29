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

    @property
    def ctx(self) -> FeatureContext:
        """Exposed so /explain can reuse the same loaded graph, model,
        and preprocessor instead of loading its own copy."""
        self._ensure_loaded()
        return self._ctx

    @property
    def model(self):
        self._ensure_loaded()
        return self._model

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

    @staticmethod
    def _confidence(scores: np.ndarray) -> np.ndarray:
        """Convert candidate scores into a relative softmax confidence."""
        scores = np.asarray(scores, dtype=float)
        shifted = scores - np.max(scores)
        exp_scores = np.exp(np.clip(shifted, -50, 50))
        return exp_scores / (exp_scores.sum() or 1.0)

    def generate_path(
        self,
        user_id: str,
        exclude_mastered_skills: set | None = None,
        goal_id: str | None = None,
        possessed_skills: set | None = None,
        experience_level: str | None = None,
        learning_style: str | None = None,
        weekly_hours: float | int | None = None,
        interests: list[str] | set[str] | None = None,
        roadmap_preferences: dict | None = None,
        completed_course_ids: set | None = None,
        mastery: dict | None = None,
        review_skills: set | None = None,
    ) -> PathGeneratorArtifact:
        """`goal_id` / `possessed_skills` / `experience_level` override
        the training-snapshot lookups for a user - how a live-profiled
        user (built up through /profile/chat, not part of the frozen
        Phase 2 training data) still gets a real roadmap."""
        try:
            self._ensure_loaded()
            logging.info(f"Generating path for user {user_id}")
            ctx = self._ctx

            if goal_id and str(goal_id).startswith("dynamic:"):
                return PathGeneratorArtifact(user_id=user_id, steps=[], state="no_candidates", message="I understand this goal, but Pathfinder currently has no validated resource catalog for this domain yet. I will not invent courses or claim unsupported coverage.")
            ordered_skills = ctx.missing_skills_for_user(
                user_id, exclude_mastered_skills, goal_id=goal_id, possessed_skills=possessed_skills
            )
            missing_set = set(ordered_skills)
            if not ordered_skills:
                return PathGeneratorArtifact(user_id=user_id, steps=[], state="mastered", message="You already have all required skills for this goal.")
            used_courses: set = set(completed_course_ids or set())
            mastery = mastery or {}
            steps = []
            candidate_found = False

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
                candidate_found = True

                feats = [
                    ctx.build_features(
                        user_id, cid, missing_set, goal_id=goal_id, experience_level=experience_level,
                        learning_style=learning_style, weekly_hours=weekly_hours, interests=interests
                    )
                    for cid in candidates
                ]
                X = pd.DataFrame([[f[c] for c in FEATURE_COLUMNS] for f in feats], columns=FEATURE_COLUMNS)
                scores = self._model.predict(X).astype(float)
                prefs = roadmap_preferences or {}
                for j, cid in enumerate(candidates):
                    title = ctx.title_by_course.get(cid, "").lower()
                    if prefs.get("more_projects") and ("guided project" in title or "practice lab" in title): scores[j] += 0.12
                    if prefs.get("more_ai") and any(x in title for x in ("deep learning", "pytorch", "llm", "rag", "model serving")): scores[j] += 0.12
                    if prefs.get("less_cloud") and ("aws" in title or "cloud" in title): scores[j] -= 0.08
                    if prefs.get("slower_pace"): scores[j] -= max(0, ctx.duration_by_course.get(cid, 0) - 10) * 0.002
                    if prefs.get("faster_pace"): scores[j] += max(0, 10 - ctx.duration_by_course.get(cid, 0)) * 0.002
                    if review_skills and skill_id in review_skills:
                        title_l=title.lower()
                        if "guided project" in title_l: scores[j] += 0.50
                        elif "practice lab" in title_l or ctx.format_by_course.get(cid)=="interactive": scores[j] += 0.10
                        else: scores[j] -= 0.08
                confidences = self._confidence(scores)
                best_i = int(np.argmax(scores))
                if float(confidences[best_i]) < self.config.min_confidence:
                    logging.info(
                        "Skipping skill %s: best candidate confidence %.3f is below min_confidence %.3f",
                        skill_id, confidences[best_i], self.config.min_confidence,
                    )
                    continue
                best_course = candidates[best_i]
                used_courses.add(best_course)

                status = "needs_review" if review_skills and skill_id in review_skills else ("current" if not steps else "locked")
                skill_mastery = float(mastery.get(skill_id, 0.0) or 0.0)
                why = f"{ctx.skill_name_by_id.get(skill_id, skill_id)} is a current gap for your goal"
                if learning_style: why += f" and this resource matches your {learning_style} learning preference"
                if weekly_hours: why += f" within your {float(weekly_hours):g}h/week study budget"
                steps.append(
                    PathStep(
                        skill_id=skill_id,
                        course_id=best_course,
                        course_title=ctx.title_by_course.get(best_course, best_course),
                        sequence_order=order_idx,
                        predicted_score=round(float(scores[best_i]), 4),
                        duration_hours=float(ctx.duration_by_course.get(best_course, 0) or 0),
                        format=str(ctx.format_by_course.get(best_course, "text")),
                        status=status, why=why, competency=ctx.skill_name_by_id.get(skill_id, skill_id),
                    )
                )

            logging.info(f"Path for {user_id}: {len(steps)} steps across {len(ordered_skills)} missing skills")
            if not candidate_found or not steps:
                return PathGeneratorArtifact(user_id=user_id, steps=[], state="no_candidates", message=("I found the goal and skill gaps, but there are no matching courses available yet." if not candidate_found else "I found the goal and courses, but none met the current recommendation confidence threshold."))
            return PathGeneratorArtifact(user_id=user_id, steps=steps, state="ok")
        except Exception as e:
            raise RecommenderException(e, sys) from e
