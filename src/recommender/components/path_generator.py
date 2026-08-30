import sys

import numpy as np
import pandas as pd
import networkx as nx

from src.recommender.constants import FEATURE_COLUMNS
from src.recommender.entity.config_entity import PathGeneratorConfig
from src.recommender.entity.artifact_entity import PathGeneratorArtifact, PathStep
from src.recommender.exception import RecommenderException
from src.recommender.logger import logging
from src.recommender.utils.main_utils import load_object
from src.recommender.utils.feature_engineering import FeatureContext
from src.recommender.llm.llm_client import get_llm_client


class PathGenerator:
    """
    Generates a roadmap where:

    1. Required skills are identified.
    2. Prerequisites are enforced first.
    3. Foundational skills are ordered sensibly.
    4. The ML ranker selects the best course for each skill.
    """

    def __init__(self, config: PathGeneratorConfig) -> None:
        self.config = config
        self._ctx: FeatureContext | None = None
        self._model = None
        self._course_skill_map: dict | None = None

    @property
    def ctx(self) -> FeatureContext:
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

        self._ctx = FeatureContext.from_preprocessor(
            preprocessor,
            graph
        )

        bridge = pd.read_csv(self.config.bridge_course_skill_path)

        course_skill_map = {}

        for _, row in (
            bridge.sort_values("skill_weight", ascending=False)
            .iterrows()
        ):
            course_skill_map.setdefault(
                row["skill_id"],
                []
            ).append(row["course_id"])

        self._course_skill_map = course_skill_map

    @staticmethod
    def _confidence(scores: np.ndarray) -> np.ndarray:
        scores = np.asarray(scores, dtype=float)

        shifted = scores - np.max(scores)
        exp_scores = np.exp(
            np.clip(shifted, -50, 50)
        )

        return exp_scores / (
            exp_scores.sum() or 1.0
        )

    def _order_skills_by_prerequisites(
        self,
        skill_ids: list[str],
        mastered_skills: set[str] | None = None,
    ) -> list[str]:
        """
        Return skills in prerequisite-first order.

        Also gives a stable, beginner-friendly ordering to skills
        that are on the same DAG level.
        """

        graph = self._ctx.graph

        mastered_skills = set(mastered_skills or set())
        required = set(skill_ids)

        # Include prerequisites needed for missing skills.
        for skill_id in list(required):
            if skill_id in graph:
                required.update(
                    nx.ancestors(graph, skill_id)
                )

        # Remove already mastered skills.
        required -= mastered_skills

        if not required:
            return []

        # Only use the relevant part of the graph.
        subgraph = graph.subgraph(required).copy()

        # Stable priority for equally valid skills.
        # This prevents Docker appearing before Python just because
        # both currently have no unmet prerequisites.
        preferred_order = [
            "python_basics",
            "git_basics",
            "linux_basics",
            "sql_basics",
            "networking_basics",

            "pandas_numpy",
            "statistics",
            "data_wrangling",

            "machine_learning",
            "deep_learning",

            "docker_basics",
            "mlops_basics",

            "rest_apis",
            "model_serving",

            "ci_cd_basics",

            "cloud_aws_basics",
            "terraform_basics",

            "kubernetes_basics",

            "monitoring_observability",
            "prometheus_grafana",
        ]

        priority = {
            skill_id: index
            for index, skill_id in enumerate(preferred_order)
        }

        def sort_key(skill_id):
            return (
                priority.get(skill_id, 10_000),
                skill_id
            )

        try:
            ordered = list(
                nx.lexicographical_topological_sort(
                    subgraph,
                    key=sort_key
                )
            )
        except Exception:
            logging.warning(
                "Could not topologically order roadmap skills. "
                "Using original skill order."
            )

            ordered = list(skill_ids)

        return ordered

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

        try:
            self._ensure_loaded()

            logging.info(
                f"Generating path for user {user_id}"
            )

            ctx = self._ctx

            if goal_id and str(goal_id).startswith("dynamic:"):
                return PathGeneratorArtifact(
                    user_id=user_id,
                    steps=[],
                    state="no_candidates",
                    message=(
                        "I understand this goal, but Pathfinder "
                        "currently has no validated resource catalog "
                        "for this domain yet."
                    ),
                )

            # -------------------------------------------------
            # 1. FIND MISSING SKILLS
            # -------------------------------------------------

            ordered_skills = ctx.missing_skills_for_user(
                user_id,
                exclude_mastered_skills,
                goal_id=goal_id,
                possessed_skills=possessed_skills,
            )

            # -------------------------------------------------
            # 2. FORCE PREREQUISITE-FIRST ORDER
            # -------------------------------------------------

            mastered_skills = set(
                exclude_mastered_skills or set()
            )

            mastered_skills |= set(
                possessed_skills or set()
            )

            ordered_skills = self._order_skills_by_prerequisites(
                ordered_skills,
                mastered_skills,
            )

            logging.info(
                "Prerequisite-ordered skills: %s",
                ordered_skills,
            )

            missing_set = set(ordered_skills)

            if not ordered_skills:
                return PathGeneratorArtifact(
                    user_id=user_id,
                    steps=[],
                    state="mastered",
                    message=(
                        "You already have all required skills "
                        "for this goal."
                    ),
                )

            used_courses = set(
                completed_course_ids or set()
            )

            mastery = mastery or {}

            steps = []
            candidate_found = False

            # -------------------------------------------------
            # 3. SELECT BEST COURSE FOR EACH SKILL
            # -------------------------------------------------

            for skill_id in ordered_skills[
                : self.config.max_path_length
            ]:

                candidates = [
                    course_id
                    for course_id in self._course_skill_map.get(
                        skill_id,
                        [],
                    )
                    if course_id not in used_courses
                ][
                    : self.config.candidate_courses_per_skill
                ]

                if not candidates:
                    continue

                candidate_found = True

                feats = [
                    ctx.build_features(
                        user_id,
                        course_id,
                        missing_set,
                        goal_id=goal_id,
                        experience_level=experience_level,
                        learning_style=learning_style,
                        weekly_hours=weekly_hours,
                        interests=interests,
                    )
                    for course_id in candidates
                ]

                X = pd.DataFrame(
                    [
                        [feature[column] for column in FEATURE_COLUMNS]
                        for feature in feats
                    ],
                    columns=FEATURE_COLUMNS,
                )

                scores = self._model.predict(X).astype(float)

                prefs = roadmap_preferences or {}

                for j, course_id in enumerate(candidates):

                    title = ctx.title_by_course.get(
                        course_id,
                        "",
                    ).lower()

                    if prefs.get("more_projects") and (
                        "guided project" in title
                        or "practice lab" in title
                    ):
                        scores[j] += 0.12

                    if prefs.get("more_ai") and any(
                        word in title
                        for word in (
                            "deep learning",
                            "pytorch",
                            "llm",
                            "rag",
                            "model serving",
                        )
                    ):
                        scores[j] += 0.12

                    if prefs.get("less_cloud") and (
                        "aws" in title
                        or "cloud" in title
                    ):
                        scores[j] -= 0.08

                    if prefs.get("slower_pace"):
                        scores[j] -= (
                            max(
                                0,
                                ctx.duration_by_course.get(
                                    course_id,
                                    0,
                                ) - 10,
                            )
                            * 0.002
                        )

                    if prefs.get("faster_pace"):
                        scores[j] += (
                            max(
                                0,
                                10
                                - ctx.duration_by_course.get(
                                    course_id,
                                    0,
                                ),
                            )
                            * 0.002
                        )

                    if (
                        review_skills
                        and skill_id in review_skills
                    ):

                        if "guided project" in title:
                            scores[j] += 0.50

                        elif (
                            "practice lab" in title
                            or ctx.format_by_course.get(
                                course_id
                            ) == "interactive"
                        ):
                            scores[j] += 0.10

                        else:
                            scores[j] -= 0.08

                confidences = self._confidence(scores)

                best_i = int(np.argmax(scores))

                if (
                    float(confidences[best_i])
                    < self.config.min_confidence
                ):
                    logging.info(
                        "Skipping skill %s because confidence %.3f "
                        "is below %.3f",
                        skill_id,
                        confidences[best_i],
                        self.config.min_confidence,
                    )

                    continue

                best_course = candidates[best_i]

                used_courses.add(best_course)

                status = (
                    "needs_review"
                    if review_skills
                    and skill_id in review_skills
                    else (
                        "current"
                        if not steps
                        else "locked"
                    )
                )

                why = (
                    f"{ctx.skill_name_by_id.get(skill_id, skill_id)} "
                    "is a current gap for your goal"
                )

                if learning_style:
                    why += (
                        f" and this resource matches your "
                        f"{learning_style} learning preference"
                    )

                if weekly_hours:
                    why += (
                        f" within your "
                        f"{float(weekly_hours):g}h/week study budget"
                    )

                steps.append(
                    PathStep(
                        skill_id=skill_id,
                        course_id=best_course,
                        course_title=ctx.title_by_course.get(
                            best_course,
                            best_course,
                        ),
                        sequence_order=len(steps) + 1,
                        predicted_score=round(
                            float(scores[best_i]),
                            4,
                        ),
                        duration_hours=float(
                            ctx.duration_by_course.get(
                                best_course,
                                0,
                            )
                            or 0
                        ),
                        format=str(
                            ctx.format_by_course.get(
                                best_course,
                                "text",
                            )
                        ),
                        status=status,
                        why=why,
                        competency=ctx.skill_name_by_id.get(
                            skill_id,
                            skill_id,
                        ),
                    )
                )

            logging.info(
                f"Path for {user_id}: "
                f"{len(steps)} steps across "
                f"{len(ordered_skills)} missing skills"
            )

            if not candidate_found or not steps:

                message = (
                    "I found the goal and skill gaps, "
                    "but there are no matching courses available yet."
                    if not candidate_found
                    else (
                        "I found the goal and courses, "
                        "but none met the current recommendation "
                        "confidence threshold."
                    )
                )

                return PathGeneratorArtifact(
                    user_id=user_id,
                    steps=[],
                    state="no_candidates",
                    message=message,
                )

            return PathGeneratorArtifact(
                user_id=user_id,
                steps=steps,
                state="ok",
            )

        except Exception as e:
            raise RecommenderException(e, sys) from e

    def generate_dynamic_path_stream(
        self,
        user_id: str,
        profile: dict,
        **kwargs,
    ):

        self._ensure_loaded()

        goal_id = (
            kwargs.get("goal_id")
            or profile.get("goal_id")
        )

        mastered_set = set(
            profile.get("skill_ids", [])
        )

        try:
            ordered_skills = self.ctx.missing_skills_for_user(
                user_id,
                mastered_set,
                goal_id=goal_id,
                possessed_skills=mastered_set,
            )

        except Exception:
            ordered_skills = []

        # IMPORTANT:
        # Apply prerequisite ordering here too.
        ordered_skills = self._order_skills_by_prerequisites(
            ordered_skills,
            mastered_set,
        )

        mastery = profile.get("mastery", {})

        hard_mastered = {
            skill_id
            for skill_id, score in mastery.items()
            if float(score) >= 0.8
        }

        all_mastered = hard_mastered | mastered_set

        ordered_skills = [
            skill_id
            for skill_id in ordered_skills
            if skill_id not in all_mastered
        ]

        if (
            profile.get(
                "roadmap_preferences",
                {},
            ).get("skip_docker")
        ):
            ordered_skills = [
                skill_id
                for skill_id in ordered_skills
                if "docker" not in skill_id.lower()
            ]

        available_catalog = {}

        for skill_id in ordered_skills[
            : self.config.max_path_length
        ]:

            candidates = self._course_skill_map.get(
                skill_id,
                [],
            )[:3]

            available_catalog[skill_id] = [
                {
                    "course_id": course_id,
                    "title": self.ctx.title_by_course.get(
                        course_id
                    ),
                }
                for course_id in candidates
            ]

        llm = get_llm_client()

        path_response = llm.generate_dynamic_path(
            user_profile=profile,
            ordered_skills=ordered_skills[
                : self.config.max_path_length
            ],
            available_catalog=available_catalog,
        )

        for step in path_response.path:
            yield step