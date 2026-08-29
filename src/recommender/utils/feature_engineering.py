"""Shared feature computation for the (user, course) ranking pair.

DataTransformation calls this to build the training matrix; PathGenerator
calls the exact same code at serving time to score candidates. Pulling
it into one place is what guarantees the two can't drift apart - a
common, easy-to-miss source of training/serving skew.
"""
import numpy as np

from src.recommender.components.skill_graph_builder import get_missing_skills_ordered

DIFFICULTY_RANK = {"beginner": 0, "intermediate": 1, "advanced": 2}


class FeatureContext:
    """Wraps everything a feature computation needs - skill vectors,
    course metadata, the text embedder, and the skill graph - behind one
    `.build_features(user_id, course_id)` call. Built fresh during
    transformation; reconstructed from the pickled preprocessor at
    serving time."""

    def __init__(
        self,
        *,
        graph,
        skill_index: dict,
        course_vectors: dict,
        course_skills: dict,
        goal_vectors: dict,
        goal_required_ids: dict,
        tfidf,
        svd,
        course_text_emb_by_id: dict,
        popularity_by_course: dict,
        difficulty_by_course: dict,
        duration_by_course: dict,
        format_by_course: dict,
        title_by_course: dict,
        max_duration: float,
        user_goal: dict,
        user_experience: dict,
        user_learning_style: dict,
        user_weekly_hours: dict,
        user_interests: dict,
        possessed_by_user: dict,
        skill_name_by_id: dict,
    ) -> None:
        self.graph = graph
        self.skill_index = skill_index
        self.n_skills = len(skill_index)
        self.course_vectors = course_vectors
        self.course_skills = course_skills
        self.goal_vectors = goal_vectors
        self.goal_required_ids = goal_required_ids
        self.tfidf = tfidf
        self.svd = svd
        self.course_text_emb_by_id = course_text_emb_by_id
        self.popularity_by_course = popularity_by_course
        self.difficulty_by_course = difficulty_by_course
        self.duration_by_course = duration_by_course
        self.format_by_course = format_by_course
        self.title_by_course = title_by_course
        self.max_duration = max_duration
        self.user_goal = user_goal
        self.user_experience = user_experience
        self.user_learning_style = user_learning_style
        self.user_weekly_hours = user_weekly_hours
        self.user_interests = user_interests
        self.possessed_by_user = possessed_by_user
        self.skill_name_by_id = skill_name_by_id
        self._n_components = svd.n_components

    @classmethod
    def from_preprocessor(cls, preprocessor: dict, graph) -> "FeatureContext":
        """Rebuilds the same context PathGenerator needs at serving time
        from the pickled preprocessor DataTransformation wrote out -
        same vectors, same fitted TF-IDF/SVD, same lookups."""
        return cls(
            graph=graph,
            skill_index=preprocessor["skill_index"],
            course_vectors=preprocessor["course_vectors"],
            course_skills=preprocessor["course_skills"],
            goal_vectors=preprocessor["goal_vectors"],
            goal_required_ids=preprocessor["goal_required_ids"],
            tfidf=preprocessor["tfidf"],
            svd=preprocessor["svd"],
            course_text_emb_by_id=preprocessor["course_text_emb_by_id"],
            popularity_by_course=preprocessor["popularity_by_course"],
            difficulty_by_course=preprocessor["difficulty_by_course"],
            duration_by_course=preprocessor["duration_by_course"],
            format_by_course=preprocessor["format_by_course"],
            title_by_course=preprocessor["title_by_course"],
            max_duration=preprocessor["max_duration"],
            user_goal=preprocessor["user_goal"],
            user_experience=preprocessor["user_experience"],
            user_learning_style=preprocessor["user_learning_style"],
            user_weekly_hours=preprocessor["user_weekly_hours"],
            user_interests=preprocessor.get("user_interests", {}),
            possessed_by_user=preprocessor["possessed_by_user"],
            skill_name_by_id=preprocessor["skill_name_by_id"],
        )

    def missing_skills_for_user(
        self,
        user_id: str,
        exclude_mastered: set | None = None,
        goal_id: str | None = None,
        possessed_skills: set | None = None,
    ) -> list:
        """`exclude_mastered` treats those skill_ids as NOT possessed for
        this call, without touching the underlying history - how
        adaptive rerouting reflects a failed assessment without
        rewriting the user's actual mastery record.

        `goal_id` / `possessed_skills` override the training-snapshot
        lookups entirely - how a user profiled live through /profile/chat
        (not part of the frozen Phase 2 training data) still gets a real
        path generated from their actual ProfileStore record."""
        possessed = set(possessed_skills) if possessed_skills is not None else set(
            self.possessed_by_user.get(user_id, set())
        )
        if exclude_mastered:
            possessed = possessed - set(exclude_mastered)
        resolved_goal = goal_id or self.user_goal.get(user_id)
        required = self.goal_required_ids.get(resolved_goal, set())
        return get_missing_skills_ordered(self.graph, possessed, required)

    def _skill_query_embedding(self, skill_ids_subset) -> np.ndarray:
        names = [self.skill_name_by_id[s] for s in skill_ids_subset if s in self.skill_name_by_id]
        text = " ".join(names) if names else ""
        return self.svd.transform(self.tfidf.transform([text]))[0]

    def build_features(
        self,
        user_id: str,
        course_id: str,
        missing_set: set,
        goal_id: str | None = None,
        experience_level: str | None = None,
        learning_style: str | None = None,
        weekly_hours: float | int | None = None,
        interests: list[str] | set[str] | None = None,
        popularity_override: float | None = None,
    ) -> dict:
        c_vec = self.course_vectors.get(course_id, np.zeros(self.n_skills))
        resolved_goal = goal_id or self.user_goal.get(user_id)
        g_vec = self.goal_vectors.get(resolved_goal, np.zeros(self.n_skills))
        c_skills = {s for s, _w in self.course_skills.get(course_id, [])}

        skill_gap_match = len(c_skills & missing_set) / len(c_skills) if c_skills else 0.0
        denom = (np.linalg.norm(c_vec) * np.linalg.norm(g_vec)) or 1.0
        goal_alignment = float(np.dot(c_vec, g_vec) / denom)

        d_course = DIFFICULTY_RANK.get(self.difficulty_by_course.get(course_id), 1)
        resolved_experience = experience_level or self.user_experience.get(user_id)
        d_user = DIFFICULTY_RANK.get(resolved_experience, 1)
        difficulty_fit = 1.0 - abs(d_course - d_user) / 2.0

        popularity = float(
            popularity_override if popularity_override is not None else self.popularity_by_course.get(course_id, 0.5)
        )
        duration = float(self.duration_by_course.get(course_id, 0) or 0)
        normalized_course_duration = duration / self.max_duration

        style_to_format = {"visual": "video", "reading": "text", "practice": "interactive"}
        resolved_style = learning_style or self.user_learning_style.get(user_id)
        target_format = self.format_by_course.get(course_id)
        learning_style_fit = 0.5 if resolved_style is None else (
            1.0 if target_format == style_to_format.get(resolved_style) else 0.0
        )

        raw_weekly_hours = weekly_hours if weekly_hours is not None else self.user_weekly_hours.get(user_id)
        try:
            weekly_hours = float(raw_weekly_hours) if raw_weekly_hours is not None else None
        except (TypeError, ValueError):
            weekly_hours = None
        if weekly_hours is None:
            time_fit = 0.5
        elif weekly_hours <= 0:
            time_fit = 0.0
        elif duration <= 0:
            time_fit = 1.0
        else:
            time_fit = min(1.0, weekly_hours / duration)

        resolved_interests = interests if interests is not None else self.user_interests.get(user_id, [])
        resolved_interests = {str(x).strip().lower() for x in (resolved_interests or [])}
        skill_name = self.skill_name_by_id.get(next(iter(c_skills), ""), "").lower()
        interest_keywords = {
            "generative_ai": {"llm", "rag", "deep learning", "pytorch", "ai"},
            "llms": {"llm", "rag", "deep learning", "pytorch"},
            "computer_vision": {"deep learning"},
            "nlp": {"llm", "rag", "deep learning"},
            "mlops": {"mlops", "model serving", "docker", "cloud", "kubernetes"},
        }
        matched = 0
        for interest in resolved_interests:
            keywords = interest_keywords.get(interest, set())
            if interest in skill_name or any(k in skill_name for k in keywords):
                matched += 1
        interest_fit = (matched / len(resolved_interests)) if resolved_interests else 0.5

        c_emb = self.course_text_emb_by_id.get(course_id)
        q_emb = self._skill_query_embedding(missing_set) if missing_set else np.zeros(self._n_components)
        if c_emb is None:
            content_similarity = 0.0
        else:
            denom2 = (np.linalg.norm(c_emb) * np.linalg.norm(q_emb)) or 1.0
            content_similarity = float(np.dot(c_emb, q_emb) / denom2)

        return {
            "skill_gap_match": skill_gap_match,
            "goal_alignment": goal_alignment,
            "difficulty_fit": difficulty_fit,
            "popularity": popularity,
            "normalized_course_duration": normalized_course_duration,
            "learning_style_fit": learning_style_fit,
            "time_fit": time_fit,
            "interest_fit": interest_fit,
            "content_similarity": content_similarity,
        }
