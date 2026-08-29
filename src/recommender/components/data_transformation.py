import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from src.recommender.constants import FEATURE_COLUMNS, RANDOM_SEED, RELEVANCE_COLUMN
from src.recommender.entity.config_entity import DataTransformationConfig
from src.recommender.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
)
from src.recommender.exception import RecommenderException
from src.recommender.logger import logging
from src.recommender.utils.feature_engineering import FeatureContext

MASTERY_SCORE_THRESHOLD = 60.0
EVAL_HOLDOUT_FRACTION = 0.2


def split_user_ids(user_ids, holdout_fraction: float = EVAL_HOLDOUT_FRACTION, seed: int = RANDOM_SEED):
    """Deterministically split users before user-dependent feature building."""
    unique_ids = np.array(sorted(set(user_ids)), dtype=object)
    if len(unique_ids) < 2:
        raise ValueError("At least two users are required for a train/evaluation split")

    rng = np.random.default_rng(seed)
    shuffled = rng.permutation(unique_ids)
    n_eval = max(1, int(round(len(unique_ids) * holdout_fraction)))
    n_eval = min(n_eval, len(unique_ids) - 1)
    eval_ids = set(shuffled[:n_eval].tolist())
    train_ids = set(shuffled[n_eval:].tolist())
    return train_ids, eval_ids


def _popularity(total: int, completed: int) -> float:
    return (completed + 1) / (total + 2)


def _temporal_rows(
    events: pd.DataFrame,
    user_ids: set,
    ctx: FeatureContext,
    *,
    initial_popularity: dict | None = None,
    popularity_events: pd.DataFrame | None = None,
    negative_samples_per_user: int = 3,
    course_ids: list | None = None,
    rng_seed: int = RANDOM_SEED,
) -> tuple[list[dict], dict]:
    """Build event rows from a single chronological stream.

    For an event at T, user mastery and catalog popularity are read before
    that event is applied. This makes every emitted feature vector causal.
    Negative samples are generated at the same timestamp/state as their
    user's positive event, never from a final future snapshot.
    """
    if not user_ids:
        return [], {}

    scoped = events[events["user_id"].isin(user_ids)].copy()
    popularity_source = popularity_events.copy() if popularity_events is not None else events.copy()

    for frame_name, frame in (("events", scoped), ("popularity_events", popularity_source)):
        frame["occurred_at"] = pd.to_datetime(frame["occurred_at"], errors="coerce")
        if frame["occurred_at"].isna().any():
            raise ValueError(f"{frame_name} contains invalid occurred_at values")

    scoped = scoped.sort_values(["occurred_at", "user_id", "event_id"], kind="mergesort")
    popularity_source = popularity_source.sort_values(
        ["occurred_at", "user_id", "event_id"], kind="mergesort"
    )

    # Start each target user from an empty historical state. For evaluation users
    # this deliberately excludes all of their held-out interactions from fitting.
    possessed = {uid: set() for uid in user_ids}
    final_event_state: dict[str, set] = {uid: set() for uid in user_ids}

    all_course_ids = list(course_ids or [])
    popularity = dict(initial_popularity or {cid: 0.5 for cid in all_course_ids})
    total_by_course = {cid: 0 for cid in all_course_ids}
    completed_by_course = {cid: 0 for cid in all_course_ids}

    rows: list[dict] = []
    rng = np.random.default_rng(rng_seed)

    target_events = set(scoped["event_id"].tolist())
    for _, ev in popularity_source.iterrows():
        cid = ev["course_id"]
        uid = ev["user_id"]
        is_target = uid in user_ids and ev["event_id"] in target_events

        if is_target:
            missing = set(ctx.missing_skills_for_user(uid, possessed_skills=possessed[uid]))
            feats = ctx.build_features(
                uid, cid, missing, popularity_override=popularity.get(cid, 0.5)
            )
            feats.update({
                "user_id": uid,
                "course_id": cid,
                RELEVANCE_COLUMN: relevance_from_event(ev),
            })
            rows.append(feats)

            # Negative candidates share exactly the same pre-event state.
            candidates = [course for course in all_course_ids if course != cid]
            if candidates and negative_samples_per_user > 0:
                sample_size = min(negative_samples_per_user, len(candidates))
                sample = rng.choice(candidates, size=sample_size, replace=False)
                for neg_cid in sample:
                    neg_feats = ctx.build_features(
                        uid, neg_cid, missing, popularity_override=popularity.get(neg_cid, 0.5)
                    )
                    neg_feats.update({
                        "user_id": uid,
                        "course_id": neg_cid,
                        RELEVANCE_COLUMN: 0,
                    })
                    rows.append(neg_feats)

        # Apply the event only after its own feature vector was created.
        total_by_course[cid] = total_by_course.get(cid, 0) + 1
        if ev["event_type"] == "completed":
            completed_by_course[cid] = completed_by_course.get(cid, 0) + 1
        popularity[cid] = _popularity(total_by_course[cid], completed_by_course[cid])

        if uid in user_ids and ev["event_id"] in target_events:
            if ev["event_type"] == "completed":
                score = pd.to_numeric(pd.Series([ev["score"]]), errors="coerce").iloc[0]
                if not pd.isna(score) and float(score) >= MASTERY_SCORE_THRESHOLD:
                    possessed[uid].add(ev["skill_id"])
            final_event_state[uid] = set(possessed[uid])

    return rows, final_event_state


def relevance_from_event(row) -> int:
    if row["event_type"] == "dropped":
        return 0
    score = row["score"]
    if pd.isna(score):
        return 1
    if score >= 85:
        return 3
    if score >= MASTERY_SCORE_THRESHOLD:
        return 2
    return 1


class DataTransformation:
    """Build training/evaluation ranking features without user or temporal leakage."""

    def __init__(self, data_ingestion_artifact: DataIngestionArtifact, config: DataTransformationConfig) -> None:
        self.data_ingestion_artifact = data_ingestion_artifact
        self.config = config

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Starting data transformation")
            paths = self.data_ingestion_artifact.file_paths
            courses = pd.read_csv(paths["courses"])
            skills = pd.read_csv(paths["skills"])
            bridge_course_skill = pd.read_csv(paths["bridge_course_skill"])
            users = pd.read_csv(paths["users"])
            bridge_goal_skill = pd.read_csv(paths["bridge_career_goal_skill"])
            events = pd.read_csv(paths["learning_events"])

            train_users, eval_users = split_user_ids(users["user_id"], seed=RANDOM_SEED)
            logging.info("User split: train=%d eval=%d seed=%d", len(train_users), len(eval_users), RANDOM_SEED)

            with open(self.config.graph_object_path, "rb") as f:
                graph = pickle.load(f)

            skill_ids = sorted(skills["skill_id"])
            skill_index = {s: i for i, s in enumerate(skill_ids)}
            n_skills = len(skill_ids)

            def to_vector(pairs):
                v = np.zeros(n_skills)
                for skill_id, weight in pairs:
                    if skill_id in skill_index:
                        v[skill_index[skill_id]] = weight
                return v

            course_skills: dict = {}
            for _, row in bridge_course_skill.iterrows():
                course_skills.setdefault(row["course_id"], []).append((row["skill_id"], float(row["skill_weight"])))
            course_vectors = {cid: to_vector(p) for cid, p in course_skills.items()}

            goal_skills: dict = {}
            for _, row in bridge_goal_skill.iterrows():
                goal_skills.setdefault(row["goal_id"], []).append((row["skill_id"], float(row["importance_weight"])))
            goal_vectors = {gid: to_vector(p) for gid, p in goal_skills.items()}
            goal_required_ids = {gid: {s for s, _w in p} for gid, p in goal_skills.items()}

            # Course text is catalog-only information, so fitting this does not use user history.
            n_components = min(self.config.svd_components, max(2, len(courses) - 1))
            tfidf = TfidfVectorizer(max_features=200, stop_words="english")
            svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_SEED)
            tfidf_matrix = tfidf.fit_transform(courses["title"])
            course_text_emb = svd.fit_transform(tfidf_matrix)
            course_text_emb_by_id = {cid: course_text_emb[i] for i, cid in enumerate(courses["course_id"])}

            difficulty_by_course = courses.set_index("course_id")["difficulty"].to_dict()
            duration_by_course = courses.set_index("course_id")["duration_hours"].to_dict()
            format_by_course = courses.set_index("course_id")["format"].to_dict()
            title_by_course = courses.set_index("course_id")["title"].to_dict()
            max_duration = max(courses["duration_hours"].max(), 1)
            user_goal = users.set_index("user_id")["career_goal_id"].to_dict()
            user_experience = users.set_index("user_id")["experience_level"].to_dict()
            user_learning_style = users.set_index("user_id")["learning_style"].to_dict()
            user_weekly_hours = pd.to_numeric(users.set_index("user_id")["weekly_hours"], errors="coerce").fillna(0).to_dict()
            user_interests = {uid: [x for x in str(v).split("|") if x] for uid, v in users.set_index("user_id")["interests"].to_dict().items()}
            skill_name_by_id = skills.set_index("skill_id")["skill_name"].to_dict()

            # Fit catalog popularity prior on TRAIN users only. Evaluation interactions are
            # never used to fit a training-side statistic.
            train_events = events[events["user_id"].isin(train_users)].copy()
            train_attempts = train_events.groupby("course_id")["event_type"].agg(
                total="count", completed_n=lambda s: (s == "completed").sum()
            )
            train_popularity = ((train_attempts["completed_n"] + 1) / (train_attempts["total"] + 2)).to_dict()

            # Preprocessor stores only training-derived event state; static user attributes are safe.
            possessed_train: dict = {}
            completed_train = train_events[train_events["event_type"] == "completed"].copy()
            completed_train["score"] = pd.to_numeric(completed_train["score"], errors="coerce")
            mastered_train = completed_train[completed_train["score"] >= MASTERY_SCORE_THRESHOLD]
            for uid, grp in mastered_train.groupby("user_id"):
                possessed_train[uid] = set(grp["skill_id"])

            base_ctx = FeatureContext(
                graph=graph,
                skill_index=skill_index,
                course_vectors=course_vectors,
                course_skills=course_skills,
                goal_vectors=goal_vectors,
                goal_required_ids=goal_required_ids,
                tfidf=tfidf,
                svd=svd,
                course_text_emb_by_id=course_text_emb_by_id,
                popularity_by_course=train_popularity,
                difficulty_by_course=difficulty_by_course,
                duration_by_course=duration_by_course,
                format_by_course=format_by_course,
                title_by_course=title_by_course,
                max_duration=max_duration,
                user_goal=user_goal,
                user_experience=user_experience,
                user_learning_style=user_learning_style,
                user_weekly_hours=user_weekly_hours,
                user_interests=user_interests,
                possessed_by_user=possessed_train,
                skill_name_by_id=skill_name_by_id,
            )

            train_rows, train_final_state = _temporal_rows(
                events,
                train_users,
                base_ctx,
                popularity_events=train_events,
                course_ids=courses["course_id"].tolist(),
                rng_seed=RANDOM_SEED,
            )

            # Evaluation rows are emitted from the same global chronological stream. Both
            # training and evaluation events update popularity only after their own timestamp,
            # so no future event can affect an earlier feature. Evaluation rows are the only
            # rows emitted for evaluation users.
            eval_rows, eval_final_state = _temporal_rows(
                events,
                eval_users,
                base_ctx,
                popularity_events=events,
                course_ids=courses["course_id"].tolist(),
                rng_seed=RANDOM_SEED,
            )

            train_df = pd.DataFrame(train_rows, columns=["user_id", "course_id", *FEATURE_COLUMNS, RELEVANCE_COLUMN])
            eval_df = pd.DataFrame(eval_rows, columns=["user_id", "course_id", *FEATURE_COLUMNS, RELEVANCE_COLUMN])
            train_df = train_df.sort_values("user_id").reset_index(drop=True)
            eval_df = eval_df.sort_values("user_id").reset_index(drop=True)

            Path(self.config.transformed_data_dir).mkdir(parents=True, exist_ok=True)
            train_path = str(Path(self.config.transformed_data_dir) / "training_features.csv")
            eval_path = str(Path(self.config.transformed_data_dir) / "evaluation_features.csv")
            train_df.to_csv(train_path, index=False)
            eval_df.to_csv(eval_path, index=False)

            preprocessor = {
                "skill_index": skill_index,
                "tfidf": tfidf,
                "svd": svd,
                "course_vectors": course_vectors,
                "course_skills": course_skills,
                "goal_vectors": goal_vectors,
                "goal_required_ids": goal_required_ids,
                "course_text_emb_by_id": course_text_emb_by_id,
                "popularity_by_course": train_popularity,
                "difficulty_by_course": difficulty_by_course,
                "duration_by_course": duration_by_course,
                "format_by_course": format_by_course,
                "title_by_course": title_by_course,
                "max_duration": max_duration,
                "user_goal": user_goal,
                "user_experience": user_experience,
                "user_learning_style": user_learning_style,
                "user_weekly_hours": user_weekly_hours,
                "user_interests": user_interests,
                "possessed_by_user": train_final_state,
                "skill_name_by_id": skill_name_by_id,
                "train_user_ids": sorted(train_users),
                "eval_user_ids": sorted(eval_users),
                "split_seed": RANDOM_SEED,
                "eval_holdout_fraction": EVAL_HOLDOUT_FRACTION,
            }
            Path(self.config.preprocessor_object_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.config.preprocessor_object_path, "wb") as f:
                pickle.dump(preprocessor, f)

            logging.info(
                "Transformation complete: train=%d rows, eval=%d rows, train labels=%s, eval labels=%s",
                len(train_df), len(eval_df),
                train_df[RELEVANCE_COLUMN].value_counts().to_dict(),
                eval_df[RELEVANCE_COLUMN].value_counts().to_dict(),
            )
            return DataTransformationArtifact(
                transformed_object_path=self.config.preprocessor_object_path,
                transformed_data_path=train_path,
                evaluation_data_path=eval_path,
                num_rows=len(train_df),
                evaluation_num_rows=len(eval_df),
                feature_columns=FEATURE_COLUMNS,
            )
        except Exception as e:
            raise RecommenderException(e, sys) from e
