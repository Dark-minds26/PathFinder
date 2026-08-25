import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from src.recommender.constants import FEATURE_COLUMNS, RELEVANCE_COLUMN
from src.recommender.entity.config_entity import DataTransformationConfig
from src.recommender.entity.artifact_entity import (
    DataIngestionArtifact,
    DataTransformationArtifact,
)
from src.recommender.exception import RecommenderException
from src.recommender.logger import logging
from src.recommender.utils.feature_engineering import FeatureContext

MASTERY_SCORE_THRESHOLD = 60.0


class DataTransformation:
    """Builds course/goal skill vectors, a TF-IDF + SVD text embedding of
    course titles, and the per-(user, course) training feature matrix the
    ranking model learns from: skill-gap match, goal alignment, difficulty
    fit, popularity, predicted time to complete, and text-content
    similarity."""

    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        config: DataTransformationConfig,
    ) -> None:
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
                course_skills.setdefault(row["course_id"], []).append(
                    (row["skill_id"], float(row["skill_weight"]))
                )
            course_vectors = {cid: to_vector(p) for cid, p in course_skills.items()}

            goal_skills: dict = {}
            for _, row in bridge_goal_skill.iterrows():
                goal_skills.setdefault(row["goal_id"], []).append(
                    (row["skill_id"], float(row["importance_weight"]))
                )
            goal_vectors = {gid: to_vector(p) for gid, p in goal_skills.items()}
            goal_required_ids = {gid: {s for s, _w in p} for gid, p in goal_skills.items()}

            n_components = min(self.config.svd_components, max(2, len(courses) - 1))
            tfidf = TfidfVectorizer(max_features=200, stop_words="english")
            svd = TruncatedSVD(n_components=n_components, random_state=42)
            tfidf_matrix = tfidf.fit_transform(courses["title"])
            course_text_emb = svd.fit_transform(tfidf_matrix)
            course_text_emb_by_id = {
                cid: course_text_emb[i] for i, cid in enumerate(courses["course_id"])
            }

            completed = events[events["event_type"] == "completed"].copy()
            completed["score"] = pd.to_numeric(completed["score"], errors="coerce")
            mastered = completed[completed["score"] >= MASTERY_SCORE_THRESHOLD]
            possessed_by_user: dict = {}
            for uid, grp in mastered.groupby("user_id"):
                possessed_by_user[uid] = set(grp["skill_id"])

            course_attempts = events.groupby("course_id")["event_type"].agg(
                total="count", completed_n=lambda s: (s == "completed").sum()
            )
            popularity_by_course = (
                (course_attempts["completed_n"] + 1) / (course_attempts["total"] + 2)
            ).to_dict()
            max_duration = max(courses["duration_hours"].max(), 1)
            difficulty_by_course = courses.set_index("course_id")["difficulty"].to_dict()
            duration_by_course = courses.set_index("course_id")["duration_hours"].to_dict()
            title_by_course = courses.set_index("course_id")["title"].to_dict()

            user_goal = users.set_index("user_id")["career_goal_id"].to_dict()
            user_experience = users.set_index("user_id")["experience_level"].to_dict()
            skill_name_by_id = skills.set_index("skill_id")["skill_name"].to_dict()

            ctx = FeatureContext(
                graph=graph,
                skill_index=skill_index,
                course_vectors=course_vectors,
                course_skills=course_skills,
                goal_vectors=goal_vectors,
                goal_required_ids=goal_required_ids,
                tfidf=tfidf,
                svd=svd,
                course_text_emb_by_id=course_text_emb_by_id,
                popularity_by_course=popularity_by_course,
                difficulty_by_course=difficulty_by_course,
                duration_by_course=duration_by_course,
                title_by_course=title_by_course,
                max_duration=max_duration,
                user_goal=user_goal,
                user_experience=user_experience,
                possessed_by_user=possessed_by_user,
                skill_name_by_id=skill_name_by_id,
            )
            missing_cache: dict = {}

            def missing_for(uid: str) -> set:
                if uid not in missing_cache:
                    missing_cache[uid] = set(ctx.missing_skills_for_user(uid))
                return missing_cache[uid]

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

            rows = []
            seen_pairs = set()
            for _, ev in events.iterrows():
                uid, cid = ev["user_id"], ev["course_id"]
                seen_pairs.add((uid, cid))
                feats = ctx.build_features(uid, cid, missing_for(uid))
                feats.update(
                    {"user_id": uid, "course_id": cid, RELEVANCE_COLUMN: relevance_from_event(ev)}
                )
                rows.append(feats)

            rng = np.random.default_rng(42)
            all_course_ids = courses["course_id"].tolist()
            for uid in users["user_id"]:
                candidates = [c for c in all_course_ids if (uid, c) not in seen_pairs]
                if not candidates:
                    continue
                sample = rng.choice(candidates, size=min(3, len(candidates)), replace=False)
                for cid in sample:
                    feats = ctx.build_features(uid, cid, missing_for(uid))
                    feats.update({"user_id": uid, "course_id": cid, RELEVANCE_COLUMN: 0})
                    rows.append(feats)

            feature_df = pd.DataFrame(rows).sort_values("user_id").reset_index(drop=True)

            Path(self.config.transformed_data_dir).mkdir(parents=True, exist_ok=True)
            data_path = str(Path(self.config.transformed_data_dir) / "training_features.csv")
            feature_df.to_csv(data_path, index=False)

            preprocessor = {
                "skill_index": skill_index,
                "tfidf": tfidf,
                "svd": svd,
                "course_vectors": course_vectors,
                "course_skills": course_skills,
                "goal_vectors": goal_vectors,
                "goal_required_ids": goal_required_ids,
                "course_text_emb_by_id": course_text_emb_by_id,
                "popularity_by_course": popularity_by_course,
                "difficulty_by_course": difficulty_by_course,
                "duration_by_course": duration_by_course,
                "title_by_course": title_by_course,
                "max_duration": max_duration,
                "user_goal": user_goal,
                "user_experience": user_experience,
                "possessed_by_user": possessed_by_user,
                "skill_name_by_id": skill_name_by_id,
            }
            Path(self.config.preprocessor_object_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.config.preprocessor_object_path, "wb") as f:
                pickle.dump(preprocessor, f)

            logging.info(
                f"Transformation complete: {len(feature_df)} rows, "
                f"label distribution {feature_df[RELEVANCE_COLUMN].value_counts().to_dict()}"
            )
            return DataTransformationArtifact(
                transformed_object_path=self.config.preprocessor_object_path,
                transformed_data_path=data_path,
                num_rows=len(feature_df),
                feature_columns=FEATURE_COLUMNS,
            )
        except Exception as e:
            raise RecommenderException(e, sys) from e
