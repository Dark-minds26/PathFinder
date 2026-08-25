"""Retrieval over the skill/goal catalog, reusing the exact TF-IDF +
SVD embedding space DataTransformation already fit in Phase 2 - no
separate vector store, no model download. Narrows a free-text message
down to the candidate ids that are actually plausible, so the
conversational agent picks from a short, grounded list instead of
inventing or misspelling an id."""
import pickle
import sys

import numpy as np
import pandas as pd

from src.recommender.exception import RecommenderException
from src.recommender.logger import logging


class RAGEngine:
    def __init__(self, preprocessor_path: str, skills_path: str, goals_path: str) -> None:
        try:
            with open(preprocessor_path, "rb") as f:
                preprocessor = pickle.load(f)
            self.tfidf = preprocessor["tfidf"]
            self.svd = preprocessor["svd"]

            skills_df = pd.read_csv(skills_path)
            goals_df = pd.read_csv(goals_path)
            self._skills = skills_df[["skill_id", "skill_name"]].to_dict("records")
            self._goals = goals_df[["goal_id", "title"]].to_dict("records")
            self._skill_emb = self.svd.transform(self.tfidf.transform(skills_df["skill_name"]))
            self._goal_emb = self.svd.transform(self.tfidf.transform(goals_df["title"]))
            logging.info(
                f"RAGEngine ready: {len(self._skills)} skills, {len(self._goals)} goals indexed"
            )
        except Exception as e:
            raise RecommenderException(e, sys) from e

    def _embed(self, text: str) -> np.ndarray:
        return self.svd.transform(self.tfidf.transform([text]))[0]

    @staticmethod
    def _top_k(query_vec: np.ndarray, candidates: list[dict], emb_matrix: np.ndarray, top_k: int) -> list[dict]:
        if len(candidates) == 0:
            return []
        q_norm = np.linalg.norm(query_vec) or 1.0
        row_norms = np.linalg.norm(emb_matrix, axis=1)
        row_norms[row_norms == 0] = 1.0
        sims = (emb_matrix @ query_vec) / (row_norms * q_norm)
        order = np.argsort(-sims)[: min(top_k, len(candidates))]
        return [candidates[i] for i in order]

    def retrieve_skills(self, query: str, top_k: int = 8) -> list[dict]:
        try:
            return self._top_k(self._embed(query), self._skills, self._skill_emb, top_k)
        except Exception as e:
            raise RecommenderException(e, sys) from e

    def retrieve_goals(self, query: str, top_k: int = 3) -> list[dict]:
        try:
            return self._top_k(self._embed(query), self._goals, self._goal_emb, top_k)
        except Exception as e:
            raise RecommenderException(e, sys) from e
