"""Orchestrates one profiling conversation: RAG narrows the catalog to
plausible candidates, the LLM client extracts structured fields
constrained to those candidates, and ProfileStore persists whatever
was confirmed. One instance per active conversation; the API layer
keeps instances alive across turns (see api/dependencies.py) so
`history` accumulates properly."""
import sys

from src.recommender.exception import RecommenderException
from src.recommender.llm.llm_client import LLMClient
from src.recommender.llm.profile_store import ProfileStore
from src.recommender.llm.rag_engine import RAGEngine
from src.recommender.logger import logging


class ConversationManager:
    def __init__(self, user_id: str, rag: RAGEngine, llm: LLMClient, store: ProfileStore) -> None:
        self.user_id = user_id
        self.rag = rag
        self.llm = llm
        self.store = store
        self.history: list[dict] = []

    def handle_turn(self, message: str) -> tuple[str, float]:
        try:
            logging.info(f"Conversation turn for user {self.user_id}: {message!r}")
            candidate_goals = self.rag.retrieve_goals(message, top_k=3)
            candidate_skills = self.rag.retrieve_skills(message, top_k=8)

            result = self.llm.profile_turn(
                history=self.history,
                user_message=message,
                candidate_goals=candidate_goals,
                candidate_skills=candidate_skills,
            )

            self.history.append({"role": "user", "content": message})
            self.history.append({"role": "assistant", "content": result["reply"]})

            self.store.update(
                self.user_id,
                goal_id=result.get("goal_id"),
                experience_level=result.get("experience_level"),
                learning_style=result.get("learning_style"),
                new_skill_ids=result.get("skill_ids") or [],
            )
            completeness = self.store.completeness(self.user_id)
            logging.info(f"Profile completeness for {self.user_id}: {completeness}")
            return result["reply"], completeness
        except Exception as e:
            raise RecommenderException(e, sys) from e
