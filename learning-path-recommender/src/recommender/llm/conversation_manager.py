import sys

from src.recommender.exception import RecommenderException
from src.recommender.logger import logging


class ConversationManager:
    """Holds per-user session state across a profiling conversation and
    coordinates RAG retrieval + LLM calls to fill in the profile."""

    def __init__(self, user_id: str) -> None:
        self.user_id = user_id
        self.history: list[dict] = []

    def handle_turn(self, message: str) -> str:
        try:
            logging.info(f"Conversation turn for user {self.user_id}")
            raise NotImplementedError("Implemented in Phase 3 - LLM integration")
        except Exception as e:
            raise RecommenderException(e, sys) from e
