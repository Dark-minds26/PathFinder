import sys

from src.recommender.exception import RecommenderException
from src.recommender.logger import logging


class RAGEngine:
    """Retrieves relevant course/skill context from the vector store
    (pgvector) to ground the conversational profiling agent."""

    def __init__(self, top_k: int = 5) -> None:
        self.top_k = top_k

    def retrieve(self, query: str) -> list[str]:
        try:
            logging.info(f"Retrieving context for: {query}")
            raise NotImplementedError("Implemented in Phase 3 - LLM integration")
        except Exception as e:
            raise RecommenderException(e, sys) from e
