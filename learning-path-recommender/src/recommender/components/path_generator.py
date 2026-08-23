import sys

from src.recommender.entity.config_entity import PathGeneratorConfig
from src.recommender.entity.artifact_entity import PathGeneratorArtifact
from src.recommender.exception import RecommenderException
from src.recommender.logger import logging


class PathGenerator:
    """Combines the skill graph's topological ordering with the
    recommender's per-skill course scores to produce one ranked,
    sequenced learning roadmap for a user."""

    def __init__(self, config: PathGeneratorConfig) -> None:
        self.config = config

    def generate_path(self, user_id: str) -> PathGeneratorArtifact:
        try:
            logging.info(f"Generating path for user {user_id}")
            # Phase 2: graph traversal + recommender scoring -> ordered path
            raise NotImplementedError("Implemented in Phase 2 - algorithms")
        except Exception as e:
            raise RecommenderException(e, sys) from e
