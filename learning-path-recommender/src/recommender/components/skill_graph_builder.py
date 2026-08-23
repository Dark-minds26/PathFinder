import sys

from src.recommender.entity.config_entity import SkillGraphConfig
from src.recommender.entity.artifact_entity import SkillGraphArtifact
from src.recommender.exception import RecommenderException
from src.recommender.logger import logging


class SkillGraphBuilder:
    """Loads skill_prerequisites edges and builds the in-memory directed
    acyclic graph (NetworkX) used for topological sorting and
    missing-skill traversal at serving time."""

    def __init__(self, config: SkillGraphConfig) -> None:
        self.config = config

    def initiate_graph_build(self) -> SkillGraphArtifact:
        try:
            logging.info("Building skill prerequisite graph")
            # Phase 2: load edges, construct nx.DiGraph, assert acyclicity
            raise NotImplementedError("Implemented in Phase 2 - data pipeline")
        except Exception as e:
            raise RecommenderException(e, sys) from e
