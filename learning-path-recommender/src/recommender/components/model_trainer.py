import sys

from src.recommender.entity.config_entity import ModelTrainerConfig
from src.recommender.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from src.recommender.exception import RecommenderException
from src.recommender.logger import logging


class ModelTrainer:
    """Trains the ranking layer of the hybrid recommender (content
    similarity features + a gradient-boosted ranker) on historical
    engagement and completion signals."""

    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        config: ModelTrainerConfig,
    ) -> None:
        self.data_transformation_artifact = data_transformation_artifact
        self.config = config

    def initiate_model_training(self) -> ModelTrainerArtifact:
        try:
            logging.info("Training recommender model")
            # Phase 2: train the ranking model, persist to trained_model_path
            raise NotImplementedError("Implemented in Phase 2 - algorithms")
        except Exception as e:
            raise RecommenderException(e, sys) from e
