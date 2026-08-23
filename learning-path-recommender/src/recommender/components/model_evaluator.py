import sys

from src.recommender.entity.config_entity import ModelEvaluatorConfig
from src.recommender.entity.artifact_entity import (
    ModelTrainerArtifact,
    ModelEvaluatorArtifact,
)
from src.recommender.exception import RecommenderException
from src.recommender.logger import logging


class ModelEvaluator:
    """Scores the trained model on held-out data (NDCG, MAP, coverage)
    and decides whether it clears the bar to replace the current one
    in the model registry."""

    def __init__(
        self,
        model_trainer_artifact: ModelTrainerArtifact,
        config: ModelEvaluatorConfig,
    ) -> None:
        self.model_trainer_artifact = model_trainer_artifact
        self.config = config

    def initiate_model_evaluation(self) -> ModelEvaluatorArtifact:
        try:
            logging.info("Evaluating trained model")
            # Phase 2: compute NDCG / MAP / coverage, compare to threshold
            raise NotImplementedError("Implemented in Phase 2 - algorithms")
        except Exception as e:
            raise RecommenderException(e, sys) from e
