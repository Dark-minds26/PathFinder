import sys

from src.recommender.entity.config_entity import DataTransformationConfig
from src.recommender.entity.artifact_entity import (
    DataValidationArtifact,
    DataTransformationArtifact,
)
from src.recommender.exception import RecommenderException
from src.recommender.logger import logging


class DataTransformation:
    """Builds course/user feature vectors: skill embeddings, difficulty
    and format encodings, and the profile vector used by the content-based
    side of the hybrid recommender."""

    def __init__(
        self,
        data_validation_artifact: DataValidationArtifact,
        config: DataTransformationConfig,
    ) -> None:
        self.data_validation_artifact = data_validation_artifact
        self.config = config

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            logging.info("Starting data transformation")
            # Phase 2: feature engineering + embedding generation
            raise NotImplementedError("Implemented in Phase 2 - data pipeline")
        except Exception as e:
            raise RecommenderException(e, sys) from e
