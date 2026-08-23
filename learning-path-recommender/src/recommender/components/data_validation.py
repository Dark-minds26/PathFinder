import sys

from src.recommender.entity.config_entity import DataValidationConfig
from src.recommender.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
)
from src.recommender.exception import RecommenderException
from src.recommender.logger import logging


class DataValidation:
    """Validates ingested data against schema.yaml and checks for drift
    before anything downstream is allowed to train on it."""

    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        config: DataValidationConfig,
    ) -> None:
        self.data_ingestion_artifact = data_ingestion_artifact
        self.config = config

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logging.info("Starting data validation")
            # Phase 2: column/type checks against schema.yaml, drift report
            raise NotImplementedError("Implemented in Phase 2 - data pipeline")
        except Exception as e:
            raise RecommenderException(e, sys) from e
