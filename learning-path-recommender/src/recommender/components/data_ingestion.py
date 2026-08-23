import sys

from src.recommender.entity.config_entity import DataIngestionConfig
from src.recommender.entity.artifact_entity import DataIngestionArtifact
from src.recommender.exception import RecommenderException
from src.recommender.logger import logging


class DataIngestion:
    """Pulls raw user, course, and learning-event records from the source
    (Postgres tables in production, seed CSVs for local dev) into the
    artifact store for the rest of the pipeline to consume."""

    def __init__(self, config: DataIngestionConfig) -> None:
        self.config = config

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Starting data ingestion")
            # Phase 2: query dim_user / dim_course / fact_learning_event
            # and write them to self.config.ingested_*_path
            raise NotImplementedError("Implemented in Phase 2 - data pipeline")
        except Exception as e:
            raise RecommenderException(e, sys) from e
