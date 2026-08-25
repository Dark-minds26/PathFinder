import shutil
import sys
from pathlib import Path

from src.recommender.entity.config_entity import DataIngestionConfig
from src.recommender.entity.artifact_entity import DataIngestionArtifact
from src.recommender.exception import RecommenderException
from src.recommender.logger import logging
from src.recommender.utils.synthetic_data import generate_seed_data

EXPECTED_TABLES = [
    "users", "courses", "skills", "skill_prerequisites",
    "career_goals", "bridge_career_goal_skill", "bridge_course_skill",
    "learning_events",
]


class DataIngestion:
    """Pulls the raw tables from the source into the artifact store.

    In production this queries dim_user / dim_course / fact_learning_event
    etc. straight from Postgres. Here - and for the competition demo, which
    has no real historical learners - the "source" is the synthetic seed
    data, generated once if it isn't already on disk.
    """

    def __init__(self, config: DataIngestionConfig) -> None:
        self.config = config

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Starting data ingestion")
            seed_dir = Path(self.config.seed_data_dir)
            if not seed_dir.exists() or not any(seed_dir.glob("*.csv")):
                logging.info(f"No seed data at {seed_dir}, generating it")
                generate_seed_data(str(seed_dir))

            ingested_dir = Path(self.config.ingested_data_dir)
            ingested_dir.mkdir(parents=True, exist_ok=True)

            file_paths = {}
            for table in EXPECTED_TABLES:
                src = seed_dir / f"{table}.csv"
                if not src.exists():
                    raise FileNotFoundError(f"Expected source table missing: {src}")
                dst = ingested_dir / f"{table}.csv"
                shutil.copyfile(src, dst)
                file_paths[table] = str(dst)

            logging.info(f"Ingested {len(file_paths)} tables into {ingested_dir}")
            return DataIngestionArtifact(
                ingested_data_dir=str(ingested_dir),
                file_paths=file_paths,
            )
        except Exception as e:
            raise RecommenderException(e, sys) from e
