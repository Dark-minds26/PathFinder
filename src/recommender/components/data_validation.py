import sys

import pandas as pd

from src.recommender.entity.config_entity import DataValidationConfig
from src.recommender.entity.artifact_entity import (
    DataIngestionArtifact,
    DataValidationArtifact,
)
from src.recommender.exception import RecommenderException
from src.recommender.logger import logging
from src.recommender.utils.main_utils import read_yaml, write_yaml


class DataValidation:
    """Validates ingested tables against schema.yaml: required columns are
    present, primary keys are unique, and foreign keys resolve to real
    rows in their parent table. Nothing downstream should train on data
    that failed this check."""

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
            schema = read_yaml(self.config.schema_file_path)
            frames = {
                table: pd.read_csv(path)
                for table, path in self.data_ingestion_artifact.file_paths.items()
            }

            report: dict = {}
            all_passed = True

            for table, spec in schema.items():
                table_report: dict = {"checks": []}
                df = frames.get(table)

                if df is None:
                    table_report["checks"].append({"missing_table": True})
                    all_passed = False
                    report[table] = table_report
                    continue

                expected_cols = set(spec.get("columns", {}).keys())
                actual_cols = set(df.columns)
                missing_cols = sorted(expected_cols - actual_cols)
                table_report["checks"].append({"missing_columns": missing_cols})
                if missing_cols:
                    all_passed = False

                pk = spec.get("primary_key")
                if pk and pk in df.columns:
                    n_dupes = int(df[pk].duplicated().sum())
                    n_nulls = int(df[pk].isna().sum())
                    table_report["checks"].append(
                        {"primary_key": pk, "duplicate_rows": n_dupes, "null_rows": n_nulls}
                    )
                    if n_dupes or n_nulls:
                        all_passed = False

                for fk_col, parent_ref in spec.get("foreign_keys", {}).items():
                    parent_table, parent_col = parent_ref.split(".")
                    parent_df = frames.get(parent_table)
                    if fk_col not in df.columns or parent_df is None:
                        continue
                    valid_ids = set(parent_df[parent_col])
                    orphaned = int((~df[fk_col].isin(valid_ids)).sum())
                    table_report["checks"].append(
                        {"foreign_key": fk_col, "references": parent_ref, "orphaned_rows": orphaned}
                    )
                    if orphaned:
                        all_passed = False

                report[table] = table_report

            write_yaml(self.config.validation_report_path, report, replace=True)
            logging.info(f"Data validation status: {all_passed}")
            return DataValidationArtifact(
                validation_status=all_passed,
                report_file_path=self.config.validation_report_path,
            )
        except Exception as e:
            raise RecommenderException(e, sys) from e
