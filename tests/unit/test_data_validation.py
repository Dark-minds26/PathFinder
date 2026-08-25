import shutil
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from src.recommender.components.data_validation import DataValidation
from src.recommender.entity.artifact_entity import DataIngestionArtifact
from src.recommender.entity.config_entity import DataValidationConfig


class TestDataValidation(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.schema_path = self.tmp / "schema.yaml"
        self.schema_path.write_text(
            "users:\n"
            "  columns:\n"
            "    user_id: str\n"
            "    goal_id: str\n"
            "  primary_key: user_id\n"
            "  foreign_keys:\n"
            "    goal_id: goals.goal_id\n"
            "goals:\n"
            "  columns:\n"
            "    goal_id: str\n"
            "  primary_key: goal_id\n"
        )

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def _run(self, users_df, goals_df):
        users_path = self.tmp / "users.csv"
        goals_path = self.tmp / "goals.csv"
        users_df.to_csv(users_path, index=False)
        goals_df.to_csv(goals_path, index=False)
        artifact = DataIngestionArtifact(
            ingested_data_dir=str(self.tmp),
            file_paths={"users": str(users_path), "goals": str(goals_path)},
        )
        config = DataValidationConfig(
            schema_file_path=str(self.schema_path),
            validation_report_path=str(self.tmp / "report.yaml"),
        )
        return DataValidation(artifact, config).initiate_data_validation()

    def test_clean_data_passes(self):
        users = pd.DataFrame({"user_id": ["u1", "u2"], "goal_id": ["g1", "g1"]})
        goals = pd.DataFrame({"goal_id": ["g1"]})
        result = self._run(users, goals)
        self.assertTrue(result.validation_status)

    def test_orphaned_foreign_key_fails(self):
        users = pd.DataFrame({"user_id": ["u1"], "goal_id": ["does_not_exist"]})
        goals = pd.DataFrame({"goal_id": ["g1"]})
        result = self._run(users, goals)
        self.assertFalse(result.validation_status)

    def test_duplicate_primary_key_fails(self):
        users = pd.DataFrame({"user_id": ["u1", "u1"], "goal_id": ["g1", "g1"]})
        goals = pd.DataFrame({"goal_id": ["g1"]})
        result = self._run(users, goals)
        self.assertFalse(result.validation_status)


if __name__ == "__main__":
    unittest.main()
