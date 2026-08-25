import os

ARTIFACT_DIR: str = "artifacts"
SEED_DATA_DIR: str = os.path.join("data", "seed")

CONFIG_FILE_PATH: str = os.path.join("config", "config.yaml")
PARAMS_FILE_PATH: str = os.path.join("config", "params.yaml")
SCHEMA_FILE_PATH: str = os.path.join("src", "recommender", "config", "schema.yaml")

RANDOM_SEED: int = 42

# Feature columns the ranking model trains on (see data_transformation.py)
FEATURE_COLUMNS: list = [
    "skill_gap_match",
    "goal_alignment",
    "difficulty_fit",
    "popularity",
    "predicted_time_to_complete",
    "content_similarity",
]
RELEVANCE_COLUMN: str = "relevance"
