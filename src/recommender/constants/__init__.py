import os

ARTIFACT_DIR: str = "artifacts"
SEED_DATA_DIR: str = os.path.join("data", "seed")
CONFIG_FILE_PATH: str = os.path.join("config", "config.yaml")
PARAMS_FILE_PATH: str = os.path.join("config", "params.yaml")
SCHEMA_FILE_PATH: str = os.path.join("src", "recommender", "config", "schema.yaml")
RANDOM_SEED: int = 42
FEATURE_COLUMNS: list = [
    "skill_gap_match",
    "goal_alignment",
    "difficulty_fit",
    "popularity",
    "normalized_course_duration",
    "learning_style_fit",
    "time_fit",
    "interest_fit",
    "content_similarity",
]
RELEVANCE_COLUMN: str = "relevance"
