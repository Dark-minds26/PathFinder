import os

ARTIFACT_DIR: str = "artifacts"
DATA_DIR: str = os.path.join("data")
RAW_USERS_FILE: str = "users.csv"
RAW_COURSES_FILE: str = "courses.csv"
RAW_EVENTS_FILE: str = "learning_events.csv"
SKILL_EDGES_FILE: str = "skill_prerequisites.csv"

CONFIG_FILE_PATH: str = os.path.join("config", "config.yaml")
PARAMS_FILE_PATH: str = os.path.join("config", "params.yaml")
SCHEMA_FILE_PATH: str = os.path.join("src", "recommender", "config", "schema.yaml")
