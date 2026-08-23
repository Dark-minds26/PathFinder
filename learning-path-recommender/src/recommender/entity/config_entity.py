from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    raw_data_dir: str
    ingested_users_path: str
    ingested_courses_path: str
    ingested_events_path: str


@dataclass
class DataValidationConfig:
    schema_file_path: str
    validation_report_path: str


@dataclass
class DataTransformationConfig:
    transformed_data_dir: str
    preprocessor_object_path: str


@dataclass
class SkillGraphConfig:
    prerequisite_edges_path: str
    graph_cache_path: str


@dataclass
class ModelTrainerConfig:
    trained_model_path: str
    expected_score: float
    model_params: dict


@dataclass
class ModelEvaluatorConfig:
    evaluation_report_path: str
    score_threshold: float


@dataclass
class ExplainerConfig:
    explainer_object_path: str
    top_k_features: int


@dataclass
class PathGeneratorConfig:
    max_path_length: int
    min_confidence: float
