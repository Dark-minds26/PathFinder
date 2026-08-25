from dataclasses import dataclass


@dataclass
class DataIngestionConfig:
    seed_data_dir: str
    ingested_data_dir: str


@dataclass
class DataValidationConfig:
    schema_file_path: str
    validation_report_path: str


@dataclass
class DataTransformationConfig:
    transformed_data_dir: str
    preprocessor_object_path: str
    svd_components: int
    graph_object_path: str


@dataclass
class SkillGraphConfig:
    prerequisite_edges_path: str
    skills_path: str
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
    candidate_courses_per_skill: int
    graph_path: str
    model_path: str
    preprocessor_path: str
    bridge_course_skill_path: str
