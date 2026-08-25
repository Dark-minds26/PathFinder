from dataclasses import dataclass, field


@dataclass
class DataIngestionArtifact:
    ingested_data_dir: str
    file_paths: dict


@dataclass
class DataValidationArtifact:
    validation_status: bool
    report_file_path: str


@dataclass
class DataTransformationArtifact:
    transformed_object_path: str
    transformed_data_path: str
    num_rows: int
    feature_columns: list


@dataclass
class SkillGraphArtifact:
    graph_object_path: str
    num_nodes: int
    num_edges: int


@dataclass
class ModelTrainerArtifact:
    trained_model_path: str
    train_score: float
    backend: str


@dataclass
class ModelEvaluatorArtifact:
    is_model_accepted: bool
    best_model_path: str
    metrics: dict


@dataclass
class ExplainerArtifact:
    explainer_object_path: str
    backend: str


@dataclass
class PathStep:
    skill_id: str
    course_id: str
    course_title: str
    sequence_order: int
    predicted_score: float


@dataclass
class PathGeneratorArtifact:
    user_id: str
    steps: list = field(default_factory=list)
