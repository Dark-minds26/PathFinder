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
    evaluation_data_path: str
    num_rows: int
    evaluation_num_rows: int
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
    feature_weights: dict = field(default_factory=dict)


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
    duration_hours: float = 0.0
    format: str = "text"
    status: str = "available"
    why: str | None = None
    competency: str | None = None


@dataclass
class PathGeneratorArtifact:
    user_id: str
    steps: list = field(default_factory=list)
    state: str = "ok"
    message: str | None = None
