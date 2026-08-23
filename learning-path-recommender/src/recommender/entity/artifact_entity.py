from dataclasses import dataclass


@dataclass
class DataIngestionArtifact:
    users_file_path: str
    courses_file_path: str
    events_file_path: str


@dataclass
class DataValidationArtifact:
    validation_status: bool
    report_file_path: str


@dataclass
class DataTransformationArtifact:
    transformed_object_path: str
    transformed_data_path: str


@dataclass
class SkillGraphArtifact:
    graph_object_path: str
    num_nodes: int
    num_edges: int


@dataclass
class ModelTrainerArtifact:
    trained_model_path: str
    train_score: float


@dataclass
class ModelEvaluatorArtifact:
    is_model_accepted: bool
    best_model_path: str


@dataclass
class ExplainerArtifact:
    explainer_object_path: str


@dataclass
class PathGeneratorArtifact:
    path_output_dir: str
