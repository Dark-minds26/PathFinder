import os

from src.recommender.constants import (
    ARTIFACT_DIR,
    CONFIG_FILE_PATH,
    PARAMS_FILE_PATH,
    SCHEMA_FILE_PATH,
    SEED_DATA_DIR,
)
from src.recommender.entity.config_entity import (
    DataIngestionConfig,
    DataValidationConfig,
    DataTransformationConfig,
    SkillGraphConfig,
    ModelTrainerConfig,
    ModelEvaluatorConfig,
    ExplainerConfig,
    PathGeneratorConfig,
)
from src.recommender.utils.main_utils import read_yaml


class ConfigurationManager:
    """Reads config.yaml / params.yaml and returns typed config entities."""

    def __init__(
        self,
        config_file_path: str = CONFIG_FILE_PATH,
        params_file_path: str = PARAMS_FILE_PATH,
    ) -> None:
        self.config = read_yaml(config_file_path) if os.path.exists(config_file_path) else {}
        self.params = read_yaml(params_file_path) if os.path.exists(params_file_path) else {}

    def get_data_ingestion_config(self) -> DataIngestionConfig:
        return DataIngestionConfig(
            seed_data_dir=SEED_DATA_DIR,
            ingested_data_dir=os.path.join(ARTIFACT_DIR, "raw_data"),
        )

    def get_data_validation_config(self) -> DataValidationConfig:
        return DataValidationConfig(
            schema_file_path=self.config.get("schema_file_path", SCHEMA_FILE_PATH),
            validation_report_path=os.path.join(ARTIFACT_DIR, "validation", "report.yaml"),
        )

    def get_data_transformation_config(self) -> DataTransformationConfig:
        return DataTransformationConfig(
            transformed_data_dir=os.path.join(ARTIFACT_DIR, "transformed"),
            preprocessor_object_path=os.path.join(ARTIFACT_DIR, "transformed", "preprocessor.pkl"),
            svd_components=self.params.get("svd_components", 24),
            graph_object_path=os.path.join(ARTIFACT_DIR, "graph", "skill_dag.pkl"),
        )

    def get_skill_graph_config(self) -> SkillGraphConfig:
        return SkillGraphConfig(
            prerequisite_edges_path=os.path.join(ARTIFACT_DIR, "raw_data", "skill_prerequisites.csv"),
            skills_path=os.path.join(ARTIFACT_DIR, "raw_data", "skills.csv"),
            graph_cache_path=os.path.join(ARTIFACT_DIR, "graph", "skill_dag.pkl"),
        )

    def get_model_trainer_config(self) -> ModelTrainerConfig:
        return ModelTrainerConfig(
            trained_model_path=os.path.join(ARTIFACT_DIR, "model", "model.pkl"),
            expected_score=self.params.get("expected_score", 0.6),
            model_params=self.params.get("model_params", {}),
        )

    def get_model_evaluator_config(self) -> ModelEvaluatorConfig:
        return ModelEvaluatorConfig(
            evaluation_report_path=os.path.join(ARTIFACT_DIR, "evaluation", "report.yaml"),
            score_threshold=self.params.get("score_threshold", 0.05),
        )

    def get_explainer_config(self) -> ExplainerConfig:
        return ExplainerConfig(
            explainer_object_path=os.path.join(ARTIFACT_DIR, "explainer", "explainer.pkl"),
            top_k_features=self.params.get("top_k_features", 5),
        )

    def get_path_generator_config(self) -> PathGeneratorConfig:
        return PathGeneratorConfig(
            max_path_length=self.params.get("max_path_length", 20),
            min_confidence=self.params.get("min_confidence", 0.5),
            candidate_courses_per_skill=self.params.get("candidate_courses_per_skill", 5),
            graph_path=os.path.join(ARTIFACT_DIR, "graph", "skill_dag.pkl"),
            model_path=os.path.join(ARTIFACT_DIR, "model", "model.pkl"),
            preprocessor_path=os.path.join(ARTIFACT_DIR, "transformed", "preprocessor.pkl"),
            bridge_course_skill_path=os.path.join(ARTIFACT_DIR, "raw_data", "bridge_course_skill.csv"),
        )
