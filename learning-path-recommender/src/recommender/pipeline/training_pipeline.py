from src.recommender.config.configuration import ConfigurationManager
from src.recommender.components.data_ingestion import DataIngestion
from src.recommender.components.data_validation import DataValidation
from src.recommender.components.data_transformation import DataTransformation
from src.recommender.components.skill_graph_builder import SkillGraphBuilder
from src.recommender.components.model_trainer import ModelTrainer
from src.recommender.components.model_evaluator import ModelEvaluator
from src.recommender.components.explainer import Explainer
from src.recommender.logger import logging


class TrainingPipeline:
    """Wires every component together in order: ingest -> validate ->
    transform -> build graph -> train -> evaluate -> explain."""

    def __init__(self) -> None:
        self.config_manager = ConfigurationManager()

    def run(self) -> None:
        logging.info("=== Training pipeline started ===")

        ingestion_artifact = DataIngestion(
            self.config_manager.get_data_ingestion_config()
        ).initiate_data_ingestion()

        validation_artifact = DataValidation(
            ingestion_artifact, self.config_manager.get_data_validation_config()
        ).initiate_data_validation()
        if not validation_artifact.validation_status:
            logging.warning(
                "Data validation failed - see %s. Continuing on synthetic seed "
                "data, but this should hard-stop the pipeline on real sources.",
                validation_artifact.report_file_path,
            )

        SkillGraphBuilder(
            self.config_manager.get_skill_graph_config()
        ).initiate_graph_build()

        transformation_artifact = DataTransformation(
            ingestion_artifact, self.config_manager.get_data_transformation_config()
        ).initiate_data_transformation()

        trainer_artifact = ModelTrainer(
            transformation_artifact, self.config_manager.get_model_trainer_config()
        ).initiate_model_training()

        evaluator_artifact = ModelEvaluator(
            trainer_artifact, transformation_artifact, self.config_manager.get_model_evaluator_config()
        ).initiate_model_evaluation()

        Explainer(
            evaluator_artifact, self.config_manager.get_explainer_config()
        ).initiate_explainer_build()

        logging.info("=== Training pipeline finished ===")
