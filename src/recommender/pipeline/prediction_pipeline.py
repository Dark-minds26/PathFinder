from src.recommender.config.configuration import ConfigurationManager
from src.recommender.components.path_generator import PathGenerator
from src.recommender.entity.artifact_entity import PathGeneratorArtifact


class PredictionPipeline:
    """Real-time path: loads the trained model + skill graph and
    generates a roadmap for a single user on request."""

    def __init__(self) -> None:
        self.config_manager = ConfigurationManager()

    def generate_path_for_user(self, user_id: str) -> PathGeneratorArtifact:
        return PathGenerator(
            self.config_manager.get_path_generator_config()
        ).generate_path(user_id)
