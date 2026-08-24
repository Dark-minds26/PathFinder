from src.recommender.config.configuration import ConfigurationManager
from src.recommender.components.path_generator import PathGenerator
from src.recommender.entity.artifact_entity import PathGeneratorArtifact


class AdaptiveReroutingPipeline:
    """Triggered when an assessment score falls below the mastery
    threshold: treats the failed skill as unmastered for this
    regeneration (without rewriting mastery history) and re-derives the
    remaining path, so foundational material comes back into the
    roadmap ahead of whatever was next."""

    def __init__(self) -> None:
        self.config_manager = ConfigurationManager()
        self._generator = PathGenerator(self.config_manager.get_path_generator_config())

    def reroute(self, user_id: str, failed_skill_id: str) -> PathGeneratorArtifact:
        return self._generator.generate_path(
            user_id, exclude_mastered_skills={failed_skill_id}
        )
