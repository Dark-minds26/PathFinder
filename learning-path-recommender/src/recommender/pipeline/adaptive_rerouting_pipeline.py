from src.recommender.config.configuration import ConfigurationManager
from src.recommender.components.skill_graph_builder import SkillGraphBuilder
from src.recommender.components.path_generator import PathGenerator
from src.recommender.entity.artifact_entity import PathGeneratorArtifact


class AdaptiveReroutingPipeline:
    """Triggered when an assessment score falls below the mastery
    threshold: marks the skill unmastered, re-derives the graph
    traversal, and regenerates the remaining path."""

    def __init__(self) -> None:
        self.config_manager = ConfigurationManager()

    def reroute(self, user_id: str, failed_skill_id: str) -> PathGeneratorArtifact:
        # Phase 2: mark failed_skill_id unmastered, rebuild traversal
        SkillGraphBuilder(self.config_manager.get_skill_graph_config())
        return PathGenerator(
            self.config_manager.get_path_generator_config()
        ).generate_path(user_id)
