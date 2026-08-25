from src.recommender.config.configuration import ConfigurationManager
from src.recommender.components.path_generator import PathGenerator
from src.recommender.entity.artifact_entity import PathGeneratorArtifact


class AdaptiveReroutingPipeline:
    """Triggered when an assessment score falls below the mastery
    threshold. Treats the failed skill as not-yet-mastered for this
    regeneration only - the user's actual completion history is never
    rewritten - so the skill (and anything gated behind it) reappears
    in the roadmap without corrupting their record.

    `goal_id` / `possessed_skills` / `experience_level` pass through to
    PathGenerator for live-profiled users who aren't in the frozen
    Phase 2 training snapshot."""

    def __init__(self) -> None:
        self.config_manager = ConfigurationManager()
        self._generator = PathGenerator(self.config_manager.get_path_generator_config())

    def reroute(
        self,
        user_id: str,
        failed_skill_id: str,
        goal_id: str | None = None,
        possessed_skills: set | None = None,
        experience_level: str | None = None,
    ) -> PathGeneratorArtifact:
        return self._generator.generate_path(
            user_id,
            exclude_mastered_skills={failed_skill_id},
            goal_id=goal_id,
            possessed_skills=possessed_skills,
            experience_level=experience_level,
        )
