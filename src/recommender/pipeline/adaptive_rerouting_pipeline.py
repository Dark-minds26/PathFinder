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
        learning_style: str | None = None,
        weekly_hours: float | int | None = None,
        interests: list[str] | set[str] | None = None,
        roadmap_preferences: dict | None = None,
        completed_course_ids: set | None = None,
        mastery: dict | None = None,
        review_skills: set | None = None,
    ) -> PathGeneratorArtifact:
        
        # 1. Force the generator to load its context so we can read it
        self._generator._ensure_loaded()

        # 2. If possessed_skills isn't explicitly provided, fetch it from the context
        if possessed_skills is None:
            possessed_skills = set(self._generator._ctx.possessed_by_user.get(user_id, []))
        else:
            possessed_skills = set(possessed_skills)

        # 3. Treat the failed skill as unmastered for this generation
        possessed_skills.discard(failed_skill_id)

        # 4. Call generate_path
        return self._generator.generate_path(
            user_id,
            exclude_mastered_skills=None, # Leave this None!
            goal_id=goal_id,
            possessed_skills=possessed_skills, # Pass the modified set
            experience_level=experience_level,
            learning_style=learning_style,
            weekly_hours=weekly_hours,
            interests=interests,
            roadmap_preferences=roadmap_preferences,
            completed_course_ids=completed_course_ids, 
            mastery=mastery, 
            review_skills=review_skills or {failed_skill_id},
        )
