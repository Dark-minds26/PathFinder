# from src.recommender.config.configuration import ConfigurationManager
# from src.recommender.components.path_generator import PathGenerator
# from src.recommender.entity.artifact_entity import PathGeneratorArtifact


# class PredictionPipeline:
#     """Real-time path: loads the trained model + skill graph and
#     generates a roadmap for a single user on request."""

#     def __init__(self) -> None:
#         self.config_manager = ConfigurationManager()

#     def generate_path_for_user(self, user_id: str) -> PathGeneratorArtifact:
#         return PathGenerator(
#             self.config_manager.get_path_generator_config()
#         ).generate_path(user_id)

from src.recommender.config.configuration import ConfigurationManager
from src.recommender.components.path_generator import PathGenerator
from src.recommender.entity.artifact_entity import PathGeneratorArtifact
from api.dependencies import get_profile_store # <-- ADDED THIS IMPORT

class PredictionPipeline:
    """Real-time path: loads the trained model + skill graph and
    generates a roadmap for a single user on request."""

    def __init__(self) -> None:
        self.config_manager = ConfigurationManager()

    def generate_path_for_user(self, user_id: str) -> PathGeneratorArtifact:
        # 1. Let your machine learning engine do its normal job
        raw_artifact = PathGenerator(
            self.config_manager.get_path_generator_config()
        ).generate_path(user_id)
        
        # 2. Fetch what the user actually knows from the live database
        profile = get_profile_store().get(user_id)
        mastered_skills = set(profile.get("skill_ids", []))
        
        # 3. INTERCEPT: Filter out any skills they already mastered
        filtered_steps = [
            step for step in raw_artifact.steps 
            if step.skill_id not in mastered_skills
        ]
        
        # 4. Return the new, perfectly clean learning path
        return PathGeneratorArtifact(
            user_id=raw_artifact.user_id,
            steps=filtered_steps,
            state=raw_artifact.state,
            message=raw_artifact.message
        )
