import sys

from src.recommender.entity.config_entity import ExplainerConfig
from src.recommender.entity.artifact_entity import (
    ModelEvaluatorArtifact,
    ExplainerArtifact,
)
from src.recommender.exception import RecommenderException
from src.recommender.logger import logging


class Explainer:
    """Wraps the accepted model in a SHAP TreeExplainer so every
    recommendation can be traced back to feature-level attributions
    (skill-gap match, goal alignment, difficulty fit, ...)."""

    def __init__(
        self,
        model_evaluator_artifact: ModelEvaluatorArtifact,
        config: ExplainerConfig,
    ) -> None:
        self.model_evaluator_artifact = model_evaluator_artifact
        self.config = config

    def initiate_explainer_build(self) -> ExplainerArtifact:
        try:
            logging.info("Building SHAP explainer")
            # Phase 2: shap.TreeExplainer(model), persist alongside the model
            raise NotImplementedError("Implemented in Phase 2 - algorithms")
        except Exception as e:
            raise RecommenderException(e, sys) from e
