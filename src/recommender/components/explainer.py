import pickle
import sys
from pathlib import Path

import numpy as np

from src.recommender.constants import FEATURE_COLUMNS
from src.recommender.entity.config_entity import ExplainerConfig
from src.recommender.entity.artifact_entity import ModelEvaluatorArtifact, ExplainerArtifact
from src.recommender.exception import RecommenderException
from src.recommender.logger import logging
from src.recommender.utils.main_utils import load_object

try:
    import shap

    _HAS_SHAP = True
except ImportError:
    _HAS_SHAP = False


class BaselineAttributionExplainer:
    """Stand-in for shap.TreeExplainer when the shap package isn't
    installed: attributes a prediction to each feature by replacing that
    feature with a baseline (the zero vector - roughly "average" since
    every feature here is a normalized similarity/fit score) and
    measuring how much the prediction moves. Not Shapley-consistent the
    way real SHAP is, but the same output shape - one contribution per
    feature, per prediction - so callers don't need to know which one
    they got."""

    def __init__(self, model, baseline: np.ndarray) -> None:
        self.model = model
        self.baseline = baseline

    def shap_values(self, X: np.ndarray) -> np.ndarray:
        X = np.atleast_2d(X)
        full_preds = self.model.predict(X)
        contributions = np.zeros_like(X, dtype=float)
        for j in range(X.shape[1]):
            X_masked = X.copy()
            X_masked[:, j] = self.baseline[j]
            masked_preds = self.model.predict(X_masked)
            contributions[:, j] = full_preds - masked_preds
        return contributions


class Explainer:
    """Wraps the accepted model so every recommendation can be traced
    back to feature-level attributions - skill-gap match, goal
    alignment, difficulty fit, and so on. Uses SHAP's TreeExplainer when
    shap is installed (exact Shapley values, fast on tree ensembles);
    falls back to baseline-perturbation attribution with the same
    per-feature output shape when it isn't, so /explain doesn't change
    behavior based on which backend trained the model."""

    def __init__(
        self,
        model_evaluator_artifact: ModelEvaluatorArtifact,
        config: ExplainerConfig,
    ) -> None:
        self.model_evaluator_artifact = model_evaluator_artifact
        self.config = config

    def initiate_explainer_build(self) -> ExplainerArtifact:
        try:
            logging.info("Building explainer")
            model = load_object(self.model_evaluator_artifact.best_model_path)
            estimator = getattr(model, "estimator", model)

            if _HAS_SHAP:
                explainer = shap.TreeExplainer(estimator)
                backend = "shap-tree-explainer"
            else:
                logging.info("shap not installed - falling back to baseline attribution")
                baseline = np.zeros(len(FEATURE_COLUMNS))
                explainer = BaselineAttributionExplainer(model, baseline)
                backend = "baseline-attribution-fallback"

            Path(self.config.explainer_object_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.config.explainer_object_path, "wb") as f:
                pickle.dump(explainer, f)

            logging.info(f"Explainer built ({backend})")
            return ExplainerArtifact(
                explainer_object_path=self.config.explainer_object_path,
                backend=backend,
            )
        except Exception as e:
            raise RecommenderException(e, sys) from e
