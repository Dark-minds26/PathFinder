import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.recommender.constants import RELEVANCE_COLUMN
from src.recommender.entity.config_entity import ModelTrainerConfig
from src.recommender.entity.artifact_entity import (
    DataTransformationArtifact,
    ModelTrainerArtifact,
)
from src.recommender.exception import RecommenderException
from src.recommender.logger import logging

try:
    import lightgbm as lgb

    _HAS_LIGHTGBM = True
except ImportError:
    _HAS_LIGHTGBM = False
    from sklearn.ensemble import GradientBoostingRegressor


class RankerModel:
    """Uniform predict(X) -> scores interface over either backend, so
    everything downstream (evaluator, explainer, path generator) doesn't
    need to know which one trained it."""

    def __init__(self, backend: str, estimator, feature_columns: list[str] | None = None) -> None:
        self.backend = backend
        self.estimator = estimator
        self.feature_columns = list(feature_columns or [])

    def predict(self, X) -> np.ndarray:
        if self.feature_columns:
            if not isinstance(X, pd.DataFrame):
                X = pd.DataFrame(X, columns=self.feature_columns)
            else:
                X = X.loc[:, self.feature_columns]
        return np.asarray(self.estimator.predict(X))


    def feature_weights(self) -> dict[str, float]:
        """Return normalized global feature importance for XAI reporting.

        LightGBM uses gain importance; the sklearn fallback exposes its
        standard feature_importances_.  The weights are model-derived and
        are never invented by the LLM.
        """
        raw = None
        if self.backend == "lightgbm-lambdarank":
            try:
                raw = self.estimator.booster_.feature_importance(importance_type="gain")
            except Exception:
                raw = getattr(self.estimator, "feature_importances_", None)
        else:
            raw = getattr(self.estimator, "feature_importances_", None)
        if raw is None:
            return {name: 0.0 for name in self.feature_columns}
        values = np.asarray(raw, dtype=float)
        total = float(values.sum())
        if total <= 0:
            return {name: 0.0 for name in self.feature_columns}
        return {name: round(float(value / total), 6) for name, value in zip(self.feature_columns, values)}


class ModelTrainer:
    """Trains the ranking layer of the hybrid recommender. Uses LightGBM's
    lambdarank objective (listwise, optimizes ranking quality directly)
    when available; falls back to a pointwise gradient-boosted regressor
    on the graded relevance label when it isn't - same feature set, same
    predict() interface, `pip install lightgbm` upgrades it with no other
    code changes."""

    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        config: ModelTrainerConfig,
    ) -> None:
        self.data_transformation_artifact = data_transformation_artifact
        self.config = config

    def initiate_model_training(self) -> ModelTrainerArtifact:
        try:
            logging.info("Training recommender model on user-level training holdout only")
            df = pd.read_csv(self.data_transformation_artifact.transformed_data_path)
            feature_cols = self.data_transformation_artifact.feature_columns
            df = df.sort_values("user_id").reset_index(drop=True)
            X = df[feature_cols]
            y = df[RELEVANCE_COLUMN].values

            if _HAS_LIGHTGBM:
                group = df.groupby("user_id", sort=False).size().values
                estimator = lgb.LGBMRanker(
                    objective="lambdarank",
                    n_estimators=self.config.model_params.get("n_estimators", 200),
                    learning_rate=self.config.model_params.get("learning_rate", 0.05),
                    max_depth=self.config.model_params.get("max_depth", 4),
                    random_state=42,
                    verbosity=-1,
                )
                estimator.fit(X, y, group=group)
                backend = "lightgbm-lambdarank"
            else:
                logging.info("lightgbm not installed - falling back to GradientBoostingRegressor")
                estimator = GradientBoostingRegressor(
                    n_estimators=self.config.model_params.get("n_estimators", 200),
                    learning_rate=self.config.model_params.get("learning_rate", 0.05),
                    max_depth=self.config.model_params.get("max_depth", 4),
                    random_state=42,
                )
                estimator.fit(X, y)
                backend = "sklearn-gbr-fallback"

            model = RankerModel(backend=backend, estimator=estimator, feature_columns=feature_cols)
            preds = model.predict(X)
            train_score = float(np.corrcoef(preds, y)[0, 1]) if len(set(y)) > 1 else 0.0
            feature_weights = model.feature_weights()
            logging.info("Model XAI feature weights: %s", feature_weights)

            Path(self.config.candidate_model_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.config.candidate_model_path, "wb") as f:
                pickle.dump(model, f)

            logging.info(f"Model trained ({backend}), train correlation={train_score:.3f}")
            return ModelTrainerArtifact(
                trained_model_path=self.config.candidate_model_path,
                train_score=train_score,
                backend=backend,
                feature_weights=feature_weights,
            )
        except Exception as e:
            raise RecommenderException(e, sys) from e
