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

    def __init__(self, backend: str, estimator) -> None:
        self.backend = backend
        self.estimator = estimator

    def predict(self, X) -> np.ndarray:
        return np.asarray(self.estimator.predict(X))


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
            logging.info("Training recommender model")
            df = pd.read_csv(self.data_transformation_artifact.transformed_data_path)
            feature_cols = self.data_transformation_artifact.feature_columns
            df = df.sort_values("user_id").reset_index(drop=True)
            X = df[feature_cols].values
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

            model = RankerModel(backend=backend, estimator=estimator)
            preds = model.predict(X)
            train_score = float(np.corrcoef(preds, y)[0, 1]) if len(set(y)) > 1 else 0.0

            Path(self.config.trained_model_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.config.trained_model_path, "wb") as f:
                pickle.dump(model, f)

            logging.info(f"Model trained ({backend}), train correlation={train_score:.3f}")
            return ModelTrainerArtifact(
                trained_model_path=self.config.trained_model_path,
                train_score=train_score,
                backend=backend,
            )
        except Exception as e:
            raise RecommenderException(e, sys) from e
