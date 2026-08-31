import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import ndcg_score

from src.recommender.constants import FEATURE_COLUMNS, RELEVANCE_COLUMN
from src.recommender.entity.config_entity import ModelEvaluatorConfig
from src.recommender.entity.artifact_entity import (
    ModelTrainerArtifact,
    DataTransformationArtifact,
    ModelEvaluatorArtifact,
)
from src.recommender.exception import RecommenderException
from src.recommender.logger import logging
from src.recommender.utils.main_utils import load_object, write_yaml

K = 5


def _average_precision_at_k(relevance: np.ndarray, k: int) -> float:
    relevance = relevance[:k]
    hits, score = 0, 0.0
    for i, rel in enumerate(relevance, start=1):
        if rel > 0:
            hits += 1
            score += hits / i
    return score / hits if hits else 0.0


class ModelEvaluator:
    """Evaluate only the feature rows belonging to users excluded from training.

    The evaluation set is produced by DataTransformation before model fitting.
    A rejected candidate is never promoted to the serving model path.
    """

    def __init__(
        self,
        model_trainer_artifact: ModelTrainerArtifact,
        data_transformation_artifact: DataTransformationArtifact,
        config: ModelEvaluatorConfig,
    ) -> None:
        self.model_trainer_artifact = model_trainer_artifact
        self.data_transformation_artifact = data_transformation_artifact
        self.config = config

    @staticmethod
    def model_feature_weights(model) -> dict[str, float]:
        """Expose normalized model-global importance for explainability."""
        if hasattr(model, "feature_weights") and callable(model.feature_weights):
            return model.feature_weights()
        return {}

    def initiate_model_evaluation(self) -> ModelEvaluatorArtifact:
        try:
            logging.info("Evaluating trained candidate model on the held-out user set")
            model = load_object(self.model_trainer_artifact.trained_model_path)
            eval_df = pd.read_csv(
                self.data_transformation_artifact.evaluation_data_path
            )
            if eval_df.empty or eval_df["user_id"].nunique() < 1:
                raise ValueError(
                    "Evaluation set is empty; refusing to evaluate or promote the model"
                )

            X_eval = eval_df[FEATURE_COLUMNS]
            eval_df = eval_df.assign(pred=model.predict(X_eval))

            ndcgs, aps, recommended = [], [], set()
            for _, group in eval_df.groupby("user_id"):
                group = group.sort_values("pred", ascending=False)
                true_rel = group[RELEVANCE_COLUMN].values.reshape(1, -1)
                pred_scores = group["pred"].values.reshape(1, -1)
                if true_rel.shape[1] >= 2 and true_rel.max() > 0:
                    ndcgs.append(float(ndcg_score(true_rel, pred_scores, k=K)))
                aps.append(_average_precision_at_k(group[RELEVANCE_COLUMN].values, K))
                recommended.update(group.head(K)["course_id"])

            total_courses = eval_df["course_id"].nunique()
            coverage = len(recommended) / total_courses if total_courses else 0.0
            metrics = {
                "ndcg_at_k": round(float(np.mean(ndcgs)), 4) if ndcgs else 0.0,
                "map_at_k": round(float(np.mean(aps)), 4) if aps else 0.0,
                "coverage": round(float(coverage), 4),
                "eval_users": int(eval_df["user_id"].nunique()),
                "k": K,
                "backend": self.model_trainer_artifact.backend,
                "methodology": "Deterministic 80/20 user-level holdout (seed=42). Users are entirely train or evaluation; evaluation rows are never used for model fitting.",
                "score_threshold": float(self.config.score_threshold),
                "acceptance_metric": "ndcg_at_k",
                "feature_weights": self.model_feature_weights(model),
            }
            is_accepted = metrics["ndcg_at_k"] >= self.config.score_threshold
            metrics["is_model_accepted"] = bool(is_accepted)

            Path(self.config.evaluation_report_path).parent.mkdir(
                parents=True, exist_ok=True
            )
            write_yaml(self.config.evaluation_report_path, metrics, replace=True)

            if is_accepted:
                Path(self.config.accepted_model_path).parent.mkdir(
                    parents=True, exist_ok=True
                )
                shutil.copy2(
                    self.model_trainer_artifact.trained_model_path,
                    self.config.accepted_model_path,
                )
                logging.info(
                    "Model accepted and promoted to %s", self.config.accepted_model_path
                )
                best_model_path = self.config.accepted_model_path
            else:
                logging.error(
                    "Model rejected: NDCG@%d=%.4f is below threshold %.4f; candidate will not be promoted",
                    K,
                    metrics["ndcg_at_k"],
                    self.config.score_threshold,
                )
                best_model_path = self.model_trainer_artifact.trained_model_path

            logging.info("Evaluation metrics: %s, accepted=%s", metrics, is_accepted)
            return ModelEvaluatorArtifact(
                is_model_accepted=is_accepted,
                best_model_path=best_model_path,
                metrics=metrics,
            )
        except Exception as e:
            raise RecommenderException(e, sys) from e
