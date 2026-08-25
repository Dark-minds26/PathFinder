import hashlib
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

EVAL_HOLDOUT_FRACTION = 0.2
K = 5


def _is_eval_user(user_id: str) -> bool:
    digest = hashlib.md5(str(user_id).encode()).hexdigest()
    return (int(digest, 16) % 100) < int(EVAL_HOLDOUT_FRACTION * 100)


def _average_precision_at_k(relevance: np.ndarray, k: int) -> float:
    relevance = relevance[:k]
    hits, score = 0, 0.0
    for i, rel in enumerate(relevance, start=1):
        if rel > 0:
            hits += 1
            score += hits / i
    return score / hits if hits else 0.0


class ModelEvaluator:
    """Scores the trained model on a held-out slice of users - NDCG@k
    and MAP@k for ranking quality, coverage for catalog diversity - and
    decides whether it clears the bar to become the registry's accepted
    model.

    Caveat worth being upfront about: the trainer currently fits on the
    full synthetic dataset, so this read on held-out *users* is a proxy
    for generalization rather than a strict train/test split. Once real
    interaction volume is large enough to afford excluding an eval slice
    from training entirely, that's the natural next tightening.
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

    def initiate_model_evaluation(self) -> ModelEvaluatorArtifact:
        try:
            logging.info("Evaluating trained model")
            model = load_object(self.model_trainer_artifact.trained_model_path)
            df = pd.read_csv(self.data_transformation_artifact.transformed_data_path)
            df["is_eval"] = df["user_id"].apply(_is_eval_user)
            eval_df = df[df["is_eval"]]
            if eval_df["user_id"].nunique() < 2:
                eval_df = df

            X_eval = eval_df[FEATURE_COLUMNS].values
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

            total_courses = df["course_id"].nunique()
            coverage = len(recommended) / total_courses if total_courses else 0.0

            metrics = {
                "ndcg_at_k": round(float(np.mean(ndcgs)), 4) if ndcgs else 0.0,
                "map_at_k": round(float(np.mean(aps)), 4) if aps else 0.0,
                "coverage": round(float(coverage), 4),
                "eval_users": int(eval_df["user_id"].nunique()),
                "k": K,
                "backend": self.model_trainer_artifact.backend,
            }
            is_accepted = metrics["ndcg_at_k"] >= self.config.score_threshold

            Path(self.config.evaluation_report_path).parent.mkdir(parents=True, exist_ok=True)
            write_yaml(self.config.evaluation_report_path, metrics, replace=True)

            logging.info(f"Evaluation metrics: {metrics}, accepted={is_accepted}")
            return ModelEvaluatorArtifact(
                is_model_accepted=is_accepted,
                best_model_path=self.model_trainer_artifact.trained_model_path,
                metrics=metrics,
            )
        except Exception as e:
            raise RecommenderException(e, sys) from e
