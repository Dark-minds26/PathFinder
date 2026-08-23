import os
import pickle
import sys
from pathlib import Path

import yaml

from src.recommender.exception import RecommenderException


def read_yaml(file_path: str) -> dict:
    try:
        with open(file_path, "r") as f:
            return yaml.safe_load(f)
    except Exception as e:
        raise RecommenderException(e, sys) from e


def write_yaml(file_path: str, content: dict, replace: bool = False) -> None:
    try:
        if replace and os.path.exists(file_path):
            os.remove(file_path)
        Path(os.path.dirname(file_path)).mkdir(parents=True, exist_ok=True)
        with open(file_path, "w") as f:
            yaml.dump(content, f)
    except Exception as e:
        raise RecommenderException(e, sys) from e


def save_object(file_path: str, obj: object) -> None:
    try:
        Path(os.path.dirname(file_path)).mkdir(parents=True, exist_ok=True)
        with open(file_path, "wb") as f:
            pickle.dump(obj, f)
    except Exception as e:
        raise RecommenderException(e, sys) from e


def load_object(file_path: str) -> object:
    try:
        with open(file_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        raise RecommenderException(e, sys) from e
