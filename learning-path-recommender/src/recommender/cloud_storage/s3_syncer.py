import os
import sys

import boto3

from src.recommender.exception import RecommenderException
from src.recommender.logger import logging


class S3Sync:
    """Pushes/pulls artifacts and the model registry to/from S3."""

    def __init__(self, bucket_name: str) -> None:
        self.bucket_name = bucket_name
        self.client = boto3.client("s3")

    def sync_folder_to_s3(self, folder: str, s3_prefix: str) -> None:
        try:
            logging.info(f"Syncing {folder} -> s3://{self.bucket_name}/{s3_prefix}")
            os.system(f"aws s3 sync {folder} s3://{self.bucket_name}/{s3_prefix}")
        except Exception as e:
            raise RecommenderException(e, sys) from e

    def sync_folder_from_s3(self, folder: str, s3_prefix: str) -> None:
        try:
            logging.info(f"Syncing s3://{self.bucket_name}/{s3_prefix} -> {folder}")
            os.system(f"aws s3 sync s3://{self.bucket_name}/{s3_prefix} {folder}")
        except Exception as e:
            raise RecommenderException(e, sys) from e
