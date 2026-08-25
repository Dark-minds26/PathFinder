"""JSON-file-backed store for live-profiled users - the ones built up
through /profile/chat, as opposed to the synthetic users baked into
the Phase 2 training snapshot. Same shape as dim_user; swappable for
a real Postgres table later with no caller-facing changes, same
reasoning as the S3 sync stand-in in cloud_storage."""
import json
import sys
from pathlib import Path
from threading import Lock

from src.recommender.exception import RecommenderException
from src.recommender.logger import logging

REQUIRED_FIELDS = ("goal_id", "experience_level", "learning_style")


class ProfileStore:
    def __init__(self, path: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = Lock()
        if not self.path.exists():
            self.path.write_text("{}")

    def _read_all(self) -> dict:
        try:
            raw = self.path.read_text().strip()
            return json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            return {}

    def _write_all(self, data: dict) -> None:
        self.path.write_text(json.dumps(data, indent=2, sort_keys=True))

    def get(self, user_id: str) -> dict:
        return self._read_all().get(user_id, {"skill_ids": []})

    def update(
        self,
        user_id: str,
        *,
        goal_id: str | None = None,
        experience_level: str | None = None,
        learning_style: str | None = None,
        new_skill_ids: list | None = None,
    ) -> dict:
        try:
            with self._lock:
                data = self._read_all()
                profile = data.get(user_id, {"skill_ids": []})
                if goal_id:
                    profile["goal_id"] = goal_id
                if experience_level:
                    profile["experience_level"] = experience_level
                if learning_style:
                    profile["learning_style"] = learning_style
                if new_skill_ids:
                    profile["skill_ids"] = sorted(set(profile.get("skill_ids", [])) | set(new_skill_ids))
                data[user_id] = profile
                self._write_all(data)
                logging.info(f"Profile updated for {user_id}: {profile}")
                return profile
        except Exception as e:
            raise RecommenderException(e, sys) from e

    def completeness(self, user_id: str) -> float:
        profile = self.get(user_id)
        have = sum(1 for f in REQUIRED_FIELDS if profile.get(f))
        have += 1 if profile.get("skill_ids") else 0
        return round(have / (len(REQUIRED_FIELDS) + 1), 4)

    def mark_unmastered(self, user_id: str, skill_id: str) -> dict:
        """Removes skill_id from a live profile's mastered set, if
        present - the live-profile equivalent of AdaptiveRerouting's
        exclude_mastered_skills, for users who aren't in the frozen
        training snapshot."""
        try:
            with self._lock:
                data = self._read_all()
                profile = data.get(user_id, {"skill_ids": []})
                profile["skill_ids"] = [s for s in profile.get("skill_ids", []) if s != skill_id]
                data[user_id] = profile
                self._write_all(data)
                return profile
        except Exception as e:
            raise RecommenderException(e, sys) from e
