"""Canonical domain models for Pathfinder's learner/resource state.

These models are intentionally independent of FastAPI request/response schemas so the
same contracts can be reused by the graph, recommender, persistence and adapters.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


MasteryStatus = Literal["validated", "needs_review", "failed"]
ResourceType = Literal["course", "project", "assessment"]


class Mastery(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skill_id: str = Field(min_length=1, max_length=100)
    score: float = Field(default=0.0, ge=0.0, le=1.0)
    status: MasteryStatus
    source: str = Field(default="self_report", min_length=1, max_length=50)
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Skill(BaseModel):
    model_config = ConfigDict(extra="forbid")

    skill_id: str = Field(min_length=1, max_length=100)
    name: str = Field(min_length=1, max_length=200)
    category: str = Field(min_length=1, max_length=80)
    is_foundational: bool = False
    prerequisite_skill_ids: list[str] = Field(default_factory=list, max_length=50)


class Goal(BaseModel):
    model_config = ConfigDict(extra="forbid")

    goal_id: str = Field(min_length=1, max_length=120)
    title: str = Field(min_length=1, max_length=200)
    domain: str = Field(min_length=1, max_length=120)
    required_skill_ids: list[str] = Field(default_factory=list, max_length=200)
    interest_ids: list[str] = Field(default_factory=list, max_length=50)
    source: Literal["curated", "dynamic"] = "curated"


class Resource(BaseModel):
    model_config = ConfigDict(extra="forbid")

    resource_id: str = Field(min_length=1, max_length=120)
    title: str = Field(min_length=1, max_length=300)
    resource_type: ResourceType = "course"
    skill_ids: list[str] = Field(min_length=1, max_length=20)
    duration_hours: float = Field(default=0.0, ge=0.0, le=10000.0)
    difficulty: Literal["beginner", "intermediate", "advanced"] = "beginner"
    format: str = Field(default="text", min_length=1, max_length=50)
    description: str | None = Field(default=None, max_length=2000)

    @field_validator("skill_ids")
    @classmethod
    def unique_skills(cls, value: list[str]) -> list[str]:
        if len(set(value)) != len(value):
            raise ValueError("skill_ids must not contain duplicates")
        return value


class User(BaseModel):
    model_config = ConfigDict(extra="forbid")

    user_id: str = Field(min_length=1, max_length=64)
    goal_id: str | None = Field(default=None, max_length=120)
    experience_level: Literal["beginner", "intermediate", "advanced"] | None = None
    learning_style: Literal["visual", "reading", "practice"] | None = None
    weekly_hours: float | None = Field(default=None, ge=0.1, le=168.0)
    interest_ids: list[str] = Field(default_factory=list, max_length=20)
    mastery: dict[str, Mastery] = Field(default_factory=dict)
    completed_resource_ids: list[str] = Field(default_factory=list, max_length=500)
    roadmap_preferences: dict[str, bool] = Field(default_factory=dict, max_length=20)

    @property
    def validated_skill_ids(self) -> set[str]:
        """Skills currently treated as mastered/hard boundaries."""
        return {sid for sid, record in self.mastery.items() if record.status == "validated"}

    @property
    def review_skill_ids(self) -> set[str]:
        return {sid for sid, record in self.mastery.items() if record.status == "needs_review"}

    @property
    def failed_skill_ids(self) -> set[str]:
        return {sid for sid, record in self.mastery.items() if record.status == "failed"}
