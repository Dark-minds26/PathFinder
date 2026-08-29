from pydantic import BaseModel, ConfigDict, Field

class ChatRequest(BaseModel):
    model_config=ConfigDict(extra="forbid")
    user_id: str = Field(min_length=1, max_length=64)
    message: str = Field(min_length=1, max_length=2000)

class ChatResponse(BaseModel):
    model_config=ConfigDict(extra="forbid")
    reply: str = Field(min_length=1, max_length=4000)
    profile_completeness: float = Field(ge=0.0, le=1.0)
    goal_id: str | None = None
    experience_level: str | None = None
    learning_style: str | None = None
    weekly_hours: float | None = Field(default=None, ge=0.1, le=168.0)
    interests: list[str] = Field(default_factory=list, max_length=5)
    roadmap_preferences: dict[str, bool] = Field(default_factory=dict, max_length=5)
    roadmap_updated: bool = False
    path: list = Field(default_factory=list, max_length=100)
    path_state: str = "none"
    path_message: str | None = None
    progress_pct: float = Field(default=0.0, ge=0.0, le=100.0)
    mastered_skills: list[str] = Field(default_factory=list, max_length=100)
    unmastered_skills: list[str] = Field(default_factory=list, max_length=100)
    mastery: dict[str,float] = Field(default_factory=dict)
    learning_history: list[dict] = Field(default_factory=list, max_length=200)
    goal_spec: dict | None = None
