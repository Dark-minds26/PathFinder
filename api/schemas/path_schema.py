from pydantic import BaseModel


class PathRequest(BaseModel):
    user_id: str


class PathStep(BaseModel):
    skill_id: str
    course_id: str
    course_title: str
    sequence_order: int
    predicted_score: float


class PathResponse(BaseModel):
    user_id: str
    path: list[PathStep]
    source: str  # "live_profile" or "training_snapshot" - which user context was used
