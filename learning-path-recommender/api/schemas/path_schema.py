from pydantic import BaseModel


class PathRequest(BaseModel):
    user_id: str


class PathStep(BaseModel):
    skill_id: str
    course_id: str
    sequence_order: int


class PathResponse(BaseModel):
    user_id: str
    path: list[PathStep]
