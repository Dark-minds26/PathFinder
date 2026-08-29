from pydantic import BaseModel


class ExplainResponse(BaseModel):
    course_id: str
    user_id: str
    explanation: str
    feature_attributions: dict[str, float]
    feature_weights: dict[str, float] = {}
