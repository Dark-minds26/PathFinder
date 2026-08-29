from pydantic import BaseModel, ConfigDict, Field
class PathRequest(BaseModel):
    model_config=ConfigDict(extra="forbid")
    user_id:str=Field(min_length=1,max_length=64)
class PathStep(BaseModel):
    model_config=ConfigDict(extra="forbid")
    skill_id:str=Field(min_length=1,max_length=100)
    course_id:str=Field(min_length=1,max_length=100)
    course_title:str=Field(min_length=1,max_length=300)
    sequence_order:int=Field(ge=1,le=100)
    predicted_score:float
    duration_hours:float=Field(ge=0,le=1000)
    format:str=Field(min_length=1,max_length=40)
    status:str="available"
    why:str|None=None
    competency:str|None=None
class PathResponse(BaseModel):
    model_config=ConfigDict(extra="forbid")
    user_id:str=Field(min_length=1,max_length=64)
    path:list[PathStep]=Field(max_length=100)
    source:str
    state:str="ok"
    message:str|None=None
    progress_pct:float=Field(default=0.0,ge=0,le=100)
    weekly_plan:list[dict]=Field(default_factory=list,max_length=14)
    assessment_score: float | None = Field(default=None, ge=0.0, le=100.0)
    assessment_status: str | None = Field(default=None, max_length=30)
    assessment_skill_id: str | None = Field(default=None, max_length=100)
