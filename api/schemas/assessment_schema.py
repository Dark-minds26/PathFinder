from pydantic import BaseModel, ConfigDict, Field
class AssessmentRequest(BaseModel):
    model_config=ConfigDict(extra="forbid")
    user_id:str=Field(min_length=1,max_length=64)
    skill_id:str=Field(min_length=1,max_length=100)
    score:float|None=Field(default=None,ge=0.0,le=100.0)
    answers:dict[str,int]|None=None
