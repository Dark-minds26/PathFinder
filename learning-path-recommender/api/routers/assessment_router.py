from fastapi import APIRouter, HTTPException

from api.schemas.path_schema import PathResponse

router = APIRouter()


@router.post("/submit", response_model=PathResponse)
def submit_assessment(user_id: str, skill_id: str, score: float) -> PathResponse:
    """Triggers adaptive rerouting when a submitted score falls below
    the mastery threshold."""
    raise HTTPException(status_code=501, detail="Implemented in Phase 2/3 - adaptive rerouting")
