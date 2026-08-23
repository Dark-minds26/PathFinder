from fastapi import APIRouter, HTTPException

from api.schemas.explain_schema import ExplainResponse

router = APIRouter()


@router.get("/{course_id}/{user_id}", response_model=ExplainResponse)
def explain_recommendation(course_id: str, user_id: str) -> ExplainResponse:
    """SHAP attributions for this course/user pair, phrased in natural
    language by the LLM."""
    raise HTTPException(status_code=501, detail="Implemented in Phase 2/3 - XAI + API wiring")
