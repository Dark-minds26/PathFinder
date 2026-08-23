from fastapi import APIRouter, HTTPException

from api.schemas.path_schema import PathRequest, PathResponse

router = APIRouter()


@router.post("/generate", response_model=PathResponse)
def generate_path(request: PathRequest) -> PathResponse:
    """Runs the graph traversal + hybrid recommender and returns a
    ranked, sequenced roadmap."""
    raise HTTPException(status_code=501, detail="Implemented in Phase 2/3 - recommender + API wiring")
