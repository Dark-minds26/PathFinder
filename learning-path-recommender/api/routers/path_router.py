from fastapi import APIRouter

from api.dependencies import get_path_generator, resolve_serving_overrides
from api.schemas.path_schema import PathRequest, PathResponse, PathStep

router = APIRouter()


@router.post("/generate", response_model=PathResponse)
def generate_path(request: PathRequest) -> PathResponse:
    """Runs the graph traversal + hybrid recommender and returns a
    ranked, sequenced roadmap. Uses this user's live profile (built up
    via /profile/chat) when one exists, otherwise falls back to the
    Phase 2 training snapshot - so both a brand-new chat user and one
    of the synthetic demo users work through the same endpoint."""
    overrides = resolve_serving_overrides(request.user_id)
    artifact = get_path_generator().generate_path(request.user_id, **overrides)
    return PathResponse(
        user_id=artifact.user_id,
        path=[PathStep(**vars(step)) for step in artifact.steps],
        source="live_profile" if overrides else "training_snapshot",
    )
