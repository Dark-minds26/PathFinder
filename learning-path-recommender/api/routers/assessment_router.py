from fastapi import APIRouter

from api.dependencies import get_profile_store, get_reroute_pipeline, resolve_serving_overrides
from api.schemas.path_schema import PathResponse, PathStep

router = APIRouter()
MASTERY_SCORE_THRESHOLD = 60.0


@router.post("/submit", response_model=PathResponse)
def submit_assessment(user_id: str, skill_id: str, score: float) -> PathResponse:
    """Triggers adaptive rerouting when a submitted score falls below
    the mastery threshold: the skill is treated as unmastered for this
    regeneration only (a live user's ProfileStore entry is also
    updated so it stays unmastered on the next request too) and the
    remaining path is regenerated around it."""
    if score >= MASTERY_SCORE_THRESHOLD:
        overrides = resolve_serving_overrides(user_id)
        from api.dependencies import get_path_generator

        artifact = get_path_generator().generate_path(user_id, **overrides)
        source = "live_profile" if overrides else "training_snapshot"
    else:
        store = get_profile_store()
        if store.get(user_id).get("goal_id"):
            store.mark_unmastered(user_id, skill_id)
        overrides = resolve_serving_overrides(user_id)
        artifact = get_reroute_pipeline().reroute(user_id, skill_id, **overrides)
        source = "live_profile" if overrides else "training_snapshot"

    return PathResponse(
        user_id=artifact.user_id,
        path=[PathStep(**vars(step)) for step in artifact.steps],
        source=source,
    )
