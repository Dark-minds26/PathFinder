from fastapi import APIRouter, HTTPException
from api.dependencies import (
    get_path_generator,
    get_profile_store,
    resolve_serving_overrides,
)
from api.schemas.path_schema import PathRequest

router = APIRouter()
NO_GOAL_MESSAGE = "Tell me what role you're targeting before generating a path."


def _step_to_dict(step) -> dict:
    return {
        "skill_id": step.skill_id,
        "course_id": step.course_id,
        "course_title": step.course_title,
        "sequence_order": step.sequence_order,
        "predicted_score": step.predicted_score,
        "duration_hours": step.duration_hours,
        "format": step.format,
        "status": step.status,
        "why": step.why,
        "competency": step.competency,
    }


@router.post("/generate")
def generate_path(request: PathRequest):
    overrides = resolve_serving_overrides(request.user_id)

    if not overrides.get("goal_id"):
        raise HTTPException(status_code=400, detail=NO_GOAL_MESSAGE)

    try:
        generator = get_path_generator()
        store = get_profile_store()

        artifact = generator.generate_path(request.user_id, **overrides)

        path_data = [_step_to_dict(step) for step in artifact.steps]

        response = {
            "source": "live_profile",
            "path": path_data,
            "state": artifact.state,
        }

        store.save_generated_path(request.user_id, response)

        return response

    except Exception as exc:
        import traceback

        traceback.print_exc()

        raise HTTPException(
            status_code=500, detail="Unable to generate the learning path right now."
        ) from exc


@router.get("/generated/{user_id}")
def get_generated_path(user_id: str):
    store = get_profile_store()

    saved_path = store.get_generated_path(user_id)

    if not saved_path:
        return {"source": "saved_profile", "path": [], "state": "not_generated"}

    return saved_path
