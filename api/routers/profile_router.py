from fastapi import APIRouter, HTTPException
from api.dependencies import (
    get_conversation_manager,
    get_path_generator,
    resolve_serving_overrides,
)
from api.schemas.profile_schema import ChatRequest, ChatResponse
import traceback

router = APIRouter()


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


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        manager = get_conversation_manager(request.user_id)

        reply, completeness, meta = manager.handle_turn(
            request.message,
            return_meta=True,
        )

        profile = meta["profile"]

        path = []
        path_state = "ok"
        path_message = "Profile updated"

        overrides = resolve_serving_overrides(request.user_id) or {}

        if profile.get("goal_id"):
            overrides["goal_id"] = profile["goal_id"]

        if overrides.get("goal_id"):
            generator = get_path_generator()

            artifact = generator.generate_path(
                request.user_id,
                **overrides,
            )

            path = [_step_to_dict(step) for step in artifact.steps]

            path_state = artifact.state

            if artifact.message:
                path_message = artifact.message

        return ChatResponse(
            reply=reply,
            profile_completeness=completeness,
            goal_id=profile.get("goal_id"),
            experience_level=profile.get("experience_level"),
            learning_style=profile.get("learning_style"),
            weekly_hours=profile.get("weekly_hours"),
            interests=profile.get("interests", []),
            roadmap_preferences=profile.get(
                "roadmap_preferences",
                {},
            ),
            roadmap_updated=meta.get(
                "recommendation_changed",
                False,
            ),
            path=path,
            path_state=path_state,
            path_message=path_message,
            progress_pct=0.0,
            mastered_skills=profile.get(
                "skill_ids",
                [],
            ),
            unmastered_skills=profile.get(
                "unmastered_skill_ids",
                [],
            ),
            mastery=profile.get(
                "mastery",
                {},
            ),
            learning_history=profile.get(
                "learning_history",
                [],
            ),
            goal_spec=profile.get("goal_spec"),
        )

    except Exception as exc:
        traceback.print_exc()

        raise HTTPException(
            status_code=502,
            detail=f"Profile extraction failed: {str(exc)}",
        ) from exc


@router.get("/{user_id}")
def get_profile(user_id: str) -> dict:
    from api.dependencies import get_profile_store

    store = get_profile_store()
    return {
        "user_id": user_id,
        "profile": store.get(user_id),
        "profile_completeness": store.completeness(user_id),
    }
