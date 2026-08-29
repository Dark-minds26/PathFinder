from fastapi import APIRouter, HTTPException
from api.dependencies import get_conversation_manager, get_path_generator, resolve_serving_overrides
from api.schemas.path_schema import PathStep
from api.schemas.profile_schema import ChatRequest, ChatResponse

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        manager = get_conversation_manager(request.user_id)
        reply, completeness, meta = manager.handle_turn(request.message, return_meta=True)
        profile = meta["profile"]
        path, state, message, progress_pct = [], "none", None, 0.0
        if meta["recommendation_changed"] and profile.get("goal_id"):
            try:
                artifact = get_path_generator().generate_path(request.user_id, **resolve_serving_overrides(request.user_id))
                path = [PathStep(**vars(step)) for step in artifact.steps]
                state, message = artifact.state, artifact.message
                required = get_path_generator().ctx.goal_required_ids.get(profile.get("goal_id"), set())
                mastery=profile.get("mastery", {})
                progress_pct = round(sum(float(mastery.get(s, 1.0 if s in profile.get("skill_ids", []) else 0.0)) for s in required)/len(required)*100, 1) if required else 0.0
            except Exception:
                state, message = "backend_failure", "I updated your profile, but I couldn't refresh the roadmap right now."
        return ChatResponse(
            reply=reply,
            profile_completeness=completeness,
            goal_id=profile.get("goal_id"),
            experience_level=profile.get("experience_level"),
            learning_style=profile.get("learning_style"),
            weekly_hours=profile.get("weekly_hours"),
            interests=profile.get("interests", []),
            roadmap_preferences=profile.get("roadmap_preferences", {}),
            roadmap_updated=bool(path) or state in {"mastered", "no_candidates", "backend_failure"},
            path=path,
            path_state=state,
            path_message=message,
            progress_pct=progress_pct,
            mastered_skills=profile.get("skill_ids", []),
            unmastered_skills=profile.get("unmastered_skill_ids", []),
            mastery=profile.get("mastery", {}), learning_history=profile.get("learning_history", []), goal_spec=profile.get("goal_spec"),
        )
    except Exception as exc:
        raise HTTPException(status_code=502, detail="Profile extraction failed; no new profile data was saved.") from exc
@router.get("/{user_id}")
def get_profile(user_id: str) -> dict:
    """Return the persisted learner profile so a fresh UI can hydrate without a chat turn."""
    from api.dependencies import get_profile_store
    store = get_profile_store()
    return {
        "user_id": user_id,
        "profile": store.get(user_id),
        "profile_completeness": store.completeness(user_id),
    }

