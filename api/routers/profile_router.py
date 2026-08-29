from fastapi import APIRouter, HTTPException
from api.dependencies import get_conversation_manager
from api.schemas.profile_schema import ChatRequest, ChatResponse
import traceback

router = APIRouter()

@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        manager = get_conversation_manager(request.user_id)
        reply, completeness, meta = manager.handle_turn(request.message, return_meta=True)
        profile = meta["profile"]

        # We no longer generate the path here! The UI handles it separately via the SSE stream.
        # Just return the updated profile state.
        return ChatResponse(
            reply=reply,
            profile_completeness=completeness,
            goal_id=profile.get("goal_id"),
            experience_level=profile.get("experience_level"),
            learning_style=profile.get("learning_style"),
            weekly_hours=profile.get("weekly_hours"),
            interests=profile.get("interests", []),
            roadmap_preferences=profile.get("roadmap_preferences", {}),
            roadmap_updated=meta.get("recommendation_changed", False), # Tells UI to trigger the SSE stream!
            path=[],
            path_state="ok",
            path_message="Profile updated",
            progress_pct=0.0,
            mastered_skills=profile.get("skill_ids", []),
            unmastered_skills=profile.get("unmastered_skill_ids", []),
            mastery=profile.get("mastery", {}),
            learning_history=profile.get("learning_history", []),
            goal_spec=profile.get("goal_spec"),
        )
    except Exception as exc:
        traceback.print_exc() # Prints the actual error to your terminal for easier debugging
        raise HTTPException(status_code=502, detail=f"Profile extraction failed: {str(exc)}") from exc

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