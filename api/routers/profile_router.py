from fastapi import APIRouter

from api.dependencies import get_conversation_manager
from api.schemas.profile_schema import ChatRequest, ChatResponse

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Conversational profiling turn: RAG narrows the skill/goal
    catalog to plausible candidates, the LLM client extracts
    structured fields constrained to those candidates, and the result
    is persisted to this user's live profile."""
    manager = get_conversation_manager(request.user_id)
    reply, completeness = manager.handle_turn(request.message)
    return ChatResponse(reply=reply, profile_completeness=completeness)
