from fastapi import APIRouter, HTTPException

from api.schemas.profile_schema import ChatRequest, ChatResponse

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    """Conversational profiling turn: RAG + LLM extract structured
    signals (skills, goal, learning style) from free text."""
    raise HTTPException(status_code=501, detail="Implemented in Phase 3 - LLM integration")
