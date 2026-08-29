from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from api.dependencies import get_path_generator, get_profile_store, resolve_serving_overrides
from api.schemas.path_schema import PathRequest

router = APIRouter()
NO_GOAL_MESSAGE = "Tell me what role you're targeting before generating a path."

@router.post("/generate")
def generate_path(request: PathRequest):
    overrides = resolve_serving_overrides(request.user_id)
    if not overrides.get("goal_id"): 
        raise HTTPException(status_code=400, detail=NO_GOAL_MESSAGE)
    
    try:
        generator = get_path_generator()
        p = get_profile_store().get(request.user_id)
        
        def stream_path_generation():
            for step in generator.generate_dynamic_path_stream(request.user_id, p, **overrides):
                yield f"data: {step.model_dump_json()}\n\n"
            yield "data: [DONE]\n\n"

        return StreamingResponse(stream_path_generation(), media_type="text/event-stream")
        
    except Exception as exc: 
        raise HTTPException(status_code=500, detail="Unable to generate the learning path right now.") from exc