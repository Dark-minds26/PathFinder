from fastapi import APIRouter, HTTPException
from api.dependencies import get_profile_store
from src.recommender.llm.llm_client import get_llm_client
from api.schemas.assessment_schema import AssessmentRequest
from api.schemas.path_schema import PathResponse

router = APIRouter()

# Temporary in-memory cache to store correct answers for dynamic grading
_temp_assessment_cache = {}

@router.get("/{skill_id}")
def get_assessment(skill_id: str):
    try:
        # 1. Dynamically generate fresh questions using the LLM!
        llm = get_llm_client()
        dynamic_test = llm.generate_assessment(skill_id)
        
        # 2. Save correct answers to grade later
        cache_key = f"test_{skill_id}"
        _temp_assessment_cache[cache_key] = {
            q["id"]: q.get("correct_index", 0) for q in dynamic_test.get("questions", [])
        }
        
        # 3. Hide correct answers from the frontend UI
        safe_questions = [
            {"id": q["id"], "question": q["question"], "options": q["options"]} 
            for q in dynamic_test.get("questions", [])
        ]
        
        return {"skill_id": skill_id, "questions": safe_questions}
    except Exception as exc:
        print(f"LLM Assessment Error: {exc}")
        raise HTTPException(status_code=500, detail="Could not dynamically generate assessment.") from exc

@router.post("/submit", response_model=PathResponse)
def submit_assessment(request: AssessmentRequest) -> PathResponse:
    cache_key = f"test_{request.skill_id}"
    correct_answers = _temp_assessment_cache.get(cache_key, {})
    
    # Grading Logic
    total = len(correct_answers) if correct_answers else len(request.answers or {})
    correct = 0
    
    if correct_answers and request.answers:
        for q_id, answer_index in request.answers.items():
            if str(correct_answers.get(q_id)) == str(answer_index):
                correct += 1
    else:
        # Fallback if cache missed
        correct = total

    score_pct = (correct / total) * 100 if total > 0 else 100.0
    status = "validated" if score_pct >= 70.0 else "needs_review"
    mastery_value = 1.0 if status == "validated" else score_pct / 100.0

    store = get_profile_store()
    before = store.get(request.user_id)
    previous_state = ((before.get("mastery_state", {}).get(request.skill_id) or {}).get("status"))

    # Update actual mastery in profile based on the test
    store.set_mastery(request.user_id, request.skill_id, mastery_value, "assessment")
    
    event_type = "assessment_validated" if status == "validated" else "assessment_review"
    store.record_history(request.user_id, {
        "type": event_type,
        "skill_id": request.skill_id,
        "score": score_pct,
        "previous_status": previous_state,
        "new_status": status,
    })

    msg = "Checkpoint validated. Skill mastered!" if status == "validated" else "Checkpoint missed. Skill added back for review."
    
    # We return an empty path here because the UI is going to trigger the SSE stream next!
    return PathResponse(
        user_id=request.user_id,
        path=[],
        source="dynamic_assessment",
        state="ok",
        message=msg,
        progress_pct=0.0,
        assessment_score=score_pct,
        assessment_status=status,
        assessment_skill_id=request.skill_id
    )