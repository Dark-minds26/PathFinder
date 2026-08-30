from fastapi import APIRouter, HTTPException
from api.dependencies import get_profile_store, get_path_generator, resolve_serving_overrides
from src.recommender.llm.llm_client import get_llm_client
from api.schemas.assessment_schema import AssessmentRequest
from api.schemas.path_schema import PathResponse, PathStep

router = APIRouter()

_temp_assessment_cache = {}

STATUS_TO_EVENT = {
    "validated": "assessment_validated",
    "failed": "assessment_failed",
    "needs_review": "assessment_needs_review",
}


def _entity_step_to_schema(step) -> PathStep:
    return PathStep(
        skill_id=step.skill_id, course_id=step.course_id, course_title=step.course_title,
        sequence_order=step.sequence_order, predicted_score=step.predicted_score,
        duration_hours=step.duration_hours, format=step.format, status=step.status,
        why=step.why, competency=step.competency,
    )


@router.get("/{skill_id}")
def get_assessment(skill_id: str):
    try:
        llm = get_llm_client()
        dynamic_test = llm.generate_assessment(skill_id)
        cache_key = f"test_{skill_id}"
        _temp_assessment_cache[cache_key] = {
            q["id"]: q.get("correct_index", 0) for q in dynamic_test.get("questions", [])
        }
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

    if request.answers:
        if correct_answers:
            total = len(correct_answers)
            correct = sum(
                1 for qid, ans in request.answers.items()
                if str(correct_answers.get(qid)) == str(ans)
            )
        else:
            # No cached answer key for this skill - fall back to the
            # demo convention that option index 1 is "correct".
            total = len(request.answers)
            correct = sum(1 for ans in request.answers.values() if int(ans) == 1)
        score_pct = (correct / total) * 100 if total > 0 else 100.0
    elif request.score is not None:
        score_pct = request.score
    else:
        score_pct = 100.0

    mastery_value = score_pct / 100.0

    store = get_profile_store()
    before = store.get(request.user_id)
    previous_state = ((before.get("mastery_state", {}).get(request.skill_id) or {}).get("status"))

    profile = store.set_mastery(request.user_id, request.skill_id, mastery_value, "assessment")
    new_status = profile["mastery_state"][request.skill_id]["status"]

    store.record_history(request.user_id, {
        "type": STATUS_TO_EVENT[new_status],
        "skill_id": request.skill_id,
        "score": score_pct,
        "previous_status": previous_state,
        "new_status": new_status,
    })

    msg = "Checkpoint validated. Skill mastered!" if new_status == "validated" else "Checkpoint missed. Skill added back for review."

    overrides = resolve_serving_overrides(request.user_id)
    path_steps: list[PathStep] = []
    if overrides.get("goal_id"):
        review_skills = {request.skill_id} if new_status != "validated" else None
        generator = get_path_generator()
        artifact = generator.generate_path(request.user_id, review_skills=review_skills, **overrides)
        path_steps = [_entity_step_to_schema(s) for s in artifact.steps]

    return PathResponse(
        user_id=request.user_id,
        path=path_steps,
        source="dynamic_assessment",
        state="ok",
        message=msg,
        progress_pct=0.0,
        assessment_score=score_pct,
        assessment_status=new_status,
        assessment_skill_id=request.skill_id,
    )