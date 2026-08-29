from fastapi import APIRouter, HTTPException
from api.dependencies import get_path_generator,get_profile_store,get_reroute_pipeline,resolve_serving_overrides
from api.schemas.assessment_schema import AssessmentRequest
from api.schemas.path_schema import PathResponse,PathStep
from src.recommender.assessment_engine import questions_for,score_answers
router=APIRouter(); REVIEW_THRESHOLD=60.0; VALIDATED_THRESHOLD=80.0

@router.get("/{skill_id}")
def get_assessment(skill_id:str): return {"skill_id":skill_id,"questions":questions_for(skill_id)}

@router.post("/submit",response_model=PathResponse)
def submit_assessment(request:AssessmentRequest)->PathResponse:
    overrides=resolve_serving_overrides(request.user_id)
    if not overrides.get("goal_id"): raise HTTPException(status_code=400,detail="Tell me what role you're targeting before assessing a skill.")
    score=request.score
    if score is None and request.answers is not None: score=score_answers(request.skill_id,request.answers)
    if score is None: raise HTTPException(status_code=422,detail="Provide either score or answers.")
    store=get_profile_store()
    before=store.get(request.user_id)
    previous_state=((before.get("mastery_state",{}).get(request.skill_id) or {}).get("status"))
    score_pct=float(score)

    if score_pct >= VALIDATED_THRESHOLD:
        status="validated"
        mastery_value=1.0
    elif score_pct >= REVIEW_THRESHOLD:
        status="needs_review"
        mastery_value=score_pct/100.0
    else:
        status="failed"
        mastery_value=score_pct/100.0

    store.set_mastery(request.user_id,request.skill_id,mastery_value,"assessment")
    event_type="assessment_validated" if status=="validated" else ("assessment_review" if status=="needs_review" else "assessment_failed")
    store.record_history(request.user_id,{
        "type":event_type,
        "skill_id":request.skill_id,
        "score":score_pct,
        "previous_status":previous_state,
        "new_status":status,
    })

    overrides=resolve_serving_overrides(request.user_id)
    if status=="validated":
        artifact=get_path_generator().generate_path(request.user_id,**overrides)
        message="Checkpoint validated. This skill is now mastered and the path moved forward."
    else:
        artifact=get_reroute_pipeline().reroute(
            request.user_id, request.skill_id, **overrides,
            review_skills={request.skill_id},
        )
        label="needs review" if status=="needs_review" else "failed"
        message=f"Checkpoint {label}. {request.skill_id.replace('_',' ').title()} was brought back into your path for review."

    p=store.get(request.user_id); required=set(get_path_generator().ctx.goal_required_ids.get(p.get("goal_id"),set())); mastery=p.get("mastery",{}); progress=round(sum(float(mastery.get(s,1.0 if s in p.get("skill_ids",[]) else 0.0)) for s in required)/len(required)*100,1) if required else 0.0
    return PathResponse(user_id=artifact.user_id,path=[PathStep(**vars(x)) for x in artifact.steps],source="adaptive_engine",state=artifact.state,message=message,progress_pct=progress,assessment_score=score_pct,assessment_status=status,assessment_skill_id=request.skill_id)
