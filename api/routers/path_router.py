from fastapi import APIRouter, HTTPException
import networkx as nx
from api.dependencies import get_path_generator, get_profile_store, resolve_serving_overrides
from api.schemas.path_schema import PathRequest, PathResponse, PathStep
router=APIRouter()
NO_GOAL_MESSAGE="Tell me what role you're targeting before generating a path."

def _progress(user_id, artifact):
    try:
        p=get_profile_store().get(user_id); goal=p.get("goal_id"); ctx=get_path_generator().ctx
        required=set(ctx.goal_required_ids.get(goal,set()))
        closure=set(required)
        for sid in required:
            if sid in ctx.graph: closure |= set(nx.ancestors(ctx.graph,sid))
        mastery=p.get("mastery",{})
        if not closure:return 0.0
        return round(sum(float(mastery.get(s,1.0 if s in p.get("skill_ids",[]) else 0.0)) for s in closure)/len(closure)*100,1)
    except Exception:return 0.0

def _plan(path,hours):
    if not hours:return []
    remaining=float(hours); day=1; out=[]
    for step in path:
        h=float(step.duration_hours or 0)
        while h>0 and day<=7:
            chunk=min(h,remaining)
            out.append({"day":day,"course_id":step.course_id,"skill_id":step.skill_id,"hours":round(chunk,1),"focus":step.course_title})
            h-=chunk; remaining-=chunk
            if remaining<=0: day+=1; remaining=float(hours)
            if day>7: break
        if day>7:break
    return out[:14]

@router.post("/generate",response_model=PathResponse)
def generate_path(request:PathRequest)->PathResponse:
    overrides=resolve_serving_overrides(request.user_id)
    if not overrides.get("goal_id"): raise HTTPException(status_code=400,detail=NO_GOAL_MESSAGE)
    try: artifact=get_path_generator().generate_path(request.user_id,**overrides)
    except Exception as exc: raise HTTPException(status_code=500,detail="Unable to generate the learning path right now.") from exc
    path=[PathStep(**vars(step)) for step in artifact.steps]
    p=get_profile_store().get(request.user_id)
    return PathResponse(user_id=artifact.user_id,path=path,source="live_profile",state=artifact.state,message=artifact.message,progress_pct=_progress(request.user_id,artifact),weekly_plan=_plan(path,p.get("weekly_hours")))
