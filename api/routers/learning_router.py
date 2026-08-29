from fastapi import APIRouter,HTTPException
from api.dependencies import get_profile_store,get_path_generator,resolve_serving_overrides
from src.recommender.project_catalog import project_for
router=APIRouter()
@router.get('/state/{user_id}')
def learner_state(user_id:str):
    p=get_profile_store().get(user_id); ctx=get_path_generator().ctx; goal=p.get('goal_id'); required=list(ctx.goal_required_ids.get(goal,set()))
    mastery=p.get('mastery',{}); skills=[]
    closure=set(required)
    for sid in required:
        if sid in ctx.graph: closure |= set(__import__('networkx').ancestors(ctx.graph,sid))
    for sid in sorted(closure):
        m=float(mastery.get(sid,1.0 if sid in p.get('skill_ids',[]) else 0.0)); status='mastered' if m>=.8 else ('needs_review' if m<.6 and sid in p.get('unmastered_skill_ids',[]) else ('learning' if m>0 else 'unknown')); skills.append({'skill_id':sid,'name':ctx.skill_name_by_id.get(sid,sid),'mastery':round(m*100,1),'status':status})
    progress=round(sum(x['mastery'] for x in skills)/len(skills),1) if skills else 0.0
    milestones=[]
    for i in range(0,len(skills),4):
        chunk=skills[i:i+4]; milestones.append({'title':f'Milestone {i//4+1}','skills':[x['skill_id'] for x in chunk],'complete':all(x['mastery']>=80 for x in chunk)})
    try:
        artifact=get_path_generator().generate_path(user_id,**resolve_serving_overrides(user_id))
        next_action=vars(artifact.steps[0]) if artifact.steps else None
    except Exception:
        next_action=None
    goal_title = None
    if goal:
        try:
            import pandas as pd
            from src.recommender.config.configuration import ConfigurationManager
            cfg = ConfigurationManager().get_data_ingestion_config()
            goals_df = pd.read_csv(f"{cfg.ingested_data_dir}/career_goals.csv")
            rows = goals_df[goals_df["goal_id"] == goal]
            if not rows.empty:
                goal_title = str(rows.iloc[0]["title"])
        except Exception:
            goal_title = None
        if not goal_title:
            goal_title = str(p.get('goal_spec', {}).get('title') or goal).replace('goal_', '').replace('_', ' ').title()
    return {'user_id':user_id,'goal_id':goal,'goal_title':goal_title,'goal_spec':p.get('goal_spec'),'skills':skills,'progress_pct':progress,'history':p.get('learning_history',[]),'completed_course_ids':p.get('completed_course_ids',[]),'milestones':milestones,'next_best_action':next_action}
@router.get('/projects/{skill_id}')
def get_project(skill_id:str): return project_for(skill_id)
@router.post('/projects/{skill_id}/complete')
def complete_project(skill_id:str,user_id:str):
    store=get_profile_store(); project=project_for(skill_id); p=store.get(user_id); old=float(p.get('mastery',{}).get(skill_id,0)); store.set_mastery(user_id,skill_id,min(.79,old+.15),'project'); store.record_history(user_id,{'type':'project_completed','skill_id':skill_id,'project_id':project['project_id']}); return {'project':project,'mastery':store.get(user_id).get('mastery',{}).get(skill_id,0),'message':'Project evidence recorded. Take the checkpoint to validate mastery.'}
@router.post('/resources/{course_id}/complete')
def complete_resource(course_id:str,user_id:str,skill_id:str):
    store=get_profile_store(); store.complete_course(user_id,course_id,skill_id); return {'course_id':course_id,'message':'Learning activity recorded.'}
