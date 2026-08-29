from fastapi import APIRouter, HTTPException
from api.dependencies import get_profile_store, get_path_generator, resolve_serving_overrides
from src.recommender.project_catalog import project_for

router = APIRouter()

@router.get('/state/{user_id}')
def learner_state(user_id: str):
    p = get_profile_store().get(user_id)
    ctx = get_path_generator().ctx
    goal = p.get('goal_id')
    required = list(ctx.goal_required_ids.get(goal, set()))
    mastery = p.get('mastery', {})
    skills = []
    
    closure = set(required)
    for sid in required:
        if sid in ctx.graph: 
            closure |= set(__import__('networkx').ancestors(ctx.graph, sid))
            
    for sid in sorted(closure):
        m = float(mastery.get(sid, 1.0 if sid in p.get('skill_ids', []) else 0.0))
        status = 'mastered' if m >= 0.8 else ('needs_review' if m < 0.6 and sid in p.get('unmastered_skill_ids', []) else ('learning' if m > 0 else 'unknown'))
        skills.append({
            'skill_id': sid,
            'name': ctx.skill_name_by_id.get(sid, sid),
            'mastery': round(m * 100, 1),
            'status': status
        })
        
    progress = round(sum(x['mastery'] for x in skills) / len(skills), 1) if skills else 0.0
    
    milestones = []
    for i in range(0, len(skills), 4):
        chunk = skills[i:i+4]
        milestones.append({
            'title': f'Milestone {i//4+1}',
            'skills': [x['skill_id'] for x in chunk],
            'complete': all(x['mastery'] >= 80 for x in chunk)
        })

    # FIX: Get next action instantly from local skills
    next_action = None
    for s in skills:
        if s['status'] in ['learning', 'needs_review', 'unknown']:
            next_action = {
                "skill_id": s['skill_id'],
                "course_title": "Continue with " + s['name'],
                "why": "This is the next unmastered skill in your prerequisite map."
            }
            break

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
            
    return {
        'user_id': user_id,
        'goal_id': goal,
        'goal_title': goal_title,
        'goal_spec': p.get('goal_spec'),
        'skills': skills,
        'progress_pct': progress,
        'history': p.get('learning_history', []),
        'completed_course_ids': p.get('completed_course_ids', []),
        'milestones': milestones,
        'next_best_action': next_action
    }

@router.get('/projects/{skill_id}')
def get_project(skill_id: str): 
    # Dynamically generate an industry-grade project brief using the LLM!
    try:
        from src.recommender.llm.llm_client import get_llm_client
        import json
        
        llm = get_llm_client()
        prompt = f"""You are a senior tech lead. Create a real-world, scalable portfolio project brief for a developer to practice the skill: {skill_id}.
        Return ONLY a JSON object matching this exact schema:
        {{"title": "Project Name", "estimated_hours": 5, "description": "2-3 sentences explaining the architecture and what they will build.", "skills": ["{skill_id}", "system_design"]}}"""
        
        resp = llm.client.chat.completions.create(
            model=llm.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4,
            response_format={"type": "json_object"}
        )
        return json.loads(resp.choices[0].message.content)
    except Exception as e:
        print(f"Project Generation Error: {e}")
        # Fallback just in case of an API timeout
        from src.recommender.project_catalog import project_for
        return project_for(skill_id)

@router.post('/projects/{skill_id}/complete')
def complete_project(skill_id: str, user_id: str):
    store = get_profile_store()
    p = store.get(user_id)
    old = float(p.get('mastery', {}).get(skill_id, 0))
    
    # Project completion gives a solid bump, capped at 79% (must take checkpoint to get 100%)
    store.set_mastery(user_id, skill_id, min(0.79, old + 0.15), 'project')
    store.record_history(user_id, {'type': 'project_completed', 'skill_id': skill_id, 'project_id': f"dynamic_{skill_id}"})
    
    return {
        'mastery': store.get(user_id).get('mastery', {}).get(skill_id, 0),
        'message': 'Project evidence recorded. Take the checkpoint to validate mastery.'
    }

@router.post('/resources/{course_id}/complete')
def complete_resource(course_id: str, user_id: str, skill_id: str):
    store = get_profile_store()
    store.complete_course(user_id, course_id, skill_id)
    
    # Get current mastery and bump it, but strictly cap it at 75% (0.75).
    # 80% (0.80) is required for full mastery, meaning they MUST take a checkpoint to clear the node.
    p = store.get(user_id)
    old_mastery = float(p.get('mastery', {}).get(skill_id, 0.0))
    new_mastery = min(0.75, old_mastery + 0.40)
    
    store.set_mastery(user_id, skill_id, new_mastery, "resource_completed")
    
    return {
        'course_id': course_id, 
        'message': 'Resource completed! Progress updated. Take the checkpoint to prove 100% mastery and advance.'
    }