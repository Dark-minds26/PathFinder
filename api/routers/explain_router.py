from fastapi import APIRouter, HTTPException
from api.dependencies import get_profile_store, get_path_generator
from src.recommender.llm.llm_client import get_llm_client
from src.recommender.utils.explain_utils import compute_attributions
from src.recommender.utils.main_utils import load_object
from src.recommender.config.configuration import ConfigurationManager

router = APIRouter()

@router.get("/{course_id}/{user_id}")
def explain_recommendation(course_id: str, user_id: str):
    generator = get_path_generator()

    if course_id not in generator.ctx.title_by_course:
        raise HTTPException(status_code=404, detail=f"Unknown course_id: {course_id}")

    course_title = generator.ctx.title_by_course[course_id]

    store = get_profile_store()
    profile = store.get(user_id)
    goal = profile.get("goal_id", "your target role").replace("dynamic:", "").replace("_", " ").title()

    cfg = ConfigurationManager()
    explainer = load_object(cfg.get_explainer_config().explainer_object_path)
    overrides = {
        "goal_id": profile.get("goal_id"),
        "possessed_skills": set(profile.get("skill_ids", [])),
        "experience_level": profile.get("experience_level"),
        "learning_style": profile.get("learning_style"),
        "weekly_hours": profile.get("weekly_hours"),
        "interests": profile.get("interests"),
    }
    attributions = compute_attributions(
        generator.ctx, generator.model, explainer, user_id, course_id, **overrides,
    )

    llm = get_llm_client()
    try:
        explanation = llm.explain(course_title, goal, attributions)
    except Exception as e:
        print(f"Explanation Error: {e}")
        explanation = "This resource bridges your current skill gaps and aligns perfectly with your overall learning goal."

    return {"explanation": explanation, "feature_attributions": attributions}