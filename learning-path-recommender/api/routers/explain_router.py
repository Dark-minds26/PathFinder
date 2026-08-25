import pandas as pd
from fastapi import APIRouter, HTTPException

from api.dependencies import get_explainer, get_llm, get_path_generator, resolve_serving_overrides
from api.schemas.explain_schema import ExplainResponse
from src.recommender.config.configuration import ConfigurationManager
from src.recommender.utils.explain_utils import compute_attributions

router = APIRouter()
_config_manager = ConfigurationManager()


@router.get("/{course_id}/{user_id}", response_model=ExplainResponse)
def explain_recommendation(course_id: str, user_id: str) -> ExplainResponse:
    """Feature attributions for this exact (course, user) pair - SHAP
    when installed, a baseline-perturbation fallback with the same
    shape when it isn't (see Explainer) - phrased in natural language
    by the LLM client (or its local stub, see llm_client.py)."""
    generator = get_path_generator()
    ctx = generator.ctx
    overrides = resolve_serving_overrides(user_id)

    if course_id not in ctx.title_by_course:
        raise HTTPException(status_code=404, detail=f"Unknown course_id: {course_id}")

    attributions = compute_attributions(
        ctx, generator.model, get_explainer(), user_id, course_id, **overrides
    )

    course_title = ctx.title_by_course[course_id]
    goal_id = overrides.get("goal_id") or ctx.user_goal.get(user_id)
    goals = pd.read_csv(f"{_config_manager.get_data_ingestion_config().ingested_data_dir}/career_goals.csv")
    goal_row = goals[goals["goal_id"] == goal_id]
    goal_title = goal_row["title"].iloc[0] if len(goal_row) else "your career goal"

    explanation = get_llm().explain(course_title, goal_title, attributions)

    return ExplainResponse(
        course_id=course_id, user_id=user_id,
        explanation=explanation, feature_attributions=attributions,
    )
