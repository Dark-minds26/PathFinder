from fastapi import APIRouter
from api.dependencies import get_profile_store, get_path_generator
from src.recommender.llm.llm_client import get_llm_client

router = APIRouter()

@router.get("/{course_id}/{user_id}")
def explain_recommendation(course_id: str, user_id: str):
    try:
        # 1. Fetch the real, human-readable course title from the context
        generator = get_path_generator()
        course_title = generator.ctx.title_by_course.get(course_id, course_id.replace("_", " ").title())

        # 2. Get the user's live profile
        store = get_profile_store()
        profile = store.get(user_id)
        
        goal = profile.get("goal_id", "your target role").replace("dynamic:", "").replace("_", " ").title()
        style = profile.get("learning_style", "practical")
        
        # 3. Ask the LLM to explain using the REAL title and a coaching tone
        llm = get_llm_client()
        prompt = f"""You are a supportive AI learning coach. The user's career goal is '{goal}' and they prefer a '{style}' learning style.
        They are about to start a learning module called '{course_title}'.
        Write exactly ONE short, highly encouraging, and conversational sentence directly addressing the user (using "you/your"), explaining why this specific module is the perfect next step for them.
        Do not use internal IDs like '{course_id}'. Focus on the real-world value."""
        
        resp = llm.client.chat.completions.create(
            model=llm.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.4
        )
        
        explanation = resp.choices[0].message.content.strip()
        
        # Clean up quotes if the LLM adds them
        if explanation.startswith('"') and explanation.endswith('"'):
            explanation = explanation[1:-1]
            
        return {"explanation": explanation}
        
    except Exception as e:
        print(f"Explanation Error: {e}")
        # Safe fallback so the UI never crashes
        return {"explanation": "This resource bridges your current skill gaps and aligns perfectly with your overall learning goal."}