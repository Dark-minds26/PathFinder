"""Prompt construction for the two LLM-backed flows: conversational
profiling and recommendation explanation. Kept separate from
llm_client.py so the prompt text can be iterated on without touching
provider wiring, and so LocalStubLLMClient's template_explanation can
reuse the same FEATURE_EXPLANATIONS the real prompt uses."""
import json

SYSTEM_PROMPT_PROFILING = """You are a learning-path assistant conducting a short conversational \
intake. Ask naturally about the user's current skills, career goal, and preferred learning \
style, and extract fields as soon as you're confident about them - don't wait until the end.

You will be given CANDIDATE_GOALS and CANDIDATE_SKILLS - the only valid ids for this system. \
Only ever return an id that appears in one of those lists; never invent one. Only fill a field \
once the user's message actually supports it.

Always reply with a single JSON object, no other text, shaped exactly like:
{"reply": "<your conversational reply to the user>",
 "extracted": {"goal_id": "<id from CANDIDATE_GOALS, or null>",
               "skill_ids": ["<id from CANDIDATE_SKILLS>", ...],
               "experience_level": "<beginner|intermediate|advanced or null>",
               "learning_style": "<visual|reading|practice or null>"}}"""

SYSTEM_PROMPT_EXPLANATION = """Given a course, a user's career goal, and a set of numeric \
feature attributions, write one short, plain-language paragraph (2-3 sentences) explaining why \
the course was recommended. Reference only the attributions provided - never invent a reason \
that isn't backed by one of them - and lead with whichever attribution has the largest \
magnitude."""

FEATURE_EXPLANATIONS = {
    "skill_gap_match": "how directly it teaches skills you're currently missing",
    "goal_alignment": "how well it lines up with your career goal",
    "difficulty_fit": "matching your current experience level",
    "popularity": "how well it's worked for learners like you",
    "predicted_time_to_complete": "fitting the time you have available",
    "content_similarity": "how closely its content matches what you need next",
}


def build_profiling_user_content(
    user_message: str, candidate_goals: list[dict], candidate_skills: list[dict]
) -> str:
    """The user-turn content sent to a real LLM: the RAG-narrowed
    candidate lists followed by what the person actually said."""
    return (
        f"CANDIDATE_GOALS: {json.dumps(candidate_goals)}\n"
        f"CANDIDATE_SKILLS: {json.dumps(candidate_skills)}\n"
        f"USER_MESSAGE: {user_message}"
    )


def build_explanation_user_content(course_title: str, goal_title: str, attributions: dict) -> str:
    ranked = dict(sorted(attributions.items(), key=lambda kv: -abs(kv[1])))
    return (
        f'COURSE: "{course_title}"\n'
        f'CAREER_GOAL: "{goal_title}"\n'
        f"ATTRIBUTIONS (most to least influential): {json.dumps(ranked)}"
    )


def template_explanation(course_title: str, attributions: dict) -> str:
    """Deterministic fallback used by LocalStubLLMClient (no LLM
    credentials configured): turns the same numeric attributions a
    real LLM would have been given into one plain sentence, so
    /explain still returns a real, human-readable answer offline."""
    ranked = sorted(attributions.items(), key=lambda kv: -abs(kv[1]))
    positive = [(name, val) for name, val in ranked if val > 0][:2]
    if not positive:
        return f'"{course_title}" was recommended based on an overall fit across your profile.'
    reasons = " and ".join(FEATURE_EXPLANATIONS.get(name, name) for name, _ in positive)
    return f'"{course_title}" was recommended mainly because of {reasons}.'
