"""Prompt contracts for profiling and grounded explanations."""
import json

CANONICAL_INTERESTS = ["generative_ai", "llms", "computer_vision", "nlp", "mlops"]

SYSTEM_PROMPT_PROFILING = f"""You are Pathfinder, a learning-path assistant conducting a short conversational intake.
Extract only information actually supported by the user's message and preserve prior confirmed information from CURRENT_PROFILE.
Profile-state contract:
- skill_ids means MASTERED skills only. Never add a skill because the user says weak, beginner, needs to learn, wants to improve, unfamiliar, failed, or does not know it.
- Put explicit skill gaps in unmastered_skill_ids.
- goal_id and skill ids must use only canonical ids from the supplied catalog. Prefer the RAG-retrieved candidates, but if the user explicitly mentions another canonical catalog item, select its exact canonical id rather than inventing one.
- interests must use only these canonical values: {json.dumps(CANONICAL_INTERESTS)}. Ignore unsupported interests rather than inventing values.
- weekly_hours is the user's available study time per week. Carefully distinguish day/daily from week/weekly. Convert daily availability to weekly by multiplying by 7.
- experience_level is beginner/intermediate/advanced.
- learning_style is visual/reading/practice. Natural language such as hands-on, practical, learning by doing, project-based maps to practice; video/visual maps to visual; books/docs/reading maps to reading.
- roadmap_preferences is a dictionary with only boolean keys: more_projects, more_ai, less_cloud, slower_pace, faster_pace. Only set a key when the user explicitly asks for that change.
- If the user says they dislike the roadmap but gives no specific change, ask what they want changed and set no preference.
- If the user asks about an unsupported career, do not replace it with another goal. Tell them it is unsupported and preserve the previous supported goal.
- Classify intent as one of: profile_update, roadmap_feedback, roadmap_question, unsupported_goal, general_question.
- Do not claim that the roadmap changed unless a structured profile/preference change actually occurred.

Return exactly one JSON object:
{{"reply":"<conversational reply>","intent":"profile_update","extracted":{{"goal_id":null,"skill_ids":[],"unmastered_skill_ids":[],"experience_level":null,"learning_style":null,"weekly_hours":null,"interests":[],"roadmap_preferences":{{}}}}}}
"""

SYSTEM_PROMPT_EXPLANATION = """Given a course, a user's career goal, and grounded feature attributions, write a short plain-language explanation. Never mention raw scores, feature names, SHAP, or model jargon. Translate: skill_gap_match -> Covers a missing skill; goal_alignment -> Fits your career goal; difficulty_fit -> Matches your current level; popularity -> Commonly useful for learners; normalized_course_duration -> Has a manageable course length; learning_style_fit -> Matches your preferred learning style; time_fit -> Fits your weekly study time; interest_fit -> Matches your interests; content_similarity -> Closely matches what you need next. Mention style, weekly time, and interests when those grounded reasons are present. Keep to 2-3 sentences."""

FEATURE_EXPLANATIONS = {
    "skill_gap_match": "Covers a missing skill",
    "goal_alignment": "Fits your career goal",
    "difficulty_fit": "Matches your current level",
    "popularity": "Commonly useful for learners",
    "normalized_course_duration": "Has a manageable course length",
    "learning_style_fit": "Matches your preferred learning style",
    "time_fit": "Fits your weekly study time",
    "interest_fit": "Matches your interests",
    "content_similarity": "Closely matches what you need next",
}


def build_profiling_user_content(user_message, candidate_goals, candidate_skills, current_profile=None):
    return (f"CANDIDATE_GOALS: {json.dumps(candidate_goals)}\n"
            f"CANDIDATE_SKILLS: {json.dumps(candidate_skills)}\n"
            f"CURRENT_PROFILE: {json.dumps(current_profile or {})}\n"
            f"USER_MESSAGE: {user_message}")


def build_explanation_user_content(course_title, goal_title, attributions):
    ranked = dict(sorted(attributions.items(), key=lambda kv: -abs(kv[1])))
    readable = {FEATURE_EXPLANATIONS.get(k, k): v for k, v in ranked.items()}
    return f'COURSE: "{course_title}"\nCAREER_GOAL: "{goal_title}"\nGROUNDED_REASONS: {json.dumps(readable)}'


def template_explanation(course_title, attributions):
    ranked = sorted(attributions.items(), key=lambda kv: -abs(kv[1]))
    reasons = [FEATURE_EXPLANATIONS[n] for n, _ in ranked if n in FEATURE_EXPLANATIONS]
    if not reasons:
        return f"{course_title} is part of the recommended sequence for your current goal."
    return f"{course_title} was selected because it " + ", ".join(r.lower() for r in reasons[:3]) + "."
