"""LLM provider abstraction: one interface, three backends.

GroqClient / OpenAIClient wrap the real SDKs (both already in
requirements.txt) against their current chat-completions APIs - correct
code, but they need a real API key and network to actually run.
LocalStubLLMClient needs neither: it's a deterministic stand-in that
matches the same structured candidate lists a real LLM would be given
by keyword overlap instead of language understanding. It exists so the
full profiling + explanation flow can be built, wired, and tested
without spending API calls or requiring credentials.

get_llm_client() picks whichever backend is actually usable, so
`pip install groq` + GROQ_API_KEY upgrades conversation quality with
zero changes anywhere else - the same graceful-degradation pattern as
the LightGBM/SHAP fallbacks in Phase 2.
"""
import os
import sys
from abc import ABC, abstractmethod

from src.recommender.exception import RecommenderException
from src.recommender.logger import logging


class LLMClient(ABC):
    @abstractmethod
    def profile_turn(
        self,
        history: list[dict],
        user_message: str,
        candidate_goals: list[dict],
        candidate_skills: list[dict],
        current_profile: dict | None = None,
    ) -> dict:
        """One profiling turn, constrained to the candidates given.

        Returns {"reply": str, "goal_id": str | None,
        "skill_ids": list[str], "experience_level": str | None,
        "learning_style": str | None} - already resolved to canonical
        IDs, never free text the caller has to re-resolve."""

    @abstractmethod
    def explain(self, course_title: str, goal_title: str, attributions: dict) -> str:
        """One short natural-language paragraph explaining why
        course_title was recommended, grounded in attributions."""


def _parse_profile_json(raw: str) -> dict:
    import json
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError) as e:
        raise RecommenderException(f"LLM returned malformed JSON: {e}", sys) from e
    if not isinstance(data, dict) or not isinstance(data.get("extracted"), dict):
        raise RecommenderException("LLM response is missing the required 'extracted' object", sys)
    if "reply" not in data:
        raise RecommenderException("LLM response is missing required field 'reply'", sys)
    extracted = data["extracted"]
    result = {
        "reply": str(data["reply"]).strip(),
        "intent": data.get("intent", "profile_update"),
        "goal_id": extracted.get("goal_id"),
        "skill_ids": extracted.get("skill_ids") or [],
        "unmastered_skill_ids": extracted.get("unmastered_skill_ids") or [],
        "experience_level": extracted.get("experience_level"),
        "learning_style": extracted.get("learning_style"),
        "weekly_hours": extracted.get("weekly_hours"),
        "interests": extracted.get("interests") or [],
        "roadmap_preferences": extracted.get("roadmap_preferences") or {},
    }
    if not result["reply"] or len(result["reply"]) > 4000:
        raise RecommenderException("LLM response has an invalid reply", sys)
    if not isinstance(result["skill_ids"], list) or not isinstance(result["unmastered_skill_ids"], list):
        raise RecommenderException("LLM response has invalid profile field types", sys)
    if not all(isinstance(x, str) and x.strip() for x in result["skill_ids"] + result["unmastered_skill_ids"]):
        raise RecommenderException("LLM response contains invalid skill ids", sys)
    allowed_intents = {"profile_update", "roadmap_feedback", "roadmap_question", "unsupported_goal", "general_question"}
    if result["intent"] not in allowed_intents:
        raise RecommenderException("LLM response contains an invalid intent", sys)
    if result["goal_id"] is not None and not isinstance(result["goal_id"], str):
        raise RecommenderException("LLM response contains an invalid goal_id", sys)
    return result


def _norm_label(value: str) -> str:
    import re
    return re.sub(r"[^a-z0-9]+", " ", str(value).strip().lower()).strip()


def _coerce_profile_result(result: dict, candidate_goals: list[dict], candidate_skills: list[dict]) -> dict:
    """Normalize harmless formatting differences from a real LLM without inventing ids."""
    valid_goals = {g["goal_id"] for g in candidate_goals}
    goal_titles = {_norm_label(g["title"]): g["goal_id"] for g in candidate_goals}
    goal_value = result.get("goal_id")
    if isinstance(goal_value, str) and goal_value not in valid_goals:
        result["goal_id"] = goal_titles.get(_norm_label(goal_value), goal_value)

    valid_skills = {s["skill_id"] for s in candidate_skills}
    skill_names = {_norm_label(s["skill_name"]): s["skill_id"] for s in candidate_skills}
    significant = {}
    for s in candidate_skills:
        tokens = [t for t in _norm_label(s["skill_name"]).split() if len(t) >= 4]
        for token in tokens:
            significant.setdefault(token, []).append(s["skill_id"])

    def coerce_skill(value: str) -> str:
        if value in valid_skills:
            return value
        norm = _norm_label(value)
        if norm in skill_names:
            return skill_names[norm]
        ids = set()
        for token in norm.split():
            if token in significant and len(significant[token]) == 1:
                ids.add(significant[token][0])
        return next(iter(ids)) if len(ids) == 1 else value

    result["skill_ids"] = [coerce_skill(x) for x in result.get("skill_ids", [])]
    result["unmastered_skill_ids"] = [coerce_skill(x) for x in result.get("unmastered_skill_ids", [])]

    style_aliases = {
        "hands on": "practice", "hands-on": "practice", "practical": "practice",
        "project based": "practice", "project-based": "practice", "learning by doing": "practice",
        "videos": "visual", "video": "visual", "visual": "visual",
        "books": "reading", "documentation": "reading", "docs": "reading", "reading": "reading",
    }
    style = result.get("learning_style")
    if isinstance(style, str):
        norm = _norm_label(style)
        result["learning_style"] = style_aliases.get(norm, style.lower().strip())

    interests_aliases = {
        "genai": "generative_ai", "generative ai": "generative_ai",
        "llm": "llms", "large language models": "llms", "language models": "llms",
        "computer vision": "computer_vision", "natural language processing": "nlp",
        "machine learning operations": "mlops",
    }
    if isinstance(result.get("interests"), list):
        result["interests"] = [interests_aliases.get(_norm_label(x), x) if isinstance(x, str) else x for x in result["interests"]]

    weekly = result.get("weekly_hours")
    if isinstance(weekly, str):
        import re
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:hours?|hrs?)\s*(?:/\s*)?(?:per|a|each)?\s*(day|daily|week|weekly)?", weekly.lower())
        if m:
            hours = float(m.group(1))
            unit = m.group(2) or "week"
            result["weekly_hours"] = hours * 7 if unit in {"day", "daily"} else hours
    return result


def _validate_profile_result(result: dict, candidate_goals: list[dict], candidate_skills: list[dict]) -> dict:
    result = _coerce_profile_result(result, candidate_goals, candidate_skills)
    valid_goals = {g["goal_id"] for g in candidate_goals}
    valid_skills = {s["skill_id"] for s in candidate_skills}
    if result["goal_id"] is not None and result["goal_id"] not in valid_goals:
        raise RecommenderException(f"LLM returned unknown goal_id: {result['goal_id']}", sys)
    unknown = (set(result["skill_ids"]) | set(result["unmastered_skill_ids"])) - valid_skills
    if unknown:
        raise RecommenderException(f"LLM returned unknown skill ids: {sorted(unknown)}", sys)
    if result["experience_level"] is not None and result["experience_level"] not in {"beginner", "intermediate", "advanced"}:
        raise RecommenderException("LLM returned invalid experience_level", sys)
    if result["learning_style"] is not None and result["learning_style"] not in {"visual", "reading", "practice"}:
        raise RecommenderException("LLM returned invalid learning_style", sys)
    if result["weekly_hours"] is not None:
        try:
            result["weekly_hours"] = float(result["weekly_hours"])
        except (TypeError, ValueError) as e:
            raise RecommenderException("LLM returned invalid weekly_hours", sys) from e
        if not 0 < result["weekly_hours"] <= 168:
            raise RecommenderException("LLM returned weekly_hours outside 0-168", sys)
    if not isinstance(result["interests"], list) or not all(isinstance(x, str) for x in result["interests"]):
        raise RecommenderException("LLM response has invalid interests", sys)
    allowed_interests = {"generative_ai", "llms", "computer_vision", "nlp", "mlops"}
    unknown_interests = set(result["interests"]) - allowed_interests
    if unknown_interests:
        raise RecommenderException(f"LLM returned unknown interests: {sorted(unknown_interests)}", sys)
    allowed_prefs = {"more_projects", "more_ai", "less_cloud", "slower_pace", "faster_pace"}
    if not isinstance(result["roadmap_preferences"], dict) or not set(result["roadmap_preferences"]).issubset(allowed_prefs):
        raise RecommenderException("LLM response has invalid roadmap_preferences", sys)
    result["roadmap_preferences"] = {k: bool(v) for k, v in result["roadmap_preferences"].items() if bool(v)}
    unmastered = set(result["unmastered_skill_ids"])
    result["skill_ids"] = [sid for sid in result["skill_ids"] if sid not in unmastered]
    return result


class GroqClient(LLMClient):
    """Backed by Groq's OpenAI-compatible chat completions API -
    chosen as the default for demo-time latency (see Phase 1)."""

    def __init__(self, model: str = "openai/gpt-oss-20b") -> None:
        import groq  # local import: only required when this backend is actually selected

        self.client = groq.Groq(api_key=os.environ["GROQ_API_KEY"])
        self.model = model

    def profile_turn(self, history, user_message, candidate_goals, candidate_skills, current_profile=None) -> dict:
        try:
            from src.recommender.llm.prompt_templates import (
                SYSTEM_PROMPT_PROFILING,
                build_profiling_user_content,
            )

            messages = [{"role": "system", "content": SYSTEM_PROMPT_PROFILING}]
            messages.extend(history[-8:])
            messages.append({
                "role": "user",
                "content": build_profiling_user_content(user_message, candidate_goals, candidate_skills, current_profile),
            })
            resp = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=0.3,
                response_format={"type": "json_object"},
            )
            return _validate_profile_result(_parse_profile_json(resp.choices[0].message.content), candidate_goals, candidate_skills)
        except Exception as e:
            raise RecommenderException(e, sys) from e

    def explain(self, course_title: str, goal_title: str, attributions: dict) -> str:
        try:
            from src.recommender.llm.prompt_templates import (
                SYSTEM_PROMPT_EXPLANATION,
                build_explanation_user_content,
            )

            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_EXPLANATION},
                    {"role": "user", "content": build_explanation_user_content(course_title, goal_title, attributions)},
                ],
                temperature=0.3,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            raise RecommenderException(e, sys) from e


class OpenAIClient(LLMClient):
    """Swap-in alternative to GroqClient - same interface, chosen via
    LLM_PROVIDER=openai when quality matters more than demo latency."""

    def __init__(self, model: str = "gpt-4o-mini") -> None:
        import openai

        self.client = openai.OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        self.model = model

    def profile_turn(self, history, user_message, candidate_goals, candidate_skills, current_profile=None) -> dict:
        try:
            from src.recommender.llm.prompt_templates import (
                SYSTEM_PROMPT_PROFILING,
                build_profiling_user_content,
            )

            messages = [{"role": "system", "content": SYSTEM_PROMPT_PROFILING}]
            messages.extend(history[-8:])
            messages.append({
                "role": "user",
                "content": build_profiling_user_content(user_message, candidate_goals, candidate_skills, current_profile),
            })
            resp = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=0.3,
                response_format={"type": "json_object"},
            )
            return _validate_profile_result(_parse_profile_json(resp.choices[0].message.content), candidate_goals, candidate_skills)
        except Exception as e:
            raise RecommenderException(e, sys) from e

    def explain(self, course_title: str, goal_title: str, attributions: dict) -> str:
        try:
            from src.recommender.llm.prompt_templates import (
                SYSTEM_PROMPT_EXPLANATION,
                build_explanation_user_content,
            )

            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT_EXPLANATION},
                    {"role": "user", "content": build_explanation_user_content(course_title, goal_title, attributions)},
                ],
                temperature=0.3,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            raise RecommenderException(e, sys) from e


_SKILL_NAME_STOPWORDS = {"basics", "fundamentals", "advanced", "and", "with"}


def _skill_name_matches(skill_name: str, text: str) -> bool:
    """Full name first ('sql basics' in text); falls back to a
    significant word from it ('sql') so 'I know some SQL' still
    matches 'SQL basics'. Still just word overlap, not understanding -
    see the class docstring."""
    import re
    name = skill_name.lower()
    if re.search(r"\b" + re.escape(name) + r"\b", text):
        return True
    words = [w for w in name.replace("/", " ").split() if w not in _SKILL_NAME_STOPWORDS and len(w) > 2]
    return any(re.search(r"\b" + re.escape(w) + r"\b", text) for w in words)


class LocalStubLLMClient(LLMClient):
    """Deterministic local stand-in with the same safe profile contract as real LLMs."""

    def profile_turn(self, history, user_message, candidate_goals, candidate_skills, current_profile=None) -> dict:
        import re
        text = user_message.lower()
        negative = re.compile(r"\b(?:weak|bad at|don't know|do not know|not familiar with|unfamiliar with|need to learn|want to learn|struggle with|struggling with|beginner in|new to|less|skip|not interested in)\b[^.!?]{0,100}")
        goal_id = next((g["goal_id"] for g in candidate_goals if g["title"].lower() in text), None)
        if goal_id is None:
            for g in candidate_goals:
                title_tokens = g["title"].lower().split()
                if title_tokens and title_tokens[0] in text and ("engineer" in text or "scientist" in text):
                    goal_id = g["goal_id"]
                    break
        mastered, unmastered = [], []
        for skill in candidate_skills:
            if not _skill_name_matches(skill["skill_name"], text):
                continue
            words = [w for w in skill["skill_name"].lower().replace("/", " ").split() if len(w) > 2 and w not in _SKILL_NAME_STOPWORDS]
            idx = next((text.find(w) for w in words if text.find(w) >= 0), -1)
            window = text[max(0, idx-70):idx+100] if idx >= 0 else text
            if negative.search(window):
                unmastered.append(skill["skill_id"])
            elif any(marker in window for marker in ("know", "comfortable", "experience", "used", "learned", "mastered", "i use", "i do", "i am")):
                mastered.append(skill["skill_id"])
        experience_level = None
        level_matches = re.findall(r"\b(beginner|intermediate|advanced)\b(?:[- ]to[- ](beginner|intermediate|advanced))?", text)
        if level_matches:
            ranks={"beginner":0,"intermediate":1,"advanced":2}
            levels=[m[1] or m[0] for m in level_matches]
            experience_level=max(levels,key=lambda x:ranks[x])
        if any(x in text for x in ("hands-on", "hands on", "practical", "learning by doing", "project-based", "project based", "learn by building")):
            learning_style = "practice"
        else:
            learning_style = next((style for style in ("visual", "reading", "practice") if style in text), None)
        weekly_hours = None
        m = re.search(r"(\d+(?:\.\d+)?)\s*(?:hours?|hrs?)\s*(?:per|a|each)?\s*(?:day|daily)", text)
        if m:
            weekly_hours = float(m.group(1)) * 7
        else:
            m = re.search(r"(\d+(?:\.\d+)?)\s*(?:hours?|hrs?)\s*(?:per|a|each)?\s*(?:week|weekly)", text)
            if m: weekly_hours = float(m.group(1))
        interests=[]
        aliases={"generative ai":"generative_ai", "genai":"generative_ai", "llm":"llms", "llms":"llms", "language models":"llms", "computer vision":"computer_vision", "nlp":"nlp", "natural language processing":"nlp", "mlops":"mlops"}
        for phrase, canonical in aliases.items():
            if phrase in text and canonical not in interests: interests.append(canonical)
        prefs={}
        if any(x in text for x in ("more practical", "more project", "more projects", "more hands-on", "more hands on")):
            prefs["more_projects"] = True; learning_style = "practice"
        if any(x in text for x in ("more ai", "more artificial intelligence", "more genai", "more llm")): prefs["more_ai"] = True
        if any(x in text for x in ("less cloud", "less aws")): prefs["less_cloud"] = True
        if "too difficult" in text or "too hard" in text: prefs["slower_pace"] = True
        if "too slow" in text or "faster path" in text: prefs["faster_pace"] = True
        noted=[]
        if goal_id: noted.append("goal: "+next(g["title"] for g in candidate_goals if g["goal_id"]==goal_id))
        if mastered: noted.append(f"{len(mastered)} mastered skill(s)")
        if unmastered: noted.append(f"{len(unmastered)} skill gap(s)")
        if experience_level: noted.append(f"level: {experience_level}")
        if learning_style: noted.append(f"style: {learning_style}")
        if weekly_hours is not None: noted.append(f"time: {weekly_hours:g} hours/week")
        if interests: noted.append("interest: "+", ".join(interests))
        if prefs: noted.append("roadmap preferences updated")
        if "ias officer" in text or "civil services" in text:
            intent="unsupported_goal"
            reply="IAS Officer is not a supported career path in the current catalog, so I did not change your existing roadmap. Supported paths include AI engineer, ML engineer, Data scientist, Backend engineer, and Frontend engineer."
        elif not noted and ("don't like" in text or "do not like" in text or "dislike" in text):
            intent="roadmap_feedback"
            reply="I can change the roadmap. What should I change: more practical projects, more AI/GenAI, less cloud, easier pacing, or faster pacing?"
        elif "supported" in text and "paths" in text:
            intent="roadmap_question"
            reply="Available paths: " + ", ".join(g["title"] for g in candidate_goals) + "."
        elif not noted and any(x in text for x in ("roadmap", "path", "course", "why")):
            intent="roadmap_question"
            reply="I can help with your learning path. Tell me whether you want to change the goal, skills, learning style, study time, or roadmap preferences."
        else:
            intent="profile_update"
            reply="Got it - "+", ".join(noted)+". Anything else?" if noted else "Tell me your target role, skills, interests, learning style, and available study time."
        return _validate_profile_result({
            "reply":reply,"intent":intent,"goal_id":goal_id,"skill_ids":mastered,"unmastered_skill_ids":unmastered,
            "experience_level":experience_level,"learning_style":learning_style,"weekly_hours":weekly_hours,
            "interests":interests,"roadmap_preferences":prefs,
        },candidate_goals,candidate_skills)

    def explain(self, course_title: str, goal_title: str, attributions: dict) -> str:
        from src.recommender.llm.prompt_templates import template_explanation
        return template_explanation(course_title, attributions)


def get_llm_client() -> LLMClient:
    provider = os.getenv("LLM_PROVIDER", "groq").lower()
    logger = logging.getLogger(__name__)
    try:
        if provider == "groq" and os.getenv("GROQ_API_KEY"):
            return GroqClient()
        if provider == "openai" and os.getenv("OPENAI_API_KEY"):
            return OpenAIClient()
    except ImportError as e:
        logger.warning(f"{provider} SDK not installed ({e}) - using LocalStubLLMClient")
        return LocalStubLLMClient()
    if provider in ("groq", "openai"):
        logger.info(f"No {provider.upper()}_API_KEY set - using LocalStubLLMClient")
    return LocalStubLLMClient()
