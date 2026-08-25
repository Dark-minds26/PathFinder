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
        extracted = data.get("extracted", {})
        return {
            "reply": data.get("reply", "").strip() or "Got it - tell me more?",
            "goal_id": extracted.get("goal_id"),
            "skill_ids": extracted.get("skill_ids") or [],
            "experience_level": extracted.get("experience_level"),
            "learning_style": extracted.get("learning_style"),
        }
    except (json.JSONDecodeError, AttributeError) as e:
        logging.warning(f"LLM returned unparseable profiling JSON: {e}")
        return {
            "reply": "Sorry, could you say that again?",
            "goal_id": None, "skill_ids": [], "experience_level": None, "learning_style": None,
        }


class GroqClient(LLMClient):
    """Backed by Groq's OpenAI-compatible chat completions API -
    chosen as the default for demo-time latency (see Phase 1)."""

    def __init__(self, model: str = "llama-3.3-70b-versatile") -> None:
        import groq  # local import: only required when this backend is actually selected

        self.client = groq.Groq(api_key=os.environ["GROQ_API_KEY"])
        self.model = model

    def profile_turn(self, history, user_message, candidate_goals, candidate_skills) -> dict:
        try:
            from src.recommender.llm.prompt_templates import (
                SYSTEM_PROMPT_PROFILING,
                build_profiling_user_content,
            )

            messages = [{"role": "system", "content": SYSTEM_PROMPT_PROFILING}]
            messages.extend(history[-8:])
            messages.append({
                "role": "user",
                "content": build_profiling_user_content(user_message, candidate_goals, candidate_skills),
            })
            resp = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=0.3,
                response_format={"type": "json_object"},
            )
            return _parse_profile_json(resp.choices[0].message.content)
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

    def profile_turn(self, history, user_message, candidate_goals, candidate_skills) -> dict:
        try:
            from src.recommender.llm.prompt_templates import (
                SYSTEM_PROMPT_PROFILING,
                build_profiling_user_content,
            )

            messages = [{"role": "system", "content": SYSTEM_PROMPT_PROFILING}]
            messages.extend(history[-8:])
            messages.append({
                "role": "user",
                "content": build_profiling_user_content(user_message, candidate_goals, candidate_skills),
            })
            resp = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=0.3,
                response_format={"type": "json_object"},
            )
            return _parse_profile_json(resp.choices[0].message.content)
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
    name = skill_name.lower()
    if name in text:
        return True
    words = [w for w in name.replace("/", " ").split() if w not in _SKILL_NAME_STOPWORDS and len(w) > 2]
    return any(w in text for w in words)


class LocalStubLLMClient(LLMClient):
    """Deterministic stand-in for local dev/testing - NOT a language
    model. Matches the user's raw message against the candidate goals
    and skills it was actually given, by word overlap instead of
    understanding the text. Enough to prove the RAG -> extraction ->
    ProfileStore -> path-generation wiring is correct end to end;
    replace with GroqClient/OpenAIClient for real conversational
    quality."""

    def profile_turn(self, history, user_message, candidate_goals, candidate_skills) -> dict:
        text = user_message.lower()

        goal_id = next(
            (g["goal_id"] for g in candidate_goals if g["title"].lower() in text), None
        )
        skill_ids = [
            s["skill_id"] for s in candidate_skills
            if _skill_name_matches(s["skill_name"], text)
        ]
        experience_level = next(
            (lvl for lvl in ("beginner", "intermediate", "advanced") if lvl in text), None
        )
        learning_style = next(
            (s for s in ("visual", "reading", "practice") if s in text), None
        )

        noted = []
        if goal_id:
            title = next(g["title"] for g in candidate_goals if g["goal_id"] == goal_id)
            noted.append(f"goal: {title}")
        if skill_ids:
            noted.append(f"{len(skill_ids)} skill(s)")
        if experience_level:
            noted.append(f"level: {experience_level}")
        if learning_style:
            noted.append(f"style: {learning_style}")
        reply = (
            "Got it - " + ", ".join(noted) + ". Anything else?"
            if noted else
            "Tell me a bit about your current skills and the role you're aiming for."
        )
        return {
            "reply": reply, "goal_id": goal_id, "skill_ids": skill_ids,
            "experience_level": experience_level, "learning_style": learning_style,
        }

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
