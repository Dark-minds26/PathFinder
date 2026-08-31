"""Conversation orchestration: grounded extraction + persistent profile state."""

import sys
from src.recommender.exception import RecommenderException
from src.recommender.goal_intelligence import normalize_goal

RECOMMENDATION_FIELDS = (
    "goal_id",
    "experience_level",
    "learning_style",
    "weekly_hours",
    "interests",
    "roadmap_preferences",
    "skill_ids",
    "unmastered_skill_ids",
    "goal_spec",
)


class ConversationManager:
    def __init__(self, user_id, rag, llm, store):
        self.user_id = user_id
        self.rag = rag
        self.llm = llm
        self.store = store
        self.history = self.store.get_chat_history(user_id)

    def handle_turn(self, message, return_meta=False):
        try:
            import re

            current = self.store.get(self.user_id)
            curated = getattr(self.rag, "_goals", [])
            dynamic = normalize_goal(message, curated, getattr(self.rag, "_skills", []))
            goals = self.rag.retrieve_goals(message, top_k=5)
            retrieved_skills = self.rag.retrieve_skills(message, top_k=12)
            text = message.lower()

            def _has_word(phrase, haystack):
                return (
                    re.search(
                        r"\b" + re.escape(phrase.lower()) + r"\b",
                        haystack,
                    )
                    is not None
                )

            exact = []
            for x in getattr(self.rag, "_skills", []):
                skill_name = x["skill_name"].lower().strip()
                # Match only the complete canonical skill name
                if _has_word(skill_name, text):
                    exact.append(x)
            # Keep RAG-ranked skills first, but expose the full canonical catalog to the
            # LLM as a safe lookup so it cannot fail simply because a valid skill was
            # mentioned outside the top-K retrieval results. The model is still instructed
            # to prefer the retrieved candidates.
            all_skills = getattr(self.rag, "_skills", [])
            skills = list(
                {
                    x["skill_id"]: x for x in retrieved_skills + exact + all_skills
                }.values()
            )
            goals, skills = self.rag.ground_profile(current, goals, skills)
            if dynamic and dynamic.source == "dynamic":
                result = {
                    "reply": f"I understand this as a {dynamic.title} goal. Pathfinder can decompose it, but the current catalog has limited curated resources for this domain. I’ll keep the goal and your profile so the path can adapt as resources become available.",
                    "goal_id": dynamic.goal_id,
                    "skill_ids": [],
                    "unmastered_skill_ids": [],
                    "experience_level": None,
                    "learning_style": None,
                    "weekly_hours": None,
                    "interests": [],
                    "roadmap_preferences": {},
                }
            else:
                result = self.llm.profile_turn(
                    history=self.history,
                    user_message=message,
                    candidate_goals=goals,
                    candidate_skills=skills,
                    current_profile=current,
                )
                # Deterministic signals are used as a consistency layer around the LLM:
                # explicit canonical goals, day-vs-week study time, and natural-language
                # learning style should never be lost because the model phrased them differently.
                if dynamic and dynamic.source == "curated":
                    result["goal_id"] = dynamic.goal_id
                import re

                if re.search(
                    r"\b(hours?|hrs?)\s*(?:per|a|each)?\s*(?:day|daily)\b", text
                ):
                    m = re.search(
                        r"(\d+(?:\.\d+)?)\s*(?:hours?|hrs?)\s*(?:per|a|each)?\s*(?:day|daily)\b",
                        text,
                    )
                    if m:
                        result["weekly_hours"] = float(m.group(1)) * 7
                elif re.search(
                    r"\b(hours?|hrs?)\s*(?:per|a|each)?\s*(?:week|weekly)\b", text
                ):
                    m = re.search(
                        r"(\d+(?:\.\d+)?)\s*(?:hours?|hrs?)\s*(?:per|a|each)?\s*(?:week|weekly)\b",
                        text,
                    )
                    if m:
                        result["weekly_hours"] = float(m.group(1))
                if any(
                    x in text
                    for x in (
                        "hands-on",
                        "hands on",
                        "practical",
                        "learning by doing",
                        "project-based",
                        "project based",
                        "learn by building",
                    )
                ):
                    result["learning_style"] = "practice"
                # Exact skill mentions are canonical and safe to merge into the LLM result.
                if exact:
                    exact_skill_ids = [
                        x["skill_id"]
                        for x in exact
                        if x["skill_id"]
                        not in set(result.get("unmastered_skill_ids", []))
                    ]
                    result["skill_ids"] = list(dict.fromkeys(exact_skill_ids))

            self.history += [
                {"role": "user", "content": message},
                {"role": "assistant", "content": result["reply"]},
            ]
            self.store.save_chat_history(self.user_id, self.history)
            goal_spec = None
            if dynamic and dynamic.source == "dynamic":
                goal_spec = {
                    "title": dynamic.title,
                    "domain": dynamic.domain,
                    "competencies": dynamic.competencies,
                    "source": dynamic.source,
                    "confidence": dynamic.confidence,
                    "required_skill_ids": dynamic.required_skill_ids,
                    "resource_available": dynamic.resource_available,
                }
            updated = self.store.update(
                self.user_id,
                goal_id=result.get("goal_id"),
                experience_level=result.get("experience_level"),
                learning_style=result.get("learning_style"),
                weekly_hours=result.get("weekly_hours"),
                new_skill_ids=result.get("skill_ids") or [],
                unmastered_skill_ids=result.get("unmastered_skill_ids") or [],
                interests=result.get("interests") or [],
                roadmap_preferences=result.get("roadmap_preferences") or {},
                goal_spec=goal_spec,
            )
            changed = any(
                updated.get(k) != current.get(k) for k in RECOMMENDATION_FIELDS
            )
            payload = {
                "profile": updated,
                "recommendation_changed": changed,
                "extracted": result,
            }
            return (
                (result["reply"], self.store.completeness(self.user_id), payload)
                if return_meta
                else (result["reply"], self.store.completeness(self.user_id))
            )
        except Exception as e:
            raise RecommenderException(e, sys) from e
