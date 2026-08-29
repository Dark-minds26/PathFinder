"""Local/demo JSON profile store. Not a production database."""
import json
import sys
from pathlib import Path
from threading import Lock
from src.recommender.exception import RecommenderException
from src.recommender.entity.domain_models import Mastery

REQUIRED_FIELDS = ("goal_id", "experience_level", "learning_style", "weekly_hours")

class ProfileStore:
    def __init__(self, path: str) -> None:
        self.path = Path(path); self.path.parent.mkdir(parents=True, exist_ok=True); self._lock = Lock()
        if not self.path.exists(): self.path.write_text("{}")
    def _read_all(self):
        try:
            raw=self.path.read_text().strip(); return json.loads(raw) if raw else {}
        except json.JSONDecodeError: return {}
    def _write_all(self, data): self.path.write_text(json.dumps(data, indent=2, sort_keys=True))
    def get(self, user_id: str) -> dict:
        profile = self._read_all().get(user_id, {"skill_ids": []})
        self._ensure_mastery_state(profile)
        return profile

    @staticmethod
    def _ensure_mastery_state(profile: dict) -> None:
        """Backfill the canonical mastery-state map from legacy fields."""
        has_legacy_mastery = bool(profile.get("mastery") or profile.get("skill_ids") or profile.get("unmastered_skill_ids") or profile.get("mastery_state"))
        if not has_legacy_mastery:
            return
        states = profile.setdefault("mastery_state", {})
        mastery = profile.get("mastery", {})
        sources = profile.get("mastery_source", {})
        for sid, raw_score in mastery.items():
            if sid in states:
                continue
            score = max(0.0, min(1.0, float(raw_score)))
            status = "validated" if score >= 0.80 else ("needs_review" if score >= 0.60 else "failed")
            states[sid] = Mastery(skill_id=sid, score=score, status=status, source=sources.get(sid, "legacy")).model_dump(mode="json")
        for sid in profile.get("skill_ids", []):
            if sid not in states:
                states[sid] = Mastery(skill_id=sid, score=0.80, status="validated", source="self_report").model_dump(mode="json")
                profile.setdefault("mastery", {})[sid] = 0.80
                profile.setdefault("mastery_source", {})[sid] = "self_report"
        for sid in profile.get("unmastered_skill_ids", []):
            if sid not in states:
                states[sid] = Mastery(skill_id=sid, score=0.0, status="needs_review", source="legacy").model_dump(mode="json")
    def update(self, user_id: str, *, goal_id=None, experience_level=None, learning_style=None, weekly_hours=None,
               new_skill_ids=None, unmastered_skill_ids=None, interests=None, roadmap_preferences=None, goal_spec=None) -> dict:
        try:
            with self._lock:
                data=self._read_all(); profile=data.get(user_id, {"skill_ids": []})
                if goal_id: profile["goal_id"]=goal_id
                if goal_spec: profile["goal_spec"]=goal_spec
                if experience_level: profile["experience_level"]=experience_level
                if learning_style: profile["learning_style"]=learning_style
                if weekly_hours is not None:
                    hours=float(weekly_hours)
                    if hours<=0 or hours>168: raise ValueError("weekly_hours must be greater than 0 and no more than 168")
                    profile["weekly_hours"]=hours
                if interests:
                    profile["interests"]=sorted(set(profile.get("interests", [])) | set(interests))
                if roadmap_preferences:
                    profile["roadmap_preferences"]={**profile.get("roadmap_preferences", {}), **{k: bool(v) for k,v in roadmap_preferences.items() if v}}
                if new_skill_ids:
                    ids=set(new_skill_ids)
                    profile["skill_ids"]=sorted(set(profile.get("skill_ids", [])) | ids)
                    mastery=profile.setdefault("mastery", {}); sources=profile.setdefault("mastery_source", {})
                    for sid in ids:
                        # A user-declared known skill is accepted as mastered for
                        # serving compatibility; assessment evidence can move it
                        # to needs_review/failed later.
                        mastery.setdefault(sid, 0.80)
                        sources.setdefault(sid, "self_report")
                if unmastered_skill_ids:
                    profile["skill_ids"]=sorted(set(profile.get("skill_ids", [])) - set(unmastered_skill_ids))
                    profile["unmastered_skill_ids"]=sorted(set(profile.get("unmastered_skill_ids", [])) | set(unmastered_skill_ids))
                self._ensure_mastery_state(profile)
                data[user_id]=profile; self._write_all(data); return profile
        except Exception as e: raise RecommenderException(e, sys) from e
    def complete_course(self,user_id,course_id,skill_id=None):
        with self._lock:
            data=self._read_all(); p=data.setdefault(user_id,{"skill_ids":[]})
            p["completed_course_ids"]=sorted(set(p.get("completed_course_ids",[]))|{course_id})
            p.setdefault("learning_history",[]).append({"type":"course_completed","course_id":course_id,"skill_id":skill_id})
            data[user_id]=p; self._write_all(data); return p

    def set_mastery(self, user_id: str, skill_id: str, mastery: float, source: str = "assessment") -> dict:
        try:
            with self._lock:
                data=self._read_all(); profile=data.get(user_id,{"skill_ids":[]})
                score = round(max(0.0,min(1.0,float(mastery))),4)
                status = "validated" if score >= 0.80 else ("needs_review" if score >= 0.60 else "failed")
                profile.setdefault("mastery", {})[skill_id]=score
                profile.setdefault("mastery_state", {})[skill_id]=Mastery(skill_id=skill_id, score=score, status=status, source=source).model_dump(mode="json")
                profile.setdefault("mastery_source", {})[skill_id]=source
                if status == "validated":
                    profile["skill_ids"]=sorted(set(profile.get("skill_ids",[]))|{skill_id})
                    profile["unmastered_skill_ids"]=[x for x in profile.get("unmastered_skill_ids",[]) if x!=skill_id]
                else:
                    # Any non-validated assessment state must reopen the skill
                    # for the adaptive engine. This keeps legacy skill_ids in
                    # sync with the canonical mastery_state contract.
                    profile["skill_ids"]=[x for x in profile.get("skill_ids",[]) if x!=skill_id]
                    profile["unmastered_skill_ids"]=sorted(set(profile.get("unmastered_skill_ids",[]))|{skill_id})
                data[user_id]=profile; self._write_all(data); return profile
        except Exception as e: raise RecommenderException(e,sys) from e

    def record_history(self, user_id: str, item: dict) -> dict:
        try:
            with self._lock:
                data=self._read_all(); p=data.setdefault(user_id,{"skill_ids":[]})
                p.setdefault("learning_history",[]).append(item); data[user_id]=p; self._write_all(data); return p
        except Exception as e: raise RecommenderException(e,sys) from e

    def completeness(self, user_id: str) -> float:
        p=self.get(user_id); checks=[bool(p.get("goal_id")), bool(p.get("experience_level")), bool(p.get("learning_style")), p.get("weekly_hours") is not None, bool(p.get("skill_ids") or p.get("unmastered_skill_ids"))]
        return round(sum(checks)/5,4)
    def mark_unmastered(self,user_id,skill_id):
        try:
            with self._lock:
                data=self._read_all(); p=data.get(user_id,{"skill_ids":[]})
                p["skill_ids"]=[s for s in p.get("skill_ids",[]) if s!=skill_id]
                p["unmastered_skill_ids"]=sorted(set(p.get("unmastered_skill_ids",[]))|{skill_id})
                data[user_id]=p; self._write_all(data); return p
        except Exception as e: raise RecommenderException(e,sys) from e
