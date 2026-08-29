"""Domain-agnostic goal understanding with grounded dynamic fallback."""
from __future__ import annotations
import re
from dataclasses import dataclass, field

DYNAMIC_BLUEPRINTS={
 "cybersecurity": ["security fundamentals","network security","identity and access management","security monitoring","incident response"],
 "product manager": ["product discovery","user research","roadmapping","product analytics","stakeholder management"],
 "ui ux": ["user research","interaction design","visual design","prototyping","usability testing"],
 "financial analyst": ["financial statements","financial modeling","data analysis","valuation","business communication"],
 "ias": ["general studies","current affairs","governance","economy","answer writing"],
 "marketing": ["market research","content strategy","analytics","campaign planning","customer segmentation"],
}

@dataclass
class GoalSpec:
    goal_id:str
    title:str
    domain:str
    competencies:list[str]=field(default_factory=list)
    source:str="curated"
    confidence:float=1.0
    required_skill_ids:list[str]=field(default_factory=list)
    resource_available:bool=True

def normalize_goal(text:str, curated:list[dict], skills:list[dict]|None=None)->GoalSpec|None:
    t=text.lower().strip()
    for g in curated:
        title=g["title"]
        if title.lower() in t or re.search(r"\b"+re.escape(title.lower())+r"\b",t):
            return GoalSpec(g["goal_id"],title,title.lower(),source="curated",confidence=1.0)
    aliases={"machine learning engineer":"ML engineer","ai engineer":"AI engineer","data scientist":"Data scientist","backend engineer":"Backend engineer","frontend engineer":"Frontend engineer","software engineer":"Backend engineer"}
    for phrase,title in aliases.items():
        if phrase in t:
            g=next((x for x in curated if x["title"]==title),None)
            if g:return GoalSpec(g["goal_id"],g["title"],g["title"].lower(),source="curated",confidence=.95)
    for domain, comps in DYNAMIC_BLUEPRINTS.items():
        if domain in t or (domain=="ui ux" and ("ui/ux" in t or "ux designer" in t)):
            return GoalSpec("dynamic:"+re.sub(r"[^a-z0-9]+","_",domain).strip("_"),domain.title(),domain,comps,"dynamic",.78,[],False)
    return None
