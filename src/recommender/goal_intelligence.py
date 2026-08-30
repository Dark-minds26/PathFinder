"""Domain-agnostic goal understanding with grounded dynamic fallback."""

from __future__ import annotations

import re
from dataclasses import dataclass, field


DYNAMIC_BLUEPRINTS = {
    "cybersecurity": [
        "security fundamentals",
        "network security",
        "identity and access management",
        "security monitoring",
        "incident response",
    ],
    "product manager": [
        "product discovery",
        "user research",
        "roadmapping",
        "product analytics",
        "stakeholder management",
    ],
    "ui ux": [
        "user research",
        "interaction design",
        "visual design",
        "prototyping",
        "usability testing",
    ],
    "financial analyst": [
        "financial statements",
        "financial modeling",
        "data analysis",
        "valuation",
        "business communication",
    ],
    "ias": [
        "general studies",
        "current affairs",
        "governance",
        "economy",
        "answer writing",
    ],
    "marketing": [
        "market research",
        "content strategy",
        "analytics",
        "campaign planning",
        "customer segmentation",
    ],
}


@dataclass
class GoalSpec:
    goal_id: str
    title: str
    domain: str
    competencies: list[str] = field(default_factory=list)
    source: str = "curated"
    confidence: float = 1.0
    required_skill_ids: list[str] = field(default_factory=list)
    resource_available: bool = True


def _normalize_text(text: str) -> str:
    """
    Normalize different user spellings into comparable text.

    Examples:
    MLOps Engineer -> mlops engineer
    ML-Ops Engineer -> ml ops engineer
    AI/ML Engineer -> ai ml engineer
    """

    text = text.lower().strip()

    text = re.sub(r"[^a-z0-9]+", " ", text)

    return re.sub(r"\s+", " ", text).strip()


def _find_curated_goal(
    curated: list[dict],
    possible_titles: list[str],
):
    """
    Find a curated goal using normalized title matching.
    """

    normalized_targets = {
        _normalize_text(title)
        for title in possible_titles
    }

    for goal in curated:
        normalized_goal = _normalize_text(goal["title"])

        if normalized_goal in normalized_targets:
            return goal

    return None


def _goal_from_curated(
    goal: dict,
    confidence: float = 1.0,
) -> GoalSpec:

    return GoalSpec(
        goal_id=goal["goal_id"],
        title=goal["title"],
        domain=_normalize_text(goal["title"]),
        source="curated",
        confidence=confidence,
    )


# ============================================================
# COMMON GOAL ALIASES
#
# Left side  = what users might type
# Right side = possible titles in your curated catalog
# ============================================================

GOAL_ALIASES = {

    # --------------------------------------------------------
    # MLOPS
    # --------------------------------------------------------

    "mlops": [
        "MLOps Engineer",
        "ML Ops Engineer",
        "Machine Learning Operations Engineer",
        "ML Platform Engineer",
    ],

    "mlops engineer": [
        "MLOps Engineer",
        "ML Ops Engineer",
        "Machine Learning Operations Engineer",
    ],

    "ml ops": [
        "MLOps Engineer",
        "ML Ops Engineer",
        "Machine Learning Operations Engineer",
    ],

    "ml ops engineer": [
        "MLOps Engineer",
        "ML Ops Engineer",
        "Machine Learning Operations Engineer",
    ],

    "machine learning operations": [
        "MLOps Engineer",
        "Machine Learning Operations Engineer",
    ],

    "machine learning operations engineer": [
        "MLOps Engineer",
        "Machine Learning Operations Engineer",
    ],

    "ml platform engineer": [
        "ML Platform Engineer",
        "MLOps Engineer",
    ],


    # --------------------------------------------------------
    # MACHINE LEARNING
    # --------------------------------------------------------

    "machine learning engineer": [
        "ML Engineer",
        "Machine Learning Engineer",
    ],

    "ml engineer": [
        "ML Engineer",
        "Machine Learning Engineer",
    ],

    "mle": [
        "ML Engineer",
        "Machine Learning Engineer",
    ],

    "machine learning developer": [
        "ML Engineer",
        "Machine Learning Engineer",
    ],

    "ml developer": [
        "ML Engineer",
        "Machine Learning Engineer",
    ],


    # --------------------------------------------------------
    # AI ENGINEERING
    # --------------------------------------------------------

    "ai engineer": [
        "AI Engineer",
        "Artificial Intelligence Engineer",
    ],

    "artificial intelligence engineer": [
        "AI Engineer",
        "Artificial Intelligence Engineer",
    ],

    "ai developer": [
        "AI Engineer",
        "Artificial Intelligence Engineer",
    ],

    "artificial intelligence developer": [
        "AI Engineer",
        "Artificial Intelligence Engineer",
    ],


    # --------------------------------------------------------
    # DATA SCIENCE
    # --------------------------------------------------------

    "data scientist": [
        "Data Scientist",
        "Data science",
    ],

    "data science": [
        "Data Scientist",
        "Data science",
    ],

    "data science engineer": [
        "Data Scientist",
    ],

    "data analyst": [
        "Data Analyst",
        "Data Analytics",
    ],

    "data analytics": [
        "Data Analyst",
        "Data Analytics",
    ],

    "business analyst": [
        "Business Analyst",
    ],


    # --------------------------------------------------------
    # DATA ENGINEERING
    # --------------------------------------------------------

    "data engineer": [
        "Data Engineer",
    ],

    "data engineering": [
        "Data Engineer",
    ],

    "etl engineer": [
        "Data Engineer",
    ],

    "etl developer": [
        "Data Engineer",
    ],

    "data pipeline engineer": [
        "Data Engineer",
    ],


    # --------------------------------------------------------
    # BACKEND
    # --------------------------------------------------------

    "backend engineer": [
        "Backend Engineer",
        "Backend Developer",
    ],

    "backend developer": [
        "Backend Engineer",
        "Backend Developer",
    ],

    "backend development": [
        "Backend Engineer",
        "Backend Developer",
    ],

    "server side developer": [
        "Backend Engineer",
        "Backend Developer",
    ],

    "api developer": [
        "Backend Engineer",
        "Backend Developer",
    ],


    # --------------------------------------------------------
    # FRONTEND
    # --------------------------------------------------------

    "frontend engineer": [
        "Frontend Engineer",
        "Frontend Developer",
    ],

    "frontend developer": [
        "Frontend Engineer",
        "Frontend Developer",
    ],

    "front end engineer": [
        "Frontend Engineer",
        "Frontend Developer",
    ],

    "front end developer": [
        "Frontend Engineer",
        "Frontend Developer",
    ],

    "web developer": [
        "Frontend Engineer",
        "Frontend Developer",
        "Web Developer",
    ],


    # --------------------------------------------------------
    # FULL STACK
    # --------------------------------------------------------

    "full stack engineer": [
        "Full Stack Engineer",
        "Full Stack Developer",
    ],

    "full stack developer": [
        "Full Stack Engineer",
        "Full Stack Developer",
    ],

    "fullstack engineer": [
        "Full Stack Engineer",
        "Full Stack Developer",
    ],

    "fullstack developer": [
        "Full Stack Engineer",
        "Full Stack Developer",
    ],

    "full stack": [
        "Full Stack Engineer",
        "Full Stack Developer",
    ],


    # --------------------------------------------------------
    # SOFTWARE ENGINEERING
    # --------------------------------------------------------

    "software engineer": [
        "Software Engineer",
    ],

    "software developer": [
        "Software Engineer",
    ],

    "software development": [
        "Software Engineer",
    ],

    "developer": [
        "Software Engineer",
    ],


    # --------------------------------------------------------
    # DEVOPS / CLOUD
    # --------------------------------------------------------

    "devops engineer": [
        "DevOps Engineer",
    ],

    "devops": [
        "DevOps Engineer",
    ],

    "cloud engineer": [
        "Cloud Engineer",
        "Cloud Computing Engineer",
    ],

    "cloud computing": [
        "Cloud Engineer",
        "Cloud Computing Engineer",
    ],

    "aws engineer": [
        "Cloud Engineer",
    ],

    "azure engineer": [
        "Cloud Engineer",
    ],

    "gcp engineer": [
        "Cloud Engineer",
    ],


    # --------------------------------------------------------
    # CYBERSECURITY
    # --------------------------------------------------------

    "cybersecurity engineer": [
        "Cybersecurity Engineer",
        "Cyber Security Engineer",
    ],

    "cyber security engineer": [
        "Cybersecurity Engineer",
        "Cyber Security Engineer",
    ],

    "security engineer": [
        "Cybersecurity Engineer",
        "Cyber Security Engineer",
    ],

    "cybersecurity": [
        "Cybersecurity Engineer",
        "Cyber Security Engineer",
    ],


    # --------------------------------------------------------
    # UI / UX
    # --------------------------------------------------------

    "ui ux designer": [
        "UI UX Designer",
        "UI/UX Designer",
        "UX Designer",
    ],

    "ui ux": [
        "UI UX Designer",
        "UI/UX Designer",
        "UX Designer",
    ],

    "ui designer": [
        "UI UX Designer",
        "UI/UX Designer",
    ],

    "ux designer": [
        "UI UX Designer",
        "UI/UX Designer",
        "UX Designer",
    ],

    "product designer": [
        "UI UX Designer",
        "UI/UX Designer",
        "Product Designer",
    ],


    # --------------------------------------------------------
    # PRODUCT MANAGEMENT
    # --------------------------------------------------------

    "product manager": [
        "Product Manager",
    ],

    "product management": [
        "Product Manager",
    ],

    "product owner": [
        "Product Manager",
        "Product Owner",
    ],


    # --------------------------------------------------------
    # AR / VR
    # --------------------------------------------------------

    "ar vr developer": [
        "AR VR Developer",
        "AR/VR Developer",
    ],

    "ar vr": [
        "AR VR Developer",
        "AR/VR Developer",
    ],

    "augmented reality developer": [
        "AR VR Developer",
        "AR/VR Developer",
    ],

    "virtual reality developer": [
        "AR VR Developer",
        "AR/VR Developer",
    ],

    "vr developer": [
        "AR VR Developer",
        "AR/VR Developer",
    ],

    "ar developer": [
        "AR VR Developer",
        "AR/VR Developer",
    ],


    # --------------------------------------------------------
    # AI SPECIALIZATIONS
    # --------------------------------------------------------

    "llm engineer": [
        "LLM Engineer",
        "AI Engineer",
    ],

    "large language model engineer": [
        "LLM Engineer",
        "AI Engineer",
    ],

    "generative ai engineer": [
        "Generative AI Engineer",
        "AI Engineer",
    ],

    "genai engineer": [
        "Generative AI Engineer",
        "AI Engineer",
    ],

    "rag engineer": [
        "RAG Engineer",
        "AI Engineer",
    ],

    "nlp engineer": [
        "NLP Engineer",
        "AI Engineer",
    ],

    "natural language processing engineer": [
        "NLP Engineer",
        "AI Engineer",
    ],

    "computer vision engineer": [
        "Computer Vision Engineer",
        "AI Engineer",
    ],

    "deep learning engineer": [
        "Deep Learning Engineer",
        "ML Engineer",
    ],
}


def normalize_goal(
    text: str,
    curated: list[dict],
    skills: list[dict] | None = None,
) -> GoalSpec | None:

    t = _normalize_text(text)

    # ========================================================
    # 1. EXACT / NORMALIZED CURATED GOAL MATCH
    # ========================================================

    for goal in curated:

        normalized_title = _normalize_text(goal["title"])

        if normalized_title in t:
            return _goal_from_curated(
                goal,
                confidence=1.0,
            )

    # ========================================================
    # 2. ALIAS MATCH
    #
    # Sort longest first so:
    #
    # "mlops engineer"
    #
    # is checked before:
    #
    # "mlops"
    # ========================================================

    for phrase in sorted(
        GOAL_ALIASES.keys(),
        key=len,
        reverse=True,
    ):

        normalized_phrase = _normalize_text(phrase)

        if normalized_phrase in t:

            goal = _find_curated_goal(
                curated,
                GOAL_ALIASES[phrase],
            )

            if goal:
                return _goal_from_curated(
                    goal,
                    confidence=0.95,
                )

    # ========================================================
    # 3. DYNAMIC FALLBACK
    # ========================================================

    for domain, competencies in DYNAMIC_BLUEPRINTS.items():

        normalized_domain = _normalize_text(domain)

        if normalized_domain in t:

            return GoalSpec(
                goal_id=(
                    "dynamic:"
                    + re.sub(
                        r"[^a-z0-9]+",
                        "_",
                        normalized_domain,
                    ).strip("_")
                ),
                title=domain.title(),
                domain=normalized_domain,
                competencies=competencies,
                source="dynamic",
                confidence=0.78,
                required_skill_ids=[],
                resource_available=False,
            )

    return None