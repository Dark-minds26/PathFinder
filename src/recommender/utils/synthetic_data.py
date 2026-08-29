"""Deterministic synthetic catalog and learner-history generator used by the demo."""
import csv
import os
import random

SKILLS: dict[str, tuple[str, str, bool, list[str]]] = {
    "python_basics": ("Python basics", "data", True, []),
    "git_basics": ("Git basics", "shared", True, []),
    "sql_basics": ("SQL basics", "data", True, []),
    "html_css_basics": ("HTML/CSS basics", "frontend", True, []),
    "docker_basics": ("Docker basics", "backend", True, []),
    "python_advanced": ("Python advanced", "data", False, ["python_basics"]),
    "statistics": ("Statistics", "data", False, ["python_basics"]),
    "rest_apis": ("REST APIs", "backend", False, ["python_basics"]),
    "databases_basics": ("Databases basics", "backend", False, ["sql_basics"]),
    "javascript_basics": ("JavaScript basics", "frontend", False, ["html_css_basics"]),
    "kubernetes_basics": ("Kubernetes basics", "backend", False, ["docker_basics"]),
    "cloud_aws_basics": ("Cloud (AWS) basics", "backend", False, ["docker_basics"]),
    "data_wrangling": ("Data wrangling", "data", False, ["python_basics", "statistics"]),
    "typescript_basics": ("TypeScript basics", "frontend", False, ["javascript_basics"]),
    "react_basics": ("React basics", "frontend", False, ["javascript_basics"]),
    "system_design": ("System design", "backend", False, ["databases_basics", "rest_apis"]),
    "backend_frameworks": ("Backend frameworks", "backend", False, ["rest_apis", "databases_basics"]),
    "data_visualization": ("Data visualization", "data", False, ["data_wrangling"]),
    "machine_learning": ("Machine learning", "data", False, ["statistics", "data_wrangling"]),
    "state_management": ("State management", "frontend", False, ["react_basics"]),
    "frontend_testing": ("Frontend testing", "frontend", False, ["react_basics"]),
    "web_performance": ("Web performance", "frontend", False, ["react_basics"]),
    "deep_learning": ("Deep learning", "data", False, ["machine_learning"]),
    "mlops_basics": ("MLOps basics", "data", False, ["machine_learning", "docker_basics"]),
    "pytorch": ("PyTorch", "ai", False, ["deep_learning"]),
    "llm_applications": ("LLM applications", "ai", False, ["deep_learning", "python_advanced"]),
    "rag_systems": ("RAG systems", "ai", False, ["llm_applications"]),
    "model_serving": ("Model serving", "ai", False, ["mlops_basics", "rest_apis"]),
}

CAREER_GOALS: dict[str, tuple[str, list[str]]] = {
    "goal_data_scientist": ("Data scientist", ["statistics", "data_wrangling", "data_visualization", "machine_learning"]),
    "goal_ml_engineer": ("ML engineer", ["machine_learning", "docker_basics", "mlops_basics", "cloud_aws_basics"]),
    "goal_ai_engineer": ("AI engineer", ["machine_learning", "deep_learning", "pytorch", "llm_applications", "rag_systems", "model_serving", "mlops_basics"]),
    "goal_backend_eng": ("Backend engineer", ["rest_apis", "databases_basics", "backend_frameworks", "system_design"]),
    "goal_frontend_eng": ("Frontend engineer", ["react_basics", "state_management", "frontend_testing"]),
}

DIFFICULTIES = ["beginner", "intermediate", "advanced"]
FORMATS = ["video", "text", "interactive"]
EXPERIENCE_LEVELS = ["beginner", "intermediate", "advanced"]
LEARNING_STYLES = ["visual", "reading", "practice"]
INTERESTS = ["generative_ai", "llms", "computer_vision", "nlp", "mlops"]
GOAL_INTERESTS = {
    "goal_data_scientist": ["nlp"],
    "goal_ml_engineer": ["mlops"],
    "goal_ai_engineer": ["generative_ai", "llms"],
    "goal_backend_eng": ["mlops"],
    "goal_frontend_eng": [],
}


def _build_prereq_graph():
    import networkx as nx
    g = nx.DiGraph()
    for skill_id, (_name, _cat, _found, prereqs) in SKILLS.items():
        g.add_node(skill_id)
        for p in prereqs:
            g.add_edge(p, skill_id)
    assert nx.is_directed_acyclic_graph(g), "SKILLS taxonomy has a cycle"
    return g


def _make_courses(rng: random.Random) -> list[dict]:
    courses, idx = [], 0
    suffixes = ["crash course", "deep dive", "guided project", "practice lab", "interview workshop"]
    for skill_id, (skill_name, *_rest) in SKILLS.items():
        for suffix in suffixes:
            idx += 1
            courses.append({
                "course_id": f"course_{idx:03d}",
                "title": f"{skill_name} {suffix}",
                "difficulty": rng.choice(DIFFICULTIES),
                "duration_hours": rng.randint(2, 40),
                "format": rng.choice(FORMATS),
                "primary_skill_id": skill_id,
            })
    return courses


def _make_users(rng: random.Random, n: int) -> list[dict]:
    goal_ids = list(CAREER_GOALS.keys())
    users = []
    for i in range(1, n + 1):
        goal = rng.choice(goal_ids)
        interests = list(GOAL_INTERESTS[goal])
        if rng.random() < 0.25:
            extra = rng.choice([x for x in INTERESTS if x not in interests])
            interests.append(extra)
        users.append({
            "user_id": f"user_{i:04d}",
            "career_goal_id": goal,
            "experience_level": rng.choice(EXPERIENCE_LEVELS),
            "learning_style": rng.choice(LEARNING_STYLES),
            "weekly_hours": rng.randint(4, 20),
            "interests": "|".join(interests),
        })
    return users


def _simulate_events(rng: random.Random, users: list[dict], courses: list[dict], graph) -> list[dict]:
    import networkx as nx
    courses_by_skill = {}
    for c in courses:
        courses_by_skill.setdefault(c["primary_skill_id"], []).append(c)
    events, idx = [], 0
    for user in users:
        goal_skills = CAREER_GOALS[user["career_goal_id"]][1]
        needed = set(goal_skills)
        for s in goal_skills:
            needed |= nx.ancestors(graph, s)
        order = [n for n in nx.topological_sort(graph) if n in needed]
        day = rng.randint(0, 60)
        for skill_id in order:
            if rng.random() > 0.85:
                break
            course = rng.choice(courses_by_skill[skill_id])
            day += rng.randint(1, 14)
            if rng.random() < 0.15:
                completion_pct, event_type, score = rng.uniform(5, 40), "dropped", ""
            else:
                completion_pct, event_type = rng.uniform(85, 100), "completed"
                score = round(rng.uniform(35, 100), 1)
            idx += 1
            events.append({
                "event_id": f"evt_{idx:06d}", "user_id": user["user_id"], "course_id": course["course_id"],
                "skill_id": skill_id, "event_type": event_type, "completion_pct": round(completion_pct, 1),
                "score": score, "occurred_at": f"2026-{1 + (day // 30) % 12:02d}-{1 + day % 28:02d}",
            })
    return events


def _write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader(); writer.writerows(rows)


def generate_seed_data(output_dir: str, num_users: int = 300, seed: int = 42) -> dict:
    rng = random.Random(seed)
    os.makedirs(output_dir, exist_ok=True)
    graph = _build_prereq_graph()
    skills_rows = [{"skill_id": sid, "skill_name": name, "category": cat, "is_foundational": int(found)} for sid, (name, cat, found, _p) in SKILLS.items()]
    prereq_rows = [{"skill_id": sid, "prerequisite_skill_id": p} for sid, (_n, _c, _f, ps) in SKILLS.items() for p in ps]
    goal_rows = [{"goal_id": gid, "title": title} for gid, (title, _s) in CAREER_GOALS.items()]
    goal_skill_rows = [{"goal_id": gid, "skill_id": sid, "importance_weight": 1.0} for gid, (_t, ss) in CAREER_GOALS.items() for sid in ss]
    courses = _make_courses(rng)
    course_rows = [{k: v for k, v in c.items() if k != "primary_skill_id"} for c in courses]
    course_skill_rows = [{"course_id": c["course_id"], "skill_id": c["primary_skill_id"], "skill_weight": 1.0} for c in courses]
    users = _make_users(rng, num_users)
    events = _simulate_events(rng, users, courses, graph)
    tables = {"skills": skills_rows, "skill_prerequisites": prereq_rows, "career_goals": goal_rows,
              "bridge_career_goal_skill": goal_skill_rows, "courses": course_rows,
              "bridge_course_skill": course_skill_rows, "users": users, "learning_events": events}
    written = {}
    for name, rows in tables.items():
        path = os.path.join(output_dir, f"{name}.csv"); _write_csv(path, rows); written[name] = path
    return written

if __name__ == "__main__":
    for name, path in generate_seed_data("data/seed").items():
        print(f"{name}: {path}")
