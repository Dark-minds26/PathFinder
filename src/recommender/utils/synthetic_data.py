"""One-time seed data generator.

Produces the raw CSVs that DataIngestion treats as "the source" (standing
in for the Postgres tables in production). This is not a pipeline
component - it runs once to bootstrap a demo-able dataset, since a
competition entry starts with zero real historical learners.

Scoped to ~24 skills / 4 career tracks / ~300 synthetic users for a fast,
fully-verifiable run. Scale num_users up, or extend SKILLS/CAREER_GOALS,
for the actual submission.
"""
import csv
import os
import random

# skill_id -> (display name, category, is_foundational, [prerequisite_skill_ids])
SKILLS: dict[str, tuple[str, str, bool, list[str]]] = {
    "python_basics":      ("Python basics",       "data",     True,  []),
    "git_basics":         ("Git basics",           "shared",   True,  []),
    "sql_basics":         ("SQL basics",           "data",     True,  []),
    "html_css_basics":    ("HTML/CSS basics",      "frontend", True,  []),
    "docker_basics":      ("Docker basics",        "backend",  True,  []),

    "python_advanced":    ("Python advanced",      "data",     False, ["python_basics"]),
    "statistics":         ("Statistics",           "data",     False, ["python_basics"]),
    "rest_apis":          ("REST APIs",             "backend",  False, ["python_basics"]),
    "databases_basics":   ("Databases basics",      "backend",  False, ["sql_basics"]),
    "javascript_basics":  ("JavaScript basics",     "frontend", False, ["html_css_basics"]),
    "kubernetes_basics":  ("Kubernetes basics",     "backend",  False, ["docker_basics"]),
    "cloud_aws_basics":   ("Cloud (AWS) basics",    "backend",  False, ["docker_basics"]),

    "data_wrangling":     ("Data wrangling",        "data",     False, ["python_basics", "statistics"]),
    "typescript_basics":  ("TypeScript basics",     "frontend", False, ["javascript_basics"]),
    "react_basics":       ("React basics",          "frontend", False, ["javascript_basics"]),
    "system_design":      ("System design",         "backend",  False, ["databases_basics", "rest_apis"]),
    "backend_frameworks": ("Backend frameworks",    "backend",  False, ["rest_apis", "databases_basics"]),

    "data_visualization": ("Data visualization",    "data",     False, ["data_wrangling"]),
    "machine_learning":   ("Machine learning",      "data",     False, ["statistics", "data_wrangling"]),
    "state_management":   ("State management",      "frontend", False, ["react_basics"]),
    "frontend_testing":   ("Frontend testing",       "frontend", False, ["react_basics"]),
    "web_performance":    ("Web performance",        "frontend", False, ["react_basics"]),

    "deep_learning":      ("Deep learning",          "data",     False, ["machine_learning"]),
    "mlops_basics":       ("MLOps basics",           "data",     False, ["machine_learning", "docker_basics"]),
}

# goal_id -> (title, [required_skill_ids])
CAREER_GOALS: dict[str, tuple[str, list[str]]] = {
    "goal_data_scientist": ("Data scientist",   ["statistics", "data_wrangling", "data_visualization", "machine_learning"]),
    "goal_ml_engineer":    ("ML engineer",      ["machine_learning", "docker_basics", "mlops_basics", "cloud_aws_basics"]),
    "goal_backend_eng":    ("Backend engineer", ["rest_apis", "databases_basics", "backend_frameworks", "system_design"]),
    "goal_frontend_eng":   ("Frontend engineer",["react_basics", "state_management", "frontend_testing"]),
}

DIFFICULTIES = ["beginner", "intermediate", "advanced"]
FORMATS = ["video", "text", "interactive"]
EXPERIENCE_LEVELS = ["beginner", "intermediate", "advanced"]
LEARNING_STYLES = ["visual", "reading", "practice"]


def _build_prereq_graph():
    import networkx as nx

    g = nx.DiGraph()
    for skill_id, (_name, _cat, _found, prereqs) in SKILLS.items():
        g.add_node(skill_id)
        for p in prereqs:
            g.add_edge(p, skill_id)  # prerequisite -> dependent
    assert nx.is_directed_acyclic_graph(g), "SKILLS taxonomy has a cycle"
    return g


def _make_courses(rng: random.Random) -> list[dict]:
    courses = []
    idx = 0
    for skill_id, (skill_name, *_rest) in SKILLS.items():
        for variant in range(2):
            idx += 1
            suffix = "crash course" if variant == 0 else "deep dive"
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
    return [
        {
            "user_id": f"user_{i:04d}",
            "career_goal_id": rng.choice(goal_ids),
            "experience_level": rng.choice(EXPERIENCE_LEVELS),
            "learning_style": rng.choice(LEARNING_STYLES),
        }
        for i in range(1, n + 1)
    ]


def _simulate_events(rng: random.Random, users: list[dict], courses: list[dict], graph) -> list[dict]:
    import networkx as nx

    courses_by_skill: dict[str, list[dict]] = {}
    for c in courses:
        courses_by_skill.setdefault(c["primary_skill_id"], []).append(c)

    events = []
    idx = 0
    for user in users:
        goal_skills = CAREER_GOALS[user["career_goal_id"]][1]
        needed: set[str] = set()
        for s in goal_skills:
            needed.add(s)
            needed |= nx.ancestors(graph, s)
        order = [n for n in nx.topological_sort(graph) if n in needed]

        day = rng.randint(0, 60)
        for skill_id in order:
            if rng.random() > 0.85:  # hasn't reached this skill yet
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
                "event_id": f"evt_{idx:06d}",
                "user_id": user["user_id"],
                "course_id": course["course_id"],
                "skill_id": skill_id,
                "event_type": event_type,
                "completion_pct": round(completion_pct, 1),
                "score": score,
                "occurred_at": f"2026-{1 + (day // 30) % 12:02d}-{1 + day % 28:02d}",
            })
    return events


def _write_csv(path: str, rows: list[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def generate_seed_data(output_dir: str, num_users: int = 300, seed: int = 42) -> dict:
    rng = random.Random(seed)
    os.makedirs(output_dir, exist_ok=True)
    graph = _build_prereq_graph()

    skills_rows = [
        {"skill_id": sid, "skill_name": name, "category": cat, "is_foundational": int(found)}
        for sid, (name, cat, found, _p) in SKILLS.items()
    ]
    prereq_rows = [
        {"skill_id": sid, "prerequisite_skill_id": p}
        for sid, (_n, _c, _f, prereqs) in SKILLS.items()
        for p in prereqs
    ]
    goal_rows = [{"goal_id": gid, "title": title} for gid, (title, _s) in CAREER_GOALS.items()]
    goal_skill_rows = [
        {"goal_id": gid, "skill_id": sid, "importance_weight": 1.0}
        for gid, (_t, skills) in CAREER_GOALS.items()
        for sid in skills
    ]
    courses = _make_courses(rng)
    course_rows = [{k: v for k, v in c.items() if k != "primary_skill_id"} for c in courses]
    course_skill_rows = [
        {"course_id": c["course_id"], "skill_id": c["primary_skill_id"], "skill_weight": 1.0}
        for c in courses
    ]
    users = _make_users(rng, num_users)
    events = _simulate_events(rng, users, courses, graph)

    tables = {
        "skills": skills_rows,
        "skill_prerequisites": prereq_rows,
        "career_goals": goal_rows,
        "bridge_career_goal_skill": goal_skill_rows,
        "courses": course_rows,
        "bridge_course_skill": course_skill_rows,
        "users": users,
        "learning_events": events,
    }
    written = {}
    for name, rows in tables.items():
        path = os.path.join(output_dir, f"{name}.csv")
        _write_csv(path, rows)
        written[name] = path
    return written


if __name__ == "__main__":
    for name, path in generate_seed_data("data/seed").items():
        print(f"{name}: {path}")
