"""Deterministic synthetic catalog and learner-history generator used by the demo."""

import csv
import os
import random

SKILLS: dict[str, tuple[str, str, bool, list[str]]] = {
    # ---------- FOUNDATIONAL (no prerequisites) ----------
    "python_basics": ("Python basics", "data", True, []),
    "git_basics": ("Git basics", "shared", True, []),
    "sql_basics": ("SQL basics", "data", True, []),
    "html_css_basics": ("HTML/CSS basics", "frontend", True, []),
    "docker_basics": ("Docker basics", "backend", True, []),
    "linux_basics": ("Linux basics", "shared", True, []),
    "networking_basics": ("Networking basics", "shared", True, []),
    "agile_basics": ("Agile basics", "shared", True, []),
    "excel_basics": ("Excel basics", "data", True, []),
    "r_basics": ("R basics", "data", True, []),
    "figma_basics": ("Figma basics", "design", True, []),
    "manual_testing": ("Manual testing", "qa", True, []),
    "security_fundamentals": ("Security fundamentals", "security", True, []),
    "product_discovery": ("Product discovery", "product", True, []),
    "stakeholder_management": ("Stakeholder management", "product", True, []),
    "ux_research": ("UX research", "design", True, []),
    "technical_writing": ("Technical writing", "writing", True, []),
    "it_support_fundamentals": ("IT support fundamentals", "sysadmin", True, []),
    "salesforce_basics": ("Salesforce basics", "crm", True, []),
    "erp_fundamentals": ("ERP fundamentals", "erp", True, []),
    "digital_marketing_fundamentals": ("Digital marketing fundamentals", "marketing", True, []),
    "jira_basics": ("Jira", "product", True, []),
    "miro_basics": ("Miro", "design", True, []),

    # ---------- DATA / ANALYTICS ----------
    "python_advanced": ("Python advanced", "data", False, ["python_basics"]),
    "pandas_numpy": ("Pandas & NumPy", "data", False, ["python_basics"]),
    "statistics": ("Statistics", "data", False, ["python_basics"]),
    "statistical_modeling_r": ("Statistical modeling (R)", "data", False, ["r_basics", "statistics"]),
    "data_wrangling": ("Data wrangling", "data", False, ["python_basics", "statistics"]),
    "power_bi": ("Power BI", "data", False, ["excel_basics", "sql_basics"]),
    "tableau": ("Tableau", "data", False, ["excel_basics", "sql_basics"]),

    # ---------- MACHINE LEARNING / AI ----------
    "machine_learning": ("Machine learning", "data", False, ["statistics", "data_wrangling"]),
    "deep_learning": ("Deep learning", "data", False, ["machine_learning"]),
    "mlops_basics": ("MLOps basics", "data", False, ["machine_learning", "docker_basics"]),
    "model_monitoring": ("Model monitoring & drift detection", "ai", False, ["mlops_basics"]),
    "pytorch": ("PyTorch", "ai", False, ["deep_learning"]),
    "llm_applications": ("LLM applications", "ai", False, ["deep_learning", "python_advanced"]),
    "vector_databases": ("Vector databases", "ai", False, ["llm_applications"]),
    "rag_systems": ("RAG systems", "ai", False, ["llm_applications", "vector_databases"]),
    "model_serving": ("Model serving", "ai", False, ["mlops_basics", "rest_apis"]),
    "prompt_engineering": ("Prompt engineering", "ai", False, ["llm_applications"]),

    # ---------- DATA ENGINEERING ----------
    "data_pipelines": ("Data pipelines", "data_eng", False, ["python_basics", "sql_basics"]),
    "spark_basics": ("Apache Spark", "data_eng", False, ["python_basics", "sql_basics"]),
    "data_warehousing": ("Data warehousing", "data_eng", False, ["data_pipelines"]),
    "etl_orchestration": ("ETL orchestration (Airflow)", "data_eng", False, ["data_pipelines"]),
    "dbt_basics": ("dbt basics", "data_eng", False, ["sql_basics"]),
    "data_governance": ("Data governance", "data", False, ["data_warehousing"]),

    # ---------- WEB: FRONTEND / BACKEND ----------
    "javascript_basics": ("JavaScript basics", "frontend", False, ["html_css_basics"]),
    "typescript_basics": ("TypeScript basics", "frontend", False, ["javascript_basics"]),
    "react_basics": ("React basics", "frontend", False, ["javascript_basics"]),
    "state_management": ("State management", "frontend", False, ["react_basics"]),
    "frontend_testing": ("Frontend testing", "frontend", False, ["react_basics"]),
    "web_performance": ("Web performance", "frontend", False, ["react_basics"]),
    "rest_apis": ("REST APIs", "backend", False, ["python_basics"]),
    "databases_basics": ("Databases basics", "backend", False, ["sql_basics"]),
    "backend_frameworks": ("Backend frameworks", "backend", False, ["rest_apis", "databases_basics"]),
    "system_design": ("System design", "backend", False, ["databases_basics", "rest_apis"]),

    # ---------- CLOUD / DEVOPS / INFRA ----------
    "kubernetes_basics": ("Kubernetes basics", "backend", False, ["docker_basics"]),
    "cloud_aws_basics": ("Cloud (AWS) basics", "backend", False, ["docker_basics"]),
    "gcp_basics": ("Google Cloud basics", "cloud", False, ["docker_basics"]),
    "azure_basics": ("Azure basics", "cloud", False, ["docker_basics"]),
    "ci_cd_basics": ("CI/CD basics", "devops", False, ["git_basics", "linux_basics"]),
    "jenkins_basics": ("Jenkins", "devops", False, ["ci_cd_basics"]),
    "ansible_basics": ("Ansible", "devops", False, ["linux_basics"]),
    "terraform_basics": ("Terraform basics", "cloud", False, ["cloud_aws_basics"]),
    "infrastructure_as_code": ("Infrastructure as code", "devops", False, ["cloud_aws_basics", "ci_cd_basics"]),
    "monitoring_observability": ("Monitoring & observability", "devops", False, ["kubernetes_basics"]),
    "prometheus_grafana": ("Prometheus & Grafana", "devops", False, ["kubernetes_basics"]),
    "cloud_architecture": ("Cloud architecture", "cloud", False, ["cloud_aws_basics", "system_design"]),
    "multi_cloud": ("Multi-cloud strategy", "cloud", False, ["cloud_architecture"]),
    "release_engineering": ("Release engineering", "devops", False, ["ci_cd_basics"]),
    "systems_administration": ("Systems administration", "sysadmin", False, ["linux_basics"]),
    "network_administration": ("Network administration", "network", False, ["networking_basics"]),

    # ---------- SECURITY ----------
    "network_security": ("Network security", "security", False, ["networking_basics", "security_fundamentals"]),
    "application_security": ("Application security", "security", False, ["security_fundamentals", "rest_apis"]),
    "identity_access_management": ("Identity & access management", "security", False, ["security_fundamentals"]),
    "penetration_testing": ("Penetration testing", "security", False, ["network_security"]),
    "burp_suite": ("Burp Suite", "security", False, ["penetration_testing"]),
    "siem_tools": ("SIEM tools (Splunk)", "security", False, ["network_security"]),

    # ---------- DATABASE ----------
    "database_administration": ("Database administration", "database", False, ["databases_basics"]),
    "database_tuning": ("Database performance tuning", "database", False, ["database_administration"]),

    # ---------- MOBILE ----------
    "kotlin_basics": ("Kotlin basics", "mobile", False, ["git_basics"]),
    "android_development": ("Android development", "mobile", False, ["kotlin_basics"]),
    "swift_basics": ("Swift basics", "mobile", False, ["git_basics"]),
    "ios_development": ("iOS development", "mobile", False, ["swift_basics"]),
    "mobile_ui_patterns": ("Mobile UI patterns", "mobile", False, ["html_css_basics"]),

    # ---------- QA ----------
    "test_automation": ("Test automation", "qa", False, ["python_basics"]),

    # ---------- PRODUCT / DESIGN ----------
    "product_roadmapping": ("Product roadmapping", "product", False, ["product_discovery", "agile_basics"]),
    "scrum_facilitation": ("Scrum facilitation", "product", False, ["agile_basics"]),
    "ui_design_principles": ("UI design principles", "design", False, ["figma_basics"]),
    "prototyping": ("Prototyping", "design", False, ["figma_basics"]),
    "design_systems": ("Design systems", "design", False, ["ui_design_principles"]),
    "sketch_basics": ("Sketch", "design", False, ["figma_basics"]),

    # ---------- GAME DEV / EMBEDDED / ROBOTICS / XR ----------
    "game_engine_basics": ("Game engine basics (Unity)", "game", False, ["git_basics"]),
    "gameplay_programming": ("Gameplay programming", "game", False, ["game_engine_basics"]),
    "graphics_programming": ("Graphics programming", "game", False, ["gameplay_programming"]),
    "embedded_c_basics": ("Embedded C basics", "embedded", False, ["git_basics"]),
    "rtos_basics": ("RTOS basics", "embedded", False, ["embedded_c_basics"]),
    "hardware_interfacing": ("Hardware interfacing", "embedded", False, ["embedded_c_basics"]),
    "robotics_fundamentals": ("Robotics fundamentals", "robotics", False, ["embedded_c_basics"]),
    "ros_basics": ("ROS basics", "robotics", False, ["robotics_fundamentals"]),
    "ar_vr_basics": ("AR/VR basics", "xr", False, ["game_engine_basics"]),

    # ---------- BLOCKCHAIN ----------
    "solidity_basics": ("Solidity basics", "blockchain", False, ["javascript_basics"]),
    "smart_contracts": ("Smart contracts", "blockchain", False, ["solidity_basics"]),
    "blockchain_architecture": ("Blockchain architecture", "blockchain", False, ["smart_contracts"]),

    # ---------- WRITING / MARKETING / CRM / ERP ----------
    "api_documentation": ("API documentation", "writing", False, ["technical_writing", "rest_apis"]),
    "seo_fundamentals": ("SEO fundamentals", "marketing", False, ["digital_marketing_fundamentals"]),
    "google_analytics": ("Google Analytics", "marketing", False, ["digital_marketing_fundamentals"]),
    "google_ads": ("Google Ads", "marketing", False, ["digital_marketing_fundamentals"]),
    "ab_testing": ("A/B testing", "marketing", False, ["google_analytics"]),
    "salesforce_apex": ("Salesforce Apex", "crm", False, ["salesforce_basics"]),
    "sap_basics": ("SAP basics", "erp", False, ["erp_fundamentals"]),
}

CAREER_GOALS: dict[str, tuple[str, list[str]]] = {
    # ---------- DATA / ANALYTICS TRACK ----------
    "goal_data_analyst": ("Data analyst", [
        "excel_basics", "sql_basics", "python_basics", "r_basics", "pandas_numpy", "power_bi", "tableau",
    ]),
    "goal_bi_analyst": ("Business intelligence analyst", [
        "excel_basics", "sql_basics", "python_basics", "pandas_numpy", "power_bi", "tableau", "data_warehousing",
    ]),
    "goal_data_scientist": ("Data scientist", [
        "excel_basics", "sql_basics", "python_basics", "r_basics", "pandas_numpy",
        "statistics", "statistical_modeling_r", "data_wrangling", "power_bi", "tableau", "machine_learning",
    ]),
    "goal_ml_engineer": ("ML engineer", [
        "python_basics", "pandas_numpy", "statistics", "data_wrangling", "machine_learning",
        "docker_basics", "mlops_basics", "cloud_aws_basics", "model_serving",
    ]),
    "goal_mlops_eng": ("MLOps engineer", [
        "linux_basics", "python_basics", "docker_basics", "machine_learning",
        "mlops_basics", "ci_cd_basics", "kubernetes_basics", "cloud_aws_basics",
        "model_serving", "model_monitoring",
    ]),
    "goal_ml_researcher": ("Machine learning researcher", [
        "python_basics", "python_advanced", "pandas_numpy", "statistics",
        "statistical_modeling_r", "data_wrangling", "machine_learning", "deep_learning",
    ]),
    "goal_ai_engineer": ("AI engineer", [
        "python_basics", "python_advanced", "pandas_numpy", "statistics", "machine_learning", "deep_learning",
        "pytorch", "llm_applications", "vector_databases", "rag_systems", "model_serving", "mlops_basics",
    ]),
    "goal_cv_eng": ("Computer vision engineer", [
        "python_basics", "pandas_numpy", "statistics", "machine_learning", "deep_learning", "pytorch",
    ]),
    "goal_nlp_eng": ("NLP engineer", [
        "python_basics", "pandas_numpy", "statistics", "machine_learning", "deep_learning", "llm_applications",
    ]),
    "goal_prompt_eng": ("Prompt engineer", [
        "python_basics", "machine_learning", "llm_applications", "prompt_engineering",
    ]),

    # ---------- DATA ENGINEERING TRACK ----------
    "goal_data_eng": ("Data engineer", [
        "python_basics", "sql_basics", "data_pipelines", "spark_basics", "data_warehousing", "etl_orchestration",
    ]),
    "goal_analytics_eng": ("Analytics engineer", [
        "sql_basics", "python_basics", "data_pipelines", "dbt_basics", "data_warehousing",
    ]),
    "goal_data_governance_analyst": ("Data governance analyst", [
        "sql_basics", "data_pipelines", "data_warehousing", "data_governance",
    ]),
    "goal_data_privacy_eng": ("Data privacy engineer", [
        "security_fundamentals", "identity_access_management", "data_warehousing", "data_governance",
    ]),

    # ---------- WEB: FRONTEND / BACKEND / FULLSTACK ----------
    "goal_frontend_eng": ("Frontend engineer", [
        "html_css_basics", "javascript_basics", "typescript_basics", "react_basics", "state_management", "frontend_testing",
    ]),
    "goal_perf_eng": ("Web performance engineer", [
        "html_css_basics", "javascript_basics", "react_basics", "web_performance", "monitoring_observability",
    ]),
    "goal_backend_eng": ("Backend engineer", [
        "python_basics", "sql_basics", "rest_apis", "databases_basics", "backend_frameworks", "system_design",
    ]),
    "goal_fullstack_eng": ("Full-stack engineer", [
        "html_css_basics", "javascript_basics", "react_basics", "state_management",
        "python_basics", "sql_basics", "rest_apis", "databases_basics", "backend_frameworks", "system_design",
    ]),
    "goal_solutions_architect": ("Solutions architect", [
        "sql_basics", "rest_apis", "databases_basics", "system_design",
        "cloud_aws_basics", "cloud_architecture", "stakeholder_management",
    ]),
    "goal_growth_eng": ("Growth engineer", [
        "javascript_basics", "digital_marketing_fundamentals", "google_analytics", "ab_testing",
    ]),

    # ---------- MOBILE ----------
    "goal_android_eng": ("Android engineer", [
        "git_basics", "html_css_basics", "kotlin_basics", "android_development", "mobile_ui_patterns",
    ]),
    "goal_ios_eng": ("iOS engineer", [
        "git_basics", "html_css_basics", "swift_basics", "ios_development", "mobile_ui_patterns",
    ]),

    # ---------- CLOUD / DEVOPS / INFRA ----------
    "goal_cloud_eng": ("Cloud engineer", [
        "linux_basics", "docker_basics", "cloud_aws_basics", "terraform_basics", "kubernetes_basics",
    ]),
    "goal_cloud_architect": ("Cloud architect", [
        "linux_basics", "docker_basics", "cloud_aws_basics", "gcp_basics", "azure_basics",
        "system_design", "cloud_architecture", "multi_cloud",
    ]),
    "goal_devops_eng": ("DevOps engineer", [
        "linux_basics", "git_basics", "docker_basics", "ci_cd_basics", "jenkins_basics",
        "kubernetes_basics", "ansible_basics", "infrastructure_as_code",
    ]),
    "goal_sre": ("Site reliability engineer", [
        "linux_basics", "docker_basics", "kubernetes_basics", "prometheus_grafana",
        "monitoring_observability", "system_design", "cloud_aws_basics", "release_engineering",
    ]),
    "goal_platform_eng": ("Platform engineer", [
        "linux_basics", "docker_basics", "kubernetes_basics", "cloud_aws_basics",
        "terraform_basics", "infrastructure_as_code", "ci_cd_basics",
    ]),
    "goal_infra_eng": ("Infrastructure engineer", [
        "linux_basics", "networking_basics", "systems_administration", "ansible_basics",
    ]),
    "goal_release_eng": ("Release engineer", [
        "git_basics", "linux_basics", "ci_cd_basics", "jenkins_basics", "release_engineering",
    ]),

    # ---------- SYSADMIN / SUPPORT / NETWORK ----------
    "goal_sysadmin": ("Systems administrator", [
        "linux_basics", "networking_basics", "it_support_fundamentals", "systems_administration",
    ]),
    "goal_it_support": ("IT support specialist", [
        "it_support_fundamentals", "networking_basics",
    ]),
    "goal_tech_support_eng": ("Technical support engineer", [
        "it_support_fundamentals", "networking_basics", "rest_apis",
    ]),
    "goal_network_eng": ("Network engineer", [
        "networking_basics", "network_administration", "security_fundamentals", "network_security",
    ]),

    # ---------- SECURITY ----------
    "goal_security_eng": ("Security engineer", [
        "networking_basics", "security_fundamentals", "network_security",
        "rest_apis", "application_security", "identity_access_management", "siem_tools",
    ]),
    "goal_pentester": ("Penetration tester", [
        "networking_basics", "security_fundamentals", "network_security", "penetration_testing", "burp_suite",
    ]),

    # ---------- DATABASE ----------
    "goal_dba": ("Database administrator", [
        "sql_basics", "databases_basics", "database_administration", "database_tuning",
    ]),

    # ---------- QA ----------
    "goal_qa_eng": ("QA engineer", [
        "manual_testing", "jira_basics", "python_basics", "test_automation", "frontend_testing",
    ]),

    # ---------- PRODUCT / DESIGN ----------
    "goal_product_manager": ("Product manager", [
        "agile_basics", "jira_basics", "product_discovery", "product_roadmapping", "stakeholder_management",
    ]),
    "goal_scrum_master": ("Scrum master", [
        "agile_basics", "jira_basics", "scrum_facilitation", "stakeholder_management",
    ]),
    "goal_ux_designer": ("UX designer", [
        "figma_basics", "miro_basics", "ux_research", "ui_design_principles", "prototyping",
    ]),
    "goal_ui_designer": ("UI designer", [
        "figma_basics", "sketch_basics", "ui_design_principles", "design_systems",
    ]),

    # ---------- GAME DEV / EMBEDDED / ROBOTICS / XR ----------
    "goal_game_dev": ("Game developer", [
        "git_basics", "game_engine_basics", "gameplay_programming", "graphics_programming",
    ]),
    "goal_embedded_eng": ("Embedded systems engineer", [
        "git_basics", "embedded_c_basics", "rtos_basics", "hardware_interfacing",
    ]),
    "goal_robotics_eng": ("Robotics engineer", [
        "git_basics", "embedded_c_basics", "robotics_fundamentals", "ros_basics",
    ]),
    "goal_ar_vr_dev": ("AR/VR developer", [
        "git_basics", "game_engine_basics", "gameplay_programming", "ar_vr_basics", "graphics_programming",
    ]),

    # ---------- BLOCKCHAIN ----------
    "goal_blockchain_dev": ("Blockchain developer", [
        "javascript_basics", "solidity_basics", "smart_contracts", "blockchain_architecture",
    ]),

    # ---------- WRITING / MARKETING / CRM / ERP ----------
    "goal_tech_writer": ("Technical writer", [
        "technical_writing", "rest_apis", "api_documentation",
    ]),
    "goal_digital_marketer": ("Digital marketing analyst", [
        "digital_marketing_fundamentals", "seo_fundamentals", "google_analytics", "google_ads",
    ]),
    "goal_seo_specialist": ("SEO specialist", [
        "digital_marketing_fundamentals", "seo_fundamentals", "google_analytics",
    ]),
    "goal_salesforce_dev": ("Salesforce developer", [
        "salesforce_basics", "salesforce_apex",
    ]),
    "goal_erp_consultant": ("ERP consultant", [
        "erp_fundamentals", "sap_basics", "stakeholder_management",
    ]),
}

DIFFICULTIES = ["beginner", "intermediate", "advanced"]
FORMATS = ["video", "text", "interactive"]
EXPERIENCE_LEVELS = ["beginner", "intermediate", "advanced"]
LEARNING_STYLES = ["visual", "reading", "practice"]

INTERESTS = [
    "generative_ai", "llms", "computer_vision", "nlp", "mlops",
    "cybersecurity", "cloud_native", "mobile_dev", "web_performance",
    "data_engineering", "product_strategy", "ux_design", "growth_marketing",
    "blockchain", "game_dev", "embedded_systems", "devops_automation",
]
GOAL_INTERESTS = {
    "goal_data_analyst": ["data_engineering"],
    "goal_bi_analyst": ["data_engineering"],
    "goal_data_scientist": ["nlp", "data_engineering"],
    "goal_ml_engineer": ["mlops"],
    "goal_mlops_eng": ["mlops", "devops_automation"],
    "goal_ml_researcher": ["generative_ai"],
    "goal_ai_engineer": ["generative_ai", "llms"],
    "goal_cv_eng": ["computer_vision"],
    "goal_nlp_eng": ["nlp", "llms"],
    "goal_prompt_eng": ["generative_ai", "llms"],
    "goal_data_eng": ["data_engineering"],
    "goal_analytics_eng": ["data_engineering"],
    "goal_data_governance_analyst": ["data_engineering", "cybersecurity"],
    "goal_data_privacy_eng": ["cybersecurity"],
    "goal_frontend_eng": ["web_performance"],
    "goal_perf_eng": ["web_performance"],
    "goal_backend_eng": ["mlops"],
    "goal_fullstack_eng": ["web_performance"],
    "goal_solutions_architect": ["cloud_native"],
    "goal_growth_eng": ["growth_marketing"],
    "goal_android_eng": ["mobile_dev"],
    "goal_ios_eng": ["mobile_dev"],
    "goal_cloud_eng": ["cloud_native"],
    "goal_cloud_architect": ["cloud_native"],
    "goal_devops_eng": ["devops_automation", "cloud_native"],
    "goal_sre": ["devops_automation", "cloud_native"],
    "goal_platform_eng": ["cloud_native", "devops_automation"],
    "goal_infra_eng": ["cloud_native"],
    "goal_release_eng": ["devops_automation"],
    "goal_sysadmin": ["cloud_native"],
    "goal_it_support": [],
    "goal_tech_support_eng": [],
    "goal_network_eng": ["cybersecurity"],
    "goal_security_eng": ["cybersecurity"],
    "goal_pentester": ["cybersecurity"],
    "goal_dba": ["data_engineering"],
    "goal_qa_eng": ["devops_automation"],
    "goal_product_manager": ["product_strategy"],
    "goal_scrum_master": ["product_strategy"],
    "goal_ux_designer": ["ux_design"],
    "goal_ui_designer": ["ux_design"],
    "goal_game_dev": ["game_dev"],
    "goal_embedded_eng": ["embedded_systems"],
    "goal_robotics_eng": ["embedded_systems"],
    "goal_ar_vr_dev": ["game_dev", "embedded_systems"],
    "goal_blockchain_dev": ["blockchain"],
    "goal_tech_writer": [],
    "goal_digital_marketer": ["growth_marketing"],
    "goal_seo_specialist": ["growth_marketing"],
    "goal_salesforce_dev": ["product_strategy"],
    "goal_erp_consultant": ["product_strategy"],
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
    """
    Create a deterministic catalog.

    Every resource belongs to exactly ONE primary skill.
    Metadata is intentionally grounded so the path router can explain:
    skill -> specific resource -> learning value.
    """

    courses = []
    idx = 0

    # Format and duration are deterministic based on resource type.
    RESOURCE_TYPES = [
        {
            "suffix": "Foundations",
            "difficulty": "beginner",
            "duration_hours": 6,
            "format": "video",
            "description_template": (
                "Learn the core concepts and practical fundamentals of {skill_name} "
                "through structured lessons and examples."
            ),
        },
        {
            "suffix": "Hands-on Practice Lab",
            "difficulty": "beginner",
            "duration_hours": 5,
            "format": "interactive",
            "description_template": (
                "Practice {skill_name} through guided exercises and practical tasks "
                "designed to build confidence with the core concepts."
            ),
        },
        {
            "suffix": "Guided Project",
            "difficulty": "intermediate",
            "duration_hours": 10,
            "format": "interactive",
            "description_template": (
                "Apply {skill_name} in a practical project and build a portfolio-ready "
                "example demonstrating how the skill is used in real work."
            ),
        },
        {
            "suffix": "Deep Dive",
            "difficulty": "advanced",
            "duration_hours": 14,
            "format": "text",
            "description_template": (
                "Explore {skill_name} in greater depth, including advanced concepts, "
                "real-world patterns, and common implementation challenges."
            ),
        },
        {
            "suffix": "Interview & Real-World Workshop",
            "difficulty": "intermediate",
            "duration_hours": 4,
            "format": "interactive",
            "description_template": (
                "Strengthen practical {skill_name} knowledge through real-world scenarios, "
                "problem solving, and interview-oriented exercises."
            ),
        },
    ]

    for skill_id, (skill_name, category, is_foundational, prereqs) in SKILLS.items():

        for resource in RESOURCE_TYPES:
            idx += 1

            # Adjust some metadata based on whether the skill is foundational.
            difficulty = resource["difficulty"]

            if is_foundational and difficulty == "advanced":
                difficulty = "intermediate"

            courses.append(
                {
                    "course_id": f"course_{idx:03d}",
                    "title": f"{skill_name} {resource['suffix']}",
                    "description": resource["description_template"].format(
                        skill_name=skill_name
                    ),
                    "difficulty": difficulty,
                    "duration_hours": resource["duration_hours"],
                    "format": resource["format"],
                    "primary_skill_id": skill_id,
                    "category": category,
                }
            )

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
        users.append(
            {
                "user_id": f"user_{i:04d}",
                "career_goal_id": goal,
                "experience_level": rng.choice(EXPERIENCE_LEVELS),
                "learning_style": rng.choice(LEARNING_STYLES),
                "weekly_hours": rng.randint(4, 20),
                "interests": "|".join(interests),
            }
        )
    return users


def _simulate_events(
    rng: random.Random, users: list[dict], courses: list[dict], graph
) -> list[dict]:
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
            events.append(
                {
                    "event_id": f"evt_{idx:06d}",
                    "user_id": user["user_id"],
                    "course_id": course["course_id"],
                    "skill_id": skill_id,
                    "event_type": event_type,
                    "completion_pct": round(completion_pct, 1),
                    "score": score,
                    "occurred_at": f"2026-{1 + (day // 30) % 12:02d}-{1 + day % 28:02d}",
                }
            )
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
        {
            "skill_id": sid,
            "skill_name": name,
            "category": cat,
            "is_foundational": int(found),
        }
        for sid, (name, cat, found, _p) in SKILLS.items()
    ]
    prereq_rows = [
        {"skill_id": sid, "prerequisite_skill_id": p}
        for sid, (_n, _c, _f, ps) in SKILLS.items()
        for p in ps
    ]
    goal_rows = [
        {"goal_id": gid, "title": title} for gid, (title, _s) in CAREER_GOALS.items()
    ]
    goal_skill_rows = [
        {"goal_id": gid, "skill_id": sid, "importance_weight": 1.0}
        for gid, (_t, ss) in CAREER_GOALS.items()
        for sid in ss
    ]
    courses = _make_courses(rng)
    course_rows = [
        {k: v for k, v in c.items() if k != "primary_skill_id"} for c in courses
    ]
    course_skill_rows = [
        {
            "course_id": c["course_id"],
            "skill_id": c["primary_skill_id"],
            "skill_weight": 1.0,
        }
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
