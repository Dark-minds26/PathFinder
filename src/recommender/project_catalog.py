PROJECTS = {
    # ---------- FOUNDATIONAL ----------
    "python_basics": (
        "Python Data Report CLI",
        ["python_basics"],
        4,
        "Build a command-line application that reads data, validates user input, and generates a useful report.",
    ),
    "git_basics": (
        "Git Workflow Practice Repository",
        ["git_basics"],
        3,
        "Create a small project repository and practice commits, branches, merges, pull requests, and version control workflows.",
    ),
    "sql_basics": (
        "SQL Sales Analysis",
        ["sql_basics"],
        5,
        "Query a sample business database using SELECT, WHERE, JOIN, GROUP BY, and aggregate functions to answer business questions.",
    ),
    "html_css_basics": (
        "Responsive Portfolio Page",
        ["html_css_basics"],
        5,
        "Build a responsive personal portfolio webpage using semantic HTML and modern CSS layouts.",
    ),
    "docker_basics": (
        "Containerized Web Application",
        ["docker_basics"],
        6,
        "Package a small application into a Docker container and configure ports, environment variables, and reproducible builds.",
    ),
    "linux_basics": (
        "Linux Automation Toolkit",
        ["linux_basics"],
        4,
        "Practice Linux navigation, permissions, processes, and shell commands by creating a small automation toolkit.",
    ),
    "networking_basics": (
        "Network Troubleshooting Lab",
        ["networking_basics"],
        5,
        "Practice IP addressing, DNS, ports, and connectivity troubleshooting using simulated networking scenarios.",
    ),
    "agile_basics": (
        "Agile Sprint Simulation",
        ["agile_basics"],
        4,
        "Plan and run a simulated agile sprint with user stories, backlog prioritization, sprint planning, and retrospectives.",
    ),
    "excel_basics": (
        "Excel Sales Dashboard",
        ["excel_basics"],
        5,
        "Analyze a business dataset using formulas, sorting, filtering, pivot tables, and charts to create a simple dashboard.",
    ),
    "r_basics": (
        "R Data Analysis Report",
        ["r_basics"],
        5,
        "Use R to load, clean, analyze, and visualize a dataset and present the findings in a structured report.",
    ),
    "figma_basics": (
        "Mobile App UI Mockup",
        ["figma_basics"],
        6,
        "Design a simple mobile application interface in Figma using frames, components, typography, and reusable design elements.",
    ),
    "manual_testing": (
        "Manual Testing Test Suite",
        ["manual_testing"],
        5,
        "Create test cases, execute them against a sample application, report defects, and document testing results.",
    ),
    "security_fundamentals": (
        "Security Risk Assessment",
        ["security_fundamentals"],
        5,
        "Analyze a sample application or system and identify common security risks, threats, and basic mitigation strategies.",
    ),
    "product_discovery": (
        "Product Discovery Case Study",
        ["product_discovery"],
        5,
        "Research a user problem, define assumptions, identify user needs, and propose a product solution.",
    ),
    "stakeholder_management": (
        "Stakeholder Communication Plan",
        ["stakeholder_management"],
        4,
        "Create a stakeholder map and communication plan for a simulated product or technology project.",
    ),
    "ux_research": (
        "UX Research Case Study",
        ["ux_research"],
        6,
        "Conduct user research, identify pain points, synthesize findings, and create actionable design recommendations.",
    ),
    "technical_writing": (
        "Technical Documentation Project",
        ["technical_writing"],
        5,
        "Write clear technical documentation for a sample software product, including setup instructions and usage guides.",
    ),
    "it_support_fundamentals": (
        "IT Support Troubleshooting Guide",
        ["it_support_fundamentals"],
        5,
        "Solve common hardware and software support scenarios and document step-by-step troubleshooting procedures.",
    ),
    "salesforce_basics": (
        "Salesforce CRM Setup",
        ["salesforce_basics"],
        6,
        "Configure a basic CRM workflow with objects, records, relationships, and simple automation.",
    ),
    "erp_fundamentals": (
        "ERP Business Process Mapping",
        ["erp_fundamentals"],
        5,
        "Map a business workflow such as procurement or inventory management to an ERP-style process.",
    ),
    "digital_marketing_fundamentals": (
        "Digital Marketing Campaign Plan",
        ["digital_marketing_fundamentals"],
        6,
        "Create a digital marketing campaign with audience targeting, channels, content ideas, and measurable objectives.",
    ),
    "jira_basics": (
        "Jira Project Board Setup",
        ["jira_basics"],
        4,
        "Create a Jira-style project board with issues, epics, priorities, workflows, and sprint organization.",
    ),
    "miro_basics": (
        "Collaborative Workshop Board",
        ["miro_basics"],
        3,
        "Create a collaborative workshop board for brainstorming, user journeys, and idea prioritization.",
    ),

    # ---------- DATA / ANALYTICS ----------
    "python_advanced": (
        "Python Data Processing Pipeline",
        ["python_advanced"],
        8,
        "Build a structured Python application using advanced functions, object-oriented programming, error handling, and modular code.",
    ),
    "pandas_numpy": (
        "Pandas and NumPy Data Analysis",
        ["pandas_numpy"],
        8,
        "Clean, transform, analyze, and summarize a real dataset using Pandas and NumPy.",
    ),
    "statistics": (
        "Statistical Analysis Case Study",
        ["statistics"],
        7,
        "Analyze a dataset using descriptive statistics, probability concepts, distributions, and hypothesis testing.",
    ),
    "statistical_modeling_r": (
        "R Statistical Modeling Project",
        ["statistical_modeling_r"],
        8,
        "Build and evaluate statistical models in R and interpret the results using real data.",
    ),
    "data_wrangling": (
        "Messy Dataset Cleaning Project",
        ["data_wrangling"],
        8,
        "Clean missing values, inconsistent formats, duplicates, and outliers to prepare a real dataset for analysis.",
    ),
    "power_bi": (
        "Power BI Business Dashboard",
        ["power_bi"],
        8,
        "Build an interactive Power BI dashboard with data modeling, transformations, KPIs, and business visualizations.",
    ),
    "tableau": (
        "Tableau Interactive Dashboard",
        ["tableau"],
        8,
        "Create an interactive Tableau dashboard that communicates important trends and insights from a dataset.",
    ),

    # ---------- MACHINE LEARNING / AI ----------
    "machine_learning": (
        "End-to-End ML Predictor",
        ["machine_learning"],
        10,
        "Train, evaluate, and compare machine learning models on a real prediction problem with reproducible preprocessing.",
    ),
    "deep_learning": (
        "Deep Learning Classification Model",
        ["deep_learning"],
        12,
        "Build and train a neural network for a classification problem and evaluate its performance.",
    ),
    "mlops_basics": (
        "Reproducible ML Pipeline",
        ["mlops_basics"],
        10,
        "Create a reproducible machine learning workflow with experiment tracking, data versioning, and pipeline stages.",
    ),
    "pytorch": (
        "PyTorch Neural Network Project",
        ["pytorch"],
        12,
        "Build, train, validate, and improve a neural network using PyTorch.",
    ),
    "llm_applications": (
        "LLM Q&A Assistant",
        ["llm_applications"],
        10,
        "Build a practical LLM application with prompt templates, structured outputs, validation, and evaluation.",
    ),
    "rag_systems": (
        "RAG Knowledge Assistant",
        ["rag_systems"],
        12,
        "Ingest documents, create embeddings, retrieve relevant context, and generate grounded answers using a RAG pipeline.",
    ),
    "model_serving": (
        "Production Model Service",
        ["model_serving"],
        12,
        "Serve a trained machine learning model behind an API with validation, health checks, containerization, and monitoring.",
    ),
    "prompt_engineering": (
        "Prompt Engineering Evaluation Lab",
        ["prompt_engineering"],
        8,
        "Design, compare, and evaluate prompts for different tasks while improving reliability and output quality.",
    ),

    # ---------- DATA ENGINEERING ----------
    "data_pipelines": (
        "Automated Data Pipeline",
        ["data_pipelines"],
        10,
        "Build a pipeline that extracts, transforms, validates, and loads data automatically.",
    ),
    "spark_basics": (
        "Apache Spark Data Processing Project",
        ["spark_basics"],
        10,
        "Process and analyze a larger dataset using Spark DataFrames and distributed transformations.",
    ),
    "data_warehousing": (
        "Data Warehouse Design Project",
        ["data_warehousing"],
        10,
        "Design a dimensional data warehouse with fact tables, dimension tables, and analytical queries.",
    ),
    "etl_orchestration": (
        "Airflow ETL Workflow",
        ["etl_orchestration"],
        10,
        "Build and schedule an ETL workflow using DAGs, task dependencies, and monitoring.",
    ),
    "dbt_basics": (
        "dbt Analytics Transformation Project",
        ["dbt_basics"],
        8,
        "Transform raw data into reliable analytical models using dbt, tests, and documentation.",
    ),
    "data_governance": (
        "Data Governance Framework",
        ["data_governance"],
        8,
        "Define data ownership, quality rules, policies, and governance processes for a sample organization.",
    ),

    # ---------- FRONTEND / BACKEND ----------
    "javascript_basics": (
        "Interactive JavaScript Web App",
        ["javascript_basics"],
        6,
        "Build an interactive browser application using JavaScript, events, functions, and DOM manipulation.",
    ),
    "typescript_basics": (
        "TypeScript Application",
        ["typescript_basics"],
        6,
        "Convert or build a JavaScript application using TypeScript types, interfaces, and safer application logic.",
    ),
    "react_basics": (
        "React Task Management App",
        ["react_basics"],
        8,
        "Build a React application using components, props, state, hooks, and reusable UI structure.",
    ),
    "state_management": (
        "React State Management App",
        ["state_management"],
        8,
        "Build a multi-component application with predictable shared state and organized data flow.",
    ),
    "frontend_testing": (
        "Frontend Testing Suite",
        ["frontend_testing"],
        7,
        "Write automated tests for frontend components and user interactions.",
    ),
    "web_performance": (
        "Website Performance Optimization",
        ["web_performance"],
        8,
        "Measure and improve page loading, rendering, assets, and frontend performance metrics.",
    ),
    "rest_apis": (
        "REST API Service",
        ["rest_apis"],
        8,
        "Build a REST API with endpoints, validation, request handling, error responses, and documentation.",
    ),
    "databases_basics": (
        "Application Database Project",
        ["databases_basics"],
        7,
        "Design a relational database and connect it to a small application.",
    ),
    "backend_frameworks": (
        "Backend Web Service",
        ["backend_frameworks"],
        10,
        "Build a backend application using a framework with routing, validation, database integration, and API endpoints.",
    ),
    "system_design": (
        "Scalable System Design Case Study",
        ["system_design"],
        10,
        "Design the architecture of a scalable application including services, databases, caching, and trade-offs.",
    ),

    # ---------- CLOUD / DEVOPS ----------
    "kubernetes_basics": (
        "Kubernetes Deployment Project",
        ["kubernetes_basics"],
        10,
        "Deploy and manage a containerized application using Kubernetes pods, deployments, services, and configuration.",
    ),
    "cloud_aws_basics": (
        "AWS Cloud Application Deployment",
        ["cloud_aws_basics"],
        10,
        "Deploy a small application using core AWS services while configuring storage, compute, networking, and permissions.",
    ),
    "gcp_basics": (
        "Google Cloud Deployment Project",
        ["gcp_basics"],
        10,
        "Deploy and configure a small application using core Google Cloud services.",
    ),
    "azure_basics": (
        "Azure Cloud Deployment Project",
        ["azure_basics"],
        10,
        "Deploy and configure a small application using core Microsoft Azure services.",
    ),
    "ci_cd_basics": (
        "CI/CD Automation Pipeline",
        ["ci_cd_basics"],
        8,
        "Create an automated pipeline that builds, tests, and deploys an application after code changes.",
    ),
    "jenkins_basics": (
        "Jenkins Automation Pipeline",
        ["jenkins_basics"],
        8,
        "Build a Jenkins pipeline that automates application builds, tests, and deployment steps.",
    ),
    "ansible_basics": (
        "Ansible Server Automation",
        ["ansible_basics"],
        8,
        "Automate server configuration and repetitive infrastructure tasks using Ansible playbooks.",
    ),
    "terraform_basics": (
        "Terraform Cloud Infrastructure",
        ["terraform_basics"],
        10,
        "Provision reproducible cloud infrastructure using Terraform configuration files and state management.",
    ),
    "infrastructure_as_code": (
        "Infrastructure as Code Project",
        ["infrastructure_as_code"],
        10,
        "Define and manage infrastructure through version-controlled configuration rather than manual setup.",
    ),
    "monitoring_observability": (
        "Application Observability Setup",
        ["monitoring_observability"],
        8,
        "Add logs, metrics, and basic observability to a running application.",
    ),
    "prometheus_grafana": (
        "Prometheus and Grafana Monitoring",
        ["prometheus_grafana"],
        8,
        "Collect application metrics with Prometheus and visualize system health through Grafana dashboards.",
    ),
    "cloud_architecture": (
        "Cloud Architecture Design",
        ["cloud_architecture"],
        10,
        "Design a reliable cloud architecture with scalability, availability, security, and cost considerations.",
    ),
    "multi_cloud": (
        "Multi-Cloud Architecture Plan",
        ["multi_cloud"],
        10,
        "Design an architecture that evaluates workloads and trade-offs across multiple cloud providers.",
    ),
    "release_engineering": (
        "Release Management Pipeline",
        ["release_engineering"],
        8,
        "Design a repeatable release workflow with versioning, testing, approvals, and rollback planning.",
    ),
    "systems_administration": (
        "Linux Server Administration Lab",
        ["systems_administration"],
        8,
        "Configure and manage users, services, storage, permissions, and basic server operations.",
    ),
    "network_administration": (
        "Network Administration Lab",
        ["network_administration"],
        8,
        "Configure and troubleshoot network services, devices, addressing, and connectivity.",
    ),

    # ---------- SECURITY ----------
    "network_security": (
        "Secure Network Design",
        ["network_security"],
        8,
        "Design a secure network using segmentation, firewalls, access controls, and monitoring concepts.",
    ),
    "application_security": (
        "Secure Web Application Review",
        ["application_security"],
        8,
        "Review a web application for common security weaknesses and apply secure development practices.",
    ),
    "identity_access_management": (
        "Identity and Access Management Design",
        ["identity_access_management"],
        8,
        "Design roles, permissions, authentication, and access policies for a sample organization.",
    ),
    "penetration_testing": (
        "Authorized Security Testing Lab",
        ["penetration_testing"],
        10,
        "Practice identifying and documenting security weaknesses in a safe, authorized lab environment.",
    ),
    "burp_suite": (
        "Web Security Testing Lab",
        ["burp_suite"],
        8,
        "Use Burp Suite in a legal training environment to inspect and test web application security.",
    ),
    "siem_tools": (
        "Security Monitoring Dashboard",
        ["siem_tools"],
        10,
        "Analyze security logs and create detections and dashboards using SIEM concepts.",
    ),

    # ---------- DATABASE ----------
    "database_administration": (
        "Database Administration Lab",
        ["database_administration"],
        8,
        "Manage database users, backups, permissions, maintenance, and operational health.",
    ),
    "database_tuning": (
        "Database Performance Optimization",
        ["database_tuning"],
        8,
        "Analyze slow queries and improve database performance using indexing and query optimization.",
    ),

    # ---------- MOBILE ----------
    "kotlin_basics": (
        "Kotlin Fundamentals App",
        ["kotlin_basics"],
        6,
        "Build a small Kotlin application using core language features and structured program logic.",
    ),
    "android_development": (
        "Android Mobile Application",
        ["android_development"],
        12,
        "Build a functional Android application with screens, navigation, data handling, and user interactions.",
    ),
    "swift_basics": (
        "Swift Fundamentals App",
        ["swift_basics"],
        6,
        "Build a small application using Swift language fundamentals and structured program logic.",
    ),
    "ios_development": (
        "iOS Mobile Application",
        ["ios_development"],
        12,
        "Build an iOS application with screens, navigation, state, and user interactions.",
    ),
    "mobile_ui_patterns": (
        "Mobile UI Design Prototype",
        ["mobile_ui_patterns"],
        6,
        "Design mobile screens using common navigation, layout, and interaction patterns.",
    ),

    # ---------- QA ----------
    "test_automation": (
        "Automated Testing Framework",
        ["test_automation"],
        10,
        "Build an automated testing suite for a sample application with repeatable test execution and reporting.",
    ),

    # ---------- PRODUCT / DESIGN ----------
    "product_roadmapping": (
        "Product Roadmap Case Study",
        ["product_roadmapping"],
        6,
        "Create a product roadmap that connects strategy, priorities, timelines, and measurable outcomes.",
    ),
    "scrum_facilitation": (
        "Scrum Ceremony Simulation",
        ["scrum_facilitation"],
        5,
        "Practice facilitating sprint planning, stand-ups, reviews, and retrospectives for a simulated team.",
    ),
    "ui_design_principles": (
        "UI Redesign Case Study",
        ["ui_design_principles"],
        6,
        "Redesign an existing interface by applying hierarchy, spacing, typography, consistency, and usability principles.",
    ),
    "prototyping": (
        "Interactive Product Prototype",
        ["prototyping"],
        8,
        "Create an interactive prototype that demonstrates user flows and key product interactions.",
    ),
    "design_systems": (
        "Reusable Design System",
        ["design_systems"],
        10,
        "Create reusable components, design tokens, typography rules, and documentation for a design system.",
    ),
    "sketch_basics": (
        "Sketch Interface Design",
        ["sketch_basics"],
        6,
        "Design a complete interface in Sketch using artboards, reusable components, and layout principles.",
    ),

    # ---------- GAME / EMBEDDED / ROBOTICS / XR ----------
    "game_engine_basics": (
        "Unity Mini Game",
        ["game_engine_basics"],
        10,
        "Build a small playable game using Unity scenes, objects, physics, and scripting.",
    ),
    "gameplay_programming": (
        "Gameplay Mechanics Project",
        ["gameplay_programming"],
        12,
        "Implement player controls, game mechanics, interactions, and progression systems.",
    ),
    "graphics_programming": (
        "Real-Time Graphics Demo",
        ["graphics_programming"],
        12,
        "Create a graphics programming demo exploring rendering, shaders, lighting, and visual effects.",
    ),
    "embedded_c_basics": (
        "Embedded C Hardware Control",
        ["embedded_c_basics"],
        8,
        "Write Embedded C code to control simulated or educational hardware inputs and outputs.",
    ),
    "rtos_basics": (
        "RTOS Task Scheduling Project",
        ["rtos_basics"],
        10,
        "Build an embedded application using multiple tasks, scheduling, synchronization, and timing concepts.",
    ),
    "hardware_interfacing": (
        "Sensor Interfacing Project",
        ["hardware_interfacing"],
        10,
        "Connect and read hardware peripherals and sensors while processing their data in an embedded application.",
    ),
    "robotics_fundamentals": (
        "Robot Navigation Simulation",
        ["robotics_fundamentals"],
        12,
        "Build a simple robotics simulation involving sensors, movement, and basic control logic.",
    ),
    "ros_basics": (
        "ROS Robot Simulation",
        ["ros_basics"],
        12,
        "Create a basic ROS application with nodes, topics, messages, and robot simulation.",
    ),
    "ar_vr_basics": (
        "AR/VR Interactive Experience",
        ["ar_vr_basics"],
        12,
        "Build a basic immersive experience using interaction, scenes, and spatial user interface concepts.",
    ),

    # ---------- BLOCKCHAIN ----------
    "solidity_basics": (
        "Solidity Smart Contract Project",
        ["solidity_basics"],
        8,
        "Write and test simple Solidity smart contracts in a safe development environment.",
    ),
    "smart_contracts": (
        "Decentralized Application Prototype",
        ["smart_contracts"],
        12,
        "Build and test smart contract functionality and connect it to a simple decentralized application interface.",
    ),
    "blockchain_architecture": (
        "Blockchain Architecture Case Study",
        ["blockchain_architecture"],
        10,
        "Design a blockchain-based system and evaluate consensus, storage, security, and scalability trade-offs.",
    ),

    # ---------- WRITING / MARKETING / CRM / ERP ----------
    "api_documentation": (
        "API Documentation Project",
        ["api_documentation"],
        6,
        "Create clear API documentation with endpoints, parameters, request examples, responses, and error handling.",
    ),
    "seo_fundamentals": (
        "SEO Website Audit",
        ["seo_fundamentals"],
        8,
        "Audit a website for keywords, content structure, technical SEO, and optimization opportunities.",
    ),
    "google_analytics": (
        "Web Analytics Report",
        ["google_analytics"],
        6,
        "Analyze website traffic and user behavior data to identify trends and actionable insights.",
    ),
    "google_ads": (
        "Google Ads Campaign Plan",
        ["google_ads"],
        6,
        "Create a search advertising campaign with keywords, audience targeting, budget, and performance metrics.",
    ),
    "ab_testing": (
        "A/B Testing Experiment",
        ["ab_testing"],
        8,
        "Design and analyze an A/B test to evaluate the impact of two product or marketing variations.",
    ),
    "salesforce_apex": (
        "Salesforce Apex Automation",
        ["salesforce_apex"],
        8,
        "Build Apex-based business logic and automation for a Salesforce application.",
    ),
    "sap_basics": (
        "SAP Business Process Case Study",
        ["sap_basics"],
        8,
        "Model a business process using core SAP and ERP workflow concepts.",
    ),
}


def project_for(skill_id: str) -> dict:
    """
    Return the deterministic project mapped to the skill.

    Every known skill has its own meaningful project.
    The fallback exists only as protection if a new skill is added
    later but has not yet been added to PROJECTS.
    """

    project = PROJECTS.get(skill_id)

    if project is None:
        skill_name = skill_id.replace("_", " ").title()

        return {
            "project_id": f"project_{skill_id}",
            "title": f"{skill_name} Practice Project",
            "skills": [skill_id],
            "estimated_hours": 6,
            "description": (
                f"Build a focused portfolio project that demonstrates "
                f"practical understanding of {skill_name}."
            ),
        }

    title, skills, hours, description = project

    return {
        "project_id": f"project_{skill_id}",
        "title": title,
        "skills": skills,
        "estimated_hours": hours,
        "description": description,
    }