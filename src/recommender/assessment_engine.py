"""Deterministic lightweight checkpoint bank for demo and production-safe tests."""

QUESTIONS = {
    "python_basics": [
        (
            "Which data structure stores key/value pairs in Python?",
            ["list", "dict", "tuple", "set"],
            1,
        ),
        ("Which keyword defines a function?", ["func", "def", "function", "lambda"], 1),
        ("What does len([1,2,3]) return?", ["2", "3", "4", "1"], 1),
    ],
    "docker_basics": [
        (
            "Which file commonly defines Docker image build steps?",
            ["docker-compose.yml", "Dockerfile", "requirements.txt", "Makefile"],
            1,
        ),
        (
            "Which command builds an image?",
            ["docker run", "docker build", "docker ps", "docker pull"],
            1,
        ),
        (
            "What isolates application dependencies in Docker?",
            ["containers", "branches", "queries", "routes"],
            1,
        ),
    ],
    "machine_learning": [
        (
            "What is overfitting?",
            [
                "Poor training only",
                "Good training but poor generalization",
                "No model",
                "Data loading failure",
            ],
            1,
        ),
        (
            "What does a validation set help estimate?",
            ["Generalization", "Disk size", "CPU speed", "API latency"],
            0,
        ),
        (
            "Which metric is common for regression?",
            ["MAE", "Accuracy", "F1 only", "BLEU"],
            0,
        ),
    ],
    "llm_applications": [
        (
            "What is a prompt?",
            [
                "A model input/instruction",
                "A database index",
                "A container",
                "A compiler",
            ],
            0,
        ),
        (
            "What is tokenization?",
            [
                "Splitting text into model tokens",
                "Encrypting text",
                "Deploying a model",
                "Scaling GPUs",
            ],
            0,
        ),
        (
            "Why evaluate an LLM application?",
            [
                "To measure behavior and quality",
                "To install Python",
                "To create a subnet",
                "To compress images",
            ],
            0,
        ),
    ],
    "rag_systems": [
        (
            "RAG combines retrieval with what?",
            ["Generation", "Compilation", "CSS", "Networking"],
            0,
        ),
        (
            "Embeddings represent content as what?",
            ["Vectors", "Tables only", "Images only", "Passwords"],
            0,
        ),
        (
            "A vector database is commonly used for what?",
            [
                "Similarity retrieval",
                "Rendering UI",
                "Building containers",
                "Compiling Java",
            ],
            0,
        ),
    ],
}


def questions_for(skill_id):
    return [
        {"id": f"{skill_id}_{i}", "question": q, "options": opts}
        for i, (q, opts, _) in enumerate(
            QUESTIONS.get(
                skill_id,
                [
                    (
                        f"Which statement best describes {skill_id.replace('_', ' ')}?",
                        ["Core concept", "Unrelated concept", "Random value", "None"],
                        0,
                    )
                ],
            ),
            1,
        )
    ]


def score_answers(skill_id, answers):
    bank = QUESTIONS.get(skill_id)
    if not bank:
        return None
    correct = 0
    for i, (_, _, idx) in enumerate(bank):
        value = answers.get(f"{skill_id}_{i + 1}", answers.get(str(i + 1)))
        if isinstance(value, int) and value == idx:
            correct += 1
    return round(correct / len(bank) * 100, 1)
