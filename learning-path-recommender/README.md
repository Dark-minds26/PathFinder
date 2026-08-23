# AI-powered personalized learning path recommender

Generates personalized, explainable learning roadmaps from a hybrid
recommender (content-based filtering + skill knowledge graph), with
SHAP-backed explanations and a RAG-driven conversational profiling agent.

## Status

Phase 1 scaffold: architecture, database schema, and the API surface
are in place. Component and LLM logic are intentionally stubbed with
`NotImplementedError`, pending Phase 2 (data pipeline + algorithms)
and Phase 3 (API + LLM integration).

## Run it

    pip install -r requirements.txt
    cp .env.example .env
    uvicorn api.main:app --reload

    pytest

## Layout

- `src/recommender/components` - single-responsibility pipeline steps
- `src/recommender/pipeline` - orchestrates components end to end
- `src/recommender/entity` - typed config/artifact contracts
- `src/recommender/llm` - RAG + conversational profiling
- `api/` - FastAPI app, routers, request/response schemas
