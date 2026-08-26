from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from api.routers import profile_router, path_router, explain_router, assessment_router

app = FastAPI(
    title="AI-powered personalized learning path recommender",
    version="0.1.0",
    description="Generates personalized, explainable learning roadmaps.",
)

app.include_router(profile_router.router, prefix="/profile", tags=["profile"])
app.include_router(path_router.router, prefix="/path", tags=["path"])
app.include_router(explain_router.router, prefix="/explain", tags=["explain"])
app.include_router(assessment_router.router, prefix="/assessment", tags=["assessment"])


@app.get("/health", tags=["health"])
def health_check() -> dict:
    """Proves the app can actually serve, not just that the process is
    alive: whether trained artifacts loaded, and which backend each
    optional-dependency component fell back to (lightgbm vs sklearn,
    groq/openai vs the local stub). Never 500s - a broken deployment
    should say so here, not on the first real request."""
    status: dict = {"status": "ok"}
    try:
        from api.dependencies import get_explainer, get_llm, get_path_generator

        generator = get_path_generator()
        status["artifacts_loaded"] = generator.ctx is not None
        status["recommender_backend"] = getattr(generator.model, "backend", "unknown")
        status["explainer_backend"] = type(get_explainer()).__name__
        status["llm_backend"] = type(get_llm()).__name__
    except Exception as e:
        status["status"] = "degraded"
        status["artifacts_loaded"] = False
        status["detail"] = f"{type(e).__name__}: {e}"
    return status


# Demo UI: a static single-page app, mounted last so it only catches
# paths none of the routers above already matched (e.g. "/", "/app.js").
_static_dir = Path(__file__).resolve().parent / "static"
if _static_dir.exists():
    app.mount("/", StaticFiles(directory=str(_static_dir), html=True), name="demo-ui")
