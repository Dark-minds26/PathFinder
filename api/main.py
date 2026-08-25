from fastapi import FastAPI

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
    return {"status": "ok"}
