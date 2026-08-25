"""Shared FastAPI dependencies: one PathGenerator, one Explainer, one
RAGEngine, one LLMClient, one ProfileStore, and one ConversationManager
per active user - all lazily created on first use and reused across
requests instead of reloading the model/graph/preprocessor every call.

DB session wiring is still pending - it arrives with a real Postgres
deployment; ProfileStore (JSON-file-backed) stands in for it here,
same reasoning as cloud_storage/s3_syncer.py.
"""
import pickle

from src.recommender.components.path_generator import PathGenerator
from src.recommender.config.configuration import ConfigurationManager
from src.recommender.llm.conversation_manager import ConversationManager
from src.recommender.llm.llm_client import LLMClient, get_llm_client
from src.recommender.llm.profile_store import ProfileStore
from src.recommender.llm.rag_engine import RAGEngine
from src.recommender.pipeline.adaptive_rerouting_pipeline import AdaptiveReroutingPipeline

_config_manager = ConfigurationManager()
_path_generator: PathGenerator | None = None
_explainer = None
_rag_engine: RAGEngine | None = None
_llm_client: LLMClient | None = None
_profile_store: ProfileStore | None = None
_reroute_pipeline: AdaptiveReroutingPipeline | None = None
_conversations: dict[str, ConversationManager] = {}


def get_path_generator() -> PathGenerator:
    global _path_generator
    if _path_generator is None:
        _path_generator = PathGenerator(_config_manager.get_path_generator_config())
    return _path_generator


def get_explainer():
    global _explainer
    if _explainer is None:
        path = _config_manager.get_explainer_config().explainer_object_path
        with open(path, "rb") as f:
            _explainer = pickle.load(f)
    return _explainer


def get_rag_engine() -> RAGEngine:
    global _rag_engine
    if _rag_engine is None:
        transform_cfg = _config_manager.get_data_transformation_config()
        ingestion_cfg = _config_manager.get_data_ingestion_config()
        _rag_engine = RAGEngine(
            preprocessor_path=transform_cfg.preprocessor_object_path,
            skills_path=f"{ingestion_cfg.ingested_data_dir}/skills.csv",
            goals_path=f"{ingestion_cfg.ingested_data_dir}/career_goals.csv",
        )
    return _rag_engine


def get_llm() -> LLMClient:
    global _llm_client
    if _llm_client is None:
        _llm_client = get_llm_client()
    return _llm_client


def get_profile_store() -> ProfileStore:
    global _profile_store
    if _profile_store is None:
        _profile_store = ProfileStore("artifacts/live_profiles.json")
    return _profile_store


def get_reroute_pipeline() -> AdaptiveReroutingPipeline:
    global _reroute_pipeline
    if _reroute_pipeline is None:
        _reroute_pipeline = AdaptiveReroutingPipeline()
    return _reroute_pipeline


def get_conversation_manager(user_id: str) -> ConversationManager:
    """One ConversationManager per user_id, kept alive across turns so
    conversation history actually accumulates. In a multi-process
    deployment this would move to a shared cache (Redis) keyed the
    same way - the interface wouldn't change."""
    if user_id not in _conversations:
        _conversations[user_id] = ConversationManager(
            user_id, rag=get_rag_engine(), llm=get_llm(), store=get_profile_store()
        )
    return _conversations[user_id]


def resolve_serving_overrides(user_id: str) -> dict:
    """Live ProfileStore entry, if one exists, formatted as the
    goal_id / possessed_skills / experience_level kwargs PathGenerator
    and explain_utils accept. Falls back to empty (meaning: use the
    frozen training-snapshot lookups) for users only known from
    synthetic training data."""
    profile = get_profile_store().get(user_id)
    if not profile.get("goal_id"):
        return {}
    return {
        "goal_id": profile.get("goal_id"),
        "possessed_skills": set(profile.get("skill_ids", [])),
        "experience_level": profile.get("experience_level"),
    }
