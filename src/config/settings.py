"""Centralized, strictly-typed settings loaded from environment variables / .env.

Every other module reads configuration through `get_settings()` rather than
calling `os.environ` directly, so there is exactly one place that knows how
config is sourced.
"""
from functools import lru_cache
from typing import Literal, Optional

from pydantic_settings import BaseSettings, SettingsConfigDict

LLMProvider = Literal["groq", "anthropic", "openai", "google"]
ExecutionMode = Literal["local", "hybrid", "benchmark"]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # GitHub
    github_token: Optional[str] = None

    # LLM providers — only the ones actually selected via llm_provider/classifier_llm_provider
    # need a key; see src/llm/llm_factory.py (critical rule: all LLM calls go through get_llm()).
    groq_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    openai_api_key: Optional[str] = None
    google_api_key: Optional[str] = None

    llm_provider: LLMProvider = "groq"
    llm_model: str = "llama-3.3-70b-versatile"
    classifier_llm_provider: LLMProvider = "groq"
    classifier_llm_model: str = "llama-3.1-8b-instant"

    # Cohere (reranker)
    cohere_api_key: Optional[str] = None

    # Voyage (code embeddings)
    voyage_api_key: Optional[str] = None

    # Execution mode: gates whether code_chunks embeddings use Voyage (hybrid/benchmark)
    # or fall back to local BGE-M3 (local) — see src/ingestion/embedding_agent.py.
    execution_mode: ExecutionMode = "local"

    # PostgreSQL
    postgres_url: str = "postgresql+psycopg://archaeology:archaeology@localhost:5432/archaeology"

    # Neo4j
    neo4j_url: str = "bolt://localhost:7687"
    neo4j_username: str = "neo4j"
    neo4j_password: str = "archaeology_dev_password"

    # Chroma
    chroma_host: str = "localhost"
    chroma_port: int = 8000

    # Redis (answer cache)
    redis_url: str = "redis://localhost:6379"

    # LangSmith / LangChain tracing — enabled from day one per project rules.
    langchain_tracing_v2: bool = True
    langchain_project: str = "code-archaeology-agent"
    langchain_api_key: Optional[str] = None

    # Local working dirs
    repo_clone_dir: str = "./data/repos"

    def apply_langsmith_env(self) -> None:
        """Push tracing config into os.environ so LangChain/LangGraph pick it up.

        Must be called before any LangGraph pipeline code runs (see project rule:
        "LangSmith tracing must be enabled from day one").
        """
        import os

        os.environ["LANGCHAIN_TRACING_V2"] = str(self.langchain_tracing_v2).lower()
        os.environ["LANGCHAIN_PROJECT"] = self.langchain_project
        if self.langchain_api_key:
            os.environ["LANGCHAIN_API_KEY"] = self.langchain_api_key


@lru_cache
def get_settings() -> Settings:
    return Settings()
