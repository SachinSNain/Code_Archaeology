"""Redis-backed answer cache, keyed on (repo_id, query).

Critical rule: the cache is checked at the FastAPI layer BEFORE the
LangGraph pipeline is invoked, not inside the graph — a cache hit must skip
the pipeline entirely (zero LLM/embedding/rerank cost), not run it and
discard the result. See src/api/main.py.

Two things this buys: demos feel instant on repeated questions, and testing
doesn't re-pay for the same LLM call every time the same query is re-run.
"""
import hashlib
import json
from functools import lru_cache
from typing import Optional

from redis import Redis

from src.config.settings import get_settings

DEFAULT_TTL_SECONDS = 60 * 60 * 24 * 7  # 7 days


class AnswerCache:
    def __init__(self, redis_url: str, ttl: int = DEFAULT_TTL_SECONDS):
        self.client = Redis.from_url(redis_url)
        self.ttl = ttl

    def cache_key(self, repo_id: str, query: str) -> str:
        h = hashlib.sha256(f"{repo_id}:{query.strip().lower()}".encode()).hexdigest()
        return f"answer:{h}"

    def get(self, repo_id: str, query: str) -> Optional[dict]:
        val = self.client.get(self.cache_key(repo_id, query))
        return json.loads(val) if val else None

    def set(self, repo_id: str, query: str, answer: dict) -> None:
        self.client.setex(self.cache_key(repo_id, query), self.ttl, json.dumps(answer, default=str))


@lru_cache(maxsize=1)
def get_answer_cache() -> AnswerCache:
    return AnswerCache(get_settings().redis_url)
