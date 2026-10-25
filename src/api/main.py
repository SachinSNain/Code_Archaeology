"""Thin FastAPI layer exposing the query endpoint (WHAT_TO_DO.MD tech stack:
"FastAPI — thin API layer exposing the query endpoint").

Also hosts the GitHub webhook that closes the Phase 5b human-approval loop:
when a tracked Living-Docs PR is merged, the proposed explanation gets
ingested into `{repo_id}_adr_rfcs` — see src/graph/nodes/docs_writer.py for
why this only happens post-merge, never on PR creation.
"""
from typing import Optional

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel

from src.cache.answer_cache import get_answer_cache
from src.graph.nodes.docs_writer import on_pr_merged
from src.graph.pipeline import run_query
from src.graph.state import Citation, EvolutionEvent

app = FastAPI(title="Code Archaeology Agent")


class QueryRequest(BaseModel):
    repo_id: str
    query: str
    target_chunk_id: Optional[str] = None
    create_docs_pr: bool = False


class QueryResponse(BaseModel):
    answer: str
    citations: list[Citation]
    provenance_class: Optional[str] = None
    provenance_confidence: Optional[float] = None
    evolution_timeline: Optional[list[EvolutionEvent]] = None
    draft_pr_url: Optional[str] = None
    query_type: Optional[str] = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(request: QueryRequest) -> QueryResponse:
    cache = get_answer_cache()
    # target_chunk_id is folded into the cache key text: the same query string
    # against a different target function is not the same question.
    cache_query = f"{request.query}::{request.target_chunk_id or ''}"

    # create_docs_pr has a side effect (opens a GitHub PR) that must happen on
    # every request it's set on, so a cache hit is not allowed to short-circuit it.
    if not request.create_docs_pr:
        cached = cache.get(request.repo_id, cache_query)
        if cached is not None:
            return QueryResponse(**cached)

    result = run_query(
        repo_id=request.repo_id,
        query=request.query,
        target_chunk_id=request.target_chunk_id,
        create_docs_pr=request.create_docs_pr,
    )
    response = QueryResponse(
        answer=result.get("answer", ""),
        citations=result.get("citations") or [],
        provenance_class=result.get("provenance_class"),
        provenance_confidence=result.get("provenance_confidence"),
        evolution_timeline=result.get("evolution_timeline"),
        draft_pr_url=result.get("draft_pr_url"),
        query_type=result.get("query_type"),
    )

    if not request.create_docs_pr:
        cache.set(request.repo_id, cache_query, response.model_dump())
    return response


@app.post("/webhooks/github")
async def github_webhook(request: Request) -> dict:
    event = request.headers.get("X-GitHub-Event")
    if event != "pull_request":
        return {"skipped": True}

    payload = await request.json()
    pr = payload.get("pull_request", {})
    if payload.get("action") != "closed" or not pr.get("merged"):
        return {"skipped": True}

    repo_id = payload.get("repository", {}).get("full_name")
    pr_number = pr.get("number")
    if not repo_id or not pr_number:
        raise HTTPException(status_code=400, detail="Missing repository or PR number in payload")

    ingested = on_pr_merged(repo_id, pr_number)
    return {"ingested": ingested}
