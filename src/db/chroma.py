"""Chroma client + collection management — the semantic similarity store.

Chroma answers "WHAT IS SEMANTICALLY SIMILAR to this?". Per critical rule #3,
each repo gets FOUR separate collections rather than one mixed collection:
mixing code, commit messages, PR discussions, and ADRs in one collection
causes retrieval pollution (a query about "why was X rewritten" would be
dominated by whichever doc type has the most rows). See WHAT_TO_DO.MD Phase 3.

Embeddings are computed externally by `src/ingestion/embedding_agent.py`
(Voyage for code, BGE-M3 for text) and passed in explicitly — collections
are created with no embedding_function so Chroma never silently computes its
own embeddings with a different (wrong) model.
"""
import re
from typing import Any, Literal, Optional

import chromadb
from chromadb.api.models.Collection import Collection

from src.config.settings import get_settings

DocType = Literal["code_chunks", "commit_messages", "pr_discussions", "adr_rfcs"]

DOC_TYPES: tuple[DocType, ...] = ("code_chunks", "commit_messages", "pr_discussions", "adr_rfcs")

_client = None


def get_client():
    global _client
    if _client is None:
        settings = get_settings()
        _client = chromadb.HttpClient(
            host=settings.chroma_host,
            port=settings.chroma_port,
            settings=chromadb.Settings(anonymized_telemetry=False),
        )
    return _client


def sanitize_repo_id(repo_id: str) -> str:
    """Chroma collection names must be alphanumeric/underscore/hyphen only.

    "owner/repo" -> "owner__repo".
    """
    return re.sub(r"[^a-zA-Z0-9_-]", "__", repo_id)


def collection_name(repo_id: str, doc_type: DocType) -> str:
    return f"{sanitize_repo_id(repo_id)}_{doc_type}"


def get_or_create_collection(repo_id: str, doc_type: DocType) -> Collection:
    if doc_type not in DOC_TYPES:
        raise ValueError(f"Unknown doc_type {doc_type!r}, must be one of {DOC_TYPES}")
    return get_client().get_or_create_collection(
        name=collection_name(repo_id, doc_type),
        metadata={"repo_id": repo_id, "doc_type": doc_type},
    )


def upsert(
    repo_id: str,
    doc_type: DocType,
    ids: list[str],
    embeddings: list[list[float]],
    documents: list[str],
    metadatas: list[dict[str, Any]],
) -> None:
    collection = get_or_create_collection(repo_id, doc_type)
    collection.upsert(ids=ids, embeddings=embeddings, documents=documents, metadatas=metadatas)


def query(
    repo_id: str,
    doc_type: DocType,
    query_embedding: list[float],
    n_results: int = 40,
    where: Optional[dict[str, Any]] = None,
) -> dict:
    collection = get_or_create_collection(repo_id, doc_type)
    return collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        where=where,
    )


def heartbeat() -> int:
    return get_client().heartbeat()


def get_existing_ids(repo_id: str, doc_type: DocType) -> set[str]:
    """All document IDs already embedded for this collection — lets a long
    embedding run resume after an interruption without redoing already-done
    work (embedding_agent.py has no separate checkpoint of its own; Chroma's
    own contents ARE the checkpoint). `include=[]` skips fetching documents/
    embeddings/metadatas, which matters here since collections can hold tens
    of thousands of rows.
    """
    collection = get_or_create_collection(repo_id, doc_type)
    result = collection.get(include=[])
    return set(result["ids"])
