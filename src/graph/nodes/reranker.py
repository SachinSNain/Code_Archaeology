"""Phase 4c — Context Ranker and Deduplicator node.

Critical rule: reranking is not optional. Cosine similarity from Chroma's
bi-encoder retrieval is "close enough" relevance; Cohere's cross-encoder
reads the query and each candidate together and produces true relevance,
which is what keeps noise out of the synthesis prompt.

Three steps, in order:
  1. Cohere Rerank over ALL retrieved material (chroma + neo4j + postgres,
     the latter two converted to synthetic text so everything is ranked on
     one scale).
  2. MMR dedup to drop near-identical chunks (e.g. the same PR discussion
     chunk surfacing from two different splits).
  3. Trim to 80% of the LLM's context window, reserving 20% for the system
     prompt + generated answer.
"""
from functools import lru_cache

import cohere
import numpy as np
import tiktoken

from src.config.settings import get_settings
from src.graph.state import ArchaeologyState, CommitMetadata, ProvenanceChain, RetrievedChunk

COHERE_RERANK_MODEL = "rerank-english-v3.0"
COHERE_RERANK_TOP_N = 40
MMR_LAMBDA = 0.5
MMR_TOP_K = 20
# 128k matches both gpt-4o and Groq's llama-3.3-70b-versatile, so this budget holds
# regardless of LLM_PROVIDER. tiktoken has no Llama/Groq encoding, so o200k_base is
# used as an approximate token-count proxy for budgeting purposes only (not exact
# for non-OpenAI models, but close enough to bound context size sanely).
LLM_CONTEXT_WINDOW_TOKENS = 128_000
CONTEXT_BUDGET_RATIO = 0.8  # reserve 20% for system prompt + generated answer


def _provenance_to_text(chain: ProvenanceChain) -> str:
    parts = [f"Function {chain.function_chunk_id}"]
    if chain.introduced_by_sha:
        parts.append(f"introduced by commit {chain.introduced_by_sha}")
    if chain.last_modified_by_sha:
        parts.append(f"last modified by commit {chain.last_modified_by_sha}")
    if chain.pr_number:
        parts.append(f"part of PR #{chain.pr_number}")
    if chain.issue_number:
        parts.append(f"addresses issue #{chain.issue_number}")
    if chain.adr_doc_id:
        parts.append(f"informed by ADR {chain.adr_doc_id}")
    if len(parts) == 1:
        parts.append("has no resolvable provenance chain")
    return ", ".join(parts) + "."


def _commit_to_text(commit: CommitMetadata) -> str:
    files = ", ".join(f["path"] for f in commit.files_changed[:5]) if commit.files_changed else "unknown files"
    return (
        f"Commit {commit.sha} by {commit.author_name or 'unknown'} "
        f"({commit.timestamp.isoformat() if commit.timestamp else 'unknown date'}): "
        f"{commit.message or ''}\nTouched: {files}"
    )


def _structured_to_chunks(state: ArchaeologyState) -> list[RetrievedChunk]:
    chunks = []
    for chain in state.get("neo4j_results") or []:
        chunks.append(
            RetrievedChunk(
                source_type="provenance_chain",
                source_id=chain.function_chunk_id,
                text=_provenance_to_text(chain),
                metadata={"pr_number": chain.pr_number, "issue_number": chain.issue_number},
            )
        )
    for commit in state.get("postgres_results") or []:
        chunks.append(
            RetrievedChunk(
                source_type="commit",
                source_id=commit.sha,
                text=_commit_to_text(commit),
                metadata={"author": commit.author_name, "timestamp": str(commit.timestamp)},
            )
        )
    return chunks


def cohere_rerank(query: str, chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    if not chunks:
        return []
    settings = get_settings()
    if not settings.cohere_api_key:
        raise RuntimeError(
            "COHERE_API_KEY is not set. Reranking is not optional in this pipeline (critical rule #7) — "
            "without it, retrieval noise passes straight into synthesis."
        )
    client = cohere.Client(api_key=settings.cohere_api_key)
    docs = [c.text for c in chunks]
    result = client.rerank(
        query=query, documents=docs, top_n=min(COHERE_RERANK_TOP_N, len(docs)), model=COHERE_RERANK_MODEL
    )
    reranked = []
    for r in result.results:
        chunk = chunks[r.index].model_copy(update={"score": r.relevance_score})
        reranked.append(chunk)
    return reranked


@lru_cache(maxsize=1)
def _dedup_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


def mmr_dedup(query: str, chunks: list[RetrievedChunk], k: int = MMR_TOP_K, lambda_mult: float = MMR_LAMBDA) -> list[RetrievedChunk]:
    """Maximal Marginal Relevance over a cheap local embedding space — chunks
    already carry a cross-encoder relevance score from Cohere; MMR here only
    needs enough semantic signal to detect near-duplicates, not to rank.
    """
    if len(chunks) <= 1:
        return chunks

    from langchain_core.vectorstores.utils import maximal_marginal_relevance

    model = _dedup_model()
    texts = [c.text for c in chunks]
    doc_embeddings = model.encode(texts, normalize_embeddings=True)
    query_embedding = model.encode([query], normalize_embeddings=True)[0]

    selected_indices = maximal_marginal_relevance(
        np.array(query_embedding), list(doc_embeddings), k=min(k, len(chunks)), lambda_mult=lambda_mult
    )
    return [chunks[i] for i in selected_indices]


def _trim_to_context_budget(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    encoding = tiktoken.get_encoding("o200k_base")  # approximate proxy tokenizer, see module docstring
    budget = int(LLM_CONTEXT_WINDOW_TOKENS * CONTEXT_BUDGET_RATIO)

    kept, used = [], 0
    for chunk in chunks:
        n_tokens = len(encoding.encode(chunk.text))
        if used + n_tokens > budget:
            break
        kept.append(chunk)
        used += n_tokens
    return kept


def context_ranker_node(state: ArchaeologyState) -> dict:
    all_chunks = list(state.get("chroma_results") or []) + _structured_to_chunks(state)
    if not all_chunks:
        return {"ranked_context": []}

    reranked = cohere_rerank(state["query"], all_chunks)
    deduped = mmr_dedup(state["query"], reranked)
    trimmed = _trim_to_context_budget(deduped)
    return {"ranked_context": trimmed}
