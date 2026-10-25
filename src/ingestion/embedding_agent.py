"""Phase 3 — EmbeddingAgent.

Embeds ingested data into four Chroma collections per repo (critical rule:
never one mixed collection — see src/db/chroma.py docstring for why).

Embedding model split (validated in Phase 1.5's offline benchmark, hard-wired
here per WHAT_TO_DO.MD Phase 3 table):
  - code_chunks:      Voyage Code Embeddings (voyage-code-2) — code-specific model
  - commit_messages:  BGE-M3, one embedding per commit
  - pr_discussions:   BGE-M3, PR body + review comments, 512 tokens / 50 overlap
  - adr_rfcs:         BGE-M3, full ADR/RFC text, 512 tokens / 50 overlap

EXECUTION_MODE gates the code_chunks model specifically: "local" mode falls
back to BGE-M3 for code too (zero API cost, no VOYAGE_API_KEY needed), while
"hybrid"/"benchmark" use Voyage as specified above. commit_messages/
pr_discussions/adr_rfcs always use BGE-M3 regardless of mode — it has no
hosted API to begin with, so there's no cost trade-off to make there.

BGE-M3 has no hosted API (unlike Voyage/Cohere/OpenAI) so it runs locally via
sentence-transformers; the model is loaded once and reused (~2GB, several
seconds to load) rather than per-batch.

Usage:
    python -m src.ingestion.embedding_agent tiangolo/fastapi [--collections code_chunks,commit_messages]
"""
import argparse
import sys
import time
from datetime import datetime
from functools import lru_cache
from typing import Literal, Optional

import voyageai
from langchain_text_splitters import TokenTextSplitter
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tqdm import tqdm

from src.config.settings import get_settings
from src.db import chroma, postgres

VOYAGE_MODEL = "voyage-code-2"
BGE_MODEL = "BAAI/bge-m3"
BGE_BATCH_SIZE = 32
TEXT_CHUNK_TOKENS = 512
TEXT_CHUNK_OVERLAP = 50

# Voyage accounts with no payment method on file are capped at 3 RPM / 10K TPM (confirmed
# via the API's own error message, not documented anywhere obvious). Both bounds are real:
# a batch that's too large in *tokens* gets rejected regardless of item count, and even
# small batches are capped at 3/minute. So ingestion batches by an estimated token budget
# (not a fixed item count) and separately enforces a minimum spacing between requests.
VOYAGE_FREE_TIER_RPM = 3
VOYAGE_FREE_TIER_TPM = 10_000
VOYAGE_MAX_TOKENS_PER_BATCH = 8_000  # headroom under the 10K TPM cap
VOYAGE_MIN_REQUEST_INTERVAL_S = 60 / VOYAGE_FREE_TIER_RPM

QueryDocType = Literal["code_chunks", "commit_messages", "pr_discussions", "adr_rfcs"]


@lru_cache(maxsize=1)
def _bge_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(BGE_MODEL)


@retry(
    # Voyage's free tier has a low requests-per-minute cap; a rate limit hit needs a wait
    # long enough for that window to roll over, not a quick network-blip-style backoff.
    retry=retry_if_exception_type(voyageai.error.RateLimitError),
    wait=wait_exponential(multiplier=2, min=20, max=90),
    stop=stop_after_attempt(10),
)
def _embed_code_voyage(texts: list[str], input_type: str = "document") -> list[list[float]]:
    settings = get_settings()
    if not settings.voyage_api_key:
        raise RuntimeError("VOYAGE_API_KEY is not set — required for code_chunks embedding (voyage-code-2).")
    client = voyageai.Client(api_key=settings.voyage_api_key)
    result = client.embed(texts, model=VOYAGE_MODEL, input_type=input_type)
    return result.embeddings


def embed_text(texts: list[str]) -> list[list[float]]:
    """BGE-M3, used for commit_messages / pr_discussions / adr_rfcs (and for
    code_chunks too when EXECUTION_MODE=local)."""
    embeddings = _bge_model().encode(texts, normalize_embeddings=True, show_progress_bar=False)
    return embeddings.tolist()


def embed_code(texts: list[str], input_type: str = "document") -> list[list[float]]:
    """Code embeddings, routed by EXECUTION_MODE: Voyage in hybrid/benchmark
    mode (code-specific model, per the Phase 1.5 benchmark), local BGE-M3 in
    local mode (zero API cost — see WHAT_TO_DO.MD "Three execution modes").
    """
    if get_settings().execution_mode == "local":
        return embed_text(texts)
    return _embed_code_voyage(texts, input_type=input_type)


def embed_query(text: str, doc_type: QueryDocType) -> list[float]:
    """Single-text embedding at query time (Phase 4b retrieval nodes)."""
    if doc_type == "code_chunks":
        return embed_code([text], input_type="query")[0]
    return embed_text([text])[0]


def _splitter() -> TokenTextSplitter:
    return TokenTextSplitter(chunk_size=TEXT_CHUNK_TOKENS, chunk_overlap=TEXT_CHUNK_OVERLAP)


def _batched(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _estimate_tokens(text: str) -> int:
    # Voyage has no public tiktoken-compatible encoding; ~4 chars/token is the standard
    # rough estimate for code/English text and only needs to be safely conservative here,
    # not exact — it's a budget guard against the 10K TPM cap, not a billing calculation.
    return max(1, len(text) // 4)


def _truncate_for_voyage(text: str) -> str:
    """A tiny minority of AST chunks (huge classes/functions the parser doesn't
    sub-chunk) are large enough to blow the 10K TPM cap on their own — up to
    ~29K estimated tokens observed in this corpus. Cap any single chunk well
    under the per-batch budget so one oversized chunk can never make a batch
    of one still fail; the stored chunk_text in Postgres is untouched, only
    what's sent to the embedding API is capped.
    """
    max_chars = (VOYAGE_MAX_TOKENS_PER_BATCH // 2) * 4  # leaves room for other chunks to share the batch
    return text if len(text) <= max_chars else text[:max_chars]


def _token_aware_batches(rows: list, max_tokens: int) -> list[list[tuple]]:
    """Groups (row, text) pairs into batches that stay under max_tokens
    (estimated), instead of a fixed item count — a fixed count is wrong here
    because a handful of oversized chunks can blow the token budget while
    hundreds of small ones would comfortably fit in one request.
    """
    batches: list[list[tuple]] = []
    current: list[tuple] = []
    current_tokens = 0
    for row in rows:
        text = _truncate_for_voyage(row.chunk_text)
        row_tokens = _estimate_tokens(text)
        if current and current_tokens + row_tokens > max_tokens:
            batches.append(current)
            current, current_tokens = [], 0
        current.append((row, text))
        current_tokens += row_tokens
    if current:
        batches.append(current)
    return batches


def embed_code_chunks(repo_id: str, limit: Optional[int] = None) -> int:
    session_factory = postgres.get_session_factory()
    with session_factory() as session:
        rows = session.query(postgres.CodeChunkRecord).filter_by(repo_id=repo_id).all()

    existing = chroma.get_existing_ids(repo_id, "code_chunks")
    rows = [r for r in rows if r.chunk_id not in existing]
    if existing:
        print(f"  resuming: {len(existing)} already embedded, {len(rows)} remaining")

    if limit:
        rows = rows[:limit]
    if not rows:
        print("  nothing left to embed" if existing else "  no code chunks found (run ast_parser first)")
        return 0

    settings = get_settings()
    is_voyage = settings.execution_mode != "local"
    if is_voyage:
        batches = _token_aware_batches(rows, VOYAGE_MAX_TOKENS_PER_BATCH)
    else:
        batches = [[(r, r.chunk_text) for r in group] for group in _batched(rows, BGE_BATCH_SIZE)]

    total = 0
    last_request_time = 0.0
    for batch in tqdm(batches, desc="embed:code_chunks"):
        if is_voyage:
            # Enforce the free tier's 3 RPM ceiling regardless of how fast the previous call returned.
            wait_needed = VOYAGE_MIN_REQUEST_INTERVAL_S - (time.time() - last_request_time)
            if wait_needed > 0:
                time.sleep(wait_needed)
        rows_in_batch = [r for r, _ in batch]
        texts = [t for _, t in batch]
        embeddings = embed_code(texts)
        last_request_time = time.time()
        ids = [r.chunk_id for r in rows_in_batch]
        metadatas = [
            {
                "repo_id": repo_id,
                "doc_type": "code_chunk",
                "source_id": r.chunk_id,
                "file_path": r.file_path,
                "timestamp": "",
                "author": "",
                "language": r.language,
            }
            for r in rows_in_batch
        ]
        chroma.upsert(repo_id, "code_chunks", ids, embeddings, texts, metadatas)
        total += len(batch)
    return total


def embed_commit_messages(repo_id: str, limit: Optional[int] = None) -> int:
    session_factory = postgres.get_session_factory()
    with session_factory() as session:
        rows = session.query(postgres.Commit).filter_by(repo_id=repo_id).all()

    existing = chroma.get_existing_ids(repo_id, "commit_messages")
    rows = [r for r in rows if r.sha not in existing]
    if existing:
        print(f"  resuming: {len(existing)} already embedded, {len(rows)} remaining")

    if limit:
        rows = rows[:limit]
    if not rows:
        print("  nothing left to embed" if existing else "  no commits found (run git_ingestion first)")
        return 0

    total = 0
    for batch in tqdm(list(_batched(rows, BGE_BATCH_SIZE)), desc="embed:commit_messages"):
        texts = [
            f"{c.message or ''}\n\nLinked PR: {c.linked_pr_number or 'none'}\n"
            f"Linked issues: {c.linked_issue_numbers or []}"
            for c in batch
        ]
        embeddings = embed_text(texts)
        ids = [c.sha for c in batch]
        metadatas = [
            {
                "repo_id": repo_id,
                "doc_type": "commit",
                "source_id": c.sha,
                "file_path": "",
                "timestamp": c.timestamp.isoformat() if c.timestamp else "",
                "author": c.author_name or "",
                "language": "",
            }
            for c in batch
        ]
        chroma.upsert(repo_id, "commit_messages", ids, embeddings, texts, metadatas)
        total += len(batch)
    return total


def embed_pr_discussions(repo_id: str, limit: Optional[int] = None) -> int:
    session_factory = postgres.get_session_factory()
    with session_factory() as session:
        rows = session.query(postgres.PullRequest).filter_by(repo_id=repo_id).all()
    if limit:
        rows = rows[:limit]
    if not rows:
        print("  no PRs found (run pr_issue_ingestion first)")
        return 0

    splitter = _splitter()
    doc_ids, doc_texts, doc_metas = [], [], []
    for pr in rows:
        comments_text = "\n\n".join(
            f"[{c.get('author')}]: {c.get('body')}" for c in (pr.review_comments or [])
        )
        full_text = f"{pr.title or ''}\n\n{pr.body or ''}\n\n{comments_text}".strip()
        if not full_text:
            continue
        for i, piece in enumerate(splitter.split_text(full_text)):
            doc_ids.append(f"pr-{pr.pr_number}-{i}")
            doc_texts.append(piece)
            doc_metas.append(
                {
                    "repo_id": repo_id,
                    "doc_type": "pr_discussion",
                    "source_id": str(pr.pr_number),
                    "file_path": "",
                    "timestamp": pr.created_at.isoformat() if pr.created_at else "",
                    "author": pr.author or "",
                    "language": "",
                }
            )

    existing = chroma.get_existing_ids(repo_id, "pr_discussions")
    if existing:
        keep = [i for i, doc_id in enumerate(doc_ids) if doc_id not in existing]
        print(f"  resuming: {len(doc_ids) - len(keep)} already embedded, {len(keep)} remaining")
        doc_ids = [doc_ids[i] for i in keep]
        doc_texts = [doc_texts[i] for i in keep]
        doc_metas = [doc_metas[i] for i in keep]

    total = 0
    for id_batch, text_batch, meta_batch in zip(
        _batched(doc_ids, BGE_BATCH_SIZE), _batched(doc_texts, BGE_BATCH_SIZE), _batched(doc_metas, BGE_BATCH_SIZE)
    ):
        embeddings = embed_text(text_batch)
        chroma.upsert(repo_id, "pr_discussions", id_batch, embeddings, text_batch, meta_batch)
        total += len(id_batch)
    return total


def embed_adr_rfcs(repo_id: str, limit: Optional[int] = None) -> int:
    session_factory = postgres.get_session_factory()
    with session_factory() as session:
        rows = session.query(postgres.ADRDocument).filter_by(repo_id=repo_id).all()
    if limit:
        rows = rows[:limit]
    if not rows:
        print("  no ADR/RFC documents found (run adr_ingestion first)")
        return 0

    splitter = _splitter()
    doc_ids, doc_texts, doc_metas = [], [], []
    for doc in rows:
        if not (doc.body or "").strip():
            continue
        for i, piece in enumerate(splitter.split_text(doc.body)):
            doc_ids.append(f"adr-{doc.doc_id}-{i}")
            doc_texts.append(piece)
            doc_metas.append(
                {
                    "repo_id": repo_id,
                    "doc_type": doc.doc_type,  # adr|rfc|design_proposal|changelog_entry|slack_thread|github_discussion
                    "source_id": doc.doc_id,
                    "file_path": doc.source_path or "",
                    "timestamp": doc.created_at.isoformat() if doc.created_at else "",
                    "author": "",
                    "language": "",
                }
            )

    existing = chroma.get_existing_ids(repo_id, "adr_rfcs")
    if existing:
        keep = [i for i, doc_id in enumerate(doc_ids) if doc_id not in existing]
        print(f"  resuming: {len(doc_ids) - len(keep)} already embedded, {len(keep)} remaining")
        doc_ids = [doc_ids[i] for i in keep]
        doc_texts = [doc_texts[i] for i in keep]
        doc_metas = [doc_metas[i] for i in keep]

    total = 0
    for id_batch, text_batch, meta_batch in zip(
        _batched(doc_ids, BGE_BATCH_SIZE), _batched(doc_texts, BGE_BATCH_SIZE), _batched(doc_metas, BGE_BATCH_SIZE)
    ):
        embeddings = embed_text(text_batch)
        chroma.upsert(repo_id, "adr_rfcs", id_batch, embeddings, text_batch, meta_batch)
        total += len(id_batch)
    return total


EMBEDDERS = {
    "code_chunks": embed_code_chunks,
    "commit_messages": embed_commit_messages,
    "pr_discussions": embed_pr_discussions,
    "adr_rfcs": embed_adr_rfcs,
}


def embed_repo(repo_id: str, collections: Optional[list[str]] = None, limit: Optional[int] = None) -> dict[str, int]:
    targets = collections or list(EMBEDDERS.keys())
    results = {}
    for name in targets:
        print(f"== {name} ==")
        results[name] = EMBEDDERS[name](repo_id, limit=limit)
        print(f"  embedded {results[name]} document(s)")
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id")
    parser.add_argument("--collections", type=str, default=None, help="Comma-separated subset of collections")
    parser.add_argument("--limit", type=int, default=None, help="Cap rows embedded per collection (testing only)")
    args = parser.parse_args()

    collections = args.collections.split(",") if args.collections else None
    embed_repo(args.repo_id, collections=collections, limit=args.limit)


if __name__ == "__main__":
    main()
