"""Phase 1.5 — Offline embedding benchmark (one-time experiment, not a runtime component).

Per WHAT_TO_DO.MD build order item 11, this runs AFTER the full pipeline
works, using real ingested chunks — there's no point benchmarking retrieval
against a synthetic corpus. Output is a markdown table (docs/embedding_benchmark.md)
that becomes a hard-wired decision for Phase 3 (src/ingestion/embedding_agent.py),
not a runtime code path.

Candidate models (see WHAT_TO_DO.MD Phase 1.5):
    - OpenAI text-embedding-3-large
    - CodeBERT (microsoft/codebert-base)     — mean-pooled last hidden state
    - BGE-M3 (BAAI/bge-m3)
    - Jina Embeddings v3
    - Voyage Code Embeddings (voyage-code-2)

Metrics: Recall@5, Precision@5, MRR, embedding latency, storage size — measured
against a hand-built (query, expected_chunk_id) label set.

Usage:
    python -m src.evaluation.benchmark_embeddings --queries src/evaluation/embedding_benchmark_queries.json
"""
import argparse
import json
import time
from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np

from src.config.settings import get_settings
from src.db import postgres

QUERIES_PATH = Path(__file__).parent / "embedding_benchmark_queries.json"
OUTPUT_PATH = Path(__file__).resolve().parents[2] / "docs" / "embedding_benchmark.md"
TOP_K = 5


@lru_cache(maxsize=1)
def _openai_client():
    from openai import OpenAI

    return OpenAI(api_key=get_settings().openai_api_key)


def embed_openai(texts: list[str]) -> list[list[float]]:
    resp = _openai_client().embeddings.create(model="text-embedding-3-large", input=texts)
    return [d.embedding for d in resp.data]


@lru_cache(maxsize=1)
def _codebert():
    import torch
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
    model = AutoModel.from_pretrained("microsoft/codebert-base")
    model.eval()
    return tokenizer, model


def embed_codebert(texts: list[str]) -> list[list[float]]:
    import torch

    tokenizer, model = _codebert()
    embeddings = []
    with torch.no_grad():
        for text in texts:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
            outputs = model(**inputs)
            mean_pooled = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
            embeddings.append(mean_pooled.tolist())
    return embeddings


@lru_cache(maxsize=1)
def _bge_m3():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("BAAI/bge-m3")


def embed_bge_m3(texts: list[str]) -> list[list[float]]:
    return _bge_m3().encode(texts, normalize_embeddings=True).tolist()


@lru_cache(maxsize=1)
def _jina_v3():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("jinaai/jina-embeddings-v3", trust_remote_code=True)


def embed_jina(texts: list[str]) -> list[list[float]]:
    return _jina_v3().encode(texts, normalize_embeddings=True).tolist()


def embed_voyage(texts: list[str]) -> list[list[float]]:
    import voyageai

    client = voyageai.Client(api_key=get_settings().voyage_api_key)
    return client.embed(texts, model="voyage-code-2", input_type="document").embeddings


CANDIDATE_MODELS: dict[str, Callable[[list[str]], list[list[float]]]] = {
    "openai-text-embedding-3-large": embed_openai,
    "codebert-base": embed_codebert,
    "bge-m3": embed_bge_m3,
    "jina-embeddings-v3": embed_jina,
    "voyage-code-2": embed_voyage,
}


def _load_chunks(repo_id: str) -> list[postgres.CodeChunkRecord]:
    session_factory = postgres.get_session_factory()
    with session_factory() as session:
        return session.query(postgres.CodeChunkRecord).filter_by(repo_id=repo_id).all()


def _cosine_topk(query_vec: np.ndarray, corpus: np.ndarray, k: int) -> list[int]:
    query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
    corpus_norm = corpus / (np.linalg.norm(corpus, axis=1, keepdims=True) + 1e-8)
    scores = corpus_norm @ query_norm
    return list(np.argsort(-scores)[:k])


def benchmark_model(name: str, embed_fn, chunks: list[postgres.CodeChunkRecord], queries: list[dict]) -> dict:
    texts = [c.chunk_text for c in chunks]
    chunk_ids = [c.chunk_id for c in chunks]

    t0 = time.time()
    corpus_embeddings = np.array(embed_fn(texts))
    corpus_latency = time.time() - t0

    hits_at_5, reciprocal_ranks, query_latencies = [], [], []
    for q in queries:
        t0 = time.time()
        query_vec = np.array(embed_fn([q["query"]])[0])
        query_latencies.append(time.time() - t0)

        top_k_idx = _cosine_topk(query_vec, corpus_embeddings, TOP_K)
        top_k_ids = [chunk_ids[i] for i in top_k_idx]

        hit = q["expected_chunk_id"] in top_k_ids
        hits_at_5.append(1 if hit else 0)
        reciprocal_ranks.append(1 / (top_k_ids.index(q["expected_chunk_id"]) + 1) if hit else 0)

    storage_bytes = corpus_embeddings.nbytes

    return {
        "model": name,
        "recall_at_5": float(np.mean(hits_at_5)),
        "precision_at_5": float(np.mean(hits_at_5)) / TOP_K,
        "mrr": float(np.mean(reciprocal_ranks)),
        "avg_query_latency_ms": float(np.mean(query_latencies)) * 1000,
        "corpus_embedding_time_s": corpus_latency,
        "storage_mb": storage_bytes / (1024 * 1024),
        "embedding_dim": corpus_embeddings.shape[1],
    }


def write_report(results: list[dict]) -> None:
    lines = [
        "# Embedding Benchmark Results",
        "",
        "Offline comparison run per WHAT_TO_DO.MD Phase 1.5. Recall@5/MRR measured against a hand-built "
        "(query, expected_chunk) label set on real ingested code chunks. The winning model per collection "
        "type is hard-wired into `src/ingestion/embedding_agent.py` (Phase 3).",
        "",
        "| Model | Recall@5 | Precision@5 | MRR | Avg query latency (ms) | Storage (MB) | Dim |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['model']} | {r['recall_at_5']:.3f} | {r['precision_at_5']:.3f} | {r['mrr']:.3f} | "
            f"{r['avg_query_latency_ms']:.1f} | {r['storage_mb']:.1f} | {r['embedding_dim']} |"
        )
    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id")
    parser.add_argument("--queries", type=Path, default=QUERIES_PATH)
    parser.add_argument("--models", type=str, default=None, help="Comma-separated subset of CANDIDATE_MODELS")
    args = parser.parse_args()

    queries = json.loads(args.queries.read_text(encoding="utf-8"))
    chunks = _load_chunks(args.repo_id)
    if not chunks:
        raise RuntimeError(f"No code chunks found for {args.repo_id} — run ast_parser first.")

    model_names = args.models.split(",") if args.models else list(CANDIDATE_MODELS.keys())
    results = [benchmark_model(name, CANDIDATE_MODELS[name], chunks, queries) for name in model_names]

    for r in results:
        print(r)
    write_report(results)


if __name__ == "__main__":
    main()
