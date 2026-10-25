"""Phase 4b — Multi-Source Retrieval nodes.

Three independent nodes (chroma_retrieval, neo4j_retrieval,
postgres_retrieval) fan out from the query classifier and fan back in to the
reranker. This is the core reason the pipeline uses LangGraph instead of a
linear LangChain chain: sequential retrieval at ~2s per store is 6s total,
parallel is ~2s total. Each node writes to its own dedicated state key
(chroma_results / neo4j_results / postgres_results), so LangGraph needs no
custom reducer to merge concurrent branch outputs — there's no shared key to
conflict on.
"""
from datetime import datetime, timedelta
from typing import Optional

from sqlalchemy import text as sql_text

from src.db import chroma, neo4j as neo4j_db, postgres
from src.graph.state import ArchaeologyState, CommitMetadata, ProvenanceChain, RetrievedChunk

CHROMA_TOP_K = 40

# Which Chroma collections are relevant per query type (Phase 4b spec).
COLLECTIONS_BY_QUERY_TYPE: dict[str, list[chroma.DocType]] = {
    "what": ["code_chunks"],
    "similar": ["code_chunks"],
    "who": [],
    "when": [],
    "why": ["code_chunks", "commit_messages", "pr_discussions", "adr_rfcs"],
    "evolution": ["code_chunks", "commit_messages", "pr_discussions", "adr_rfcs"],
}


def _lookup_chunk_text(repo_id: str, chunk_id: str) -> Optional[str]:
    session_factory = postgres.get_session_factory()
    with session_factory() as session:
        row = (
            session.query(postgres.CodeChunkRecord)
            .filter_by(repo_id=repo_id, chunk_id=chunk_id)
            .one_or_none()
        )
        return row.chunk_text if row else None


def chroma_retrieval_node(state: ArchaeologyState) -> dict:
    from src.ingestion.embedding_agent import embed_query  # local import: avoids loading BGE-M3/Voyage clients for nodes that never call this

    repo_id = state["repo_id"]
    query_type = state.get("query_type", "why")
    collections = COLLECTIONS_BY_QUERY_TYPE.get(query_type, ["code_chunks"])

    query_text = state["query"]
    target_chunk_id = state.get("target_chunk_id")
    code_query_text = query_text
    if target_chunk_id:
        chunk_text = _lookup_chunk_text(repo_id, target_chunk_id)
        if chunk_text:
            code_query_text = f"{query_text}\n\n{chunk_text}"

    results: list[RetrievedChunk] = []
    for collection in collections:
        query_embedding = embed_query(
            code_query_text if collection == "code_chunks" else query_text, collection
        )
        raw = chroma.query(repo_id, collection, query_embedding, n_results=CHROMA_TOP_K, where={"repo_id": repo_id})
        ids = raw.get("ids", [[]])[0]
        docs = raw.get("documents", [[]])[0]
        metas = raw.get("metadatas", [[]])[0]
        dists = raw.get("distances", [[]])[0]
        for doc_id, doc_text, meta, dist in zip(ids, docs, metas, dists):
            results.append(
                RetrievedChunk(
                    source_type=meta.get("doc_type", collection),
                    source_id=meta.get("source_id", doc_id),
                    text=doc_text,
                    metadata=meta,
                    score=1.0 - dist if dist is not None else None,  # cosine distance -> similarity
                )
            )

    return {"chroma_results": results}


def find_target_chunk_via_chroma(repo_id: str, query_text: str) -> Optional[str]:
    from src.ingestion.embedding_agent import embed_query

    embedding = embed_query(query_text, "code_chunks")
    raw = chroma.query(repo_id, "code_chunks", embedding, n_results=1, where={"repo_id": repo_id})
    metas = raw.get("metadatas", [[]])[0]
    return metas[0].get("source_id") if metas else None


def neo4j_retrieval_node(state: ArchaeologyState) -> dict:
    repo_id = state["repo_id"]
    target_chunk_id = state.get("target_chunk_id")

    resolved_here = False
    if not target_chunk_id:
        target_chunk_id = find_target_chunk_via_chroma(repo_id, state["query"])
        resolved_here = True

    if not target_chunk_id:
        return {"neo4j_results": []}

    records = neo4j_db.fetch_provenance_chain(repo_id, target_chunk_id)
    chains: list[ProvenanceChain] = []
    for rec in records:
        introduced = rec.get("introduced") or {}
        modified = rec.get("modified") or {}
        pr = rec.get("pr") or {}
        issue = rec.get("issue") or {}
        adr = rec.get("adr") or {}
        chains.append(
            ProvenanceChain(
                function_chunk_id=target_chunk_id,
                introduced_by_sha=introduced.get("sha"),
                last_modified_by_sha=modified.get("sha"),
                pr_number=pr.get("pr_number"),
                issue_number=issue.get("issue_number"),
                adr_doc_id=adr.get("doc_id"),
                raw_path_records=[rec],
            )
        )
    if not chains:
        # No provenance chain at all is itself meaningful — Phase 4d reads this
        # sparsity as a signal (e.g. "orphaned"), so we still return a stub.
        chains.append(ProvenanceChain(function_chunk_id=target_chunk_id))

    update: dict = {"neo4j_results": chains}
    if resolved_here:
        update["target_chunk_id"] = target_chunk_id
    return update


def postgres_retrieval_node(state: ArchaeologyState) -> dict:
    repo_id = state["repo_id"]
    target_chunk_id = state.get("target_chunk_id")
    session_factory = postgres.get_session_factory()

    file_path = None
    if target_chunk_id:
        with session_factory() as session:
            chunk_row = (
                session.query(postgres.CodeChunkRecord)
                .filter_by(repo_id=repo_id, chunk_id=target_chunk_id)
                .one_or_none()
            )
            file_path = chunk_row.file_path if chunk_row else None

    with session_factory() as session:
        if file_path:
            rows = session.execute(
                sql_text(
                    """
                    SELECT sha, message, author_name, timestamp, linked_pr_number,
                           linked_issue_numbers, files_changed
                    FROM commits
                    WHERE repo_id = :repo_id
                      AND EXISTS (
                        SELECT 1 FROM jsonb_array_elements(files_changed) elem
                        WHERE elem->>'path' = :file_path
                      )
                    ORDER BY timestamp DESC
                    LIMIT 50
                    """
                ),
                {"repo_id": repo_id, "file_path": file_path},
            ).mappings().all()
        else:
            cutoff = datetime.utcnow() - timedelta(days=730)
            rows = session.execute(
                sql_text(
                    """
                    SELECT sha, message, author_name, timestamp, linked_pr_number,
                           linked_issue_numbers, files_changed
                    FROM commits
                    WHERE repo_id = :repo_id AND timestamp >= :cutoff
                    ORDER BY timestamp DESC
                    LIMIT 50
                    """
                ),
                {"repo_id": repo_id, "cutoff": cutoff},
            ).mappings().all()

    results = [
        CommitMetadata(
            sha=r["sha"],
            message=r["message"],
            author_name=r["author_name"],
            timestamp=r["timestamp"],
            linked_pr_number=r["linked_pr_number"],
            linked_issue_numbers=r["linked_issue_numbers"] or [],
            files_changed=r["files_changed"] or [],
        )
        for r in rows
    ]
    return {"postgres_results": results}


# Maps the Phase 4a strategy label to the concrete set of node names LangGraph
# should fan out to (see pipeline.py). Kept separate from
# query_classifier.route_after_classification, which documents the strategy
# labels themselves per WHAT_TO_DO.MD's literal routing function.
STRATEGY_TO_NODES: dict[str, list[str]] = {
    "chroma_only": ["chroma_retrieval"],
    "postgres_only": ["postgres_retrieval"],
    "neo4j_postgres": ["neo4j_retrieval", "postgres_retrieval"],
    "all_parallel": ["chroma_retrieval", "neo4j_retrieval", "postgres_retrieval"],
}


def route_to_retrieval_nodes(state: ArchaeologyState) -> list[str]:
    return STRATEGY_TO_NODES[state["retrieval_strategy"]]
