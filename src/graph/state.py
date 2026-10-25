"""ArchaeologyState — the single state object that flows through every
LangGraph node in the query-time pipeline (Phase 4/5).

Every node receives the full state and returns a partial update (LangGraph
merges dict updates into state); nodes must not have side effects beyond
state updates and database writes (critical rule — keeps the graph debuggable
via LangSmith traces, since a node's effect is fully visible in its diff).
"""
from datetime import datetime
from typing import Any, Literal, Optional, TypedDict

from pydantic import BaseModel

QueryType = Literal["what", "when", "who", "why", "evolution", "similar"]
RetrievalStrategy = Literal["chroma_only", "postgres_only", "neo4j_postgres", "all_parallel"]

# The seven provenance classes (critical rule: not a binary dead-code check).
ProvenanceClass = Literal[
    "well_documented",
    "legacy_migration",
    "hotfix",
    "experimental_prototype",
    "generated_code",
    "third_party_import",
    "orphaned",
]


class RetrievedChunk(BaseModel):
    source_type: str  # "code_chunk" | "commit" | "pr_discussion" | doc_type from ADRDocument
    source_id: str
    text: str
    metadata: dict[str, Any]
    score: Optional[float] = None  # cosine similarity from Chroma, overwritten by rerank score in 4c


class ProvenanceChain(BaseModel):
    function_chunk_id: str
    introduced_by_sha: Optional[str] = None
    last_modified_by_sha: Optional[str] = None
    pr_number: Optional[int] = None
    issue_number: Optional[int] = None
    adr_doc_id: Optional[str] = None
    raw_path_records: list[dict[str, Any]] = []


class CommitMetadata(BaseModel):
    sha: str
    message: Optional[str] = None
    author_name: Optional[str] = None
    timestamp: Optional[datetime] = None
    linked_pr_number: Optional[int] = None
    linked_issue_numbers: list[int] = []
    files_changed: list[dict[str, Any]] = []


class Citation(BaseModel):
    source_type: str  # "pr" | "commit" | "issue" | "adr"
    source_id: str
    url: str
    excerpt: str


class EvolutionEvent(BaseModel):
    date: datetime
    event_type: str  # initial_implementation|refactor|security_fix|performance_optimization|migration|deprecation
    description: str
    pr_number: Optional[int] = None
    issue_number: Optional[int] = None
    pr_url: Optional[str] = None
    files_changed: list[str] = []
    commit_shas: list[str] = []


class ArchaeologyState(TypedDict, total=False):
    # Input
    repo_id: str
    query: str
    target_chunk_id: Optional[str]
    create_docs_pr: bool  # opt-in: run Phase 5b and open a draft docs PR for this answer

    # Phase 4a outputs
    query_type: QueryType
    retrieval_strategy: RetrievalStrategy

    # Phase 4b outputs
    chroma_results: list[RetrievedChunk]
    neo4j_results: list[ProvenanceChain]
    postgres_results: list[CommitMetadata]

    # Phase 4c outputs
    ranked_context: list[RetrievedChunk]

    # Phase 4d outputs
    provenance_class: ProvenanceClass
    provenance_confidence: float

    # Phase 4e outputs (conditional)
    evolution_timeline: Optional[list[EvolutionEvent]]

    # Phase 5a outputs
    answer: str
    citations: list[Citation]

    # Phase 5b outputs
    draft_pr_url: Optional[str]

    # Evaluation outputs (Phase 6)
    ragas_scores: Optional[dict[str, float]]
