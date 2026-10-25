"""SQLAlchemy models + connection for the PostgreSQL store.

Postgres answers "WHEN was this written? WHO changed it?" — fast relational
queries, timestamp sorting, joins across commits/PRs/issues. See
`WHAT_TO_DO.MD` "Core architectural principle" for why this is a separate
store from Neo4j (causality) and Chroma (semantic similarity).

Every table carries `repo_id` (critical rule #1) so the schema supports
multiple repositories without migration.
"""
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    ARRAY,
    DateTime,
    ForeignKey,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
)
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, sessionmaker

from src.config.settings import get_settings


class Base(DeclarativeBase):
    pass


class Commit(Base):
    __tablename__ = "commits"
    __table_args__ = (UniqueConstraint("repo_id", "sha", name="uq_commits_repo_sha"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    sha: Mapped[str] = mapped_column(String(40), nullable=False, index=True)
    message: Mapped[Optional[str]] = mapped_column(Text)
    author_name: Mapped[Optional[str]] = mapped_column(String)
    author_email: Mapped[Optional[str]] = mapped_column(String)
    timestamp: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), index=True)
    parent_shas: Mapped[Optional[list[str]]] = mapped_column(ARRAY(String))
    files_changed: Mapped[Optional[list]] = mapped_column(JSONB)  # [{path, additions, deletions, patch}]
    linked_issue_numbers: Mapped[Optional[list[int]]] = mapped_column(ARRAY(Integer))
    linked_pr_number: Mapped[Optional[int]] = mapped_column(Integer, index=True)


class PullRequest(Base):
    __tablename__ = "pull_requests"
    __table_args__ = (UniqueConstraint("repo_id", "pr_number", name="uq_pr_repo_number"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    pr_number: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    title: Mapped[Optional[str]] = mapped_column(Text)
    body: Mapped[Optional[str]] = mapped_column(Text)
    author: Mapped[Optional[str]] = mapped_column(String)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    merged_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), index=True)
    merge_commit_sha: Mapped[Optional[str]] = mapped_column(String(40), index=True)
    linked_issue_numbers: Mapped[Optional[list[int]]] = mapped_column(ARRAY(Integer))
    labels: Mapped[Optional[list[str]]] = mapped_column(ARRAY(String))
    review_comments: Mapped[Optional[list]] = mapped_column(JSONB)  # [{author, body, created_at, path, line}]
    files_touched: Mapped[Optional[list[str]]] = mapped_column(ARRAY(String))


class Issue(Base):
    __tablename__ = "issues"
    __table_args__ = (UniqueConstraint("repo_id", "issue_number", name="uq_issue_repo_number"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    issue_number: Mapped[int] = mapped_column(Integer, nullable=False, index=True)
    title: Mapped[Optional[str]] = mapped_column(Text)
    body: Mapped[Optional[str]] = mapped_column(Text)
    author: Mapped[Optional[str]] = mapped_column(String)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    closed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    labels: Mapped[Optional[list[str]]] = mapped_column(ARRAY(String))
    comment_thread: Mapped[Optional[list]] = mapped_column(JSONB)  # [{author, body, created_at}]
    linked_pr_numbers: Mapped[Optional[list[int]]] = mapped_column(ARRAY(Integer))


class ADRDocument(Base):
    """Architectural decision records / RFCs / design proposals / changelog entries (Phase 1c)."""

    __tablename__ = "adr_documents"
    __table_args__ = (UniqueConstraint("repo_id", "doc_id", name="uq_adr_repo_doc"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    doc_id: Mapped[str] = mapped_column(String, nullable=False)  # stable slug, e.g. file path or discussion id
    doc_type: Mapped[str] = mapped_column(String, nullable=False)  # adr|rfc|design_proposal|changelog_entry|slack_thread|github_discussion
    title: Mapped[Optional[str]] = mapped_column(Text)
    source_path: Mapped[Optional[str]] = mapped_column(String)
    body: Mapped[Optional[str]] = mapped_column(Text)
    created_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))


class CodeChunkRecord(Base):
    """Structural metadata for AST-parsed chunks (Phase 2a).

    The chunk text is also embedded into Chroma (Phase 3) and the function
    node is mirrored into Neo4j (Phase 2b); this table is the durable,
    queryable source of truth so Phase 2b/3 can be re-run independently
    without re-cloning and re-parsing the repo.
    """

    __tablename__ = "code_chunks"
    __table_args__ = (UniqueConstraint("repo_id", "chunk_id", name="uq_chunk_repo_id"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    chunk_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    file_path: Mapped[str] = mapped_column(String, nullable=False, index=True)
    function_name: Mapped[Optional[str]] = mapped_column(String)
    class_name: Mapped[Optional[str]] = mapped_column(String)
    language: Mapped[str] = mapped_column(String, nullable=False)
    start_line: Mapped[int] = mapped_column(Integer, nullable=False)
    end_line: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    last_modified_sha: Mapped[Optional[str]] = mapped_column(String(40))
    docstring: Mapped[Optional[str]] = mapped_column(Text)
    calls: Mapped[Optional[list[str]]] = mapped_column(ARRAY(String))
    called_by: Mapped[Optional[list[str]]] = mapped_column(ARRAY(String))


class RepoProfileRecord(Base):
    """Persisted RepoProfile output of Phase 0 (RepoConfigAgent)."""

    __tablename__ = "repo_profiles"
    __table_args__ = (UniqueConstraint("repo_id", name="uq_repo_profile_repo_id"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    repo_url: Mapped[str] = mapped_column(String, nullable=False)
    primary_languages: Mapped[Optional[list[str]]] = mapped_column(ARRAY(String))
    commit_quality_score: Mapped[Optional[float]] = mapped_column()
    pr_density: Mapped[Optional[float]] = mapped_column()
    has_adrs: Mapped[bool] = mapped_column(default=False)
    adr_paths: Mapped[Optional[list[str]]] = mapped_column(ARRAY(String))
    has_issue_linking: Mapped[bool] = mapped_column(default=False)
    slack_export_path: Mapped[Optional[str]] = mapped_column(String)
    total_commits: Mapped[Optional[int]] = mapped_column(Integer)
    date_range_start: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    date_range_end: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    fingerprinted_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)


class PendingDocPR(Base):
    """Tracks Living Docs Writer PRs awaiting human review (Phase 5b).

    Nothing in `proposed_text` is indexed into Chroma until `merged` flips to
    True (critical rule: never index an unreviewed LLM explanation, or a
    hallucination becomes "evidence" for future queries). A webhook or the
    polling fallback in docs_writer.py flips `merged`/`ingested` after the PR
    is actually merged on GitHub.
    """

    __tablename__ = "pending_doc_prs"
    __table_args__ = (UniqueConstraint("repo_id", "pr_number", name="uq_pending_doc_pr"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    pr_number: Mapped[int] = mapped_column(Integer, nullable=False)
    chunk_id: Mapped[str] = mapped_column(String, nullable=False)
    proposed_text: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)
    merged: Mapped[bool] = mapped_column(default=False)
    ingested: Mapped[bool] = mapped_column(default=False)


class IngestionFailure(Base):
    """Items that permanently failed during ingestion (retries exhausted) and were
    skipped rather than aborting the whole run — see pr_issue_ingestion.py. A single
    unfetchable PR/issue (e.g. a persistent upstream 503) must not block the other
    thousands of items behind it on a multi-hour run.
    """

    __tablename__ = "ingestion_failures"

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    source: Mapped[str] = mapped_column(String, nullable=False)  # "pull_requests" | "issues"
    item_number: Mapped[int] = mapped_column(Integer, nullable=False)
    error: Mapped[str] = mapped_column(Text, nullable=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow)


class IngestionCheckpoint(Base):
    """Tracks incremental-sync progress per repo/source so large repos are never fully re-walked.

    E.g. (repo_id="kubernetes/kubernetes", source="git_commits", last_value=<HEAD sha at last run>)
    or (repo_id=..., source="pr_issues", last_value=<last fetched PR number>).
    """

    __tablename__ = "ingestion_checkpoints"
    __table_args__ = (UniqueConstraint("repo_id", "source", name="uq_checkpoint_repo_source"),)

    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    repo_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    source: Mapped[str] = mapped_column(String, nullable=False)  # "git_commits" | "pull_requests" | "issues"
    last_value: Mapped[str] = mapped_column(String, nullable=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=datetime.utcnow, onupdate=datetime.utcnow)


_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    if _engine is None:
        _engine = create_engine(get_settings().postgres_url, pool_pre_ping=True, future=True)
    return _engine


def get_session_factory():
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine(), expire_on_commit=False, future=True)
    return _SessionLocal


def init_db() -> None:
    """Create all tables. Idempotent — safe to call on every startup."""
    Base.metadata.create_all(get_engine())
