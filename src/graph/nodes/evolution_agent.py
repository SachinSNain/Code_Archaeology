"""Phase 4e — Architecture Evolution Agent (conditional node).

"This is the most impressive demo output in the entire system." — triggered
when query_type == "evolution", OR when the Neo4j traversal reveals a
function with 5+ distinct commit chains spanning more than a year (i.e. the
component has enough history to be worth narrating even if the user didn't
explicitly ask for a timeline).

Pipeline: pull full commit history for the component from Neo4j -> enrich
with PR/issue bodies from Postgres -> cluster commits into "migration
events" (temporally close + same files + semantically similar PR
descriptions) -> synthesize one narrated EvolutionEvent per cluster with
the configured LLM (src/llm/llm_factory.get_llm()).
"""
from datetime import datetime, timezone
from functools import lru_cache
from typing import Literal, Optional

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from src.db import neo4j as neo4j_db, postgres
from src.graph.state import ArchaeologyState, EvolutionEvent
from src.llm.llm_factory import get_llm

CLUSTER_MAX_GAP_DAYS = 14
CLUSTER_MIN_FILE_OVERLAP = 0.2  # jaccard similarity of touched-file sets
CLUSTER_MIN_PR_SIMILARITY = 0.6
MIN_CHAINS_FOR_AUTO_TRIGGER = 5
MIN_SPAN_DAYS_FOR_AUTO_TRIGGER = 365

EventType = Literal[
    "initial_implementation", "refactor", "security_fix", "performance_optimization", "migration", "deprecation"
]


class _EventSynthesis(BaseModel):
    event_type: EventType
    description: str


@lru_cache(maxsize=1)
def _similarity_model():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


def should_run_evolution_agent(state: ArchaeologyState) -> bool:
    if state.get("query_type") == "evolution":
        return True
    chains = state.get("neo4j_results") or []
    shas = {c.introduced_by_sha for c in chains if c.introduced_by_sha} | {
        c.last_modified_by_sha for c in chains if c.last_modified_by_sha
    }
    if len(shas) < MIN_CHAINS_FOR_AUTO_TRIGGER:
        return False
    return True  # timestamp span is cheap to check but requires a DB hit; the node itself re-verifies


def _fetch_full_history(repo_id: str, chunk_id: str) -> list[dict]:
    records = neo4j_db.fetch_all_history_for_function(repo_id, chunk_id)
    commits = []
    for rec in records:
        commit = rec.get("c") or {}
        if not commit.get("timestamp"):
            continue
        commits.append(
            {
                "sha": commit.get("sha"),
                "timestamp": datetime.fromisoformat(commit["timestamp"]),
                "message": commit.get("message"),
                "pr_number": (rec.get("pr") or {}).get("pr_number"),
                "issue_number": (rec.get("issue") or {}).get("issue_number"),
            }
        )
    commits.sort(key=lambda c: c["timestamp"])
    return commits


def _enrich_with_postgres(repo_id: str, commits: list[dict]) -> None:
    shas = [c["sha"] for c in commits if c["sha"]]
    pr_numbers = {c["pr_number"] for c in commits if c["pr_number"]}
    if not shas:
        return

    session_factory = postgres.get_session_factory()
    with session_factory() as session:
        commit_rows = {
            r.sha: r
            for r in session.query(postgres.Commit).filter(postgres.Commit.repo_id == repo_id, postgres.Commit.sha.in_(shas)).all()
        }
        pr_rows = (
            {
                r.pr_number: r
                for r in session.query(postgres.PullRequest)
                .filter(postgres.PullRequest.repo_id == repo_id, postgres.PullRequest.pr_number.in_(pr_numbers))
                .all()
            }
            if pr_numbers
            else {}
        )

    for c in commits:
        commit_row = commit_rows.get(c["sha"])
        c["files_changed"] = [f["path"] for f in (commit_row.files_changed or [])] if commit_row else []
        pr_row = pr_rows.get(c["pr_number"])
        c["pr_title"] = pr_row.title if pr_row else None
        c["pr_body"] = pr_row.body if pr_row else None
        c["pr_url"] = (
            f"https://github.com/{repo_id}/pull/{c['pr_number']}" if c["pr_number"] else None
        )


def _jaccard(a: set, b: set) -> float:
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _cluster_commits(commits: list[dict]) -> list[list[dict]]:
    if not commits:
        return []

    texts = [f"{c.get('pr_title') or ''} {c.get('pr_body') or ''}".strip() or (c.get("message") or "") for c in commits]
    model = _similarity_model()
    embeddings = model.encode(texts, normalize_embeddings=True)

    clusters: list[list[dict]] = [[commits[0]]]
    cluster_embeddings: list = [embeddings[0]]

    for i in range(1, len(commits)):
        commit = commits[i]
        prev = clusters[-1][-1]
        gap_days = (commit["timestamp"] - prev["timestamp"]).days
        file_overlap = _jaccard(set(commit.get("files_changed", [])), set(prev.get("files_changed", [])))
        pr_similarity = float(embeddings[i] @ cluster_embeddings[-1])

        same_event = (
            gap_days <= CLUSTER_MAX_GAP_DAYS
            and (file_overlap >= CLUSTER_MIN_FILE_OVERLAP or pr_similarity >= CLUSTER_MIN_PR_SIMILARITY)
        )
        if same_event:
            clusters[-1].append(commit)
            cluster_embeddings[-1] = (cluster_embeddings[-1] + embeddings[i]) / 2
        else:
            clusters.append([commit])
            cluster_embeddings.append(embeddings[i])

    return clusters


def _synthesize_event(repo_id: str, cluster: list[dict]) -> EvolutionEvent:
    summary_lines = []
    for c in cluster:
        line = f"- {c['timestamp'].date()} [{c['sha'][:10]}] {c.get('message', '').strip().splitlines()[0] if c.get('message') else ''}"
        if c.get("pr_title"):
            line += f" (PR: {c['pr_title']})"
        summary_lines.append(line)

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are summarizing one cluster of related commits as a single architectural evolution "
                "event for a code archaeology timeline. Classify it as one of: initial_implementation, "
                "refactor, security_fix, performance_optimization, migration, deprecation. Write a one-"
                "sentence description of what changed and why, grounded only in the commits/PRs given.",
            ),
            ("human", "{commits}"),
        ]
    )
    chain = prompt | get_llm(purpose="default", temperature=0.2).with_structured_output(_EventSynthesis)
    result: _EventSynthesis = chain.invoke({"commits": "\n".join(summary_lines)})

    pr_numbers = {c["pr_number"] for c in cluster if c.get("pr_number")}
    issue_numbers = {c["issue_number"] for c in cluster if c.get("issue_number")}
    files = sorted({f for c in cluster for f in c.get("files_changed", [])})

    return EvolutionEvent(
        date=cluster[0]["timestamp"],
        event_type=result.event_type,
        description=result.description,
        pr_number=next(iter(pr_numbers), None),
        issue_number=next(iter(issue_numbers), None),
        pr_url=cluster[0].get("pr_url"),
        files_changed=files,
        commit_shas=[c["sha"] for c in cluster],
    )


def evolution_agent_node(state: ArchaeologyState) -> dict:
    repo_id = state["repo_id"]
    target_chunk_id = state.get("target_chunk_id")
    if not target_chunk_id:
        return {"evolution_timeline": None}

    commits = _fetch_full_history(repo_id, target_chunk_id)
    if len(commits) < 2:
        return {"evolution_timeline": []}

    span_days = (commits[-1]["timestamp"] - commits[0]["timestamp"]).days
    if state.get("query_type") != "evolution" and (
        len(commits) < MIN_CHAINS_FOR_AUTO_TRIGGER or span_days < MIN_SPAN_DAYS_FOR_AUTO_TRIGGER
    ):
        return {"evolution_timeline": None}  # auto-trigger condition didn't actually hold

    _enrich_with_postgres(repo_id, commits)
    clusters = _cluster_commits(commits)
    timeline = [_synthesize_event(repo_id, cluster) for cluster in clusters]

    return {"evolution_timeline": timeline}
