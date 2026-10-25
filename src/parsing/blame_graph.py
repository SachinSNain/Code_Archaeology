"""Phase 2b — BlameGraphBuilder.

For each code chunk produced by ASTCodeParserAgent, reconstructs the
provenance chain and stores it as a graph in Neo4j:

    (:Function)-[:INTRODUCED_BY]->(:Commit)-[:PART_OF_PR]->(:PR)-[:ADDRESSES]->(:Issue)-[:INFORMED_BY]->(:ADR)
    (:Function)-[:LAST_MODIFIED_BY]->(:Commit)
    (:Commit)-[:MODIFIES]->(:Function)

`INTRODUCED_BY` (earliest commit) and the full `MODIFIES` history are found
via `git log -L <start>,<end>:<path>` — git's line-range history tracker,
which follows a specific line range through renames/reflows far more
accurately than a file-level `git log` would. `LAST_MODIFIED_BY` reuses the
`last_modified_sha` already computed by ast_parser.py's blame pass, so we
don't recompute it.

Handling missing links (critical rule): a Function node is created even when
its INTRODUCED_BY/PART_OF_PR/ADDRESSES edges can't be resolved — sparse
provenance is itself a signal Phase 4d's ProvenanceClassifier reads, not an
error condition.

Usage:
    python -m src.parsing.blame_graph tiangolo/fastapi [--limit N]
"""
import argparse
import re
import sys
from typing import Optional

from tqdm import tqdm

from src.db import neo4j as neo4j_db
from src.db import postgres
from src.ingestion._git_utils import clone_or_open, repo_id_to_url

COMMIT_HEADER_RE = re.compile(r"^commit ([0-9a-f]{40})", re.MULTILINE)


def _line_range_commit_shas(repo, rel_path: str, start: int, end: int) -> list[str]:
    """All commits that ever touched this exact line range, newest-first."""
    try:
        output = repo.git.log("-L", f"{start},{end}:{rel_path}")
    except Exception:
        return []
    return COMMIT_HEADER_RE.findall(output)


def _load_commit_lookup(repo_id: str) -> dict[str, postgres.Commit]:
    session_factory = postgres.get_session_factory()
    with session_factory() as session:
        rows = session.query(postgres.Commit).filter_by(repo_id=repo_id).all()
        return {c.sha: c for c in rows}


def _load_pr_lookup(repo_id: str) -> dict[int, postgres.PullRequest]:
    session_factory = postgres.get_session_factory()
    with session_factory() as session:
        rows = session.query(postgres.PullRequest).filter_by(repo_id=repo_id).all()
        return {p.pr_number: p for p in rows}


def _load_issue_lookup(repo_id: str) -> dict[int, postgres.Issue]:
    session_factory = postgres.get_session_factory()
    with session_factory() as session:
        rows = session.query(postgres.Issue).filter_by(repo_id=repo_id).all()
        return {i.issue_number: i for i in rows}


def _load_adr_lookup(repo_id: str) -> list[postgres.ADRDocument]:
    session_factory = postgres.get_session_factory()
    with session_factory() as session:
        return session.query(postgres.ADRDocument).filter_by(repo_id=repo_id).all()


def _commit_node_kwargs(repo_id: str, sha: str, commit_lookup: dict) -> dict:
    row = commit_lookup.get(sha)
    return {
        "sha": sha,
        "repo_id": repo_id,
        "message": (row.message[:500] if row and row.message else None),
        "author": row.author_name if row else None,
        "timestamp": row.timestamp.isoformat() if row and row.timestamp else None,
    }


def _build_function_graph(
    tx,
    *,
    repo_id: str,
    chunk,
    touching_shas: list[str],
    commit_lookup: dict,
    pr_lookup: dict,
    issue_lookup: dict,
) -> None:
    neo4j_db.merge_function(
        tx,
        chunk_id=chunk.chunk_id,
        repo_id=repo_id,
        file_path=chunk.file_path,
        function_name=chunk.function_name,
        language=chunk.language,
    )
    function_uid = neo4j_db.uid(repo_id, chunk.chunk_id)

    all_shas = list(touching_shas)
    if chunk.last_modified_sha and chunk.last_modified_sha not in all_shas:
        all_shas.append(chunk.last_modified_sha)

    linked_prs: set[int] = set()
    for sha in all_shas:
        neo4j_db.merge_commit(tx, **_commit_node_kwargs(repo_id, sha, commit_lookup))
        commit_uid = neo4j_db.uid(repo_id, sha)
        neo4j_db.merge_edge(tx, label_a="Commit", uid_a=commit_uid, rel="MODIFIES", label_b="Function", uid_b=function_uid)

        commit_row = commit_lookup.get(sha)
        if commit_row and commit_row.linked_pr_number:
            linked_prs.add(commit_row.linked_pr_number)

    # INTRODUCED_BY: the oldest commit in the line-range history (git log lists newest-first).
    if touching_shas:
        introduced_sha = touching_shas[-1]
        neo4j_db.merge_commit(tx, **_commit_node_kwargs(repo_id, introduced_sha, commit_lookup))
        neo4j_db.merge_edge(
            tx, label_a="Function", uid_a=function_uid, rel="INTRODUCED_BY",
            label_b="Commit", uid_b=neo4j_db.uid(repo_id, introduced_sha),
        )

    # LAST_MODIFIED_BY: precomputed by ast_parser.py's per-file blame pass.
    if chunk.last_modified_sha:
        neo4j_db.merge_edge(
            tx, label_a="Function", uid_a=function_uid, rel="LAST_MODIFIED_BY",
            label_b="Commit", uid_b=neo4j_db.uid(repo_id, chunk.last_modified_sha),
        )

    # Commit -> PR -> Issue chain.
    for pr_number in linked_prs:
        pr_row = pr_lookup.get(pr_number)
        neo4j_db.merge_pr(
            tx, pr_number=pr_number, repo_id=repo_id,
            title=pr_row.title if pr_row else None,
            merged_at=pr_row.merged_at.isoformat() if pr_row and pr_row.merged_at else None,
        )
        pr_uid = neo4j_db.uid(repo_id, pr_number)
        for sha in all_shas:
            commit_row = commit_lookup.get(sha)
            if commit_row and commit_row.linked_pr_number == pr_number:
                neo4j_db.merge_edge(
                    tx, label_a="Commit", uid_a=neo4j_db.uid(repo_id, sha),
                    rel="PART_OF_PR", label_b="PR", uid_b=pr_uid,
                )

        if pr_row:
            for issue_number in (pr_row.linked_issue_numbers or []):
                issue_row = issue_lookup.get(issue_number)
                neo4j_db.merge_issue(
                    tx, issue_number=issue_number, repo_id=repo_id,
                    title=issue_row.title if issue_row else None,
                    created_at=issue_row.created_at.isoformat() if issue_row and issue_row.created_at else None,
                )
                neo4j_db.merge_edge(
                    tx, label_a="PR", uid_a=pr_uid, rel="ADDRESSES",
                    label_b="Issue", uid_b=neo4j_db.uid(repo_id, issue_number),
                )


def _link_issues_to_adrs(repo_id: str) -> int:
    """Best-effort Issue -[:INFORMED_BY]->(:ADR) link: an ADR/RFC/design doc
    that textually references an issue number is treated as having informed it.
    """
    issue_lookup = _load_issue_lookup(repo_id)
    adrs = _load_adr_lookup(repo_id)
    linked = 0
    with neo4j_db.session() as s:
        for adr in adrs:
            referenced = {int(n) for n in re.findall(r"#(\d+)", adr.body or "")}
            for issue_number in referenced:
                if issue_number not in issue_lookup:
                    continue
                s.execute_write(
                    neo4j_db.merge_adr, doc_id=adr.doc_id, repo_id=repo_id,
                    title=adr.title, doc_type=adr.doc_type,
                )
                s.execute_write(
                    neo4j_db.merge_edge, label_a="Issue", uid_a=neo4j_db.uid(repo_id, issue_number),
                    rel="INFORMED_BY", label_b="ADR", uid_b=neo4j_db.uid(repo_id, adr.doc_id),
                )
                linked += 1
    return linked


def build_blame_graph(repo_id: str, limit: Optional[int] = None) -> int:
    postgres.init_db()
    neo4j_db.init_constraints()

    repo = clone_or_open(repo_id, repo_id_to_url(repo_id))

    session_factory = postgres.get_session_factory()
    with session_factory() as session:
        chunks = session.query(postgres.CodeChunkRecord).filter_by(repo_id=repo_id).all()

    if limit:
        chunks = chunks[:limit]

    print(f"Building blame graph for {len(chunks)} chunk(s) in {repo_id}")

    commit_lookup = _load_commit_lookup(repo_id)
    pr_lookup = _load_pr_lookup(repo_id)
    issue_lookup = _load_issue_lookup(repo_id)

    processed = 0
    with neo4j_db.session() as s:
        for chunk in tqdm(chunks, desc=f"blame_graph:{repo_id}"):
            touching_shas = _line_range_commit_shas(repo, chunk.file_path, chunk.start_line, chunk.end_line)
            s.execute_write(
                _build_function_graph,
                repo_id=repo_id,
                chunk=chunk,
                touching_shas=touching_shas,
                commit_lookup=commit_lookup,
                pr_lookup=pr_lookup,
                issue_lookup=issue_lookup,
            )
            processed += 1

    adr_links = _link_issues_to_adrs(repo_id)
    neo4j_db.close_driver()

    print(f"Done. Built graph for {processed} function(s); linked {adr_links} issue->ADR edge(s).")
    return processed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id")
    parser.add_argument("--limit", type=int, default=None, help="Cap chunks processed (testing only)")
    args = parser.parse_args()
    build_blame_graph(args.repo_id, limit=args.limit)


if __name__ == "__main__":
    main()
