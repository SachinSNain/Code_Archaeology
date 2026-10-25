"""Phase 1a — GitIngestionAgent.

Walks every commit with GitPython and stores SHA, message, author, timestamp,
parent SHAs, and per-file diffs into Postgres. Parses commit messages for
issue references (Fixes/Closes/Resolves/Related to #N).

Incremental sync (critical for large repos): the last-processed commit SHA is
checkpointed in `ingestion_checkpoints`. Subsequent runs walk only the range
`<checkpoint>..HEAD` instead of the full history.

Usage:
    python -m src.ingestion.git_ingestion tiangolo/fastapi
"""
import re
import sys
from datetime import datetime, timezone
from typing import Optional

import git
from sqlalchemy.dialects.postgresql import insert
from tqdm import tqdm

from src.db import postgres
from src.ingestion._git_utils import clone_or_open, repo_id_to_url

ISSUE_LINK_RE = re.compile(r"\b(?:fixes|closes|resolves|related to)\s+#(\d+)", re.IGNORECASE)
PR_REF_RE = re.compile(r"\(#(\d+)\)\s*$")

BATCH_SIZE = 500
MAX_PATCH_CHARS = 20_000  # guard against pathological diffs bloating JSONB storage
CHECKPOINT_SOURCE = "git_commits"


def parse_linked_issues(message: str) -> list[int]:
    return sorted({int(n) for n in ISSUE_LINK_RE.findall(message)})


def parse_linked_pr(message: str) -> Optional[int]:
    first_line = message.strip().splitlines()[0] if message.strip() else ""
    match = PR_REF_RE.search(first_line)
    return int(match.group(1)) if match else None


def get_checkpoint(repo_id: str) -> Optional[str]:
    session_factory = postgres.get_session_factory()
    with session_factory() as session:
        row = (
            session.query(postgres.IngestionCheckpoint)
            .filter_by(repo_id=repo_id, source=CHECKPOINT_SOURCE)
            .one_or_none()
        )
        return row.last_value if row else None


def set_checkpoint(repo_id: str, sha: str) -> None:
    session_factory = postgres.get_session_factory()
    with session_factory() as session:
        stmt = insert(postgres.IngestionCheckpoint).values(
            repo_id=repo_id, source=CHECKPOINT_SOURCE, last_value=sha, updated_at=datetime.utcnow()
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["repo_id", "source"],
            set_={"last_value": sha, "updated_at": datetime.utcnow()},
        )
        session.execute(stmt)
        session.commit()


def _build_files_changed(commit: git.Commit) -> list[dict]:
    parent = commit.parents[0] if commit.parents else None
    stats_files = commit.stats.files  # {path: {insertions, deletions, lines}}

    patches_by_path: dict[str, str] = {}
    try:
        diff_index = commit.diff(parent, create_patch=True, R=True) if parent else commit.diff(
            git.NULL_TREE, create_patch=True
        )
        for diff_item in diff_index:
            path = diff_item.b_path or diff_item.a_path
            try:
                patch_text = diff_item.diff.decode("utf-8", errors="replace")
            except Exception:
                patch_text = ""
            if len(patch_text) > MAX_PATCH_CHARS:
                patch_text = patch_text[:MAX_PATCH_CHARS] + f"\n<diff truncated at {MAX_PATCH_CHARS} chars>"
            patches_by_path[path] = patch_text
    except Exception:
        pass  # binary-heavy or malformed commits: fall back to stats-only, no patch text

    files_changed = []
    for path, stat in stats_files.items():
        files_changed.append(
            {
                "path": path,
                "additions": stat.get("insertions", 0),
                "deletions": stat.get("deletions", 0),
                "patch": patches_by_path.get(path, ""),
            }
        )
    return files_changed


def _commit_to_row(repo_id: str, commit: git.Commit) -> dict:
    message = commit.message
    return {
        "repo_id": repo_id,
        "sha": commit.hexsha,
        "message": message,
        "author_name": commit.author.name,
        "author_email": commit.author.email,
        "timestamp": datetime.fromtimestamp(commit.committed_date, tz=timezone.utc),
        "parent_shas": [p.hexsha for p in commit.parents],
        "files_changed": _build_files_changed(commit),
        "linked_issue_numbers": parse_linked_issues(message),
        "linked_pr_number": parse_linked_pr(message),
    }


def _upsert_batch(session_factory, rows: list[dict]) -> None:
    if not rows:
        return
    with session_factory() as session:
        stmt = insert(postgres.Commit).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["repo_id", "sha"],
            set_={
                "message": stmt.excluded.message,
                "author_name": stmt.excluded.author_name,
                "author_email": stmt.excluded.author_email,
                "timestamp": stmt.excluded.timestamp,
                "parent_shas": stmt.excluded.parent_shas,
                "files_changed": stmt.excluded.files_changed,
                "linked_issue_numbers": stmt.excluded.linked_issue_numbers,
                "linked_pr_number": stmt.excluded.linked_pr_number,
            },
        )
        session.execute(stmt)
        session.commit()


def ingest_commits(repo_id: str, repo_url: Optional[str] = None) -> int:
    postgres.init_db()
    url = repo_url or repo_id_to_url(repo_id)
    repo = clone_or_open(repo_id, url)
    session_factory = postgres.get_session_factory()

    checkpoint = get_checkpoint(repo_id)
    rev_range = f"{checkpoint}..HEAD" if checkpoint else "HEAD"

    # iter_commits yields newest-first; reverse so we process oldest-first and
    # can checkpoint safely mid-run (resuming skips only fully-processed commits).
    commits = list(repo.iter_commits(rev_range))
    commits.reverse()

    if not commits:
        print(f"No new commits for {repo_id} (checkpoint={checkpoint}). Up to date.")
        return 0

    print(f"Ingesting {len(commits)} commit(s) for {repo_id} (checkpoint={checkpoint or 'none'})")

    batch: list[dict] = []
    processed = 0
    for commit in tqdm(commits, desc=f"git_ingestion:{repo_id}"):
        batch.append(_commit_to_row(repo_id, commit))
        if len(batch) >= BATCH_SIZE:
            _upsert_batch(session_factory, batch)
            set_checkpoint(repo_id, commit.hexsha)
            processed += len(batch)
            batch = []

    if batch:
        _upsert_batch(session_factory, batch)
        set_checkpoint(repo_id, commits[-1].hexsha)
        processed += len(batch)

    print(f"Done. Ingested {processed} commit(s). Checkpoint now at {commits[-1].hexsha}.")
    return processed


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python -m src.ingestion.git_ingestion <owner/repo>")
        sys.exit(1)
    ingest_commits(sys.argv[1])


if __name__ == "__main__":
    main()
