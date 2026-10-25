"""Phase 1b — PRIssueIngestionAgent.

Uses PyGithub to pull every PR (title, body, review comments, linked issues,
labels, merge commit SHA, files touched) and every issue (title, body,
comment thread, labels, linked PRs) into Postgres.

Rate limiting is a real production concern, not a toy one: GitHub allows
5,000 authenticated API calls/hour, and a repo like VSCode has 10,000+ PRs
each requiring 2-3 detail calls (review comments, files touched). This module
checkpoints the last-fetched PR/issue number so a run interrupted by a rate
limit reset can resume from a computed page offset instead of re-paging from
item 1, and backs off with exponential wait + explicit rate-limit-reset sleep.

PR-to-commit linking: GitHub returns the merge commit SHA per PR; after
ingesting PRs we UPDATE commits.linked_pr_number by joining on that SHA,
completing the commit -> PR edge of the provenance chain (consumed by
BlameGraphBuilder in Phase 2b).

Usage:
    python -m src.ingestion.pr_issue_ingestion tiangolo/fastapi [--limit N]
"""
import argparse
import sys
import time
from datetime import datetime, timezone
from typing import Optional

import requests
from github import Github, RateLimitExceededException
from github.GithubException import GithubException
from sqlalchemy import text
from sqlalchemy.dialects.postgresql import insert
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tqdm import tqdm

from src.config.settings import get_settings
from src.db import postgres

PER_PAGE = 30  # PyGithub/GitHub API default page size
CHECKPOINT_SOURCE_PRS = "pull_requests"
CHECKPOINT_SOURCE_ISSUES = "issues"


def get_github_client() -> Github:
    token = get_settings().github_token
    return Github(token, per_page=PER_PAGE) if token else Github(per_page=PER_PAGE)


def _sleep_until_rate_limit_reset(gh: Github) -> None:
    rate = gh.get_rate_limit().core
    reset_ts = rate.reset.replace(tzinfo=timezone.utc).timestamp()
    wait_s = max(reset_ts - time.time(), 0) + 5
    print(f"Rate limit exhausted. Sleeping {wait_s:.0f}s until reset...")
    time.sleep(wait_s)


@retry(
    # GithubException covers rate limits / API-level errors; requests.RequestException
    # (including the RetryError requests itself raises after exhausting its own retries
    # on 5xx responses, e.g. transient 503s from GitHub) covers transport-level failures.
    # Both are real production occurrences on a multi-hour ingestion run, not edge cases.
    retry=retry_if_exception_type((GithubException, requests.exceptions.RequestException)),
    wait=wait_exponential(multiplier=2, min=2, max=60),
    stop=stop_after_attempt(8),
)
def _call_with_backoff(fn, *args, **kwargs):
    gh = kwargs.pop("_gh", None)
    try:
        return fn(*args, **kwargs)
    except RateLimitExceededException:
        if gh is not None:
            _sleep_until_rate_limit_reset(gh)
        raise


def get_checkpoint(repo_id: str, source: str) -> Optional[int]:
    session_factory = postgres.get_session_factory()
    with session_factory() as session:
        row = (
            session.query(postgres.IngestionCheckpoint)
            .filter_by(repo_id=repo_id, source=source)
            .one_or_none()
        )
        return int(row.last_value) if row else None


def set_checkpoint(repo_id: str, source: str, value: int) -> None:
    session_factory = postgres.get_session_factory()
    with session_factory() as session:
        stmt = insert(postgres.IngestionCheckpoint).values(
            repo_id=repo_id, source=source, last_value=str(value), updated_at=datetime.utcnow()
        )
        stmt = stmt.on_conflict_do_update(
            index_elements=["repo_id", "source"],
            set_={"last_value": str(value), "updated_at": datetime.utcnow()},
        )
        session.execute(stmt)
        session.commit()


def _pr_to_row(repo_id: str, pr) -> dict:
    review_comments = [
        {
            "author": c.user.login if c.user else None,
            "body": c.body,
            "created_at": c.created_at.isoformat() if c.created_at else None,
            "path": c.path,
            # PyGithub's PullRequestComment doesn't wrap "line" as an attribute (only
            # "position"/"original_position"), but the GitHub API returns it — read it
            # from raw_data instead of leaving inline-comment line numbers empty.
            "line": c.raw_data.get("line") or c.position,
        }
        for c in pr.get_review_comments()
    ]
    files_touched = [f.filename for f in pr.get_files()]
    linked_issues = sorted(
        {int(n) for n in __import__("re").findall(r"#(\d+)", pr.body or "")}
    )

    return {
        "repo_id": repo_id,
        "pr_number": pr.number,
        "title": pr.title,
        "body": pr.body,
        "author": pr.user.login if pr.user else None,
        "created_at": pr.created_at,
        "merged_at": pr.merged_at,
        "merge_commit_sha": pr.merge_commit_sha,
        "linked_issue_numbers": linked_issues,
        "labels": [l.name for l in pr.labels],
        "review_comments": review_comments,
        "files_touched": files_touched,
    }


def _issue_to_row(repo_id: str, issue) -> dict:
    comment_thread = [
        {
            "author": c.user.login if c.user else None,
            "body": c.body,
            "created_at": c.created_at.isoformat() if c.created_at else None,
        }
        for c in issue.get_comments()
    ]
    linked_prs = sorted(
        {int(n) for n in __import__("re").findall(r"#(\d+)", issue.body or "")}
    )
    return {
        "repo_id": repo_id,
        "issue_number": issue.number,
        "title": issue.title,
        "body": issue.body,
        "author": issue.user.login if issue.user else None,
        "created_at": issue.created_at,
        "closed_at": issue.closed_at,
        "labels": [l.name for l in issue.labels],
        "comment_thread": comment_thread,
        "linked_pr_numbers": linked_prs,
    }


def _upsert_prs(rows: list[dict]) -> None:
    if not rows:
        return
    session_factory = postgres.get_session_factory()
    with session_factory() as session:
        stmt = insert(postgres.PullRequest).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["repo_id", "pr_number"],
            set_={c.name: getattr(stmt.excluded, c.name) for c in postgres.PullRequest.__table__.columns if c.name not in ("id", "repo_id", "pr_number")},
        )
        session.execute(stmt)
        session.commit()


def _upsert_issues(rows: list[dict]) -> None:
    if not rows:
        return
    session_factory = postgres.get_session_factory()
    with session_factory() as session:
        stmt = insert(postgres.Issue).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["repo_id", "issue_number"],
            set_={c.name: getattr(stmt.excluded, c.name) for c in postgres.Issue.__table__.columns if c.name not in ("id", "repo_id", "issue_number")},
        )
        session.execute(stmt)
        session.commit()


def _record_failure(repo_id: str, source: str, item_number: int, error: Exception) -> None:
    session_factory = postgres.get_session_factory()
    with session_factory() as session:
        session.add(
            postgres.IngestionFailure(repo_id=repo_id, source=source, item_number=item_number, error=str(error)[:2000])
        )
        session.commit()
    print(f"  SKIPPED {source} #{item_number} after exhausting retries: {error}")


def link_commits_to_prs(repo_id: str) -> int:
    """Complete the commit -> PR provenance edge via merge_commit_sha join."""
    session_factory = postgres.get_session_factory()
    with session_factory() as session:
        result = session.execute(
            text(
                """
                UPDATE commits
                SET linked_pr_number = pr.pr_number
                FROM pull_requests pr
                WHERE commits.repo_id = pr.repo_id
                  AND commits.sha = pr.merge_commit_sha
                  AND commits.repo_id = :repo_id
                """
            ),
            {"repo_id": repo_id},
        )
        session.commit()
        return result.rowcount


def ingest_pull_requests(repo_id: str, limit: Optional[int] = None) -> int:
    postgres.init_db()
    gh = get_github_client()
    repo = gh.get_repo(repo_id)

    # Checkpoint stores a COUNT of PRs successfully processed, not a PR number:
    # PR numbers share a namespace with issues and have gaps, so `count // PER_PAGE`
    # is the only reliable way to recompute which page to resume from (pages are
    # slices of the creation-ordered list, not of the PR-number space).
    processed_count = int(get_checkpoint(repo_id, CHECKPOINT_SOURCE_PRS) or 0)
    pulls = repo.get_pulls(state="all", sort="created", direction="asc")
    total = pulls.totalCount

    start_page, in_page_offset = divmod(processed_count, PER_PAGE)
    print(f"Ingesting PRs for {repo_id}: {total} total, resuming at item {processed_count} (page {start_page})")

    processed_this_run = 0
    page_index = start_page
    with tqdm(total=total, initial=processed_count, desc=f"pr_ingestion:{repo_id}") as pbar:
        while True:
            page = _call_with_backoff(pulls.get_page, page_index, _gh=gh)
            if not page:
                break
            start_idx = in_page_offset if page_index == start_page else 0
            for pr in page[start_idx:]:
                try:
                    row = _call_with_backoff(_pr_to_row, repo_id, pr, _gh=gh)
                    _upsert_prs([row])
                except Exception as exc:
                    # A single permanently-unfetchable PR (retries exhausted) must not
                    # abort a multi-hour run — record it and move on to the next item.
                    _record_failure(repo_id, CHECKPOINT_SOURCE_PRS, pr.number, exc)
                # Checkpoint per item (not per page) regardless of success/failure: a
                # crash mid-page must lose at most the one in-flight item, and a
                # permanently-failing item must not block everything behind it.
                processed_count += 1
                set_checkpoint(repo_id, CHECKPOINT_SOURCE_PRS, processed_count)
                processed_this_run += 1
                pbar.update(1)
                if limit and processed_this_run >= limit:
                    break
            page_index += 1
            if (limit and processed_this_run >= limit) or len(page) < PER_PAGE:
                break

    linked = link_commits_to_prs(repo_id)
    print(f"Done. Ingested {processed_this_run} PR(s) this run. Linked {linked} commit(s) to their PR via merge_commit_sha.")
    return processed_this_run


def ingest_issues(repo_id: str, limit: Optional[int] = None) -> int:
    postgres.init_db()
    gh = get_github_client()
    repo = gh.get_repo(repo_id)

    processed_count = int(get_checkpoint(repo_id, CHECKPOINT_SOURCE_ISSUES) or 0)
    issues = repo.get_issues(state="all", sort="created", direction="asc")
    total = issues.totalCount

    start_page, in_page_offset = divmod(processed_count, PER_PAGE)
    print(f"Ingesting issues for {repo_id}: {total} total, resuming at item {processed_count} (page {start_page})")

    processed_this_run = 0
    page_index = start_page
    with tqdm(total=total, initial=processed_count, desc=f"issue_ingestion:{repo_id}") as pbar:
        while True:
            page = _call_with_backoff(issues.get_page, page_index, _gh=gh)
            if not page:
                break
            start_idx = in_page_offset if page_index == start_page else 0
            for issue in page[start_idx:]:
                processed_count += 1  # count position in the paginated stream, PRs included, so resume offsets stay valid
                set_checkpoint(repo_id, CHECKPOINT_SOURCE_ISSUES, processed_count)
                try:
                    # issue.pull_request is a lazily-loaded PyGithub property (GitHub's Issues API
                    # also returns PRs; skip them, PRs are handled separately) — it makes its own
                    # API call under the hood, so it needs the same retry protection as any other
                    # call here, not just the ones already wrapped explicitly.
                    is_pr = _call_with_backoff(lambda i=issue: i.pull_request is not None, _gh=gh)
                    if is_pr:
                        continue
                    row = _call_with_backoff(_issue_to_row, repo_id, issue, _gh=gh)
                    _upsert_issues([row])
                except Exception as exc:
                    _record_failure(repo_id, CHECKPOINT_SOURCE_ISSUES, issue.number, exc)
                    continue
                processed_this_run += 1
                pbar.update(1)
                if limit and processed_this_run >= limit:
                    break
            page_index += 1
            if (limit and processed_this_run >= limit) or len(page) < PER_PAGE:
                break

    print(f"Done. Ingested {processed_this_run} issue(s) this run.")
    return processed_this_run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("repo_id")
    parser.add_argument("--limit", type=int, default=None, help="Cap items ingested per source (testing only)")
    args = parser.parse_args()

    ingest_pull_requests(args.repo_id, limit=args.limit)
    ingest_issues(args.repo_id, limit=args.limit)


if __name__ == "__main__":
    main()
