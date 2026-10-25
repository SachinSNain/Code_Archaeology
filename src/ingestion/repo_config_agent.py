"""Phase 0 — RepoConfigAgent.

Fingerprints a repository before ingestion begins so downstream agents can
calibrate behavior: a repo with poor commit hygiene needs more weight on PR
discussions; a repo with no ADRs skips that ingestion step; a repo with no
issue linking cannot build a complete provenance chain and must flag gaps
explicitly (see Phase 4d ProvenanceClassifier).

Usage:
    python -m src.ingestion.repo_config_agent tiangolo/fastapi
"""
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from pydantic import BaseModel
from github import Github, GithubException

from src.config.repo_profiles import get_known_hints
from src.config.settings import get_settings
from src.db import postgres
from src.ingestion._git_utils import clone_or_open, repo_id_to_url

# tree-sitter-supported languages this project can actually chunk (Phase 2a).
# Anything outside this map is still detected/reported but AST-parsing will
# skip it.
EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".go": "go",
    ".java": "java",
    ".rs": "rust",
}

CONVENTIONAL_COMMIT_RE = re.compile(
    r"^(feat|fix|chore|docs|refactor|test|style|perf|build|ci|revert)(\(.+\))?!?:\s"
)
ISSUE_LINK_RE = re.compile(r"\b(fixes|closes|resolves|related to)\s+#(\d+)", re.IGNORECASE)
PR_REF_RE = re.compile(r"\(#(\d+)\)\s*$")

FALLBACK_ADR_PATHS = [
    "docs/adr/",
    "docs/architecture/",
    "docs/proposals/",
    "keps/",
    "rfcs/",
    "proposals/",
    "DECISIONS.md",
    "ARCHITECTURE.md",
]

COMMIT_SAMPLE_SIZE = 2000  # cap for heuristic scoring on large repos


class RepoProfile(BaseModel):
    repo_id: str
    repo_url: str
    primary_languages: list[str]
    commit_quality_score: float  # 0.0-1.0
    pr_density: float  # PRs per month average
    has_adrs: bool
    adr_paths: list[str]
    has_issue_linking: bool
    slack_export_path: Optional[str] = None
    total_commits: int
    date_range: tuple[datetime, datetime]


def _detect_primary_languages(repo_root: Path, top_n: int = 5) -> list[str]:
    counts: Counter[str] = Counter()
    for path in repo_root.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        lang = EXTENSION_TO_LANGUAGE.get(path.suffix.lower())
        if lang:
            counts[lang] += 1
    return [lang for lang, _ in counts.most_common(top_n)]


def _score_commit_quality(messages: list[str]) -> float:
    if not messages:
        return 0.0
    conventional = sum(1 for m in messages if CONVENTIONAL_COMMIT_RE.match(m.strip()))
    non_trivial_length = sum(1 for m in messages if len(m.strip()) >= 15)
    has_body = sum(1 for m in messages if "\n\n" in m.strip())

    conventional_ratio = conventional / len(messages)
    length_ratio = non_trivial_length / len(messages)
    body_ratio = has_body / len(messages)

    # Conventional-commit usage is the strongest signal; body/length are supporting signals.
    score = 0.5 * conventional_ratio + 0.3 * length_ratio + 0.2 * body_ratio
    return round(min(score, 1.0), 3)


def _detect_issue_linking(messages: list[str]) -> bool:
    if not messages:
        return False
    linked = sum(1 for m in messages if ISSUE_LINK_RE.search(m))
    return (linked / len(messages)) > 0.05  # >5% of sampled commits reference an issue


def _estimate_pr_density_via_api(repo_id: str, first_commit_dt: datetime, token: Optional[str]) -> Optional[float]:
    """One cheap Search API call for total PR count; avoids paging through every PR."""
    try:
        gh = Github(token) if token else Github()
        result = gh.search_issues(query=f"repo:{repo_id} type:pr")
        total_prs = result.totalCount
    except GithubException:
        return None

    months = max((datetime.now(timezone.utc) - first_commit_dt).days / 30.44, 1.0)
    return round(total_prs / months, 2)


def _estimate_pr_density_via_git_log(messages: list[str], first_commit_dt: datetime) -> float:
    """Fallback when no GitHub token / API unavailable: count PR-reference patterns in
    commit subjects, e.g. GitHub's default squash-merge message "Some fix (#1234)".
    """
    pr_refs = sum(1 for m in messages if PR_REF_RE.search(m.strip().splitlines()[0]))
    months = max((datetime.now(timezone.utc) - first_commit_dt).days / 30.44, 1.0)
    return round(pr_refs / months, 2)


def _detect_adrs(repo_id: str, repo_root: Path) -> tuple[bool, list[str]]:
    candidates = list(FALLBACK_ADR_PATHS)
    hints = get_known_hints(repo_id)
    if hints:
        candidates = hints.adr_paths + candidates

    found = []
    for rel_path in candidates:
        if (repo_root / rel_path).exists():
            found.append(rel_path)
    return (len(found) > 0, found)


def fingerprint_repo(repo_id: str, repo_url: Optional[str] = None) -> RepoProfile:
    settings = get_settings()
    url = repo_url or repo_id_to_url(repo_id)
    repo = clone_or_open(repo_id, url)

    total_commits = int(repo.git.rev_list("--count", "HEAD"))
    all_commits = list(repo.iter_commits("HEAD", max_count=COMMIT_SAMPLE_SIZE))
    if not all_commits:
        raise ValueError(f"No commits found in {repo_id}")

    messages = [c.message for c in all_commits]
    head_dt = datetime.fromtimestamp(all_commits[0].committed_date, tz=timezone.utc)

    # NOTE: `git log --reverse -1` does NOT return the oldest commit — max-count
    # is applied during traversal (newest-first) *before* --reverse reorders the
    # output, so it just returns the single newest commit again. The correct way
    # to find the repo's first commit is via the root commit(s) of the DAG.
    root_shas = repo.git.rev_list("--max-parents=0", "HEAD").split()
    if root_shas:
        root_commit = repo.commit(root_shas[0])
        first_dt = datetime.fromtimestamp(root_commit.committed_date, tz=timezone.utc)
    else:
        first_dt = head_dt

    primary_languages = _detect_primary_languages(Path(repo.working_tree_dir))
    commit_quality_score = _score_commit_quality(messages)
    has_issue_linking = _detect_issue_linking(messages)
    has_adrs, adr_paths = _detect_adrs(repo_id, Path(repo.working_tree_dir))

    pr_density = _estimate_pr_density_via_api(repo_id, first_dt, settings.github_token)
    if pr_density is None:
        pr_density = _estimate_pr_density_via_git_log(messages, first_dt)

    return RepoProfile(
        repo_id=repo_id,
        repo_url=url,
        primary_languages=primary_languages,
        commit_quality_score=commit_quality_score,
        pr_density=pr_density,
        has_adrs=has_adrs,
        adr_paths=adr_paths,
        has_issue_linking=has_issue_linking,
        slack_export_path=None,
        total_commits=total_commits,
        date_range=(first_dt, head_dt),
    )


def save_repo_profile(profile: RepoProfile) -> None:
    postgres.init_db()
    session_factory = postgres.get_session_factory()
    with session_factory() as session:
        existing = (
            session.query(postgres.RepoProfileRecord)
            .filter_by(repo_id=profile.repo_id)
            .one_or_none()
        )
        record = existing or postgres.RepoProfileRecord(repo_id=profile.repo_id)
        record.repo_url = profile.repo_url
        record.primary_languages = profile.primary_languages
        record.commit_quality_score = profile.commit_quality_score
        record.pr_density = profile.pr_density
        record.has_adrs = profile.has_adrs
        record.adr_paths = profile.adr_paths
        record.has_issue_linking = profile.has_issue_linking
        record.slack_export_path = profile.slack_export_path
        record.total_commits = profile.total_commits
        record.date_range_start = profile.date_range[0]
        record.date_range_end = profile.date_range[1]
        record.fingerprinted_at = datetime.utcnow()
        if not existing:
            session.add(record)
        session.commit()


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python -m src.ingestion.repo_config_agent <owner/repo>")
        sys.exit(1)

    repo_id = sys.argv[1]
    profile = fingerprint_repo(repo_id)
    save_repo_profile(profile)

    print(profile.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
