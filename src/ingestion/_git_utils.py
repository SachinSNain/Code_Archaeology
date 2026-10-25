"""Shared local-clone helper for ingestion agents that need a working tree
(RepoConfigAgent for fingerprinting, GitIngestionAgent for commit walking,
ASTCodeParserAgent for reading file contents).

Kept as a private module (leading underscore) rather than a public API —
it exists purely to avoid duplicating clone/open logic across three callers.
"""
import re
from pathlib import Path

import git

from src.config.settings import get_settings


def repo_id_to_url(repo_id: str) -> str:
    return f"https://github.com/{repo_id}.git"


def local_clone_path(repo_id: str) -> Path:
    settings = get_settings()
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "__", repo_id)
    return Path(settings.repo_clone_dir) / safe_name


def _sync_working_tree_to_default_branch(repo: git.Repo) -> None:
    """Fast-forward the checked-out working tree to the remote's default branch HEAD.

    `fetch()` alone updates remote-tracking refs but not the working tree, which
    would leave `git blame` / file reads (AST parsing, ADR scanning) looking at
    stale content relative to what GitIngestionAgent just recorded in Postgres.
    """
    default_ref = repo.remotes.origin.refs.HEAD.reference
    repo.git.checkout(default_ref.remote_head)
    repo.git.reset("--hard", default_ref.path)


def clone_or_open(repo_id: str, repo_url: str | None = None) -> git.Repo:
    """Clone the repo if not present locally, otherwise open, fetch, and sync to HEAD."""
    path = local_clone_path(repo_id)
    if path.exists() and (path / ".git").exists():
        repo = git.Repo(path)
        repo.remotes.origin.fetch()
        _sync_working_tree_to_default_branch(repo)
        return repo

    path.parent.mkdir(parents=True, exist_ok=True)
    url = repo_url or repo_id_to_url(repo_id)
    return git.Repo.clone_from(url, path)
