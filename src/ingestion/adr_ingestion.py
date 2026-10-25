"""Phase 1c — ADRIngestionAgent.

Scans the repo for architectural decision records, RFCs, design proposals,
and CHANGELOG entries, using RepoProfile.adr_paths from Phase 0 to know where
to look, falling back to common conventions (docs/, rfcs/, proposals/,
DECISIONS.md, ARCHITECTURE.md) when nothing is known in advance.

Per-repo knowledge (see src/config/repo_profiles.py):
  - Kubernetes: docs/proposals/ and keps/
  - Airflow: docs/apache-airflow-providers/
  - LangChain: GitHub Discussions (best-effort via GraphQL) + docs/
  - FastAPI: no ADR directory -> issues tagged 'design' or 'discussion'

Each ingested document is tagged with a `doc_type`: adr, rfc, design_proposal,
changelog_entry, slack_thread, github_discussion. This enables filtered
retrieval in Phase 4 (a "why" query weights ADRs/RFCs highly; a "what" query
ignores them entirely).

Usage:
    python -m src.ingestion.adr_ingestion tiangolo/fastapi
"""
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests
from sqlalchemy.dialects.postgresql import insert
from unstructured.partition.auto import partition
from unstructured.partition.md import partition_md
from unstructured.partition.text import partition_text

from src.config.repo_profiles import get_known_hints
from src.config.settings import get_settings
from src.db import postgres
from src.ingestion._git_utils import clone_or_open, repo_id_to_url

DOC_EXTENSIONS = {".md", ".mdx", ".rst", ".txt", ".pdf"}
FALLBACK_SCAN_PATHS = ["docs/", "rfcs/", "proposals/", "DECISIONS.md", "ARCHITECTURE.md"]


def _extract_title(text: str, fallback: str) -> str:
    lines = text.splitlines()
    # Prefer an actual markdown heading over the first non-empty line: unstructured's
    # markdown partitioner emits admonition markers (e.g. "/// warning") as their own
    # element before the real heading, which would otherwise win as the "title".
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("#"):
            return stripped.lstrip("#").strip()
    for line in lines:
        stripped = line.strip()
        if stripped:
            return stripped
    return fallback


def _partition_file(path: Path) -> str:
    # Dispatch markdown/text directly rather than through partition.auto: auto-detection
    # shells out to libmagic, which has no native Windows install and makes every file
    # fail with "failed to find libmagic" there. .md/.txt are the overwhelming majority
    # of ADR/RFC sources in practice; partition.auto is kept only as the fallback for
    # everything else (PDFs, etc.), where libmagic's absence will surface per-file
    # instead of silently zeroing out ingestion on Windows.
    try:
        suffix = path.suffix.lower()
        if suffix in (".md", ".mdx"):
            elements = partition_md(filename=str(path))
        elif suffix in (".txt",):
            elements = partition_text(filename=str(path))
        else:
            elements = partition(filename=str(path))
        return "\n\n".join(el.text for el in elements if getattr(el, "text", None))
    except Exception as exc:
        print(f"  skip (unparseable): {path} ({exc})")
        return ""


def _scan_local_docs(repo_root: Path, candidate_paths: list[str], doc_type: str) -> list[dict]:
    docs = []
    for rel_path in candidate_paths:
        target = repo_root / rel_path
        if not target.exists():
            continue
        files = [target] if target.is_file() else [
            p for p in target.rglob("*") if p.is_file() and p.suffix.lower() in DOC_EXTENSIONS
        ]
        for file_path in files:
            body = _partition_file(file_path)
            if not body.strip():
                continue
            rel = str(file_path.relative_to(repo_root)).replace("\\", "/")
            docs.append(
                {
                    "doc_id": rel,
                    "doc_type": doc_type,
                    "title": _extract_title(body, fallback=file_path.name),
                    "source_path": rel,
                    "body": body,
                    "created_at": None,
                }
            )
    return docs


def _scan_labeled_issues(repo_id: str, label_hints: list[str], doc_type: str) -> list[dict]:
    """Fallback for repos with no ADR directory (e.g. FastAPI): treat issues
    tagged 'design'/'discussion' as design-decision documents. Reads from
    Postgres `issues` (already ingested in Phase 1b) rather than hitting the
    GitHub API again.
    """
    if not label_hints:
        return []
    session_factory = postgres.get_session_factory()
    docs = []
    with session_factory() as session:
        issues = session.query(postgres.Issue).filter_by(repo_id=repo_id).all()
        for issue in issues:
            labels = set(issue.labels or [])
            if not labels.intersection(label_hints):
                continue
            comments_text = "\n\n".join(
                f"[{c.get('author')}]: {c.get('body')}" for c in (issue.comment_thread or [])
            )
            body = f"{issue.body or ''}\n\n{comments_text}".strip()
            if not body:
                continue
            docs.append(
                {
                    "doc_id": f"issue-{issue.issue_number}",
                    "doc_type": doc_type,
                    "title": issue.title or f"Issue #{issue.issue_number}",
                    "source_path": f"issues/{issue.issue_number}",
                    "body": body,
                    "created_at": issue.created_at,
                }
            )
    return docs


def _scan_github_discussions(repo_id: str, token: Optional[str]) -> list[dict]:
    """Best-effort: GitHub Discussions live behind the GraphQL API (no REST
    endpoint), so this degrades gracefully to an empty list rather than
    failing ingestion when no token is available or the repo has Discussions
    disabled — consistent with Phase 0's rule that missing data sources must
    be flagged, not treated as a hard failure.
    """
    if not token:
        return []
    owner, name = repo_id.split("/")
    query = """
    query($owner: String!, $name: String!, $after: String) {
      repository(owner: $owner, name: $name) {
        discussions(first: 50, after: $after) {
          pageInfo { hasNextPage endCursor }
          nodes { number title bodyText createdAt }
        }
      }
    }
    """
    docs, after = [], None
    try:
        while True:
            resp = requests.post(
                "https://api.github.com/graphql",
                json={"query": query, "variables": {"owner": owner, "name": name, "after": after}},
                headers={"Authorization": f"Bearer {token}"},
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()["data"]["repository"]["discussions"]
            for node in data["nodes"]:
                docs.append(
                    {
                        "doc_id": f"discussion-{node['number']}",
                        "doc_type": "github_discussion",
                        "title": node["title"],
                        "source_path": f"discussions/{node['number']}",
                        "body": node["bodyText"],
                        "created_at": node["createdAt"],
                    }
                )
            if not data["pageInfo"]["hasNextPage"]:
                break
            after = data["pageInfo"]["endCursor"]
    except Exception as exc:
        print(f"  GitHub Discussions unavailable for {repo_id}: {exc}")
    return docs


def _upsert_docs(repo_id: str, docs: list[dict]) -> None:
    if not docs:
        return
    rows = []
    for d in docs:
        created_at = d["created_at"]
        if isinstance(created_at, str):
            created_at = datetime.fromisoformat(created_at.replace("Z", "+00:00"))
        rows.append({**d, "repo_id": repo_id, "created_at": created_at})

    session_factory = postgres.get_session_factory()
    with session_factory() as session:
        stmt = insert(postgres.ADRDocument).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["repo_id", "doc_id"],
            set_={
                "doc_type": stmt.excluded.doc_type,
                "title": stmt.excluded.title,
                "source_path": stmt.excluded.source_path,
                "body": stmt.excluded.body,
                "created_at": stmt.excluded.created_at,
            },
        )
        session.execute(stmt)
        session.commit()


def ingest_adrs(repo_id: str) -> int:
    postgres.init_db()
    settings = get_settings()

    session_factory = postgres.get_session_factory()
    with session_factory() as session:
        profile = session.query(postgres.RepoProfileRecord).filter_by(repo_id=repo_id).one_or_none()

    hints = get_known_hints(repo_id)
    adr_paths = list((profile.adr_paths if profile else []) or []) or list(hints.adr_paths if hints else [])
    adr_paths = adr_paths + FALLBACK_SCAN_PATHS
    doc_type = hints.adr_doc_type if hints else "adr"

    repo = clone_or_open(repo_id, repo_id_to_url(repo_id))
    repo_root = Path(repo.working_tree_dir)

    docs = _scan_local_docs(repo_root, adr_paths, doc_type)

    if hints and hints.issue_label_hints:
        docs += _scan_labeled_issues(repo_id, hints.issue_label_hints, doc_type)

    docs += _scan_github_discussions(repo_id, settings.github_token)

    _upsert_docs(repo_id, docs)
    print(f"Ingested {len(docs)} ADR/RFC/design document(s) for {repo_id} (doc_type={doc_type}).")
    return len(docs)


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python -m src.ingestion.adr_ingestion <owner/repo>")
        sys.exit(1)
    ingest_adrs(sys.argv[1])


if __name__ == "__main__":
    main()
