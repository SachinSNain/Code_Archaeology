"""Known per-repository configuration hints.

RepoConfigAgent (Phase 0) fingerprints a repo automatically, but some repos
have well-known, non-discoverable conventions (e.g. Kubernetes KEPs live in
`keps/`, not `docs/adr/`). This table seeds the fingerprinter with hints;
the fingerprinter still verifies each path exists in the actual repo before
trusting it, so a stale entry here degrades gracefully rather than breaking
ingestion.
"""
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class KnownRepoHints:
    repo_id: str
    adr_paths: List[str] = field(default_factory=list)
    adr_doc_type: str = "adr"
    issue_label_hints: List[str] = field(default_factory=list)
    notes: Optional[str] = None


KNOWN_REPO_HINTS: dict[str, KnownRepoHints] = {
    "kubernetes/kubernetes": KnownRepoHints(
        repo_id="kubernetes/kubernetes",
        adr_paths=["docs/proposals/", "keps/"],
        adr_doc_type="design_proposal",
        notes="KEPs (Kubernetes Enhancement Proposals) are the primary ADR source.",
    ),
    "apache/airflow": KnownRepoHints(
        repo_id="apache/airflow",
        adr_paths=["docs/apache-airflow-providers/"],
        adr_doc_type="rfc",
        notes="AIPs (Airflow Improvement Proposals) are tracked mostly on the wiki/mailing list; "
        "in-repo docs are supplementary.",
    ),
    "langchain-ai/langchain": KnownRepoHints(
        repo_id="langchain-ai/langchain",
        adr_paths=["docs/"],
        adr_doc_type="design_proposal",
        notes="Most architectural discussion happens in GitHub Discussions, pulled separately via API.",
    ),
    "tiangolo/fastapi": KnownRepoHints(
        repo_id="tiangolo/fastapi",
        adr_paths=[],
        adr_doc_type="changelog_entry",
        issue_label_hints=["design", "discussion"],
        notes="FastAPI has no formal ADR directory; design discussion lives in issues tagged "
        "'design' or 'discussion', and in docs_src/ changelog entries.",
    ),
}


def get_known_hints(repo_id: str) -> Optional[KnownRepoHints]:
    return KNOWN_REPO_HINTS.get(repo_id)
