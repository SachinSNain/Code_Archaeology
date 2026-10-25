"""Neo4j driver + Cypher query library — the provenance/blame graph store.

Neo4j answers "WHAT CAUSED this?" via multi-hop traversal:
function -> commit -> PR -> issue -> ADR. See WHAT_TO_DO.MD Phase 2b for the
node/edge schema this module implements.

Community Edition does not support composite uniqueness constraints, so every
node carries a synthetic `uid` property (f"{repo_id}:{natural_key}") with a
single-property uniqueness constraint. This keeps repo_id-scoped uniqueness
correct without requiring Enterprise Edition.
"""
from contextlib import contextmanager
from typing import Any, Iterable, Optional

from neo4j import Driver, GraphDatabase

from src.config.settings import get_settings

_driver: Optional[Driver] = None


def get_driver() -> Driver:
    global _driver
    if _driver is None:
        settings = get_settings()
        _driver = GraphDatabase.driver(
            settings.neo4j_url, auth=(settings.neo4j_username, settings.neo4j_password)
        )
    return _driver


def close_driver() -> None:
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None


@contextmanager
def session():
    with get_driver().session() as s:
        yield s


CONSTRAINTS = [
    "CREATE CONSTRAINT function_uid IF NOT EXISTS FOR (f:Function) REQUIRE f.uid IS UNIQUE",
    "CREATE CONSTRAINT commit_uid IF NOT EXISTS FOR (c:Commit) REQUIRE c.uid IS UNIQUE",
    "CREATE CONSTRAINT pr_uid IF NOT EXISTS FOR (p:PR) REQUIRE p.uid IS UNIQUE",
    "CREATE CONSTRAINT issue_uid IF NOT EXISTS FOR (i:Issue) REQUIRE i.uid IS UNIQUE",
    "CREATE CONSTRAINT adr_uid IF NOT EXISTS FOR (a:ADR) REQUIRE a.uid IS UNIQUE",
    "CREATE INDEX function_repo_id IF NOT EXISTS FOR (f:Function) ON (f.repo_id)",
    "CREATE INDEX commit_repo_id IF NOT EXISTS FOR (c:Commit) ON (c.repo_id)",
]


def init_constraints() -> None:
    with session() as s:
        for stmt in CONSTRAINTS:
            s.run(stmt)


def uid(repo_id: str, natural_key: Any) -> str:
    return f"{repo_id}:{natural_key}"


# ---- Node upserts -----------------------------------------------------

def merge_function(tx, *, chunk_id: str, repo_id: str, file_path: str,
                    function_name: Optional[str], language: str) -> None:
    tx.run(
        """
        MERGE (f:Function {uid: $uid})
        SET f.chunk_id = $chunk_id, f.repo_id = $repo_id, f.file_path = $file_path,
            f.function_name = $function_name, f.language = $language
        """,
        uid=uid(repo_id, chunk_id), chunk_id=chunk_id, repo_id=repo_id,
        file_path=file_path, function_name=function_name, language=language,
    )


def merge_commit(tx, *, sha: str, repo_id: str, message: Optional[str],
                  author: Optional[str], timestamp: Optional[str]) -> None:
    tx.run(
        """
        MERGE (c:Commit {uid: $uid})
        SET c.sha = $sha, c.repo_id = $repo_id, c.message = $message,
            c.author = $author, c.timestamp = $timestamp
        """,
        uid=uid(repo_id, sha), sha=sha, repo_id=repo_id, message=message,
        author=author, timestamp=timestamp,
    )


def merge_pr(tx, *, pr_number: int, repo_id: str, title: Optional[str],
             merged_at: Optional[str]) -> None:
    tx.run(
        """
        MERGE (p:PR {uid: $uid})
        SET p.pr_number = $pr_number, p.repo_id = $repo_id, p.title = $title,
            p.merged_at = $merged_at
        """,
        uid=uid(repo_id, pr_number), pr_number=pr_number, repo_id=repo_id,
        title=title, merged_at=merged_at,
    )


def merge_issue(tx, *, issue_number: int, repo_id: str, title: Optional[str],
                 created_at: Optional[str]) -> None:
    tx.run(
        """
        MERGE (i:Issue {uid: $uid})
        SET i.issue_number = $issue_number, i.repo_id = $repo_id, i.title = $title,
            i.created_at = $created_at
        """,
        uid=uid(repo_id, issue_number), issue_number=issue_number, repo_id=repo_id,
        title=title, created_at=created_at,
    )


def merge_adr(tx, *, doc_id: str, repo_id: str, title: Optional[str], doc_type: str) -> None:
    tx.run(
        """
        MERGE (a:ADR {uid: $uid})
        SET a.doc_id = $doc_id, a.repo_id = $repo_id, a.title = $title, a.doc_type = $doc_type
        """,
        uid=uid(repo_id, doc_id), doc_id=doc_id, repo_id=repo_id, title=title, doc_type=doc_type,
    )


# ---- Edge upserts -------------------------------------------------------

def merge_edge(tx, *, label_a: str, uid_a: str, rel: str, label_b: str, uid_b: str) -> None:
    tx.run(
        f"""
        MATCH (a:{label_a} {{uid: $uid_a}}), (b:{label_b} {{uid: $uid_b}})
        MERGE (a)-[:{rel}]->(b)
        """,
        uid_a=uid_a, uid_b=uid_b,
    )


# ---- Provenance traversal (Phase 2b / used by Phase 4b Neo4j retrieval) ----

PROVENANCE_QUERY = """
MATCH (f:Function {uid: $function_uid})
OPTIONAL MATCH path1 = (f)-[:INTRODUCED_BY]->(introduced:Commit)-[:PART_OF_PR]->(pr:PR)-[:ADDRESSES]->(issue:Issue)
OPTIONAL MATCH path2 = (f)-[:LAST_MODIFIED_BY]->(modified:Commit)
OPTIONAL MATCH (issue)-[:INFORMED_BY]->(adr:ADR)
RETURN f, introduced, pr, issue, modified, adr
"""


def fetch_provenance_chain(repo_id: str, chunk_id: str) -> list[dict]:
    with session() as s:
        result = s.run(PROVENANCE_QUERY, function_uid=uid(repo_id, chunk_id))
        return [record.data() for record in result]


def fetch_all_history_for_function(repo_id: str, chunk_id: str) -> list[dict]:
    """All commits that ever touched this function, chronologically (Phase 4e evolution agent)."""
    query = """
    MATCH (f:Function {uid: $function_uid})<-[:MODIFIES]-(c:Commit)
    OPTIONAL MATCH (c)-[:PART_OF_PR]->(pr:PR)
    OPTIONAL MATCH (pr)-[:ADDRESSES]->(issue:Issue)
    RETURN c, pr, issue
    ORDER BY c.timestamp ASC
    """
    with session() as s:
        result = s.run(query, function_uid=uid(repo_id, chunk_id))
        return [record.data() for record in result]
