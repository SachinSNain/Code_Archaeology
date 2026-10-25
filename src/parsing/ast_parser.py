"""Phase 2a — ASTCodeParserAgent.

Parses every source file with tree-sitter and chunks at the function/class
boundary. Critical rule: chunking is NEVER done by line count or sliding
window — a chunk is exactly the byte range of one AST node (a function,
method, or class definition), so a chunk is always semantically complete.

tree-sitter is fault-tolerant: a syntactically broken file still yields a
partial AST (error nodes) instead of raising, so history that contains
temporarily-broken commits doesn't crash ingestion.

`last_modified_sha` is computed once per file via a single `git blame` call
(not once per chunk) and then looked up per chunk by line range — blame is
the expensive operation here, so batching it per file instead of per function
avoids O(functions) subprocess calls on large files.

Usage:
    python -m src.parsing.ast_parser tiangolo/fastapi
"""
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

import git
from pydantic import BaseModel
from sqlalchemy.dialects.postgresql import insert
from tqdm import tqdm
from tree_sitter import Language, Node, Parser

import tree_sitter_go as tsgo
import tree_sitter_java as tsjava
import tree_sitter_javascript as tsjavascript
import tree_sitter_python as tspython
import tree_sitter_rust as tsrust
import tree_sitter_typescript as tstypescript

from src.db import postgres
from src.ingestion._git_utils import clone_or_open, repo_id_to_url

SKIP_DIRS = {
    ".git", "node_modules", "dist", "build", "venv", ".venv", "__pycache__",
    ".mypy_cache", "vendor", ".tox", "site-packages", "egg-info",
}

EXTENSION_TO_LANGUAGE = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".mjs": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".go": "go",
    ".java": "java",
    ".rs": "rust",
}

FUNCTION_NODE_TYPES = {
    "python": {"function_definition"},
    "javascript": {"function_declaration", "method_definition", "generator_function_declaration"},
    "typescript": {"function_declaration", "method_definition", "generator_function_declaration"},
    "tsx": {"function_declaration", "method_definition", "generator_function_declaration"},
    "go": {"function_declaration", "method_declaration"},
    "java": {"method_declaration", "constructor_declaration"},
    "rust": {"function_item"},
}

CLASS_NODE_TYPES = {
    "python": {"class_definition"},
    "javascript": {"class_declaration"},
    "typescript": {"class_declaration", "interface_declaration"},
    "tsx": {"class_declaration", "interface_declaration"},
    "go": set(),  # Go has no classes
    "java": {"class_declaration", "interface_declaration"},
    "rust": {"impl_item", "struct_item", "trait_item"},
}

CALL_NODE_TYPES = {
    "python": {"call"},
    "javascript": {"call_expression"},
    "typescript": {"call_expression"},
    "tsx": {"call_expression"},
    "go": {"call_expression"},
    "java": {"method_invocation"},
    "rust": {"call_expression"},
}


class CodeChunk(BaseModel):
    chunk_id: str
    repo_id: str
    file_path: str
    function_name: Optional[str] = None
    class_name: Optional[str] = None
    language: str
    start_line: int
    end_line: int
    chunk_text: str
    last_modified_sha: Optional[str] = None
    docstring: Optional[str] = None
    calls: list[str] = []
    called_by: list[str] = []  # populated later via static analysis in blame_graph.py


@lru_cache(maxsize=None)
def get_parser(language: str) -> Parser:
    if language == "python":
        lang = Language(tspython.language())
    elif language == "javascript":
        lang = Language(tsjavascript.language())
    elif language == "typescript":
        lang = Language(tstypescript.language_typescript())
    elif language == "tsx":
        lang = Language(tstypescript.language_tsx())
    elif language == "go":
        lang = Language(tsgo.language())
    elif language == "java":
        lang = Language(tsjava.language())
    elif language == "rust":
        lang = Language(tsrust.language())
    else:
        raise ValueError(f"Unsupported language: {language}")
    return Parser(lang)


def _node_name(node: Node, source: bytes) -> Optional[str]:
    for field in ("name", "type"):
        child = node.child_by_field_name(field)
        if child is not None:
            return source[child.start_byte:child.end_byte].decode("utf-8", "replace")
    return None


def _extract_docstring(node: Node, language: str, source: bytes) -> Optional[str]:
    if language == "python":
        body = node.child_by_field_name("body")
        if body is not None and body.named_child_count > 0:
            first = body.named_children[0]
            if first.type == "expression_statement" and first.named_child_count > 0:
                expr = first.named_children[0]
                if expr.type == "string":
                    text = source[expr.start_byte:expr.end_byte].decode("utf-8", "replace")
                    return text.strip().strip('"""').strip("'''").strip()
        return None

    # Generic fallback: consecutive leading comment lines directly above the node.
    comments = []
    prev = node.prev_sibling
    while prev is not None and prev.type == "comment":
        comments.insert(0, source[prev.start_byte:prev.end_byte].decode("utf-8", "replace"))
        prev = prev.prev_sibling
    return "\n".join(comments) if comments else None


def _extract_calls(node: Node, language: str, source: bytes) -> list[str]:
    call_types = CALL_NODE_TYPES.get(language, set())
    if not call_types:
        return []
    names: set[str] = set()

    def walk(n: Node) -> None:
        if n.type in call_types:
            callee = n.child_by_field_name("function") or n.child_by_field_name("name")
            if callee is None and n.named_child_count > 0:
                callee = n.named_children[0]
            if callee is not None:
                text = source[callee.start_byte:callee.end_byte].decode("utf-8", "replace")
                simple = text.replace("::", ".").split(".")[-1].strip()
                if simple.isidentifier():
                    names.add(simple)
        for child in n.children:
            walk(child)

    walk(node)
    return sorted(names)


def _walk_chunks(node: Node, language: str, source: bytes, enclosing_class: Optional[str], out: list[dict]) -> None:
    func_types = FUNCTION_NODE_TYPES.get(language, set())
    class_types = CLASS_NODE_TYPES.get(language, set())

    if node.type in func_types:
        out.append(
            {
                "kind": "function",
                "node": node,
                "function_name": _node_name(node, source),
                "class_name": enclosing_class,
            }
        )
        # Still descend, to catch nested/closure functions inside this one.
        for child in node.children:
            _walk_chunks(child, language, source, enclosing_class, out)
        return

    if node.type in class_types:
        class_name = _node_name(node, source)
        out.append({"kind": "class", "node": node, "function_name": None, "class_name": class_name})
        for child in node.children:
            _walk_chunks(child, language, source, class_name, out)
        return

    for child in node.children:
        _walk_chunks(child, language, source, enclosing_class, out)


def parse_file(repo_id: str, file_path: Path, rel_path: str, blame_line_shas: list[Optional[str]]) -> list[CodeChunk]:
    language = EXTENSION_TO_LANGUAGE.get(file_path.suffix.lower())
    if language is None:
        return []

    try:
        source = file_path.read_bytes()
    except OSError:
        return []

    parser = get_parser(language)
    tree = parser.parse(source)  # tolerant of syntax errors: yields a partial tree, never raises

    raw_chunks: list[dict] = []
    _walk_chunks(tree.root_node, language, source, None, raw_chunks)

    chunks = []
    for i, raw in enumerate(raw_chunks):
        node: Node = raw["node"]
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        chunk_text = source[node.start_byte:node.end_byte].decode("utf-8", "replace")

        shas_in_range = [s for s in blame_line_shas[start_line - 1:end_line] if s]
        last_modified_sha = shas_in_range[-1] if shas_in_range else None

        chunks.append(
            CodeChunk(
                chunk_id=f"{rel_path}:{start_line}-{end_line}:{i}",
                repo_id=repo_id,
                file_path=rel_path,
                function_name=raw["function_name"],
                class_name=raw["class_name"],
                language="typescript" if language == "tsx" else language,
                start_line=start_line,
                end_line=end_line,
                chunk_text=chunk_text,
                last_modified_sha=last_modified_sha,
                docstring=_extract_docstring(node, language, source),
                calls=_extract_calls(node, language, source),
                called_by=[],
            )
        )
    return chunks


def _blame_line_shas(repo: git.Repo, rel_path: str) -> list[Optional[str]]:
    """One `git blame` per file -> flat list where index i is the commit SHA
    that last touched line (i+1). Binary/unreadable files degrade to all-None
    rather than raising, since blame is a best-effort enrichment, not required
    for a chunk to be valid.
    """
    try:
        blame_entries = repo.blame("HEAD", rel_path)
    except Exception:
        return []
    shas: list[Optional[str]] = []
    for commit, lines in blame_entries:
        shas.extend([commit.hexsha] * len(lines))
    return shas


def _iter_source_files(repo_root: Path):
    for path in repo_root.rglob("*"):
        if not path.is_file():
            continue
        if any(part in SKIP_DIRS for part in path.parts):
            continue
        if path.suffix.lower() in EXTENSION_TO_LANGUAGE:
            yield path


def _upsert_chunks(repo_id: str, chunks: list[CodeChunk]) -> None:
    if not chunks:
        return
    rows = [
        {
            "repo_id": c.repo_id,
            "chunk_id": c.chunk_id,
            "file_path": c.file_path,
            "function_name": c.function_name,
            "class_name": c.class_name,
            "language": c.language,
            "start_line": c.start_line,
            "end_line": c.end_line,
            "chunk_text": c.chunk_text,
            "last_modified_sha": c.last_modified_sha,
            "docstring": c.docstring,
            "calls": c.calls,
            "called_by": c.called_by,
        }
        for c in chunks
    ]
    session_factory = postgres.get_session_factory()
    with session_factory() as session:
        stmt = insert(postgres.CodeChunkRecord).values(rows)
        stmt = stmt.on_conflict_do_update(
            index_elements=["repo_id", "chunk_id"],
            set_={
                c.name: getattr(stmt.excluded, c.name)
                for c in postgres.CodeChunkRecord.__table__.columns
                if c.name not in ("id", "repo_id", "chunk_id")
            },
        )
        session.execute(stmt)
        session.commit()


def parse_repo(repo_id: str) -> int:
    postgres.init_db()
    repo = clone_or_open(repo_id, repo_id_to_url(repo_id))
    repo_root = Path(repo.working_tree_dir)

    files = list(_iter_source_files(repo_root))
    print(f"Found {len(files)} source file(s) to parse for {repo_id}")

    total_chunks = 0
    batch: list[CodeChunk] = []
    for file_path in tqdm(files, desc=f"ast_parser:{repo_id}"):
        rel_path = str(file_path.relative_to(repo_root)).replace("\\", "/")
        blame_shas = _blame_line_shas(repo, rel_path)
        chunks = parse_file(repo_id, file_path, rel_path, blame_shas)
        batch.extend(chunks)
        total_chunks += len(chunks)
        if len(batch) >= 500:
            _upsert_chunks(repo_id, batch)
            batch = []
    _upsert_chunks(repo_id, batch)

    print(f"Done. Parsed {total_chunks} function/class chunk(s) for {repo_id}.")
    return total_chunks


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python -m src.parsing.ast_parser <owner/repo>")
        sys.exit(1)
    parse_repo(sys.argv[1])


if __name__ == "__main__":
    main()
