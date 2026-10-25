"""Verifies the critical rule: chunks are AST function/class boundaries, never line-count windows."""
from pathlib import Path

from src.parsing.ast_parser import parse_file


def _write_and_parse(tmp_path: Path, filename: str, source: str, repo_id: str = "test/repo"):
    file_path = tmp_path / filename
    file_path.write_text(source, encoding="utf-8")
    return parse_file(repo_id, file_path, filename, blame_line_shas=[])


def test_python_function_chunk_boundaries_match_def_exactly(tmp_path):
    source = (
        "def helper():\n"
        "    return 1\n"
        "\n"
        "\n"
        "def main():\n"
        "    x = helper()\n"
        "    return x + 1\n"
    )
    chunks = _write_and_parse(tmp_path, "sample.py", source)
    names = {c.function_name for c in chunks}
    assert names == {"helper", "main"}

    main_chunk = next(c for c in chunks if c.function_name == "main")
    assert main_chunk.start_line == 5
    assert main_chunk.end_line == 7
    assert main_chunk.chunk_text.startswith("def main():")
    assert "helper" in main_chunk.calls


def test_python_class_and_method_both_chunked(tmp_path):
    source = (
        "class Greeter:\n"
        "    def greet(self, name):\n"
        "        return f'hello {name}'\n"
    )
    chunks = _write_and_parse(tmp_path, "sample.py", source)
    kinds = {(c.class_name, c.function_name) for c in chunks}
    assert ("Greeter", None) in kinds  # the class itself is a chunk
    assert ("Greeter", "greet") in kinds  # the method is a separate chunk, with enclosing class recorded


def test_python_docstring_extracted(tmp_path):
    source = 'def documented():\n    """This explains the function."""\n    return 42\n'
    chunks = _write_and_parse(tmp_path, "sample.py", source)
    assert len(chunks) == 1
    assert "This explains the function." in chunks[0].docstring


def test_unsupported_extension_yields_no_chunks(tmp_path):
    chunks = _write_and_parse(tmp_path, "README.md", "# Just markdown, not code\n")
    assert chunks == []


def test_syntactically_broken_file_does_not_raise(tmp_path):
    # tree-sitter must be fault-tolerant: a partial/broken AST, not a crash.
    broken_source = "def broken(:\n    this is not valid python at all !!!\n"
    chunks = _write_and_parse(tmp_path, "broken.py", broken_source)
    assert isinstance(chunks, list)  # did not raise
