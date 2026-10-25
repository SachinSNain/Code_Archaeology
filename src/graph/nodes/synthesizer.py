"""Phase 5a — Answer Synthesis Agent node.

Generates a structured, cited answer from the ranked context, provenance
class, and optional evolution timeline. Every factual claim must be grounded
in retrieved evidence — the system prompt explicitly instructs the model to
say so when context doesn't support a claim, rather than inferring, since an
ungrounded-but-plausible answer here is worse than an honest "unknown."

Citation URLs are built programmatically from repo_id + source_id rather than
trusted from the LLM output, so a hallucinated PR number still points at a
real (possibly 404) GitHub URL instead of a fabricated one.
"""
import re

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel

from src.graph.state import ArchaeologyState, Citation, ProvenanceClass, RetrievedChunk
from src.llm.llm_factory import get_llm

# Phase 4d table: synthesis prompt adjustment per provenance class.
PROVENANCE_ADJUSTMENTS: dict[ProvenanceClass, str] = {
    "well_documented": "Standard synthesis.",
    "legacy_migration": "This code was migrated from an older system. The original rationale may predate this repository.",
    "hotfix": "This appears to be an emergency fix. Look for the issue it resolved.",
    "experimental_prototype": "This was experimental code. It may not represent a deliberate design decision.",
    "generated_code": "This is generated code. Query the generator configuration, not this file.",
    "third_party_import": "This appears to be imported from a third-party source. The design decisions are upstream.",
    "orphaned": "No provenance chain could be reconstructed. This code may be safe to remove.",
}

GENERATED_CODE_SHORT_CIRCUIT_ANSWER = (
    "This code is generated (provenance class: generated_code). Its content is produced by a generator/template "
    "rather than hand-authored, so there is no independent design rationale to reconstruct here — the real "
    "decision context lives in whatever generates it (e.g. a schema, proto definition, or codegen script). "
    "Query the generator's configuration or source instead of this file."
)

SYSTEM_PROMPT = """You are a code archaeology assistant. Answer questions about why code \
exists by reasoning from the provided historical evidence. Every claim you make must be \
grounded in the provided context. If the context does not support a claim, say so explicitly \
rather than inferring.

Tag every factual claim inline with its source, e.g. [PR #4431], [Commit a3f92b], [Issue #891], \
[ADR some-doc-id]. Structure the answer with these sections:
- Direct answer (2-3 sentences)
- Historical context (what motivated this)
- Key decision points (alternatives considered, from PR discussions, if any)
- Current state (how it evolved to what it is now)

Also return a separate structured list of every citation used, each with: source_type (pr|commit|issue|adr), \
source_id (the number/sha/doc_id, no '#' or prefix), and a short excerpt from the source material \
supporting the claim."""


class _CitationOut(BaseModel):
    source_type: str
    source_id: str
    excerpt: str


class _SynthesisOutput(BaseModel):
    answer: str
    citations: list[_CitationOut]


def _build_github_url(repo_id: str, source_type: str, source_id: str) -> str:
    base = f"https://github.com/{repo_id}"
    if source_type == "pr":
        return f"{base}/pull/{source_id}"
    if source_type == "commit":
        return f"{base}/commit/{source_id}"
    if source_type == "issue":
        return f"{base}/issues/{source_id}"
    if source_type == "adr":
        return f"{base}/blob/HEAD/{source_id}"
    if source_type == "code_chunk":
        # chunk_id format is "{file_path}:{start_line}-{end_line}:{index}" (see
        # src/parsing/ast_parser.py) — parse it back out into a line-anchored blob URL.
        match = re.match(r"^(?P<path>.+):(?P<start>\d+)-(?P<end>\d+):\d+$", source_id)
        if match:
            return f"{base}/blob/HEAD/{match['path']}#L{match['start']}-L{match['end']}"
    return base


def _format_context(chunks: list[RetrievedChunk]) -> str:
    lines = []
    for c in chunks:
        score_str = f"{c.score:.3f}" if c.score is not None else "n/a"
        lines.append(f"[{c.source_type} {c.source_id}] (score={score_str})\n{c.text[:1500]}")
    return "\n\n---\n\n".join(lines) if lines else "(no retrieved context)"


def _format_evolution(state: ArchaeologyState) -> str:
    timeline = state.get("evolution_timeline")
    if not timeline:
        return "(not applicable to this query)"
    lines = [f"{e.date.date()} — {e.event_type}: {e.description}" for e in timeline]
    return "\n".join(lines)


def synthesizer_node(state: ArchaeologyState) -> dict:
    provenance_class: ProvenanceClass = state.get("provenance_class", "well_documented")

    if provenance_class == "generated_code":
        return {"answer": GENERATED_CODE_SHORT_CIRCUIT_ANSWER, "citations": []}

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            (
                "human",
                "Provenance context: {provenance_context}\n\n"
                "Retrieved context (ranked by relevance):\n{context}\n\n"
                "Evolution timeline (if available):\n{evolution}\n\n"
                "Query: {query}",
            ),
        ]
    )
    chain = prompt | get_llm(purpose="default", temperature=0.1).with_structured_output(_SynthesisOutput)
    result: _SynthesisOutput = chain.invoke(
        {
            "provenance_context": f"{provenance_class}: {PROVENANCE_ADJUSTMENTS[provenance_class]}",
            "context": _format_context(state.get("ranked_context") or []),
            "evolution": _format_evolution(state),
            "query": state["query"],
        }
    )

    repo_id = state["repo_id"]
    citations = [
        Citation(
            source_type=c.source_type,
            source_id=c.source_id,
            url=_build_github_url(repo_id, c.source_type, c.source_id),
            excerpt=c.excerpt,
        )
        for c in result.citations
    ]

    return {"answer": result.answer, "citations": citations}
