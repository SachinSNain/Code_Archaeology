"""Phase 4a — Query Classifier node.

Per the updated brief: deterministic Python first, LLM second. Keyword/regex
rules handle the common cases at effectively zero cost and <5ms latency; an
LLM call (via llm_factory.get_llm("classifier"), critical rule #1 — no node
instantiates a chat model directly) is reserved for queries the rules can't
confidently classify. The fallthrough rate is tracked and logged — if it
exceeds 10%, the rules need to be improved (that threshold is a maintenance
signal, not a hard failure).

Rules are checked in priority order because query text often contains more
than one trigger word (e.g. "why did the evolution of the auth system..."
should route as "evolution", not "why") — most specific/least ambiguous
categories are checked first, "what" last since its triggers are the most
generic.
"""
import logging
import re
from typing import Optional

from pydantic import BaseModel, Field

from src.graph.state import ArchaeologyState, QueryType, RetrievalStrategy

logger = logging.getLogger(__name__)

QUERY_TYPE_TO_STRATEGY: dict[QueryType, RetrievalStrategy] = {
    "what": "chroma_only",
    "when": "postgres_only",
    "who": "neo4j_postgres",
    "why": "all_parallel",
    "evolution": "all_parallel",  # Neo4j temporal + Postgres + Chroma, per WHAT_TO_DO.MD Phase 4a table
    "similar": "chroma_only",
}

# (query_type, regex) in priority order — first match wins.
RULES: list[tuple[QueryType, re.Pattern]] = [
    (
        "evolution",
        re.compile(
            # "evol\w*" catches both evolve/evolved/evolving AND evolution/evolutionary (different
            # roots sharing the "evol" prefix — "evolv\w*" alone would miss "evolution").
            r"\b(evol\w*|history of|timeline|over time|progression|migrat\w+\s+(from|to)|changed over)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "similar",
        re.compile(
            r"\b(similar|analogous|comparable|duplicate\w*|other implementations?|other places|same pattern)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "who",
        re.compile(
            r"^\s*who\b|\bwho\s+(wrote|added|modified|committed|authored|introduced|created|fixed|reviewed|approved|merged|owns?|last)\b"
            r"|\bauthor of\b|\bwhich\s+(developer|contributor|author|maintainer)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "when",
        re.compile(
            r"^\s*when\b|\bwhen\s+(was|did|is)\b|\bwhat\s+(date|year|time)\b|\btimestamp of\b|\bdate\s+(added|created|introduced)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "why",
        re.compile(
            r"^\s*why\b|\bwhy\s+(is|was|does|did)\b|\bwhat motivated\b|\brationale\b|\breasoning\b|\bwhat alternatives\b",
            re.IGNORECASE,
        ),
    ),
    (
        "what",
        re.compile(
            r"^\s*what\b|\bwhat does\b|\bexplain\b|\bdescribe\b|\bwhat is the purpose\b|\bwhat happens\b",
            re.IGNORECASE,
        ),
    ),
]


class _FallthroughStats:
    """Module-level counters for the rule-vs-LLM fallthrough rate."""

    def __init__(self) -> None:
        self.total = 0
        self.fallthrough = 0

    def record(self, used_fallback: bool) -> None:
        self.total += 1
        if used_fallback:
            self.fallthrough += 1
        rate = self.fallthrough / self.total
        if used_fallback and self.total >= 20 and rate > 0.10:
            logger.warning(
                "Query classifier LLM-fallthrough rate is %.1f%% (%d/%d) — above the 10%% "
                "threshold; the deterministic rules likely need more coverage.",
                rate * 100,
                self.fallthrough,
                self.total,
            )


_stats = _FallthroughStats()


def classify_query_deterministic(query: str) -> Optional[QueryType]:
    for query_type, pattern in RULES:
        if pattern.search(query):
            return query_type
    return None


class _LLMClassification(BaseModel):
    query_type: QueryType = Field(description="One of: what, when, who, why, evolution, similar")


def classify_query_llm_fallback(query: str) -> QueryType:
    from src.llm.llm_factory import get_llm

    llm = get_llm(purpose="classifier", temperature=0)
    prompt = (
        "Classify this developer question about a codebase into exactly one type: "
        "what (asks what code does), when (asks about timing/dates), who (asks about authorship), "
        "why (asks for rationale/motivation), evolution (asks how something changed over time), "
        "similar (asks for comparable/duplicate implementations).\n\n"
        f"Question: {query}"
    )
    result: _LLMClassification = llm.with_structured_output(_LLMClassification).invoke(prompt)
    return result.query_type


def classify_query(query: str) -> tuple[QueryType, str]:
    """Returns (query_type, method) where method is "rule" or "llm_fallback"."""
    rule_result = classify_query_deterministic(query)
    if rule_result is not None:
        _stats.record(used_fallback=False)
        return rule_result, "rule"

    llm_result = classify_query_llm_fallback(query)
    _stats.record(used_fallback=True)
    return llm_result, "llm_fallback"


def query_classifier_node(state: ArchaeologyState) -> dict:
    query_type, method = classify_query(state["query"])
    logger.info("query_classifier: %r -> %s (via %s)", state["query"], query_type, method)
    strategy = QUERY_TYPE_TO_STRATEGY[query_type]
    return {"query_type": query_type, "retrieval_strategy": strategy}


def route_after_classification(state: ArchaeologyState) -> str:
    strategy = state["retrieval_strategy"]
    if strategy == "chroma_only":
        return "chroma_retrieval"
    elif strategy == "postgres_only":
        return "postgres_retrieval"
    elif strategy == "neo4j_postgres":
        return "parallel_neo4j_postgres"
    else:
        return "full_parallel_retrieval"
