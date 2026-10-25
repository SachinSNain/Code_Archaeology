"""LangGraph graph assembly + compilation — the query-time pipeline (Phase 4/5).

LangSmith tracing is enabled here, before any node executes (critical rule
#8: "Set the environment variables before writing any pipeline code" — this
module is the pipeline's entry point, so it's the one place that's
guaranteed to run before a single node fires).

Graph shape:
    query_classifier
        -> (fan-out, subset per retrieval_strategy)
           chroma_retrieval / neo4j_retrieval / postgres_retrieval
        -> (fan-in) context_ranker
        -> provenance_classifier
        -> (conditional) evolution_agent -> synthesizer
                          | (skip)      ->
        -> (conditional) docs_writer [only if state["create_docs_pr"]]
        -> END
"""
from langgraph.graph import END, START, StateGraph

from src.config.settings import get_settings

get_settings().apply_langsmith_env()

from src.graph.nodes.docs_writer import docs_writer_node
from src.graph.nodes.evolution_agent import evolution_agent_node, should_run_evolution_agent
from src.graph.nodes.provenance_classifier import provenance_classifier_node
from src.graph.nodes.query_classifier import query_classifier_node
from src.graph.nodes.reranker import context_ranker_node
from src.graph.nodes.retrieval import (
    chroma_retrieval_node,
    neo4j_retrieval_node,
    postgres_retrieval_node,
    route_to_retrieval_nodes,
)
from src.graph.nodes.synthesizer import synthesizer_node
from src.graph.state import ArchaeologyState

RETRIEVAL_NODE_NAMES = ["chroma_retrieval", "neo4j_retrieval", "postgres_retrieval"]


def route_after_provenance(state: ArchaeologyState) -> str:
    return "evolution_agent" if should_run_evolution_agent(state) else "synthesizer"


def route_after_synthesis(state: ArchaeologyState) -> str:
    return "docs_writer" if state.get("create_docs_pr") else END


def build_pipeline():
    workflow = StateGraph(ArchaeologyState)

    workflow.add_node("query_classifier", query_classifier_node)
    workflow.add_node("chroma_retrieval", chroma_retrieval_node)
    workflow.add_node("neo4j_retrieval", neo4j_retrieval_node)
    workflow.add_node("postgres_retrieval", postgres_retrieval_node)
    workflow.add_node("context_ranker", context_ranker_node)
    workflow.add_node("provenance_classifier", provenance_classifier_node)
    workflow.add_node("evolution_agent", evolution_agent_node)
    workflow.add_node("synthesizer", synthesizer_node)
    workflow.add_node("docs_writer", docs_writer_node)

    workflow.add_edge(START, "query_classifier")

    # Fan-out: route_to_retrieval_nodes returns a subset of RETRIEVAL_NODE_NAMES
    # depending on retrieval_strategy; LangGraph runs each returned node in parallel.
    workflow.add_conditional_edges("query_classifier", route_to_retrieval_nodes, RETRIEVAL_NODE_NAMES)

    # Fan-in: each retrieval node writes its own state key, so no reducer is needed.
    for node_name in RETRIEVAL_NODE_NAMES:
        workflow.add_edge(node_name, "context_ranker")

    workflow.add_edge("context_ranker", "provenance_classifier")
    workflow.add_conditional_edges(
        "provenance_classifier", route_after_provenance, ["evolution_agent", "synthesizer"]
    )
    workflow.add_edge("evolution_agent", "synthesizer")
    workflow.add_conditional_edges("synthesizer", route_after_synthesis, ["docs_writer", END])
    workflow.add_edge("docs_writer", END)

    return workflow.compile()


_pipeline = None


def get_pipeline():
    global _pipeline
    if _pipeline is None:
        _pipeline = build_pipeline()
    return _pipeline


def run_query(repo_id: str, query: str, target_chunk_id: str | None = None, create_docs_pr: bool = False) -> ArchaeologyState:
    pipeline = get_pipeline()
    initial_state: ArchaeologyState = {
        "repo_id": repo_id,
        "query": query,
        "target_chunk_id": target_chunk_id,
        "create_docs_pr": create_docs_pr,
    }
    return pipeline.invoke(initial_state)
