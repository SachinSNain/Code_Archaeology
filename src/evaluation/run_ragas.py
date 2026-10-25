"""Phase 6 — Evaluation pipeline.

Runs the full query-time pipeline against a held-out (question,
expected_answer, expected_source_chunks) test set and scores it with RAGAS
(context recall/precision, faithfulness, answer relevance) plus a DeepEval
GEval metric for citation accuracy. Run after every significant architecture
change — NOT on every live query, which is what src/graph/pipeline.py is for.

Usage:
    python -m src.evaluation.run_ragas [--test-set path/to/test_set.json]
"""
import argparse
import json
import time
from pathlib import Path
from statistics import mean

from deepeval.metrics import GEval
from deepeval.models.base_model import DeepEvalBaseLLM
from deepeval.test_case import LLMTestCase, LLMTestCaseParams
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness

from src.graph.pipeline import run_query
from src.llm.llm_factory import get_llm

DEFAULT_TEST_SET_PATH = Path(__file__).parent / "test_set.json"


class _GroqDeepEvalModel(DeepEvalBaseLLM):
    """DeepEval's metrics (GEval included) default to OpenAI; this routes the
    citation-accuracy judge through llm_factory.get_llm() instead, so
    evaluation works with whatever LLM_PROVIDER is configured (critical rule:
    all LLM calls go through the factory), not hardcoded to OpenAI.
    """

    def load_model(self):
        return get_llm(purpose="default", temperature=0)

    def generate(self, prompt: str) -> str:
        return self.model.invoke(prompt).content

    async def a_generate(self, prompt: str) -> str:
        result = await self.model.ainvoke(prompt)
        return result.content

    def get_model_name(self) -> str:
        return "llm_factory:default"

# Target baselines from WHAT_TO_DO.MD Phase 6 — set before building, evaluated against after.
BASELINES = {
    "context_recall": 0.75,
    "faithfulness": 0.85,
    "citation_accuracy": 0.90,
    "latency_seconds": 8.0,
}


def load_test_set(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def run_pipeline_for_test_set(test_cases: list[dict]) -> list[dict]:
    rows = []
    for case in test_cases:
        start = time.time()
        result_state = run_query(case["repo_id"], case["question"], target_chunk_id=case.get("target_chunk_id"))
        latency = time.time() - start

        contexts = [c.text for c in (result_state.get("ranked_context") or [])]
        rows.append(
            {
                "question": case["question"],
                "answer": result_state.get("answer", ""),
                "contexts": contexts or ["(no context retrieved)"],
                "ground_truth": case["expected_answer"],
                "citations": result_state.get("citations", []),
                "latency": latency,
            }
        )
    return rows


def run_ragas_metrics(rows: list[dict]):
    dataset = EvaluationDataset.from_list(
        [
            {
                "user_input": r["question"],
                "response": r["answer"],
                "retrieved_contexts": r["contexts"],
                "reference": r["ground_truth"],
            }
            for r in rows
        ]
    )
    # RAGAS's metrics default to OpenAI; wrap our configured LLM (Groq by default) and a
    # local HuggingFace embedder instead so evaluation doesn't require OPENAI_API_KEY.
    from langchain_community.embeddings import HuggingFaceEmbeddings

    ragas_llm = LangchainLLMWrapper(get_llm(purpose="default", temperature=0))
    ragas_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2"))
    return evaluate(
        dataset,
        metrics=[context_recall, context_precision, faithfulness, answer_relevancy],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )


def run_citation_accuracy(rows: list[dict]) -> float | None:
    """DeepEval GEval metric: does each citation's excerpt actually support
    the claim it's attached to in the answer? (Additional metric per Phase 6,
    since RAGAS has no built-in citation-grounding check.)
    """
    metric = GEval(
        name="CitationAccuracy",
        criteria=(
            "Given the answer (actual_output) and a supporting excerpt (context), determine whether the "
            "excerpt genuinely supports a claim made in the answer. Score 1.0 if it clearly does, 0.0 if it "
            "does not or is unrelated."
        ),
        evaluation_params=[LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.CONTEXT],
        model=_GroqDeepEvalModel(),
    )
    scores = []
    for r in rows:
        for citation in r["citations"]:
            excerpt = citation.excerpt if hasattr(citation, "excerpt") else citation.get("excerpt", "")
            if not excerpt:
                continue
            test_case = LLMTestCase(input=r["question"], actual_output=r["answer"], context=[excerpt])
            try:
                metric.measure(test_case)
                scores.append(metric.score)
            except Exception as exc:
                # A single judge-call failure (e.g. the LLM provider's rate limit) must not
                # discard every score computed so far — report on what succeeded.
                print(f"  citation_accuracy: skipped one citation after judge error: {exc}")
    return mean(scores) if scores else None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-set", type=Path, default=DEFAULT_TEST_SET_PATH)
    args = parser.parse_args()

    test_cases = load_test_set(args.test_set)
    print(f"Running pipeline against {len(test_cases)} test case(s)...")
    rows = run_pipeline_for_test_set(test_cases)

    print("\n== RAGAS ==")
    ragas_result = run_ragas_metrics(rows)
    ragas_scores = ragas_result.to_pandas().mean(numeric_only=True).to_dict()
    for name, value in ragas_scores.items():
        print(f"  {name}: {value:.3f}")

    print("\n== DeepEval ==")
    try:
        citation_accuracy = run_citation_accuracy(rows)
    except Exception as exc:
        print(f"  citation_accuracy: FAILED ({exc})")
        citation_accuracy = None
    else:
        print(f"  citation_accuracy: {citation_accuracy:.3f}" if citation_accuracy is not None else "  citation_accuracy: n/a (no citations produced)")

    avg_latency = mean(r["latency"] for r in rows)
    print(f"\n  avg_latency_seconds: {avg_latency:.2f}")

    print("\n== Baseline comparison ==")
    checks = {
        "context_recall": ragas_scores.get("context_recall"),
        "faithfulness": ragas_scores.get("faithfulness"),
        "citation_accuracy": citation_accuracy,
        "latency_seconds": avg_latency,
    }
    for name, baseline in BASELINES.items():
        actual = checks.get(name)
        if actual is None:
            print(f"  {name}: n/a")
            continue
        passed = actual <= baseline if name == "latency_seconds" else actual >= baseline
        print(f"  {name}: {actual:.3f} vs baseline {baseline} -> {'PASS' if passed else 'FAIL'}")


if __name__ == "__main__":
    main()
