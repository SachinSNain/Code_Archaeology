"""Mode 3 — LLM provider benchmark.

Runs the Phase 6 RAGAS/DeepEval evaluation suite once per candidate LLM
provider (temporarily overriding LLM_PROVIDER/LLM_MODEL) and aggregates the
results into docs/model_benchmark.md — turning model selection into a
documented experiment instead of a preference, per WHAT_TO_DO.MD's "Three
execution modes" section.

Costs real money across multiple providers — run once, after the pipeline is
stable, not on every change (per WHAT_TO_DO.MD build order step 14).

Usage:
    python scripts/run_model_benchmark.py --providers groq,anthropic,openai
"""
import argparse
import sys
import time
from pathlib import Path
from statistics import mean

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.config.settings import get_settings
from src.evaluation.run_ragas import load_test_set, run_citation_accuracy, run_pipeline_for_test_set, run_ragas_metrics

OUTPUT_PATH = Path(__file__).resolve().parent.parent / "docs" / "model_benchmark.md"

# (provider, model) candidates — matches WHAT_TO_DO.MD "Benchmarking phase: multiple providers".
CANDIDATES: dict[str, tuple[str, str]] = {
    "groq": ("groq", "llama-3.3-70b-versatile"),
    "anthropic": ("anthropic", "claude-sonnet-4-6"),
    "openai": ("openai", "gpt-4o"),
    "google": ("google", "gemini-1.5-pro"),
}

PROVIDER_KEY_ATTR = {
    "groq": "groq_api_key",
    "anthropic": "anthropic_api_key",
    "openai": "openai_api_key",
    "google": "google_api_key",
}


def _has_key(provider: str) -> bool:
    return bool(getattr(get_settings(), PROVIDER_KEY_ATTR[provider], None))


def benchmark_provider(provider: str, model: str, test_cases: list[dict]) -> dict:
    import src.llm.llm_factory as llm_factory

    # Force this provider for the run regardless of .env, then clear the cached
    # client afterward so the next provider in the loop isn't stuck reusing it.
    settings = get_settings()
    original_provider, original_model = settings.llm_provider, settings.llm_model
    settings.llm_provider, settings.llm_model = provider, model
    llm_factory._cached_llm.cache_clear()

    try:
        start = time.time()
        rows = run_pipeline_for_test_set(test_cases)
        wall_time = time.time() - start

        ragas_result = run_ragas_metrics(rows)
        ragas_scores = ragas_result.to_pandas().mean(numeric_only=True).to_dict()
        citation_accuracy = run_citation_accuracy(rows)
        avg_latency = mean(r["latency"] for r in rows)

        return {
            "provider": provider,
            "model": model,
            "faithfulness": ragas_scores.get("faithfulness"),
            "context_recall": ragas_scores.get("context_recall"),
            "citation_accuracy": citation_accuracy,
            "avg_latency_s": avg_latency,
            "total_wall_time_s": wall_time,
        }
    finally:
        settings.llm_provider, settings.llm_model = original_provider, original_model
        llm_factory._cached_llm.cache_clear()


def write_report(results: list[dict]) -> None:
    lines = [
        "# LLM Provider Benchmark (Mode 3)",
        "",
        "Same RAGAS/DeepEval test set (src/evaluation/test_set.json) run once per provider via "
        "scripts/run_model_benchmark.py, with every node's LLM calls going through "
        "src/llm/llm_factory.get_llm() so only the provider/model changes between runs.",
        "",
        "| Provider | Model | Faithfulness | Context Recall | Citation Accuracy | Avg latency (s) |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        def fmt(x):
            return f"{x:.3f}" if isinstance(x, (int, float)) else "n/a"

        lines.append(
            f"| {r['provider']} | {r['model']} | {fmt(r['faithfulness'])} | {fmt(r['context_recall'])} | "
            f"{fmt(r['citation_accuracy'])} | {fmt(r['avg_latency_s'])} |"
        )
    OUTPUT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--providers", type=str, default=None, help="Comma-separated subset of CANDIDATES")
    parser.add_argument("--test-set", type=Path, default=None)
    args = parser.parse_args()

    providers = args.providers.split(",") if args.providers else list(CANDIDATES.keys())
    test_cases = load_test_set(args.test_set) if args.test_set else load_test_set(
        Path(__file__).resolve().parent.parent / "src" / "evaluation" / "test_set.json"
    )

    results = []
    for provider in providers:
        if not _has_key(provider):
            print(f"Skipping {provider}: no API key configured.")
            continue
        model = CANDIDATES[provider][1]
        print(f"== Benchmarking {provider} ({model}) ==")
        results.append(benchmark_provider(provider, model, test_cases))

    if not results:
        print("No providers had API keys configured — nothing to benchmark.")
        return

    for r in results:
        print(r)
    write_report(results)


if __name__ == "__main__":
    main()
