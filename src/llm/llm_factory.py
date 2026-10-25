"""Provider-agnostic LLM factory.

Critical rule: every node that needs an LLM calls `get_llm()` — no node
instantiates `ChatGroq`/`ChatAnthropic`/`ChatOpenAI`/`ChatGoogleGenerativeAI`
directly. This is the only place provider-specific code lives, so the whole
pipeline's LLM can be swapped via `LLM_PROVIDER` in `.env` with zero code
changes elsewhere — and it's what makes the Mode 3 provider benchmark
(docs/model_benchmark.md) possible: the same nodes run unmodified against
Groq, Claude, GPT-4o, or Gemini.

Provider SDKs are imported lazily inside each branch so an unused provider
(e.g. `langchain_google_genai`, if GOOGLE_API_KEY is never set) doesn't have
to be installed for the others to work.
"""
from functools import lru_cache
from typing import Literal

from langchain_core.language_models.chat_models import BaseChatModel

from src.config.settings import LLMProvider, get_settings

Purpose = Literal["default", "classifier", "benchmark_openai", "benchmark_anthropic"]


def _build(provider: LLMProvider, model: str, temperature: float) -> BaseChatModel:
    settings = get_settings()

    if provider == "groq":
        from langchain_groq import ChatGroq

        return ChatGroq(model=model, api_key=settings.groq_api_key, temperature=temperature)
    if provider == "anthropic":
        from langchain_anthropic import ChatAnthropic

        return ChatAnthropic(model=model, api_key=settings.anthropic_api_key, temperature=temperature)
    if provider == "openai":
        from langchain_openai import ChatOpenAI

        return ChatOpenAI(model=model, api_key=settings.openai_api_key, temperature=temperature)
    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        return ChatGoogleGenerativeAI(model=model, google_api_key=settings.google_api_key, temperature=temperature)
    raise ValueError(f"Unknown LLM provider: {provider}")


@lru_cache(maxsize=32)
def _cached_llm(provider: LLMProvider, model: str, temperature: float) -> BaseChatModel:
    return _build(provider, model, temperature)


def get_llm(purpose: Purpose = "default", temperature: float = 0.0) -> BaseChatModel:
    """Returns the configured LLM for the given purpose.

    purpose: "default" (main synthesis/reasoning nodes) | "classifier" (query
    classifier's rare LLM fallback for ambiguous queries) | "benchmark_openai" |
    "benchmark_anthropic" (Mode 3 provider comparison — bypasses LLM_PROVIDER
    to force a specific provider regardless of the active config).
    """
    settings = get_settings()

    if purpose == "classifier":
        return _cached_llm(settings.classifier_llm_provider, settings.classifier_llm_model, temperature)
    if purpose == "benchmark_openai":
        return _cached_llm("openai", "gpt-4o", temperature)
    if purpose == "benchmark_anthropic":
        return _cached_llm("anthropic", "claude-sonnet-4-6", temperature)
    return _cached_llm(settings.llm_provider, settings.llm_model, temperature)
