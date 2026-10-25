# Code Archaeology Agent

Answers *"why is this code written this way?"* by traversing git history, PR discussions,
issues, and architectural decision records across any GitHub repository.

Given a function, class, or file, the system reconstructs the decision context behind it:
what motivated it, what PR introduced it, what issue drove that PR, what alternatives were
debated, and how the component evolved over time — grounded entirely in the repository's
actual historical record, not just a summary of what the code does.

Default target repository for development/testing: [`tiangolo/fastapi`](https://github.com/tiangolo/fastapi).

## Architecture: three stores, three question types

| Store | Question it answers | Why this store |
|---|---|---|
| **PostgreSQL** | WHEN was this written? WHO changed it? | Fast relational queries, timestamp sorting, joins across commits/PRs/issues |
| **Neo4j** | WHAT CAUSED this? (provenance chain) | Multi-hop graph traversal: function → commit → PR → issue → decision |
| **Chroma** | WHAT IS SEMANTICALLY SIMILAR to this? | Embedding-space retrieval across code, commits, PR discussions, ADRs |

Every schema in every store carries a `repo_id` field so the system can later support
multiple repositories without migration.

## Pipeline phases

0. `RepoConfigAgent` — fingerprints a repo (languages, commit quality, PR density, ADR locations)
1. `GitIngestionAgent` / `PRIssueIngestionAgent` / `ADRIngestionAgent` — parallel ingestion into Postgres
2. `ASTCodeParserAgent` (tree-sitter, function/class-level chunks) → `BlameGraphBuilder` (Neo4j provenance graph)
3. `EmbeddingAgent` — four Chroma collections per repo (`code_chunks`, `commit_messages`, `pr_discussions`, `adr_rfcs`)
4. LangGraph query pipeline — deterministic query classifier (LLM fallback only for ambiguous queries) → parallel multi-source retrieval → Cohere rerank + MMR dedup → provenance classifier → (conditional) evolution timeline agent
5. Answer synthesis with citations + Living Docs Writer (opens a human-reviewed GitHub PR; never auto-merges)
6. Evaluation — RAGAS + DeepEval against a held-out (question, answer, contexts) test set

Every LLM call goes through `src/llm/llm_factory.get_llm()` — no node instantiates a provider
SDK directly. Swap `LLM_PROVIDER` in `.env` (`groq` | `anthropic` | `openai` | `google`) to change
the whole pipeline's model with zero code changes. `EXECUTION_MODE=local` also routes `code_chunks`
embeddings to local BGE-M3 instead of Voyage, so the system runs at zero API cost with just a
Groq/Anthropic/OpenAI/Google key for the LLM layer.

See [WHAT_TO_DO.MD](WHAT_TO_DO.MD) for the full architecture brief and rationale, and
[docs/embedding_benchmark.md](docs/embedding_benchmark.md) / [docs/model_benchmark.md](docs/model_benchmark.md)
for the offline embedding and LLM provider comparisons that determined the defaults wired into
Phase 3 and `llm_factory.py`.

## Getting started

```bash
cp .env.example .env               # fill in API keys
docker compose up -d               # Postgres + Neo4j + Chroma + Redis
python -m venv .venv && .venv/Scripts/activate  # Windows
pip install -r requirements.txt
python scripts/init_db.py          # create Postgres tables + Neo4j constraints
```

Then run ingestion for the default repo:

```bash
python -m src.ingestion.repo_config_agent tiangolo/fastapi
python -m src.ingestion.git_ingestion tiangolo/fastapi
python -m src.ingestion.pr_issue_ingestion tiangolo/fastapi
python -m src.parsing.ast_parser tiangolo/fastapi
python -m src.parsing.blame_graph tiangolo/fastapi
python -m src.ingestion.embedding_agent tiangolo/fastapi
```

## Observability

LangSmith tracing is enabled from the first line of pipeline code (`LANGCHAIN_TRACING_V2=true`).
Every LangGraph node execution — inputs/outputs, latency, token usage, cost — is traced
automatically once `LANGCHAIN_API_KEY` is set.
