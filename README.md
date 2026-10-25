# Code Archaeology Agent

> Answers **"why is this code written this way?"** by traversing git history, pull-request
> discussions, issues, and architectural decision records of any GitHub repository.

A LangGraph multi-agent system that reconstructs the *decision context* behind a function, class,
or file: what motivated it, which PR introduced it, which issue drove that PR, what alternatives
were debated, and how the component evolved. Every answer is grounded in the repository's actual
historical record and returned with citations, not a summary of what the code does.

Default development target: [`tiangolo/fastapi`](https://github.com/tiangolo/fastapi).

---

## Highlights

- **Multi-agent LangGraph pipeline** with query routing, parallel fan-out retrieval, conditional
  branches, and a human-gated documentation step.
- **Three purpose-built stores** (PostgreSQL, Neo4j, Chroma), each answering a different class of question.
- **tree-sitter AST chunking** at function/class level, so retrieval returns meaningful code units.
- **Cohere reranking with MMR deduplication** over multi-source candidates.
- **7-category provenance classifier** that labels how a piece of code came to exist.
- **Human-in-the-loop**: the Living Docs Writer opens a GitHub PR for review and never auto-merges.
- **Provider-agnostic LLM layer**: switch between Groq, Anthropic, OpenAI, and Google with one env var.
- **Built-in evaluation**: RAGAS + DeepEval harness and an LLM provider benchmark.
- **Observability**: LangSmith tracing of every node (inputs/outputs, latency, tokens, cost).

## Architecture

### Three stores, three question types

| Store | Question it answers | Why this store |
|---|---|---|
| **PostgreSQL** | WHEN was this written? WHO changed it? | Relational queries, timestamp sorting, joins across commits/PRs/issues |
| **Neo4j** | WHAT CAUSED this? | Multi-hop provenance traversal: function → commit → PR → issue → decision |
| **Chroma** | WHAT IS SEMANTICALLY SIMILAR? | Embedding retrieval across code, commits, PR discussions, and ADRs |

Every schema carries a `repo_id` so multiple repositories are supported without migration.

### Query-time pipeline

```
                          ┌─ chroma_retrieval ──┐
query_classifier ─ route ─┼─ neo4j_retrieval ───┼─ context_ranker ─ provenance_classifier
                          └─ postgres_retrieval ┘   (Cohere + MMR)            │
                                                                              ▼
                          docs_writer ◀─ synthesizer ◀─ evolution_agent (only for "evolution" queries)
                       (only if create_docs_pr)         (citations)
```

| Node | Role |
|---|---|
| `query_classifier` | Deterministic rules classify the query (what/when/who/why/evolution/similar); an LLM is used only for ambiguous queries |
| `*_retrieval` | Parallel retrieval from the stores selected by the retrieval strategy |
| `context_ranker` | Cohere rerank + MMR dedup, trimmed to the context window |
| `provenance_classifier` | Assigns one of seven provenance classes with a confidence score |
| `evolution_agent` | Builds a timeline of changes (conditional) |
| `synthesizer` | Produces the answer with citations |
| `docs_writer` | Opens a human-reviewed GitHub PR with generated docs (conditional) |

### Provenance classes

`well_documented` · `legacy_migration` · `hotfix` · `experimental_prototype` ·
`generated_code` · `third_party_import` · `orphaned`

### Ingestion pipeline

| Phase | Component | Output |
|---|---|---|
| 0 | `RepoConfigAgent` | Fingerprints a repo (languages, commit quality, PR density, ADR locations) |
| 1 | `GitIngestionAgent`, `PRIssueIngestionAgent`, `ADRIngestionAgent` | Commits, PRs, issues, ADRs in Postgres |
| 2 | `ASTCodeParserAgent`, `BlameGraphBuilder` | Function/class chunks and the Neo4j provenance graph |
| 3 | `EmbeddingAgent` | Four Chroma collections per repo: `code_chunks`, `commit_messages`, `pr_discussions`, `adr_rfcs` |

## Project structure

```
src/
├── api/          FastAPI app (/query, /health, /webhooks/github)
├── cache/        Redis-backed answer cache
├── config/       Settings and per-repo profiles
├── db/           Postgres, Neo4j and Chroma clients
├── evaluation/   RAGAS/DeepEval harness, test set, embedding benchmark
├── graph/        LangGraph state, nodes and pipeline assembly
├── ingestion/    Repo config, git, PR/issue, ADR and embedding agents
├── llm/          Provider-agnostic LLM factory
└── parsing/      tree-sitter AST parser and blame graph builder
scripts/          DB initialisation and LLM provider benchmark
tests/            Unit tests
```

## Getting started

### Prerequisites

- Python 3.11+
- Docker (for Postgres, Neo4j, Chroma, Redis)
- A GitHub token and an API key for at least one LLM provider (Groq has a free tier)

### Setup

```bash
git clone https://github.com/SachinSNain/Code_Archaeology.git
cd Code_Archaeology

cp .env.example .env                # fill in API keys
docker compose up -d                # Postgres + Neo4j + Chroma + Redis

python -m venv .venv
source .venv/bin/activate           # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python scripts/init_db.py           # create Postgres tables + Neo4j constraints
```

### Ingest a repository

```bash
python -m src.ingestion.repo_config_agent tiangolo/fastapi
python -m src.ingestion.git_ingestion tiangolo/fastapi
python -m src.ingestion.pr_issue_ingestion tiangolo/fastapi
python -m src.parsing.ast_parser tiangolo/fastapi
python -m src.parsing.blame_graph tiangolo/fastapi
python -m src.ingestion.embedding_agent tiangolo/fastapi
```

### Run the API

```bash
uvicorn src.api.main:app --reload
```

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"repo_id": "tiangolo/fastapi", "query": "Why does the dependency injection system cache results?"}'
```

The response contains the `answer`, `citations`, `provenance_class`, `provenance_confidence`,
`evolution_timeline` (for evolution queries), `draft_pr_url`, and `query_type`.

| Endpoint | Purpose |
|---|---|
| `POST /query` | Run the full pipeline. Set `create_docs_pr: true` to open a docs PR for review |
| `POST /webhooks/github` | On merged PRs, ingests the new PR so the knowledge base stays current |
| `GET /health` | Liveness check |

## Configuration

All configuration lives in `.env` (see [.env.example](.env.example)).

| Variable | Purpose |
|---|---|
| `LLM_PROVIDER` / `LLM_MODEL` | `groq` \| `anthropic` \| `openai` \| `google`, and the model name |
| `CLASSIFIER_LLM_PROVIDER` / `CLASSIFIER_LLM_MODEL` | Model for the query classifier's rare LLM fallback |
| `EXECUTION_MODE` | `local` (BGE-M3 embeddings, zero embedding API cost), `hybrid` (Voyage for code embeddings), or `benchmark` |
| `COHERE_API_KEY` | Reranker |
| `GITHUB_TOKEN` | Ingestion and docs PR creation |
| `LANGCHAIN_TRACING_V2`, `LANGCHAIN_API_KEY` | LangSmith tracing |

Every LLM call goes through `src/llm/llm_factory.get_llm()`; no node instantiates a provider SDK
directly, so changing `LLM_PROVIDER` swaps the model for the whole pipeline with no code changes.

## Evaluation

The evaluation harness runs the full pipeline against a held-out test set of
(question, expected answer, expected source chunks) triplets.

```bash
python -m src.evaluation.run_ragas                        # RAGAS + DeepEval
python -m src.evaluation.benchmark_embeddings             # offline embedding comparison
python scripts/run_model_benchmark.py --providers groq,anthropic,openai   # LLM provider comparison (costs money)
```

| Tool | Metrics |
|---|---|
| **RAGAS** | context recall, context precision, faithfulness, answer relevancy |
| **DeepEval** | citation accuracy (GEval) |

Run it after significant architecture changes, not on every live query.
Design targets are set in [What_To_Do.md](What_To_Do.md#phase-6-evaluation-pipeline).

## Observability

LangSmith tracing is enabled at the top of the pipeline module. With `LANGCHAIN_API_KEY` set, every
LangGraph node execution is traced automatically with inputs/outputs, latency, token usage, and cost.

## Testing

```bash
pytest tests/unit
```

## Further reading

- [What_To_Do.md](What_To_Do.md): full architecture brief, phase-by-phase design, and rationale
- [Interview.md](Interview.md): design decisions and trade-offs explained

## Tech stack

Python · LangGraph · LangChain · FastAPI · PostgreSQL · Neo4j · Chroma · Redis · tree-sitter ·
Cohere · Voyage / BGE-M3 · RAGAS · DeepEval · LangSmith · Docker
