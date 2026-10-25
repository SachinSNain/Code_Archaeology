# Code Archaeology Agent — Project Brief for Claude Code

## Target repository for initial build and testing

**Repository:** `tiangolo/fastapi`
**URL:** https://github.com/tiangolo/fastapi

Use this repo for all development, testing, and initial evaluation. Reasons: small enough to fully ingest in minutes (not hours), has clean and descriptive PR descriptions giving the archaeology agent rich decision context, well-linked issues, and enough history (since 2018) to produce interesting evolution timelines. Once the full pipeline is verified working on FastAPI, point it at `kubernetes/kubernetes` or `langchain-ai/langchain` for a more impressive demo.

---

## What this document is

This is the complete context, architecture, and implementation brief for building the **Code Archaeology Agent** — a production-grade Agentic AI system that answers the question *"why is this code written this way?"* by traversing git history, PR discussions, issues, and architectural decision records across any GitHub repository.

Read this entire document before writing a single line of code. Every architectural decision here was made deliberately and the reasoning is explained. Do not substitute simpler alternatives without understanding what you are trading off.

---

## Project identity

**What it does:** Given any function, class, or file in a GitHub repository, the system reconstructs the full decision context behind it — what motivated it, what PR introduced it, what issue drove that PR, what alternatives were debated, and how that component has evolved over time.

**What it does NOT do:** It does not just summarize what the code does (any LLM can do that). It explains *why* the code exists in its current form, grounded in the actual historical record of the repository.

**Target repositories (repo-agnostic by design):**
- Kubernetes (has KEP design proposals — excellent ADR source)
- LangChain (rapid architectural evolution — good for evolution timelines)
- Apache Airflow (AIP RFCs — rich decision documents)
- FastAPI (clean PR discussions)
- Any private GitHub repository

**Resume/portfolio positioning:** This project demonstrates Agentic AI, RAG, LangGraph stateful pipelines, multi-store data architecture (vector DB + graph DB + relational DB), and production engineering practices (evaluation, observability, human-in-the-loop). Every architectural decision should be explainable and evidence-based.

---

## Core architectural principle: three stores, three question types

The system uses **three separate databases**. This is the most important design decision in the project. Do not collapse them into one.

| Store | Question it answers | Why this store specifically |
|---|---|---|
| **PostgreSQL** | WHEN was this written? WHO changed it? | Fast relational queries, timestamp sorting, joins across commits/PRs/issues |
| **Neo4j** | WHAT CAUSED this? (provenance chain) | Multi-hop graph traversal: function → commit → PR → issue → decision. SQL recursive CTEs can do this but become unmaintainable. Vector search cannot do this at all. |
| **Chroma** | WHAT IS SEMANTICALLY SIMILAR to this? | Embedding-space retrieval across code, commit messages, PR discussions, ADRs |

Every schema in every store must include a `repo_id` field from day one. This is the foundation for a future multi-repository knowledge graph (not in scope for v1, but the schema must support it without migration).

---

## Technology stack — full list with justifications

### Core framework
- **LangGraph** — orchestrates the multi-agent pipeline as a stateful graph with conditional edges and parallel node execution. NOT just LangChain chains, which are linear and cannot handle conditional routing or parallel retrieval.
- **LangChain** — used within individual nodes for document loading, prompt templates, RAG chains, and output parsers.

### Databases
- **PostgreSQL** — structured metadata store (commits, PRs, issues, authors, timestamps, file paths)
- **Neo4j** — blame graph (nodes: functions, commits, PRs, issues; edges: introduced_by, linked_to, references, modified_by)
- **Chroma** — vector store (four collections per repo: code_chunks, commit_messages, pr_discussions, adr_rfcs)

### Parsing and ingestion
- **GitPython** — git operations (clone, log, blame, diff)
- **PyGithub** — GitHub REST API (PRs, issues, review comments, rate-limit-aware)
- **tree-sitter** — AST parsing for code chunking at function/class boundary. Handles 40+ languages with one API. Do NOT chunk code by line count or character count.
- **Unstructured.io** — parsing ADR documents, RFCs, PDF design docs

### Embeddings
- **Voyage Code Embeddings** — for `code_chunks` collection (code-specific model, selected via offline benchmark)
- **BGE-M3** — for `commit_messages`, `pr_discussions`, `adr_rfcs` collections (general text, strong multilingual performance)
- See "Offline Embedding Benchmark" section below for how this decision was validated.

### Retrieval and reranking
- **Cohere Rerank API** — cross-encoder reranker applied to top-40 retrieved chunks before synthesis. This is not optional — without reranking, retrieval quality degrades significantly because cosine similarity alone does not measure true relevance.
- **MMR (Maximal Marginal Relevance)** via LangChain — deduplicates retrieved chunks that are semantically near-identical

### LLM — provider-agnostic by design

The LLM layer must be abstracted behind a factory (`src/llm/llm_factory.py`) so any provider can be swapped via config with zero code changes. This is how production systems are built and is a strong talking point in interviews.

**Primary development model: Groq (free tier)**
- Model: `llama-3.3-70b-versatile` or `qwen-qwq-32b` (check current Groq catalog)
- Why: Free, extremely fast inference, sufficient reasoning quality for all nodes during development
- Used for all nodes during development and testing

**Final synthesis and demo: Anthropic API (Claude claude-sonnet-4-6)**
- Used only for the final Explanation Agent during demo and benchmarking
- One API call per query — everything else is deterministic or uses Groq

**Benchmarking phase: multiple providers**
- Run the same evaluation against Groq (Llama/Qwen), Claude claude-sonnet-4-6, GPT-4o, Gemini
- Compare faithfulness, citation accuracy, latency, cost
- This turns model selection into a documented experiment rather than an assumption

**Query classifier: deterministic Python — NO LLM call**
- Simple keyword/regex rules handle >95% of query classification correctly
- Zero API cost, <5ms latency
- Example: `if any(w in query.lower() for w in ["when", "added", "introduced"]): return "when"`
- Reserve LLM classification only for ambiguous queries that fail the rule-based classifier

### Evaluation
- **RAGAS** — measures Context Recall, Context Precision, Faithfulness, Answer Relevance on (question, answer, contexts) triplets
- **DeepEval** — additional metrics: hallucination rate, citation accuracy
- **LangSmith** — per-node tracing across the LangGraph pipeline (execution time, token usage, retrieved docs, costs)

### Observability
- **LangSmith** — primary observability tool. Enabled via `LANGCHAIN_TRACING_V2=true` environment variable. Captures every LangGraph node automatically.
- OpenTelemetry / Grafana — optional, only needed if deploying as a long-running service

### Infrastructure
- **Docker Compose** — local development environment (Postgres, Neo4j, Chroma all containerized)
- **FastAPI** — thin API layer exposing the query endpoint
- **Pydantic** — all data models and state objects typed strictly

---

## Complete system architecture — phase by phase

### Phase 0: Repository configuration

**Agent:** `RepoConfigAgent`

**What it does:**
Accepts any GitHub repo URL and optional configuration (primary language, Jira workspace URL, Slack export path). Fingerprints the repository to understand its characteristics before ingestion begins.

**Fingerprinting outputs (stored as `RepoProfile`):**
- Primary language(s) detected
- Commit message quality score (does the repo use conventional commits? are messages descriptive?)
- PR density (average PRs per month — indicates how much decision context exists in PRs)
- Whether ADR/RFC documents exist and where (Kubernetes uses `docs/proposals/`, Airflow uses `docs/apache-airflow-providers/`, most repos use `docs/adr/` or `DECISIONS.md`)
- Whether issues are linked from commit messages (via `Fixes #123`, `Closes #456` patterns)

**Why this matters:** Downstream agents use the `RepoProfile` to calibrate behavior. A repo with poor commit message quality needs more weight on PR discussions. A repo with no ADRs skips that ingestion step entirely. A repo with no issue linking cannot build a complete provenance chain and must flag gaps explicitly.

**Data model:**
```python
class RepoProfile(BaseModel):
    repo_id: str                    # "{owner}/{repo}" e.g. "kubernetes/kubernetes"
    repo_url: str
    primary_languages: List[str]
    commit_quality_score: float     # 0.0–1.0
    pr_density: float               # PRs per month average
    has_adrs: bool
    adr_paths: List[str]
    has_issue_linking: bool
    slack_export_path: Optional[str]
    total_commits: int
    date_range: Tuple[datetime, datetime]
```

---

### Phase 1: Multi-source data ingestion

Three parallel ingestion agents. Each stores raw structured data in PostgreSQL and raw text in preparation for embedding.

#### 1a: `GitIngestionAgent`

**What it does:** Walks every commit using GitPython. For each commit, extracts: SHA, message, author, email, timestamp, parent SHAs, and the per-file diff. Parses commit messages for issue references (`Fixes #`, `Closes #`, `Resolves #`, `Related to #`).

**Key engineering detail — incremental sync:** Large repos (Kubernetes has 100k+ commits) cannot be fully re-ingested on every run. Implement checkpoint-based incremental sync: store the last-processed commit SHA in Postgres, and on subsequent runs, walk only from HEAD to that checkpoint.

**PostgreSQL schema:**
```sql
CREATE TABLE commits (
    id UUID PRIMARY KEY,
    repo_id VARCHAR NOT NULL,       -- ALWAYS include repo_id
    sha VARCHAR(40) NOT NULL,
    message TEXT,
    author_name VARCHAR,
    author_email VARCHAR,
    timestamp TIMESTAMPTZ,
    parent_shas VARCHAR[],
    files_changed JSONB,            -- [{path, additions, deletions, patch}]
    linked_issue_numbers INT[],     -- parsed from message
    linked_pr_number INT,
    UNIQUE(repo_id, sha)
);
```

#### 1b: `PRIssueIngestionAgent`

**What it does:** Uses PyGithub to pull every PR (title, body, review comments, linked issues, labels, merge commit SHA, files touched) and every issue (title, body, comment thread, labels, linked PRs).

**Key engineering detail — rate limiting:** GitHub gives 5,000 API calls/hour authenticated. A repo like VSCode has 10,000+ PRs. Implement exponential backoff and checkpoint the last-fetched PR number so ingestion can resume after rate limit resets. This is a real production problem, not a toy concern.

**Key engineering detail — PR-to-commit linking:** GitHub's API returns the merge commit SHA for each PR. Use this to link `commits.linked_pr_number` during ingestion, creating the commit→PR edge in the provenance chain.

**PostgreSQL schema:**
```sql
CREATE TABLE pull_requests (
    id UUID PRIMARY KEY,
    repo_id VARCHAR NOT NULL,
    pr_number INT NOT NULL,
    title TEXT,
    body TEXT,
    author VARCHAR,
    created_at TIMESTAMPTZ,
    merged_at TIMESTAMPTZ,
    merge_commit_sha VARCHAR(40),
    linked_issue_numbers INT[],
    labels VARCHAR[],
    review_comments JSONB,          -- [{author, body, created_at, path, line}]
    files_touched VARCHAR[],
    UNIQUE(repo_id, pr_number)
);

CREATE TABLE issues (
    id UUID PRIMARY KEY,
    repo_id VARCHAR NOT NULL,
    issue_number INT NOT NULL,
    title TEXT,
    body TEXT,
    author VARCHAR,
    created_at TIMESTAMPTZ,
    closed_at TIMESTAMPTZ,
    labels VARCHAR[],
    comment_thread JSONB,           -- [{author, body, created_at}]
    linked_pr_numbers INT[],
    UNIQUE(repo_id, issue_number)
);
```

#### 1c: `ADRIngestionAgent`

**What it does:** Scans the repo for architectural decision records, RFCs, design proposals, and CHANGELOG entries. Uses the `RepoProfile.adr_paths` from Phase 0 to know where to look. Falls back to scanning `docs/`, `rfcs/`, `proposals/`, `DECISIONS.md`, `ARCHITECTURE.md` if no known paths.

**Per-repo knowledge:**
- Kubernetes: `docs/proposals/` and `keps/` (Kubernetes Enhancement Proposals)
- Apache Airflow: `docs/apache-airflow-providers/` and AIPs
- LangChain: GitHub Discussions (pull via API) + `docs/`
- FastAPI: GitHub Issues tagged `design` or `discussion`

**Document types to tag:** Each ingested document gets a `doc_type` metadata field: `adr`, `rfc`, `design_proposal`, `changelog_entry`, `slack_thread`, `github_discussion`. This enables filtered retrieval in Phase 4.

---

### Phase 1.5: Offline embedding benchmark (run once, not at runtime)

This is a **one-time offline experiment**, not a system component. It runs during the build/research phase and its output is a documented decision that gets hard-wired into Phase 3.

**What to do:**
1. Manually construct 50–100 (query, expected_chunk) pairs from the target repository. Example: query = "Why was the retry logic rewritten in 2022?", expected_chunk = the PR description from PR #4821 that rewrote it.
2. Embed all chunks with each candidate model.
3. For each query, retrieve top-5 chunks with each model.
4. Measure Recall@5 (did the expected chunk appear in the top 5?), Precision@K, MRR, latency, and storage size.

**Candidate models to evaluate:**
- OpenAI `text-embedding-3-large`
- CodeBERT (`microsoft/codebert-base`)
- BGE-M3 (`BAAI/bge-m3`)
- Jina Embeddings v3
- Voyage Code Embeddings (`voyage-code-2`)

**Expected finding (to validate):** Code-specific models (Voyage, CodeBERT) outperform general models on `code_chunks` retrieval. General models (BGE-M3, Jina) perform comparably or better on `pr_discussions` and `adr_rfcs` because those are natural language, not code.

**Output:** A markdown table in `docs/embedding_benchmark.md` with all metrics. This table gets referenced in the README and in interviews. The winning models get hard-wired into Phase 3.

**Why this matters for the resume:** An interviewer who asks "why did you choose this embedding model?" gets a data-driven answer backed by a reproducible experiment. This is a significantly stronger signal than "I read it was good for code."

---

### Phase 2: AST parsing and blame graph construction

#### 2a: `ASTCodeParserAgent`

**What it does:** For every source file in the repository, parses the AST using tree-sitter and chunks at the function/class/module boundary.

**Critical design rule:** Do NOT chunk by line count, character count, or sliding window. A 500-line file should become 12 function-level chunks, each semantically complete. Line-count chunking splits a function across two chunks, destroying its meaning.

**tree-sitter specifics:**
- Install language grammars: `tree-sitter-python`, `tree-sitter-javascript`, `tree-sitter-typescript`, `tree-sitter-go`, `tree-sitter-java`, `tree-sitter-rust` — whichever the repo uses
- Parse incrementally: tree-sitter can re-parse only the changed subtree when a file is modified, making re-ingestion fast
- Fault-tolerant: tree-sitter produces a partial AST for syntactically broken files rather than crashing — important for real repos that have broken files in history

**Chunk metadata (stored per chunk):**
```python
class CodeChunk(BaseModel):
    chunk_id: str
    repo_id: str                    # always
    file_path: str
    function_name: Optional[str]
    class_name: Optional[str]
    language: str
    start_line: int
    end_line: int
    chunk_text: str
    last_modified_sha: str          # from git blame
    docstring: Optional[str]        # extracted from AST if present
    calls: List[str]                # function names this chunk calls
    called_by: List[str]            # populated later via static analysis
```

#### 2b: `BlameGraphBuilder`

**What it does:** For each code chunk, runs `git blame` to get the commit SHA of every line, then traverses the provenance chain and stores it as a graph in Neo4j.

**Graph schema (Neo4j nodes and edges):**
```
Nodes:
  (:Function {chunk_id, repo_id, file_path, function_name, language})
  (:Commit   {sha, repo_id, message, author, timestamp})
  (:PR       {pr_number, repo_id, title, merged_at})
  (:Issue    {issue_number, repo_id, title, created_at})
  (:ADR      {doc_id, repo_id, title, doc_type})

Edges:
  (:Function)-[:INTRODUCED_BY]->(:Commit)
  (:Function)-[:LAST_MODIFIED_BY]->(:Commit)
  (:Commit)-[:PART_OF_PR]->(:PR)
  (:PR)-[:ADDRESSES]->(:Issue)
  (:Issue)-[:INFORMED_BY]->(:ADR)
  (:Commit)-[:MODIFIES]->(:Function)
  (:Function)-[:CALLS]->(:Function)    -- for cross-reference resolution
```

**Key Cypher query (provenance traversal — used in Phase 4b):**
```cypher
MATCH path = (:Function {chunk_id: $chunk_id, repo_id: $repo_id})
  -[:INTRODUCED_BY|LAST_MODIFIED_BY]->(:Commit)
  -[:PART_OF_PR]->(:PR)
  -[:ADDRESSES]->(:Issue)
RETURN path
```

**Handling missing links:** Not every commit has a linked PR. Not every PR has a linked issue. The graph must handle sparse data gracefully — a node exists in the graph even if its outgoing edges are missing. The `ProvenanceClassifier` in Phase 4d uses the shape of this graph (which edges exist, which are missing) to classify the code's provenance type.

---

### Phase 3: Embedding and vector storage

**Agent:** `EmbeddingAgent`

**Four Chroma collections per repository** (prefixed with `{repo_id}_`):

| Collection | Content | Embedding model | Chunk size |
|---|---|---|---|
| `{repo_id}_code_chunks` | AST-parsed function/class bodies | Voyage Code Embeddings (from benchmark) | Function-level (variable) |
| `{repo_id}_commit_messages` | Full commit message + linked PR/issue numbers | BGE-M3 | One per commit |
| `{repo_id}_pr_discussions` | PR body + all review comments concatenated | BGE-M3 | 512 tokens with 50-token overlap |
| `{repo_id}_adr_rfcs` | Full ADR/RFC documents | BGE-M3 | 512 tokens with 50-token overlap |

**Metadata on every Chroma document (enables filtered retrieval):**
```python
{
    "repo_id": "kubernetes/kubernetes",
    "doc_type": "code_chunk" | "commit" | "pr_discussion" | "adr",
    "source_id": str,               # chunk_id, commit SHA, PR number, or doc_id
    "file_path": str,               # for code chunks
    "timestamp": str,               # ISO format, for temporal filtering
    "author": str,
    "language": str,                # for code chunks
}
```

**Why four collections instead of one:** Mixed-type collections cause retrieval pollution. A query about "why was the auth module rewritten" should retrieve PR discussions and ADRs with high weight, and code chunks with low weight. Separate collections let you query each with appropriate weight and merge results. A single collection forces you to retrieve the top-N across all types, which gets dominated by whichever type has more documents.

---

### Phase 4: Query-time pipeline (LangGraph state graph)

This is the core of the system. Every query from a developer runs through this graph.

#### LangGraph state object

This state object flows through every node. Every node receives the full state and returns an updated state. Nodes must not have side effects outside of state updates and database writes.

```python
class ArchaeologyState(TypedDict):
    # Input
    repo_id: str
    query: str                          # developer's natural language question
    target_chunk_id: Optional[str]      # if developer clicked on a specific function

    # Phase 4a outputs
    query_type: str                     # "what", "when", "who", "why", "evolution", "similar"
    retrieval_strategy: str             # "chroma_only", "postgres_only", "neo4j_postgres", "all_parallel"

    # Phase 4b outputs
    chroma_results: List[RetrievedChunk]
    neo4j_results: List[ProvenanceChain]
    postgres_results: List[CommitMetadata]

    # Phase 4c outputs
    ranked_context: List[RetrievedChunk]    # reranked + deduplicated, trimmed to context window

    # Phase 4d outputs
    provenance_class: str               # one of 7 classes (see below)
    provenance_confidence: float

    # Phase 4e outputs (conditional — only for "evolution" query type)
    evolution_timeline: Optional[List[EvolutionEvent]]

    # Phase 5a outputs
    answer: str
    citations: List[Citation]           # [{source_type, source_id, url, excerpt}]

    # Phase 5b outputs
    draft_pr_url: Optional[str]         # GitHub PR created for human review

    # Evaluation outputs (Phase 6)
    ragas_scores: Optional[Dict[str, float]]
```

#### Node 4a: Query Classifier

**What it does:** Classifies the incoming query into one of six types and selects the appropriate retrieval strategy.

**Implementation:** Do NOT use a full LLM call here. Use a lightweight embedding-based classifier trained on ~200 labeled examples. Target latency: <50ms. A full GPT-4 call here adds 500ms of overhead just to decide what to retrieve.

**Query types and retrieval strategies:**

| Query type | Example | Retrieval strategy |
|---|---|---|
| `what` | "What does this function do?" | Chroma `code_chunks` only |
| `when` | "When was this authentication module added?" | PostgreSQL only |
| `who` | "Who last modified the retry logic?" | PostgreSQL + Neo4j |
| `why` | "Why is this implemented this way?" | All three stores, parallel |
| `evolution` | "How has the auth system evolved?" | Neo4j temporal + PostgreSQL + Chroma |
| `similar` | "Show me similar implementations in the codebase" | Chroma `code_chunks` only |

**LangGraph routing:**
```python
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
```

#### Node 4b: Multi-Source Retrieval (parallel)

**What it does:** Fires retrieval against all selected stores simultaneously using LangGraph's parallel node execution. This is the core reason for using LangGraph over a simple LangChain chain.

**Parallel execution in LangGraph:** Define separate nodes for Chroma retrieval, Neo4j retrieval, and Postgres retrieval. Use a fan-out edge from the router to all three, and a fan-in edge that merges their outputs into `ArchaeologyState` before passing to the reranker. Sequential retrieval at 2 seconds each = 6 seconds total. Parallel = 2 seconds total.

**Chroma retrieval specifics:**
- Query against whichever collections are relevant to the query type
- Apply metadata filters: `repo_id` always, `doc_type` based on query type, `timestamp` range if the query is temporal
- Retrieve top-40 candidates (more than needed — the reranker trims this down)

**Neo4j retrieval specifics:**
- Run the provenance traversal Cypher query from Phase 2b
- If `target_chunk_id` is set: start from that specific function node
- If query is text only: first find the relevant function via Chroma, then traverse from that node
- Return the full provenance chain as structured data, not embedded text

**PostgreSQL retrieval specifics:**
- Structured queries: "all commits touching `auth.py` in the last 2 years, sorted by timestamp"
- Join commits to PRs to issues to get the full structured metadata chain
- Returns structured rows, not embedded text

#### Node 4c: Context Ranker and Deduplicator

**What it does:** Takes the raw retrieved chunks from all sources, re-ranks them by true relevance, deduplicates near-identical chunks, and trims to fit the LLM context window.

**Steps in order:**
1. **Cohere Rerank:** Pass the original query + all retrieved text chunks to `co.rerank()`. This cross-encoder model reads the query and each chunk together and scores true relevance. Much more accurate than cosine similarity alone. Returns re-scored chunks.
2. **MMR deduplication:** Among the top-scored chunks, apply Maximal Marginal Relevance to filter out chunks that are semantically near-identical to a higher-ranked chunk. Prevents the LLM from receiving the same information five times.
3. **Context window trimming:** Keep top chunks up to 80% of the model's context window. Reserve 20% for the system prompt and generated answer.

**Why Cohere Rerank specifically:** It's a cross-encoder, meaning it processes the query and document together as a pair rather than independently. This captures query-document interaction that bi-encoder cosine similarity misses. It's also an API call (no local model to run), making it easy to integrate.

#### Node 4d: Provenance Classifier

**What it does:** Analyzes the shape of the provenance chain returned from Neo4j and classifies the code into one of seven categories. This replaces the original binary "dead code / not dead code" logic.

**This is a reasoning task — use GPT-4 here**, not a classifier. The input is: the code chunk text, the provenance chain (which edges exist, which are missing), and the commit messages associated with it.

**Seven provenance classes:**

| Class | Signal pattern | Synthesis prompt adjustment |
|---|---|---|
| `well_documented` | Full chain: function→commit→PR→issue, PR has rich description | Standard synthesis |
| `legacy_migration` | Old commits, no linked issues, commit message says "migrate" or "port" | "This code was migrated from an older system. The original rationale may predate this repository." |
| `hotfix` | Commit message starts with "fix", "hotfix", "urgent". No PR. Committed directly to main. | "This appears to be an emergency fix. Look for the issue it resolved." |
| `experimental_prototype` | Branch name contains "experiment", "spike", "poc". Never linked to a formal issue. | "This was experimental code. It may not represent a deliberate design decision." |
| `generated_code` | File path contains "generated", "autogenerated", "proto". Commit message is "regenerate" or "update generated files". | "This is generated code. Query the generator configuration, not this file." |
| `third_party_import` | File was added in a single large commit. Author email is external. No subsequent modifications. | "This appears to be imported from a third-party source. The design decisions are upstream." |
| `orphaned` | No linked PR. No linked issue. Commit message is one word or generic. No subsequent references. True dead code. | "No provenance chain could be reconstructed. This code may be safe to remove." |

**LangGraph routing after classification:** Each class routes to a differently-parameterized synthesis prompt in Phase 5a. The `generated_code` class short-circuits the answer with a special message rather than synthesizing an explanation.

#### Node 4e: Architecture Evolution Agent (conditional)

**Triggered when:** `query_type == "evolution"` OR when the Neo4j traversal reveals a function node with 5+ distinct commit chains spanning more than 1 year.

**What it does:** Reconstructs the architectural evolution timeline of a component by:
1. Querying Neo4j for ALL commits that ever touched this component, ordered chronologically
2. Querying PostgreSQL for the PR and issue metadata for each commit
3. Clustering commits into "migration events" — groups of related changes that together constitute a meaningful architectural shift
4. Synthesizing a chronological narrative with GPT-4

**Clustering logic:** A "migration event" is a cluster of commits where:
- The commits are temporally close (within 2 weeks of each other)
- The commits touch the same set of files
- The PR descriptions share semantic similarity (detected via embedding cosine similarity)

**Output structure:**
```python
class EvolutionEvent(BaseModel):
    date: datetime
    event_type: str         # "initial_implementation", "refactor", "security_fix",
                            # "performance_optimization", "migration", "deprecation"
    description: str        # GPT-4 synthesized summary
    pr_number: Optional[int]
    issue_number: Optional[int]
    pr_url: Optional[str]
    files_changed: List[str]
    commit_shas: List[str]
```

**Example output (auth module evolution):**
```
2019-03 — Initial implementation: JWT-based authentication [PR #312]
2020-07 — Hotfix: Session fallback added after JWT library vulnerability [Issue #891]
2021-02 — Performance: Redis cache layer added for token validation [PR #1204, ADR-07]
2022-09 — Security: Token rotation introduced [PR #2891, Security ADR-15]
2023-11 — Migration: Full OAuth 2.0 migration [PR #4431, RFC-22]
```

This is the most impressive demo output in the entire system. Build it well.

---

### Phase 5: Answer synthesis and human-gated documentation

#### Node 5a: Answer Synthesis Agent

**What it does:** Receives the ranked context, provenance class, and optional evolution timeline. Generates a structured answer with every claim cited to a specific source.

**Prompt structure (parameterized by provenance class):**
```
System: You are a code archaeology assistant. Answer questions about why code 
exists by reasoning from the provided historical evidence. Every claim you make 
must be grounded in the provided context. If the context does not support a 
claim, say so explicitly rather than inferring.

Provenance context: [provenance_class and its implications]

Retrieved context (ranked by relevance):
[ranked_context chunks with source labels]

Evolution timeline (if available):
[evolution_timeline]

Query: [original developer query]

Answer format:
- Direct answer (2-3 sentences)
- Historical context (what motivated this)
- Key decision points (what alternatives were considered, from PR discussions)
- Current state (how it evolved to what it is now)
- Citations: [list every claim's source with PR/commit/issue URL]
```

**Citation format:** Every factual claim in the answer must be tagged with its source. Use inline citations like `[PR #4431]`, `[Commit a3f92b]`, `[Issue #891]`, `[ADR-15]`. The citation list at the end must include the full URL to each source on GitHub.

#### Node 5b: Living Docs Writer with Human Approval Gate

**What it does:** Generates a proposed documentation comment and creates a GitHub PR for human review. Does NOT write directly to the repository. Does NOT index the explanation until the PR is merged.

**Why the approval gate exists:** If an LLM-generated explanation contains a hallucination and gets indexed immediately, it becomes "evidence" retrieved in future queries, which reinforces the incorrect explanation. Human approval breaks this feedback loop.

**Workflow:**
1. Format the synthesized answer as a structured code comment (language-appropriate format: Python docstring, JSDoc, Go doc comment, etc.)
2. Create a draft GitHub PR via the API that adds this comment to the relevant file
3. Tag the PR: `ai-generated`, `archaeology-agent`, `needs-review`
4. Store a webhook listener entry: when this specific PR is merged, trigger re-ingestion of the comment into Chroma
5. On merge: ingest the (now human-validated) explanation into `{repo_id}_adr_rfcs` collection

**The flywheel effect:** Each approved explanation enriches the knowledge base. Future queries about the same component retrieve the human-validated explanation as high-quality context, improving answer quality over time. The knowledge base becomes progressively more trustworthy, not progressively noisier.

---

### Phase 6: Evaluation pipeline

**When it runs:** Against a held-out test set of 50–100 (question, expected_answer, expected_source_chunks) triplets. Run after every significant architecture change. Not run on every live query (too slow and expensive).

**RAGAS metrics (automatic given the triplet format):**
- `context_recall` — did the retrieved context contain the information needed to answer?
- `context_precision` — how much of the retrieved context was actually relevant?
- `faithfulness` — does the generated answer stay within the bounds of the retrieved context? (hallucination detection)
- `answer_relevance` — does the answer actually address the question?

**Additional metrics (DeepEval):**
- `citation_accuracy` — does each cited PR/commit/issue actually contain the claim it's cited for?
- `hallucination_rate` — percentage of answers containing claims not grounded in retrieved context

**LangSmith per-node metrics (automatic via tracing):**
- Execution time per node
- Token count and cost per node
- Number of documents retrieved at each step
- Reranker score distribution

**Target baselines (set these before building, evaluate against them after):**
- Context Recall > 0.75
- Faithfulness > 0.85
- Citation Accuracy > 0.90
- Answer latency < 8 seconds end-to-end

---

### Phase 7: Observability

**LangSmith setup (required, minimal effort):**
```bash
export LANGCHAIN_TRACING_V2=true
export LANGCHAIN_PROJECT="code-archaeology-agent"
export LANGCHAIN_API_KEY="your_key"
```

That environment configuration alone gives you automatic traces for every LangGraph node execution including: inputs/outputs, execution time, token usage, cost, errors, and retries.

**What to instrument manually** (in addition to automatic LangSmith tracing):
- Log retrieval counts and scores at each retrieval node
- Log which query_type was classified for each query
- Log provenance_class distribution over time (are most queries hitting "orphaned"? That suggests poor commit hygiene in the target repo)
- Log reranker score distributions (a flat distribution suggests retrieval quality is poor)

---

## Repository structure

```
code-archaeology-agent/
├── README.md                           # Project overview + benchmark results tables
├── docker-compose.yml                  # Postgres + Neo4j + Chroma + Redis
├── .env.example                        # Template with all keys — copy to .env and fill in
├── docs/
│   ├── embedding_benchmark.md          # Results of Phase 1.5 offline experiment
│   └── model_benchmark.md             # Mode 3 LLM comparison table (faithfulness, latency, cost)
│
├── src/
│   ├── config/
│   │   ├── settings.py                 # Pydantic settings (env vars)
│   │   └── repo_profiles.py            # Known repo configurations
│   │
│   ├── ingestion/
│   │   ├── git_ingestion.py            # Phase 1a
│   │   ├── pr_issue_ingestion.py       # Phase 1b
│   │   ├── adr_ingestion.py            # Phase 1c
│   │   └── embedding_agent.py          # Phase 3
│   │
│   ├── parsing/
│   │   ├── ast_parser.py               # Phase 2a — tree-sitter chunking
│   │   └── blame_graph.py              # Phase 2b — Neo4j graph builder
│   │
│   ├── graph/
│   │   ├── state.py                    # ArchaeologyState TypedDict
│   │   ├── nodes/
│   │   │   ├── query_classifier.py     # Phase 4a
│   │   │   ├── retrieval.py            # Phase 4b — parallel retrieval nodes
│   │   │   ├── reranker.py             # Phase 4c
│   │   │   ├── provenance_classifier.py # Phase 4d
│   │   │   ├── evolution_agent.py      # Phase 4e
│   │   │   ├── synthesizer.py          # Phase 5a
│   │   │   └── docs_writer.py          # Phase 5b
│   │   └── pipeline.py                 # LangGraph graph assembly + compilation
│   │
│   ├── llm/
│   │   └── llm_factory.py              # Provider-agnostic LLM factory — all nodes use this
│   │
│   ├── cache/
│   │   └── answer_cache.py             # Redis-backed answer cache keyed on (repo_id, query_hash)
│   │
│   ├── evaluation/
│   │   ├── test_set.json               # 50–100 labeled (question, answer, contexts) triplets
│   │   ├── run_ragas.py                # Phase 6 evaluation runner
│   │   └── benchmark_embeddings.py     # Phase 1.5 offline embedding comparison
│   │
│   ├── api/
│   │   └── main.py                     # FastAPI endpoints
│   │
│   └── db/
│       ├── postgres.py                 # SQLAlchemy models + connection
│       ├── neo4j.py                    # Neo4j driver + Cypher query library
│       └── chroma.py                   # Chroma client + collection management
│
└── tests/
    ├── unit/
    └── integration/
```

---

## LLM factory — `src/llm/llm_factory.py`

This is a required component. No node in the pipeline should directly instantiate `ChatOpenAI`, `ChatAnthropic`, or `ChatGroq`. Everything goes through `get_llm()`.

```python
# src/llm/llm_factory.py

def get_llm(purpose: str = "default") -> BaseChatModel:
    """
    Returns the configured LLM for the given purpose.
    purpose: "default" | "classifier" | "benchmark_openai" | "benchmark_anthropic"
    
    Swap the entire pipeline's LLM by changing LLM_PROVIDER in .env.
    No other code changes required.
    """
    provider = settings.LLM_PROVIDER if purpose == "default" else settings.CLASSIFIER_LLM_PROVIDER
    model = settings.LLM_MODEL if purpose == "default" else settings.CLASSIFIER_LLM_MODEL

    if provider == "groq":
        return ChatGroq(model=model, api_key=settings.GROQ_API_KEY)
    elif provider == "anthropic":
        return ChatAnthropic(model=model, api_key=settings.ANTHROPIC_API_KEY)
    elif provider == "openai":
        return ChatOpenAI(model=model, api_key=settings.OPENAI_API_KEY)
    elif provider == "google":
        return ChatGoogleGenerativeAI(model=model, google_api_key=settings.GOOGLE_API_KEY)
    else:
        raise ValueError(f"Unknown LLM provider: {provider}")
```

Every node that needs an LLM calls `get_llm()`. The benchmark phase runs the full pipeline multiple times with different `.env` configurations and records the RAGAS scores for each. This is the experiment that justifies your model choice with data.

---

## Three execution modes

The system supports three modes configured via `EXECUTION_MODE` in `.env`. This is not optional — it is what makes the project usable during development without burning API credits.

**Mode 1 — Local (default during development)**
```
EXECUTION_MODE=local
LLM_PROVIDER=groq
LLM_MODEL=llama-3.3-70b-versatile
```
- All retrieval: local (Chroma, Neo4j, PostgreSQL — all self-hosted via Docker)
- Embeddings: BGE-M3 running locally via HuggingFace (no API key needed)
- LLM: Groq free tier
- Cost: $0 during development

**Mode 2 — Hybrid (recommended for demo)**
```
EXECUTION_MODE=hybrid
LLM_PROVIDER=anthropic
LLM_MODEL=claude-sonnet-4-6
```
- All retrieval: local (unchanged)
- Embeddings: Voyage API for code chunks, BGE-M3 local for text
- LLM: Claude claude-sonnet-4-6 for final synthesis only — one API call per query
- Cost: ~$0.01–0.03 per query

**Mode 3 — Benchmark**
```
EXECUTION_MODE=benchmark
```
- Runs the full evaluation suite against all configured providers sequentially
- Outputs a comparison table: provider, model, faithfulness, citation accuracy, latency, cost per query
- This table goes into `docs/model_benchmark.md` and the README
- Run once after the pipeline is stable. Do not run repeatedly — it costs money.

---

## Answer caching — `src/cache/answer_cache.py`

Implement a simple Redis-backed cache keyed on `(repo_id, query_hash)`. This is required for two reasons: demos are more impressive when repeated questions are instant, and it prevents paying for the same LLM call twice during testing.

```python
import hashlib, json
from redis import Redis

class AnswerCache:
    def __init__(self, redis_url: str):
        self.client = Redis.from_url(redis_url)
        self.ttl = 60 * 60 * 24 * 7  # 7 days

    def cache_key(self, repo_id: str, query: str) -> str:
        h = hashlib.sha256(f"{repo_id}:{query.strip().lower()}".encode()).hexdigest()
        return f"answer:{h}"

    def get(self, repo_id: str, query: str) -> dict | None:
        key = self.cache_key(repo_id, query)
        val = self.client.get(key)
        return json.loads(val) if val else None

    def set(self, repo_id: str, query: str, answer: dict):
        key = self.cache_key(repo_id, query)
        self.client.setex(key, self.ttl, json.dumps(answer))
```

The cache check happens at the FastAPI layer before the LangGraph pipeline is invoked. Cache miss → run pipeline → store result → return. Cache hit → return immediately, zero LLM cost. Add Redis to Docker Compose alongside Postgres, Neo4j, and Chroma.

---

## Build order — what to build first

Build in this exact sequence. Each phase produces something testable before moving to the next.

1. **Docker Compose environment** — get Postgres, Neo4j, Chroma, and Redis running locally first. Verify connections. Nothing else works without this.

2. **LLM factory (`src/llm/llm_factory.py`)** — build this second, before any agent code. Every subsequent node will use `get_llm()`. Set `LLM_PROVIDER=groq` in `.env` and verify a basic Groq API call works.

3. **Answer cache (`src/cache/answer_cache.py`)** — build this third. Wire it into the FastAPI endpoint before the pipeline exists. Verify Redis set/get works.

4. **Phase 0 (RepoConfigAgent)** — pick one small repo to test with (FastAPI is good: manageable size, clean PRs, good commit messages). Get the RepoProfile working.

5. **Phase 1a (GitIngestionAgent)** — ingest commits into Postgres. Verify with SQL queries. Test incremental sync.

6. **Phase 1b (PRIssueIngestionAgent)** — ingest PRs and issues. Verify linkage between commits and PRs.

7. **Phase 2a (ASTCodeParser)** — parse the repo's code with tree-sitter. Verify chunk boundaries are at function/class level, not arbitrary line counts.

8. **Phase 2b (BlameGraphBuilder)** — build the Neo4j graph. Verify the provenance chain with a manual Cypher query on a known function.

9. **Phase 3 (EmbeddingAgent)** — embed everything into Chroma. Verify retrieval with a test query before building any agents on top.

10. **Phase 4 (LangGraph pipeline)** — build nodes one at a time. Test each node in isolation before wiring the graph. The deterministic query classifier first, then retrieval nodes, then reranker, then provenance classifier, then evolution agent last.

11. **Phase 5 (Synthesis + Docs Writer)** — get the answer quality working before worrying about the PR creation workflow.

12. **Phase 6 (Evaluation)** — build the test set and run RAGAS. Tune retrieval and synthesis based on scores.

13. **Phase 1.5 (Embedding Benchmark)** — run this after the full pipeline works so you have a real retrieval setup to evaluate models against.

14. **Mode 3 (Model Benchmark)** — run last, after everything is stable. Compare Groq, Claude, GPT-4o on the RAGAS test set. Document results in `docs/model_benchmark.md`.

---

## Critical rules — do not violate these

1. **Every LLM call goes through `llm_factory.get_llm()`.**  Never instantiate `ChatGroq`, `ChatAnthropic`, or `ChatOpenAI` directly in a node. The factory is the only place provider-specific code lives.

2. **Every schema includes `repo_id`.** Every Postgres table, every Neo4j node, every Chroma metadata dict. This is non-negotiable for future multi-repo support.

3. **The query classifier is deterministic Python first, LLM second.** Implement keyword/regex rules that handle the common cases. Only fall through to an LLM call for queries that the rules cannot classify. Log the fallthrough rate — if it exceeds 10%, improve the rules.

4. **Answer cache is checked before the LangGraph pipeline is invoked.** Cache hit = return immediately, zero LLM cost. Cache miss = run pipeline, store result, return. This is wired at the FastAPI layer, not inside LangGraph.

5. **Do not chunk code by line count.** Use tree-sitter AST boundaries. A function that spans lines 100–180 is one chunk. Period.

3. **Do not use one Chroma collection for everything.** Four collections per repo: `code_chunks`, `commit_messages`, `pr_discussions`, `adr_rfcs`. Mixed collections degrade retrieval quality.

4. **The Query Classifier must not use a full LLM call.** Use an embedding-based classifier. Target <50ms. An LLM call here wastes 500ms on every single query just to decide what to retrieve.

5. **The Living Docs Writer must not write directly to the repository.** Always create a PR for human review. Never auto-merge. The hallucination feedback loop this prevents is not a theoretical concern — it will happen.

6. **The Provenance Classifier has seven classes, not two.** "Dead code" is only one of seven possible conclusions. Jumping to "dead code" on missing provenance is incorrect — missing provenance most often means hotfix or legacy migration, not dead code.

7. **Reranking is not optional.** Without Cohere Rerank after retrieval, the top-40 chunks going into synthesis will contain substantial noise. The reranker is what makes the synthesis quality acceptable.

8. **LangSmith tracing must be enabled from day one.** Set the environment variables before writing any pipeline code. Debugging a LangGraph pipeline without traces is extremely painful.

---

## Environment variables

```bash
# GitHub
GITHUB_TOKEN=                       # Personal access token with repo read scope

# LLM providers — only the ones you have keys for need to be filled
GROQ_API_KEY=                       # Primary dev model — free at console.groq.com
ANTHROPIC_API_KEY=                  # Demo/benchmark — console.anthropic.com
OPENAI_API_KEY=                     # Benchmark only — optional
GOOGLE_API_KEY=                     # Benchmark only — optional

# LLM config — change these to switch providers without touching code
LLM_PROVIDER=groq                   # groq | anthropic | openai | google
LLM_MODEL=llama-3.3-70b-versatile   # model name for chosen provider
CLASSIFIER_LLM_PROVIDER=groq        # provider for query classifier fallback
CLASSIFIER_LLM_MODEL=llama-3.1-8b-instant  # fast cheap model for classification

# Cohere (reranker — free tier at dashboard.cohere.com)
COHERE_API_KEY=

# Voyage (code embeddings — free tier at dash.voyageai.com)
VOYAGE_API_KEY=

# PostgreSQL
POSTGRES_URL=postgresql://localhost:5432/archaeology

# Neo4j
NEO4J_URL=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=

# Chroma
CHROMA_HOST=localhost
CHROMA_PORT=8000

# Redis (answer cache)
REDIS_URL=redis://localhost:6379

# LangSmith (observability — enable from day one)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=code-archaeology-agent
LANGCHAIN_API_KEY=
```

---

## What "done" looks like

The system is complete when a developer can:

1. Point it at any GitHub repository URL
2. Ask "why is `function_name` in `file_path` written this way?"
3. Receive a structured answer with: historical context, the PR and issue that motivated it, alternatives that were considered (from PR review comments), how the component evolved over time, and citations with working GitHub URLs for every claim
4. See a draft PR created in the repository with a proposed documentation comment for human review
5. View per-node execution traces in LangSmith
6. Run the RAGAS evaluation suite and see scores above the target baselines

That is the complete system. Build it in the order specified. Do not skip the evaluation phase — it is what makes this project defensible to a technical audience.