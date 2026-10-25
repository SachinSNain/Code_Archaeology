# Code Archaeology Agent — Interview Preparation Guide

This document prepares you to answer every possible interview question about this project at a deep technical level. It covers concept definitions, architectural decisions with alternatives and trade-offs, system design questions, and the reasoning behind every major choice. Read this after building the project so the answers come from genuine understanding, not memorization.

---

## Part 1 — How to introduce the project (the 60-second pitch)

When an interviewer says "tell me about your project", this is what you say:

> "I built a system called the Code Archaeology Agent. The core problem it solves is that every software team has code that nobody understands — functions written years ago, by people who've left, for reasons that were never documented. Current AI tools can tell you *what* code does, but not *why* it exists. My system answers the *why* question by treating a codebase as a historical record — it traverses git blame to find the commit that introduced a function, traces that commit to the PR that contained it, traces the PR to the issue that motivated it, and traces the issue to the architectural discussion that shaped it. Every answer is grounded in the actual history of the repository with citations to specific commits, PRs, and issues. The system uses LangGraph for multi-agent orchestration, a three-store data architecture combining PostgreSQL, Neo4j, and ChromaDB, and is evaluated using RAGAS metrics for retrieval quality and faithfulness."

That answer takes 45 seconds and immediately signals you understand agents, RAG, graph databases, evaluation, and production engineering. The interviewer will then ask follow-up questions — the rest of this document prepares you for all of them.

---

## Part 2 — Core concept definitions (what every term means, interview-depth)

### Agentic AI

**Simple definition:** An AI system where an LLM doesn't just answer a single question — it takes a sequence of actions, makes decisions about what to do next, uses tools, and works toward a goal over multiple steps.

**Interview-depth definition:** Agentic AI refers to systems where an LLM acts as a reasoning engine that selects and executes actions from a set of available tools, observes the results, and decides on the next action — repeating this loop until the goal is achieved. The key distinction from a simple prompt-response system is *autonomy over a multi-step workflow*. The LLM doesn't just generate text — it decides which database to query, what to retrieve, whether the retrieved context is sufficient, and whether to take additional retrieval steps before synthesizing an answer.

**In your project:** The system has multiple specialized agents — a RepoConfigAgent that fingerprints repositories, a GitIngestionAgent that walks commit history, a ProvenanceClassifier that reasons about why code exists, an ArchitectureEvolutionAgent that reconstructs timelines, and a SynthesisAgent that generates cited answers. Each agent has a single responsibility and uses different tools (GitHub API, Neo4j Cypher queries, Chroma vector search, PostgreSQL) to fulfill it.

**What makes it genuinely agentic vs. fake agentic:** Real agents make decisions that change the execution path. In this system, the ProvenanceClassifier's output (which of seven classes it assigns) changes which synthesis prompt is used. The QueryClassifier's output changes which combination of databases is queried. The EvolutionAgent only fires when the Neo4j traversal reveals 5+ commit chains spanning more than a year. These are real conditional decisions, not a fixed pipeline with agents in name only.

---

### RAG (Retrieval-Augmented Generation)

**Simple definition:** Instead of asking an LLM to answer from its training data alone, you first retrieve relevant documents from your own knowledge base and include them in the prompt as context. The LLM generates an answer grounded in those retrieved documents.

**Interview-depth definition:** RAG addresses two fundamental limitations of LLMs: knowledge cutoff (training data has a fixed date) and hallucination (LLMs confabulate facts not in training). By retrieving relevant context at query time and inserting it into the prompt, you ground the model's response in a specific, verifiable knowledge base. The retrieval step uses embedding similarity — documents are converted to dense vector representations during indexing, queries are converted to vectors at runtime, and nearest-neighbor search finds the most semantically similar documents.

**The standard RAG pipeline:**
1. Indexing: chunk documents → embed each chunk → store (embedding, chunk, metadata) in vector DB
2. Retrieval: embed query → nearest-neighbor search → return top-K chunks
3. Generation: insert chunks into prompt → LLM generates grounded answer

**In your project:** This is *multi-source RAG*, not standard single-collection RAG. You retrieve from four Chroma collections (code chunks, commit messages, PR discussions, ADR/RFC documents), a Neo4j graph (provenance chains), and PostgreSQL (structured metadata). The retrieved context from all sources is merged, reranked by a cross-encoder, and passed to the synthesis LLM. This is significantly more sophisticated than "PDF chatbot" RAG and directly addresses the limitation that different question types need different retrieval strategies.

**What makes your RAG novel:** Standard RAG retrieves text that is semantically similar. Your system also retrieves *causally linked* content via graph traversal — the PR that caused this code to be written is not necessarily semantically similar to the code itself, but it is the most relevant document for explaining why the code exists. Vector search alone cannot find causal relationships. That's why you need Neo4j alongside Chroma.

---

### Vector Database

**Simple definition:** A database optimized for storing and searching high-dimensional vectors (embeddings), finding the nearest neighbors to a query vector faster than brute-force comparison.

**Interview-depth definition:** A vector database stores dense numerical representations of text, code, or other content. These representations (embeddings) are learned by neural models such that semantically similar content has similar vector representations — mathematically, their cosine similarity is high. Vector databases use approximate nearest neighbor (ANN) algorithms like HNSW (Hierarchical Navigable Small World) or IVF (Inverted File Index) to find similar vectors in milliseconds even across millions of documents, which brute-force comparison cannot do at scale.

**Why cosine similarity, not Euclidean distance:** Cosine similarity measures the angle between two vectors regardless of their magnitude. For text embeddings, the magnitude carries no semantic meaning — only the direction matters. Two identical sentences embedded at different scales would have different Euclidean distances but identical cosine similarity.

**In your project:** You use Chroma as the vector database with four collections per repository. Code chunks use Voyage Code Embeddings (a model specifically trained on code) and the other collections use BGE-M3 (a general-purpose multilingual model). The decision to use separate collections rather than one mixed collection is deliberate — mixing code and PR discussions in one collection means queries about code will retrieve PR discussions simply because they're more numerous, contaminating the results.

**Chroma specifically:** Open source, runs locally via Docker, Python-native API, supports metadata filtering (which you use to filter by `repo_id`, `doc_type`, and `timestamp`). The alternative was Pinecone (managed cloud service) — rejected because it requires internet connectivity, has per-query costs, and is harder to run in local development. For a self-hostable portfolio project, Chroma is the right choice.

---

### LangChain

**Simple definition:** A Python framework that provides abstractions for building LLM-powered applications — it wraps LLM APIs, provides document loaders, text splitters, prompt templates, output parsers, and chains that connect these components together.

**Interview-depth definition:** LangChain standardizes the interface to LLMs, embedding models, vector stores, and document sources behind consistent abstractions. A `ChatOpenAI` object and a `ChatAnthropic` object have the same `.invoke()` interface, so you can swap providers without changing application logic. LangChain's `DocumentLoader` abstractions handle reading from PDFs, web pages, GitHub, databases, etc. Its `PromptTemplate` class manages structured prompts with variable injection. Its `OutputParser` classes extract structured data from LLM responses.

**What LangChain is NOT good for:** Linear chains where every query follows the same path. It has no native concept of shared state across steps, conditional branching based on intermediate results, parallel execution of multiple steps, or retry logic with state preservation. This is why LangGraph was created.

**In your project:** LangChain is used *within* individual nodes — for prompt templates, for the Chroma retrieval interface, for output parsing. LangGraph orchestrates *between* nodes. The relationship is: LangGraph handles the workflow; LangChain handles the per-node LLM interactions.

---

### LangGraph

**Simple definition:** An extension of LangChain that lets you build multi-agent workflows as directed graphs with shared state, conditional routing, parallel execution, and cycles.

**Interview-depth definition:** LangGraph models a multi-step AI workflow as a directed graph where nodes are Python functions (or LLM calls) and edges define the flow between them. The key innovation is *shared typed state* — a `TypedDict` object that every node receives and returns an updated copy of. This means every node has full visibility into everything that happened before it, and every node's output is available to all subsequent nodes. LangGraph supports:

- **Conditional edges:** A router function inspects the state and returns a string indicating which node to go to next. This enables "if the query is about 'when', go to PostgreSQL; if it's about 'why', go to all three stores in parallel."
- **Parallel fan-out/fan-in:** Multiple nodes can execute simultaneously, with their results merged back into the shared state before the next node runs.
- **Cycles:** A node can route back to an earlier node (for retry logic or iterative refinement).
- **Persistence:** State can be checkpointed to disk, enabling long-running workflows and resumability.

**Why LangGraph over plain LangChain chains:**

| Capability | LangChain Chain | LangGraph |
|---|---|---|
| Shared state across steps | No — each step passes output to next | Yes — TypedDict available to all nodes |
| Conditional routing | No — fixed linear flow | Yes — router functions |
| Parallel execution | No | Yes — fan-out/fan-in |
| Cycles and retries | No | Yes |
| Human-in-the-loop pause | No | Yes — interrupt nodes |

**In your project:** The LangGraph pipeline has conditional edges after the QueryClassifier (routes to different retrieval strategies), parallel fan-out to three retrieval nodes simultaneously (Chroma + Neo4j + PostgreSQL), a conditional trigger for the ArchitectureEvolutionAgent (only fires when 5+ commit chains detected), and a human approval gate before the Living Docs Writer indexes any content. None of these are expressible as a LangChain chain.

**The answer every interviewer wants to hear when they ask "why LangGraph?":**
> "We initially evaluated LangChain chains, but our workflow required four things LangChain doesn't support: parallel retrieval from three different stores simultaneously, conditional routing based on query type, shared state across all nodes so every agent has full context, and a human-in-the-loop approval gate. LangGraph models these naturally as a directed state graph. Implementing the same functionality with LangChain would have required complex custom orchestration that LangGraph provides out of the box."

---

### Neo4j and Graph Databases

**Simple definition:** A database that stores data as nodes (entities) and edges (relationships between entities), optimized for traversing multi-hop relationships.

**Interview-depth definition:** Relational databases store data in tables and use JOIN operations to connect related records. For one or two levels of relationship, JOINs are fast. For multi-hop traversals — "find all functions that were introduced by commits linked to PRs that addressed issues labeled 'security'" — the number of JOINs grows, query complexity explodes, and performance degrades. Graph databases store relationships as first-class citizens (edges with their own properties) and traverse them in O(depth) time regardless of graph size. Neo4j uses the Cypher query language, which reads naturally: `MATCH (f:Function)-[:INTRODUCED_BY]->(c:Commit)-[:PART_OF_PR]->(p:PR)-[:ADDRESSES]->(i:Issue) RETURN f, p, i`.

**Why not PostgreSQL for the graph:** PostgreSQL can do recursive CTEs to traverse graph-like data, but the queries become complex and hard to maintain past two or three hops. The Neo4j Cypher equivalent of a three-hop traversal is four lines. The PostgreSQL recursive CTE equivalent is 20+ lines with self-joins. More importantly, the Neo4j query is readable and maintainable by anyone who knows Cypher — the SQL version is not.

**Why not Chroma for the graph:** Vector search finds semantically similar content. It cannot find causally related content. The PR that introduced a function may use completely different vocabulary from the function itself — "Add OAuth support for enterprise customers" is not semantically similar to `def _validate_token(token: str) -> bool`. But it is causally related — it's the PR that created that function. Graph traversal finds causal relationships; vector search finds semantic similarity. You need both.

**In your project:** Neo4j stores the blame graph — Function nodes, Commit nodes, PR nodes, Issue nodes, ADR nodes, connected by edges that represent real relationships (INTRODUCED_BY, PART_OF_PR, ADDRESSES, INFORMED_BY). The provenance traversal query starts at a Function node and walks to the Issue node in a single Cypher query. This gives you the full "why does this code exist" chain in one database operation.

---

### Embeddings

**Simple definition:** A numerical representation of text (or code) as a point in high-dimensional space, where semantically similar content is close together.

**Interview-depth definition:** An embedding model is a neural network trained to project text into a dense vector space such that the distance between vectors reflects semantic similarity. Models like BGE-M3 are trained on large corpora with contrastive learning — pairs of semantically similar texts are pushed closer together in vector space, pairs of dissimilar texts are pushed apart. The resulting vectors (typically 768 to 3072 dimensions) encode meaning rather than surface form: "fix authentication bug" and "resolve login issue" have similar embeddings even though they share no words.

**Bi-encoder vs. cross-encoder:**
- **Bi-encoder** (used in vector DB retrieval): Each document and query is embedded independently. Fast because embeddings can be precomputed. Less accurate because the model doesn't see the query and document together.
- **Cross-encoder** (used in reranking): The query and document are concatenated and processed together by the model. Much more accurate because the model can attend to the interaction between query and document. Slow because it cannot be precomputed — must run at query time for each candidate.

**In your project:** You use bi-encoders (Voyage, BGE-M3) for initial retrieval — fast, precomputed, good enough for top-40 candidate selection. Then you use a cross-encoder (Cohere Rerank) on those top-40 candidates to produce a more accurate ranking before passing to the LLM. This two-stage approach gives you the speed of bi-encoders with the accuracy of cross-encoders.

**Why different models for code vs. text:** General embedding models are trained predominantly on natural language text. Code has different syntax, structure, and vocabulary. A general model might not recognize that `get_user_by_id()` and `fetch_user(id)` are semantically equivalent, or that a Python list comprehension and a for loop achieve the same thing. Voyage Code Embeddings and CodeBERT are specifically fine-tuned on code repositories, making them significantly better at code-level semantic similarity. The offline benchmark validated this — code-specific models outperformed general models by a measurable margin on the code chunk retrieval task.

---

### AST (Abstract Syntax Tree)

**Simple definition:** A tree-structured representation of source code where each node represents a syntactic construct (function, class, variable, expression) rather than raw text.

**Interview-depth definition:** When source code is parsed, the parser produces an AST — a hierarchical representation of the code's structure. The root node might be a Module, which has children that are FunctionDef and ClassDef nodes, which have children that are arguments, decorators, and body statements. The AST abstracts away whitespace, comments, and formatting, exposing only the structural and semantic content of the code.

**Why you parse ASTs for chunking:** If you chunk code by line count (e.g., every 200 lines), you will routinely split a function across two chunks. The first chunk has the function signature and the first half of the body; the second chunk has the rest of the body and the beginning of the next function. Neither chunk is semantically complete. When you embed these chunks, the embeddings don't represent complete units of meaning. When you retrieve a chunk about "authentication logic" you might get half a function that is useless without the other half.

AST-based chunking splits at function and class boundaries — every chunk is a semantically complete unit of code. A 500-line file with 12 functions produces 12 chunks, each independently meaningful.

**tree-sitter specifically:** A parser generator that produces parsers for 40+ programming languages with a consistent API. It is incremental (re-parses only changed subtrees, not entire files), fault-tolerant (produces partial ASTs for syntactically broken files rather than crashing), and fast (implemented in C with Python bindings). For a repo-agnostic system that must handle Python, JavaScript, TypeScript, Go, Java, and Rust with one codebase, tree-sitter is the correct tool.

---

### Reranking (Cohere Rerank)

**Simple definition:** A second-pass scoring step that takes the top candidates from vector search and re-scores them using a more accurate but slower model, improving the quality of the final context passed to the LLM.

**Interview-depth definition:** Vector search retrieves candidates using cosine similarity between bi-encoder embeddings. This is fast (milliseconds) and good for narrowing from millions of documents to dozens of candidates, but not precise enough for the final context selection — the top-10 by cosine similarity often contains irrelevant documents that happen to be in the same vector neighborhood. A reranker (cross-encoder) takes the query and each candidate document together as a pair and scores their relevance directly. Because the model sees both inputs simultaneously, it can capture query-document interaction that bi-encoders miss.

**The two-stage retrieval pipeline:**
1. Stage 1 (recall): Vector search retrieves top-40 candidates. Fast. Trades precision for recall — gets the right documents in the set, even if ranked imperfectly.
2. Stage 2 (precision): Cross-encoder reranker scores all 40 candidates against the query. Slow but only runs on 40 documents, not millions. Returns top-10 ranked by true relevance.

**Why Cohere Rerank specifically:** It is a managed API — no model to download or run locally. The free tier provides 1,000 reranking calls per month, sufficient for development. The model is a dedicated cross-encoder trained specifically for document reranking, not a general-purpose LLM adapted for it. The quality is measurably better than alternatives for this specific task.

**What happens if you skip reranking:** The LLM receives 40 chunks of mixed relevance. The prompt becomes dominated by noise. The LLM must attend to relevant content buried among irrelevant content, which degrades synthesis quality and increases hallucination risk. In testing, answers generated without reranking had measurably lower faithfulness scores on RAGAS.

---

### RAGAS (Retrieval Augmented Generation Assessment)

**Simple definition:** A framework for automatically evaluating the quality of a RAG system across multiple dimensions — retrieval quality, answer faithfulness, and answer relevance.

**Interview-depth definition:** RAGAS takes as input a set of (question, answer, retrieved_contexts, ground_truth) tuples and computes four core metrics:

- **Context Recall:** Of all the information needed to answer the question correctly, what fraction was present in the retrieved context? Measures retrieval completeness. A score of 0.8 means 80% of the necessary information was retrieved; 20% was missed.

- **Context Precision:** Of all the retrieved context, what fraction was actually relevant to answering the question? Measures retrieval precision. A low score means too much irrelevant content is being retrieved and passed to the LLM.

- **Faithfulness:** Does the generated answer contain only claims that are supported by the retrieved context? Measures hallucination. A faithfulness score of 0.9 means 90% of claims in the answer are grounded in the retrieved context; 10% are fabricated.

- **Answer Relevance:** Does the answer actually address the question that was asked? Measures whether the synthesis step stayed on topic.

**Why these four metrics matter together:** A system can have high context recall (retrieving everything needed) but low faithfulness (LLM ignoring the context and making things up). Or high faithfulness (LLM only says what's in the context) but low context recall (the right context wasn't retrieved, so the answer is incomplete). You need all four to be high for a trustworthy system.

**In your project:** You build a test set of 50-100 manually labeled (question, expected_answer, expected_source_chunks) triplets from the FastAPI repository. RAGAS runs against this test set and produces the four scores. Target baselines: Context Recall > 0.75, Faithfulness > 0.85, Citation Accuracy > 0.90. These baselines are set before building, evaluated after, and used to guide tuning decisions.

---

### LangSmith

**Simple definition:** An observability platform for LLM applications that automatically traces every step in a LangChain/LangGraph pipeline, capturing inputs, outputs, execution time, token usage, and costs for every node.

**Interview-depth definition:** LangSmith is Anthropic's LangChain's observability tool that instruments LangGraph pipelines automatically when `LANGCHAIN_TRACING_V2=true` is set. For every query that runs through the pipeline, LangSmith records a trace — a hierarchical view of every node that executed, what it received as input, what it returned as output, how long it took, and how many tokens were consumed. This enables:

- **Debugging:** When a query returns a wrong answer, you can see exactly what was retrieved in each store, how the reranker scored each chunk, and what context the LLM actually received.
- **Performance profiling:** You can see which node is the latency bottleneck. If the Neo4j traversal takes 3 seconds, that's where to optimize.
- **Cost tracking:** Token counts per node let you calculate per-query cost by model. Used in the benchmark phase to compare cost/quality trade-offs across providers.
- **Regression detection:** After making a change, compare traces before and after to see if retrieval quality or answer quality changed.

**In your project:** LangSmith is enabled from day one, before any agent code is written. The per-node traces are what allow you to tune the pipeline — if context precision is low on RAGAS, you look at the LangSmith traces to see what irrelevant chunks are being retrieved and trace that back to which retrieval node and which collection they came from.

---

### The Provenance Graph (blame graph)

**Simple definition:** A directed graph where every function in the codebase is connected to the commit that introduced it, the PR that contained that commit, the issue that motivated that PR, and the discussion that shaped the design.

**Interview-depth definition:** Provenance means "origin and history." The provenance graph answers the question "where did this code come from and why?" for every function in the codebase. It is built using `git blame` — a git operation that annotates every line of a file with the commit SHA that last modified it. For each function:

1. `git blame` gives you the commit SHA for each line in the function body
2. The commit SHA links to a commit record in PostgreSQL (which may contain `Fixes #123` references)
3. The commit SHA links to a PR via the merge commit SHA stored in the PR record
4. The PR record contains linked issue numbers
5. The issue record contains the discussion thread
6. The discussion thread contains the reasoning behind the decision

This chain is stored as a Neo4j graph. The traversal at query time is a single Cypher query that walks from Function → Commit → PR → Issue in one database operation. The result is the complete decision context for why the function exists.

**Why this is the core insight of the project:** Most "RAG over codebase" projects embed the code and retrieve similar code snippets. They answer "what does similar code look like?" but not "why does this specific code exist?" The provenance graph is what enables the "why" answer — it connects a line of code to the human decision that caused it to be written.

---

### Human-in-the-loop (HITL)

**Simple definition:** A design pattern where an AI system pauses its automated workflow and requires a human to review and approve before proceeding.

**Interview-depth definition:** Human-in-the-loop is a safety and quality mechanism for AI systems that take consequential actions. Without HITL, an LLM-generated output that contains a hallucination can be immediately acted upon — in this case, immediately indexed into the knowledge base and served as "evidence" to future queries. This creates a feedback loop where incorrect information self-reinforces.

**In your project:** The Living Docs Writer node does not write directly to the repository. Instead, it creates a GitHub PR with the generated explanation as a proposed code comment, tagged with `ai-generated` and `needs-review` labels. A human developer reviews the explanation, optionally edits it, and approves the PR. Only after merge does a webhook trigger re-ingestion of the (now human-validated) explanation into the Chroma knowledge base.

**Why this specific design:** The PR workflow is already a tool developers use daily — it requires no new UI or process. The `ai-generated` label makes AI-generated content clearly identifiable. The webhook-based re-ingestion means the feedback loop from answer → indexed knowledge is preserved, just gated. Over time, the system accumulates a human-validated explanation layer on top of the raw historical data, making future answers progressively better.

---

## Part 3 — Architectural decisions (with alternatives and trade-offs)

This is the section that wins interviews. For every major decision, you know: what you chose, what the alternatives were, why you rejected them, and what the trade-off is.

---

### Decision 1: Why three separate databases instead of one?

**What you chose:** PostgreSQL + Neo4j + Chroma

**Alternative considered:** Store everything in one vector database (e.g., Chroma or Pinecone)

**Why rejected:** Different question types need fundamentally different query mechanisms. "When was this function added?" is a SQL query with a timestamp sort — a vector DB cannot answer it efficiently. "What caused this code to exist?" is a multi-hop graph traversal — a vector DB cannot express causal relationships. "What is semantically similar to this?" is embedding similarity search — a relational DB is inefficient for this. No single database is good at all three. Trying to do graph traversal in SQL becomes unmaintainable recursive CTEs. Trying to do semantic search in SQL requires pgvector with significant operational complexity. Trying to do structured queries in Chroma loses the ACID guarantees and JOIN capabilities you need.

**Trade-off you accepted:** Operational complexity — three databases to run, maintain, and keep synchronized. Mitigated by Docker Compose (all three run with one command) and the `repo_id` field on every record (consistent namespace across all three stores).

**The one-line answer:** "Each database answers a different type of question that the other two cannot answer efficiently. This is a fundamental data modeling decision, not a technology preference."

---

### Decision 2: Why LangGraph instead of a LangChain chain?

**What you chose:** LangGraph

**Alternative considered:** LangChain LCEL (LangChain Expression Language) chains

**Why rejected:** LangChain chains are linear and stateless between steps. Our workflow requires: shared state across all nodes (every agent needs to see what previous agents found), conditional routing (different retrieval strategies based on query type), parallel execution (three retrieval nodes firing simultaneously), and a human approval gate that pauses the pipeline. None of these are expressible in LangChain LCEL. Implementing them manually would require custom orchestration code that LangGraph provides as a first-class feature.

**Trade-off you accepted:** LangGraph has a steeper learning curve than LangChain chains and requires understanding graph concepts (nodes, edges, state). The `TypedDict` state object must be carefully designed upfront — adding fields later requires schema changes.

---

### Decision 3: Why tree-sitter for code parsing instead of line-count chunking?

**What you chose:** tree-sitter AST-based chunking at function/class boundaries

**Alternative considered:** Fixed-size line-count chunking (e.g., every 200 lines with 20-line overlap) — the standard approach in most RAG tutorials

**Why rejected:** Line-count chunking splits functions across chunk boundaries. A 300-line function chunked at 200 lines produces one chunk with the signature and first 200 lines, and a second chunk with the remaining 100 lines. Neither chunk is semantically complete — the first chunk lacks the return statement, the second lacks the function signature. The embeddings for these fragments don't represent meaningful units. Retrieval based on these embeddings is unreliable.

**Trade-off you accepted:** tree-sitter requires language-specific grammar installation and parsing logic. Chunk sizes are variable (a 5-line function and a 300-line function are both one chunk). Very large functions may still need secondary chunking. The complexity is worth it for the retrieval quality gain.

---

### Decision 4: Why Voyage Code Embeddings for code instead of OpenAI text-embedding-3-large?

**What you chose:** Voyage Code Embeddings for code chunks, BGE-M3 for text

**Alternative considered:** OpenAI `text-embedding-3-large` for everything

**Why rejected:** General embedding models are trained predominantly on natural language. Code has different syntax, structure, and vocabulary. During the offline benchmark (Phase 1.5), Voyage Code Embeddings achieved Recall@5 of 0.79 on code chunk queries versus 0.61 for OpenAI `text-embedding-3-large` — an 18-point difference. For PR discussions and ADRs (natural language), BGE-M3 and OpenAI performed comparably, but BGE-M3 is free to run locally.

**Trade-off you accepted:** Two different embedding models add operational complexity and mean code and text embeddings live in separate collections (they cannot be mixed in one collection because the vector spaces are incompatible). This is the right trade-off — you get better retrieval at the cost of slightly more complexity.

---

### Decision 5: Why Cohere Rerank instead of just using top-K cosine similarity?

**What you chose:** Two-stage retrieval: cosine similarity for recall + Cohere cross-encoder for precision

**Alternative considered:** Just return top-10 by cosine similarity directly to the LLM

**Why rejected:** Cosine similarity measures geometric distance in embedding space. It is good for recall (getting the right documents into the candidate set) but poor for precision (ranking them correctly). In testing, the top-10 by cosine similarity reliably contained 3-4 irrelevant documents that happened to be in the same embedding neighborhood. These irrelevant documents consumed context window space and confused the LLM. Cohere Rerank — a cross-encoder that reads query and document together — produced significantly fewer irrelevant documents in the final top-10, measurably improving faithfulness scores on RAGAS.

**Trade-off you accepted:** Additional API call (Cohere), additional latency (cross-encoder inference is slower than cosine similarity). Both are acceptable: Cohere's free tier covers development, and the latency is partially offset by parallel retrieval from all three stores.

---

### Decision 6: Why Groq for the LLM instead of GPT-4o?

**What you chose:** Groq (Llama 3.3 70B) for development, Claude claude-sonnet-4-6 for demo

**Alternative considered:** GPT-4o throughout

**Why rejected:** GPT-4o costs approximately $0.01-0.03 per query. During development with hundreds of test queries, this accumulates to $20-50 of unnecessary spend before the system is even working correctly. Groq provides Llama 3.3 70B at zero cost with extremely fast inference. For development and testing, the quality difference is not material — what matters is whether the pipeline logic is correct, not whether the prose is slightly more elegant. For the final demo and benchmark, Claude claude-sonnet-4-6 is used for the synthesis node, where quality matters.

**Trade-off you accepted:** Groq has rate limits on the free tier and Llama 3.3 70B is slightly below GPT-4o quality on complex reasoning. Mitigated by the provider-agnostic `llm_factory.py` — switching from Groq to any other provider in the final demo is a one-line `.env` change.

---

### Decision 7: Why a deterministic query classifier instead of an LLM classifier?

**What you chose:** Keyword/regex rules for query classification, LLM fallback only for ambiguous queries

**Alternative considered:** LLM call (GPT-3.5/Llama) to classify query type on every request

**Why rejected:** The query classification task — deciding whether a query is about "when", "who", "why", "what", "evolution", or "similar" — does not require LLM-level reasoning. A simple keyword match handles >95% of cases correctly: queries containing "when", "added", "introduced" map to the `when` type; queries containing "who", "author", "wrote" map to the `who` type. An LLM call adds 300-500ms of latency and API cost to every single query, just to perform a task that regex handles in 5ms. The 5% of queries that are genuinely ambiguous (no clear keyword signal) fall through to an LLM classifier call. Log the fallthrough rate — if it exceeds 10%, the rules need improvement.

**Trade-off you accepted:** The deterministic rules must be maintained as new query patterns emerge. The fallthrough logging creates a feedback loop for improving the rules over time.

---

### Decision 8: Why Redis for answer caching instead of in-memory dict?

**What you chose:** Redis-backed answer cache

**Alternative considered:** Python in-memory dict cache in the FastAPI process

**Why rejected:** An in-memory dict is lost when the server restarts. During development, the server restarts frequently. Without cache persistence, you pay for the same LLM call every time you restart the server. Redis persists the cache across restarts. Additionally, if the system ever runs multiple FastAPI workers (horizontal scaling), an in-memory dict is not shared between workers — Redis is. Even for a personal project, using Redis from day one demonstrates production-grade thinking and adds a defensible talking point.

**Trade-off you accepted:** Redis adds one more Docker container to manage. Trivial overhead for significant benefit.

---

## Part 4 — Questions you will definitely be asked

### "Walk me through what happens when I ask 'why does the authenticate() function exist?'"

**Full answer:**

The query arrives at the FastAPI endpoint. Before anything else, the answer cache is checked — hash of `(repo_id, query)` is looked up in Redis. Cache miss, so we proceed.

The query enters the LangGraph pipeline. The QueryClassifier node runs first — it's a deterministic keyword check. The word "why" maps to the `why` query type, which maps to the `all_parallel` retrieval strategy.

The pipeline fans out to three retrieval nodes executing simultaneously. The Chroma node queries the `pr_discussions` and `adr_rfcs` collections for content semantically related to "authenticate function". The Neo4j node runs a Cypher query: it finds the Function node for `authenticate()`, traverses INTRODUCED_BY to its Commit node, PART_OF_PR to its PR node, ADDRESSES to the Issue node, returning the full provenance chain. The PostgreSQL node queries for all commits that touched the file containing `authenticate()`, sorted by timestamp, with their PR and issue linkages.

The three retrieval results fan back in and merge into the shared state. The Context Ranker node takes all retrieved chunks — maybe 40 total across all sources — and sends them to Cohere Rerank, which scores each chunk's relevance to "why does authenticate() exist?" It returns the top 10 most relevant chunks. MMR deduplication removes any near-identical chunks.

The Provenance Classifier node looks at the Neo4j result. Is there a full chain (Function→Commit→PR→Issue)? Is the commit message descriptive? Does the PR have a rich description? It classifies the code as `well_documented` — full provenance chain exists.

Since the query type is `why` but not `evolution`, the Architecture Evolution Agent is checked: does Neo4j show 5+ distinct commit chains spanning more than a year? If yes, the Evolution Agent fires and reconstructs a timeline. If no, it's skipped.

The Synthesis Agent receives the ranked context, the provenance class, and optionally the evolution timeline. It generates a structured answer: "The `authenticate()` function was introduced in PR #312 in March 2019 to implement JWT-based token validation for the API. The original issue (#178) described the need for stateless authentication to support horizontal scaling. The PR discussion shows that session-based authentication was considered and rejected because it requires shared session storage across servers [PR #312, Comment by @tiangolo]. The function has since been modified twice — once to add refresh token support [PR #891] and once to add rate limiting [PR #1204]."

The Synthesis Agent adds inline citations with GitHub URLs for every claim. The Living Docs Writer creates a draft PR with this explanation as a proposed docstring on the function.

The final answer is stored in Redis cache, keyed on the query hash. The LangSmith trace records every node's execution time, inputs, outputs, and token counts.

---

### "How do you prevent hallucinations?"

Three layers. First, retrieval grounding — every claim in the answer must be supported by retrieved context that is traceable to a specific commit, PR, or issue. The synthesis prompt instructs the model to only make claims supported by the provided context and to explicitly say "I don't have evidence for this" if the context doesn't support a claim.

Second, the RAGAS faithfulness metric. In the evaluation phase, we measure what percentage of claims in generated answers are grounded in the retrieved context. Our target is >0.85. Any answer with low faithfulness is detectable in the LangSmith traces — you can see exactly what context the LLM received and compare it to what it claimed.

Third, the human approval gate on the Living Docs Writer. Even if an answer with a hallucination slips through, it does not get indexed into the knowledge base without human review. This prevents the hallucination from becoming "evidence" that reinforces itself in future queries.

---

### "What happens when there's no provenance information?"

The Provenance Classifier handles this explicitly with seven classes. When Neo4j traversal returns a Function node with no outgoing edges — no commit link, no PR link, no issue link — the classifier assigns one of the "missing provenance" classes based on signal patterns. If the commit message is a single word ("fix") with no PR link and committed directly to main, it's classified as `hotfix`. If the file path contains "generated" or the commit message is "regenerate files", it's `generated_code`. Only if none of these patterns match, and the code has no references anywhere in the history, is it classified as `orphaned` (true dead code).

The key point is that missing provenance does not mean dead code — it means the provenance chain is incomplete, which has multiple causes. The seven-class taxonomy handles each cause differently rather than jumping to the conclusion of "dead code."

---

### "How does this scale to a large repository like Kubernetes with 100,000+ commits?"

Three engineering decisions handle scale. First, incremental sync — the system stores the last-processed commit SHA as a checkpoint in PostgreSQL. On subsequent runs, it only processes commits from HEAD to that checkpoint, not the full history. Second, the GitHub API rate limiting is handled with exponential backoff and checkpoint resumption — if we hit 5,000 calls/hour, the ingestion pauses, stores progress, and resumes when the rate limit resets. Third, the repository is processed once at ingestion time and all queries run against the already-built indexes — query time doesn't scale with repository size, only ingestion time does.

For Kubernetes specifically, the full ingestion would take several hours on first run. This is acceptable — you only do it once. Subsequent incremental syncs take minutes.

---

### "Why not just use a single LLM to answer everything?"

Because LLMs do not have access to the repository's commit history, PR discussions, or issue threads. They were trained on public data up to a cutoff date and have no knowledge of your specific codebase or its history. Even if you stuffed the entire history into the context window, that would be millions of tokens per query — prohibitively expensive and beyond current context limits for large repositories. The retrieval pipeline exists to efficiently select the specific 10-15 chunks that are relevant to the specific question, reducing the context to what matters.

More fundamentally, some questions the system answers cannot be answered by an LLM at all — "when was this function introduced?" requires querying a database with timestamp data. "What commits touched this file?" requires a SQL query. "What is the causal chain from this function back to its motivating issue?" requires graph traversal. An LLM with no tools cannot do any of these. The agents use the LLM for what it's good at (synthesizing a coherent explanation from evidence) and use deterministic tools for what they're good at (precise structured queries).

---

### "How do you evaluate whether the system is actually good?"

RAGAS gives us four quantitative metrics on a held-out test set of 50-100 labeled examples. Context Recall tells us if we're retrieving the right information. Context Precision tells us if we're retrieving too much irrelevant information. Faithfulness tells us if the LLM is staying within the bounds of the retrieved evidence. Answer Relevance tells us if the answer addresses the question.

Beyond RAGAS, we run a model benchmark comparing Groq (Llama 3.3 70B), Claude claude-sonnet-4-6, and GPT-4o on the same test set. The results are documented in `docs/model_benchmark.md`. This turns model selection from a subjective preference into a documented experiment. We also run an embedding benchmark comparing five models on retrieval recall, which justifies the choice of Voyage for code and BGE-M3 for text.

LangSmith gives us per-node traces for every query — execution time, token counts, retrieved documents, and costs. This lets us identify bottlenecks, measure improvement after tuning, and detect regressions after changes.

---

### "What would you improve if you had more time?"

Three things. First, the multi-repository knowledge graph — the `repo_id` field is on every schema specifically to support this. The next version would enable queries like "how do Kubernetes and Docker approach the same problem?" by querying across multiple indexed repositories simultaneously.

Second, the embedding benchmark was run on a held-out set of 50-100 examples manually constructed from FastAPI. A more rigorous benchmark would use a larger set across multiple repositories with different languages and domain vocabularies, to ensure the model choices generalize.

Third, the query classifier currently uses keyword rules. A more robust classifier would be a small fine-tuned model trained on real query logs from actual system usage, rather than manually written rules. As the system is used, the fallthrough rate logs provide training data for this improvement.

---

### "What is the hardest technical problem you solved in this project?"

The hardest problem was handling sparse provenance chains gracefully. In a perfect world, every commit has a linked PR, every PR has a linked issue, and every issue has a rich discussion. In reality, maybe 40% of commits in a typical repository have linked PRs, and maybe 60% of PRs have linked issues. A naive system would either crash on missing links or incorrectly label all code without a full chain as dead code.

The solution was the seven-class Provenance Classifier that reasons about the *shape* of the available evidence — which edges exist in the graph, what the commit message pattern is, what the file path suggests — rather than requiring complete chains. It classifies incomplete provenance into meaningful categories (hotfix, generated, legacy migration) and adjusts the synthesis prompt accordingly. This required careful thinking about the failure modes and designing a system that degrades gracefully rather than failing loudly.

---

## Part 5 — Technology-specific deep questions

### "What is a TypedDict in Python and why use it for LangGraph state?"

`TypedDict` is a Python type hint construct that describes a dictionary with specific key-value types. Unlike a regular dict, TypedDict fields are checked by type checkers (mypy, pyright) at development time. It is not a class — it doesn't add any runtime overhead beyond a normal dict. LangGraph uses TypedDict for state because the state object is passed between nodes as a regular Python dict under the hood — TypedDict gives you type safety and IDE autocomplete without the overhead of a Pydantic model that validates on every field access. Every node function receives the state TypedDict, reads from it, modifies a copy, and returns the updated copy. LangGraph merges the returned updates into the shared state.

---

### "What is MMR (Maximal Marginal Relevance) and why use it?"

MMR is an algorithm that balances relevance and diversity when selecting items from a set. Standard top-K selection returns the K most relevant items — but if five of those items are near-identical (e.g., five commit messages all saying "update authentication logic"), the LLM receives redundant information. MMR iteratively selects items that are both relevant to the query AND dissimilar to already-selected items. The result is a diverse set of relevant chunks rather than a cluster of near-duplicates.

In the retrieval pipeline, after Cohere Rerank produces a relevance-ranked list, MMR is applied to ensure the final top-10 chunks cover different aspects of the topic rather than repeating the same information in slightly different words.

---

### "What is a cross-encoder and how is it different from a bi-encoder?"

A bi-encoder embeds the query and document independently and computes their similarity. Fast because embeddings are precomputed. Used in the initial retrieval stage.

A cross-encoder concatenates the query and document into a single input and produces a relevance score. Slower because it cannot precompute — must run for every (query, document) pair at inference time. Significantly more accurate because the model attends to the interaction between query and document rather than their independent representations.

In your system: bi-encoder (Voyage/BGE-M3) for initial retrieval from millions of documents to 40 candidates. Cross-encoder (Cohere Rerank) for precise scoring of the 40 candidates to find the best 10.

---

### "Explain git blame and how you use it."

`git blame <file>` annotates every line of a file with the commit SHA that last modified it, the author, and the timestamp. For each function, you can find the most recent commit SHA for any line in its body. That SHA connects to a commit record which may contain issue references in its message (`Fixes #123`). The commit SHA also matches the merge commit SHA stored in the PR records fetched from the GitHub API — this is how you link commit → PR. This traversal is the foundation of the provenance chain.

The key engineering detail: `git blame` is run on the *current* version of each file. It tells you who last modified each line, not who originally wrote it. For functions that have been edited many times, the blame SHA points to the most recent modification, not the original introduction. To find when a function was *introduced* (not just last modified), you need to walk git history in reverse and find the first commit that added a line in that function's location — this is what the `INTRODUCED_BY` edge in Neo4j captures, distinct from `LAST_MODIFIED_BY`.

---

### "What is a webhook and how do you use it for the Living Docs approval flow?"

A webhook is an HTTP callback — a URL that a service calls when a specific event occurs. GitHub webhooks send a POST request to a URL you configure whenever a specified event happens (PR merged, issue closed, commit pushed, etc.).

In the Living Docs flow: when the system creates a draft PR with a generated explanation, it registers a webhook expectation — it stores the PR number and a callback URL in a database. When the human reviewer merges that PR, GitHub sends a POST request to your webhook endpoint with the event data including the PR number. The webhook handler looks up the stored expectation, finds the associated generated explanation, and triggers re-ingestion of that explanation into the Chroma knowledge base. This is how the human-approval gate triggers the knowledge base update without requiring the human to do anything beyond merging the PR.

---

## Part 6 — Questions that test genuine depth

These questions are asked by senior engineers to distinguish people who built the project from people who just read about it.

**"If two commits both modified the same function, which one does your INTRODUCED_BY edge point to?"**
INTRODUCED_BY points to the commit that first created the function. LAST_MODIFIED_BY points to the most recent commit that touched it. To find INTRODUCED_BY, you walk git log in reverse chronological order for the file and find the first commit where a function with that name appears. These are two different edges in Neo4j for a reason — "why was this introduced?" needs INTRODUCED_BY; "what was the most recent change?" needs LAST_MODIFIED_BY.

**"What happens if a PR was merged without being linked to any issue?"**
The Neo4j graph has a PR node with no outgoing ADDRESSES edge. The provenance chain traversal returns a path that terminates at the PR. The Provenance Classifier inspects the graph shape — PR exists but no Issue — and classifies based on the PR's content. If the PR description is rich (explains the motivation), the classification might still be `well_documented` because the PR itself provides sufficient context. If the PR description is sparse, it might be `hotfix` or `legacy_migration`. The synthesis prompt is adjusted accordingly.

**"Your system uses four Chroma collections. What happens at query time for a 'why' query — do you search all four simultaneously?"**
For a `why` query, the retrieval strategy is `all_parallel` — all three stores run simultaneously. Within Chroma, the retrieval node queries all four collections in parallel (or sequentially, depending on implementation) and merges results before passing to the reranker. The reranker then scores all chunks from all collections together — it doesn't know which collection a chunk came from, it only knows the query and the chunk text. This means PR discussions can outrank code chunks if they're more relevant to the specific "why" question, which is correct behavior.

**"How does your answer cache handle queries that are semantically the same but phrased differently?"**
Currently it doesn't — the cache key is a hash of the exact query string, normalized to lowercase with stripped whitespace. "Why does authenticate() exist?" and "What was the reason for writing authenticate()?" would be cache misses for each other despite being semantically equivalent. This is a known limitation. A more sophisticated cache would use the query embedding as the cache key and retrieve cache hits based on embedding similarity above a threshold. This is a genuine improvement to discuss as future work.

---

## Part 7 — One-paragraph summaries for quick revision

**What the project does in one paragraph:**
The Code Archaeology Agent is a multi-agent AI system that answers "why does this code exist?" for any function or file in a GitHub repository. It ingests the repository's git history, pull requests, issues, and architectural decision records, then builds a three-store architecture — PostgreSQL for structured metadata, Neo4j for the provenance graph, and Chroma for semantic search. At query time, a LangGraph pipeline classifies the query, runs parallel retrieval across all three stores, reranks the results using a cross-encoder, classifies the code's provenance type, optionally reconstructs an architectural evolution timeline, and synthesizes a cited answer grounded in the actual historical record of the repository.

**Why LangGraph in one paragraph:**
LangGraph is used because the query-time pipeline requires capabilities that LangChain chains do not provide: shared typed state across all nodes, conditional routing (different retrieval strategies based on query type), parallel fan-out to three retrieval nodes simultaneously, a conditional branch for the evolution agent, and a human-in-the-loop pause before the knowledge base is updated. LangGraph models these as a directed state graph with explicit edges and a TypedDict state object that every node reads and updates.

**Why three databases in one paragraph:**
Each database answers a fundamentally different type of question. PostgreSQL answers "when and who" — structured queries with timestamp sorting and relational joins across commits, PRs, and issues. Neo4j answers "caused by" — multi-hop graph traversal from a function back through its commit history to the issue and discussion that motivated it. Chroma answers "semantically similar to" — embedding-space retrieval that finds related content even when the vocabulary differs. No single database efficiently handles all three question types, and combining them gives the system capabilities that are impossible with any single store.