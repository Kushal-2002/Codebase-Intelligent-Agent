# Codebase Intelligence Agent

An **Agentic AI system** for understanding large code repositories using **Tree-sitter**, **Vector Search (ChromaDB)**, **GraphRAG**, and **LLM-powered planning**.

Instead of relying only on semantic search, the agent reasons over a repository's call graph, executes tools, gathers observations, and generates answers grounded in the source code.

---

# Features

- Function-level code indexing using Tree-sitter
- Semantic code search using Sentence Transformers
- ChromaDB vector database
- Repository call graph generation
- GraphRAG using function relationships
- LLM-powered planner
- Tool execution framework
- Multi-step agent loop
- Context-grounded answer generation

---

# Architecture

```
                    User Question
                           │
                           ▼
                    Planner (LLM)
                           │
        ┌──────────────────┼──────────────────┐
        ▼                  ▼                  ▼
 search_code()      get_callers()      get_dependencies()
        │                  │                  │
        └──────────────────┼──────────────────┘
                           │
                           ▼
                  Repository Observations
                           │
                           ▼
                    Planner (LLM)
                 Continue / Finish?
                           │
                           ▼
                 Final Answer Generator
                           │
                           ▼
                      Final Response
```

---

# Tech Stack

- Python
- Tree-sitter
- Sentence Transformers
- ChromaDB
- NetworkX
- LangGraph + LangChain (tool-calling agent)
- Ollama (Llama 3.1, local) or Google Gemini API

---

# Repository Indexing

## 1. Parse Repository

The repository is parsed using Tree-sitter.

Supported:

- Function definitions
- Function calls

---

## 2. Build Function Chunks

Each function becomes an independent retrieval unit.

Example

```cpp
connect_any_tracker(...)
```

---

## 3. Generate Embeddings

Embeddings are generated using

```
all-MiniLM-L6-v2
```

---

## 4. Store in ChromaDB

Each chunk stores

- Function name
- Source code
- Metadata

---

## 5. Build Call Graph

Tree-sitter extracts every function invocation.

Example

```
main
   │
   ▼
connect_any_tracker
   │
   ▼
connect_to_addr
```

The graph is stored using NetworkX.

---

# GraphRAG

Instead of retrieving only similar code,

the system expands retrieval using the call graph.

Supported operations

- Find callers
- Find dependencies

This produces repository-aware context instead of isolated code snippets.

---

# Agent Workflow

The project follows an Agentic AI architecture.

## Planner

The planner receives the user's question.

It decides which tool should be executed.

Example

```
Question

↓

search_code(connect_any_tracker)
```

---

## Available Tools

### search_code(function)

Semantic retrieval from ChromaDB.

---

### get_callers(function)

Returns every function that calls the target function.

---

### get_dependencies(function)

Returns every function invoked by the target function.

---

# Tool Execution

Each selected tool executes independently.

Example

```
search_code(connect_any_tracker)

↓

Retrieve source code

↓

Return observation
```

---

# Observation Loop

After executing a tool,

the observation is appended to the conversation.

Example

```
Question

↓

Planner

↓

Tool

↓

Observation

↓

Planner

↓

Tool

↓

Observation

↓

Planner

↓

Finish
```

This allows iterative reasoning instead of one-shot retrieval.

---

# Final Answer Generation

Once the planner decides enough information has been collected,

the accumulated observations are passed to another LLM prompt.

The answer generator produces the final response using only repository context.

---

# Example

Question

```
How does connect_any_tracker work?
```

Planner

```
search_code(connect_any_tracker)
```

Observation

```
Function Source Code
```

Planner

```
finish
```

Answer

```
connect_any_tracker iterates through the list of trackers,
attempts to establish a connection using connect_to_addr,
and returns the first successful socket connection.
```

---

# Web Interface

Paste any public GitHub link and ask questions about that codebase:

```bash
streamlit run app.py
```

1. Paste a link such as `https://github.com/antirez/kilo` in the sidebar and click **Load repository**.
   The repo is shallow-cloned, parsed with Tree-sitter, embedded and turned into a call graph
   (Lua, 68 files / 1,300 functions, takes about 15 seconds).
2. Ask questions in the chat. Each answer shows the agent's tool calls and the functions it used.
3. Previously indexed repositories stay available in the dropdown; tick
   **Re-clone and re-index** to pull the latest code.

Each repository gets its own folder in `workspaces/` (git-ignored) with its clone,
vector index and call graph. The cloned code is only parsed, never built or run.
Only C and C++ repositories are supported for now.

---

# LangGraph Agent

`agent_langgraph.py` rebuilds the agent loop as a LangGraph `StateGraph`:

```
seed ──> planner ──(tool call)──> tools ──┐
            ^                             │
            └─────────────────────────────┘
         planner ──(done / step limit)──> answer ──> END
```

- **seed**: semantic search, so the planner sees which functions exist
- **planner**: LLM with the three tools bound via native tool calling (no regex parsing)
- **tools**: LangGraph `ToolNode` running `search_code`, `get_callers`, `get_dependencies`
- **answer**: grounded answer from the collected observations

The tools and the answer prompt are shared with `agent.py`, so the benchmark
compares orchestration only. Print the graph with `python agent_langgraph.py --diagram`.

---

# Choosing the LLM

All LLM calls go through `llm.py`:

```bash
# local (default)
ollama pull llama3.1
python agent_langgraph.py

# Google Gemini
cp .env.example .env        # then put your GOOGLE_API_KEY in .env
LLM_PROVIDER=gemini python agent_langgraph.py
```

`LLM_MODEL` overrides the model name. The original `llama3` does not support
tool calling in Ollama, so the default is `llama3.1`.

---

# Evaluation

```bash
python eval/run_eval.py                          # vector, graph, agent, langgraph
python eval/run_eval.py --systems agent,langgraph
LLM_PROVIDER=gemini python eval/run_eval.py
python eval/run_eval.py --retrieval-only         # no LLM needed
```

Results are saved per question in `eval/results/`.

---

# Project Structure

```
Codebase-Intelligence-Agent/
│
├── agent.py                 # Hand-written agent loop
├── agent_langgraph.py       # Same agent as a LangGraph state machine
├── planner.py               # Planner LLM (hand-written agent)
├── llm.py                   # Ollama / Gemini switch used everywhere
├── tools.py                 # Tool implementations (switchable index)
├── app.py                   # Streamlit web interface
├── workspace.py             # Clone a GitHub repo into workspaces/ and index it
├── indexer.py               # Tree-sitter parsing, embeddings, call graph
│
├── ingest.py                # Index the bundled sample repo
├── build_call_graph.py      # Same as ingest.py (kept for old instructions)
├── eval/                    # Benchmark questions, runner, results
│
├── chroma_db/
│
├── data/
│   ├── graph.pkl
│   ├── call_graph.pkl
│   └── function_lookup.pkl
│
├── repository/
│
└── README.md
```

---

# Current Capabilities

- Semantic code search
- Function-level retrieval
- Call graph traversal
- GraphRAG
- Tool-based reasoning
- Agent loop
- Repository-grounded answer generation

---

# Planned Improvements

- Recursive GraphRAG (multi-hop traversal)
- AST-aware retrieval
- GPT-5 planner and answer generator
- Repository summarization
- Cross-file reasoning
- Code editing agent
- Support for multiple programming languages

---

# Workflow Summary

```
Repository

↓

Tree-sitter

↓

Function Chunks

↓

Embeddings

↓

ChromaDB

↓

User Question

↓

Planner

↓

Tool Selection

↓

GraphRAG

↓

Observation

↓

Planner

↓

Finish

↓

Answer LLM

↓

Final Response
```

---

# Future Vision

The long-term goal is to build an AI software engineering assistant capable of

- Understanding unfamiliar codebases
- Explaining complex execution flows
- Navigating dependencies
- Assisting in debugging
- Suggesting code modifications
- Generating documentation
- Answering repository-specific questions through autonomous tool usage