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
- Ollama
- Llama 3

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

# Project Structure

```
Codebase-Intelligence-Agent/
│
├── agent.py                 # Agent loop
├── planner.py               # Planner LLM
├── tools.py                 # Tool implementations
│
├── build_embeddings.py
├── build_call_graph.py
├── build_lookup.py
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
- LangGraph integration
- GPT-5 planner and answer generator
- Repository summarization
- Cross-file reasoning
- Code editing agent
- Streamlit interface
- Evaluation benchmark
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