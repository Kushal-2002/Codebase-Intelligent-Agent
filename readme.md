# Codebase Intelligence Agent

A lightweight Codebase Intelligence Agent that combines **Tree-Sitter**, **Vector Search (ChromaDB)**, **LLM-based Question Answering**, and **Call Graph Analysis** to understand and answer questions about large code repositories.

## Overview

This project indexes source code repositories by extracting function definitions, generating embeddings, and storing them in a vector database. Users can then ask natural language questions about the codebase and receive answers grounded in the repository source code.

To improve retrieval quality, the system also builds a function-level call graph and performs graph-based context expansion (Graph-RAG), allowing related functions and dependencies to be included in the retrieved context.

---

## Features

### Function Extraction

* Parses source code using Tree-Sitter.
* Extracts all function definitions.
* Captures:

  * Function name
  * Source code
  * File path

### Semantic Code Search

* Uses SentenceTransformers (`all-MiniLM-L6-v2`) for embeddings.
* Stores embeddings in ChromaDB.
* Retrieves semantically relevant functions for natural language queries.

### LLM-Powered Code Understanding

* Uses Ollama + Llama 3.
* Answers questions using retrieved repository context.
* Restricts responses to repository information.

### Call Graph Construction

Builds a repository-wide call graph by identifying:

```text
Function A
   ↓
Function B
```

relationships.

Example:

```text
main
 └── connect_any_tracker
          └── connect_to_addr
```

### Cross-File Dependency Tracking

Functions are uniquely identified using:

```text
<file_path>::<function_name>
```

Example:

```text
repository/client.cpp::main
repository/server.cpp::main
```

This eliminates ambiguity when multiple files contain functions with identical names.

### Graph-RAG

The retrieval pipeline is extended using dependency information.

Instead of retrieving only the matched function:

```text
Question
   ↓
Vector Search
   ↓
Matched Function
```

the system performs:

```text
Question
   ↓
Vector Search
   ↓
Matched Function
   ↓
Call Graph Expansion
   ↓
Dependency Functions
   ↓
LLM
```

This provides richer repository context and improves reasoning about implementation details.

---

## Project Architecture

```text
Repository
    │
    ▼
Tree-Sitter Parsing
    │
    ▼
Function Extraction
    │
    ▼
Embedding Generation
    │
    ▼
ChromaDB Storage
    │
    ▼
Semantic Retrieval
    │
    ▼
Call Graph Expansion
    │
    ▼
Context Construction
    │
    ▼
Ollama (Llama 3)
    │
    ▼
Answer Generation
```

---

## Components

### ingest.py

Responsible for:

* Parsing repository files
* Extracting function definitions
* Generating embeddings
* Storing functions in ChromaDB
* Building function lookup tables

Outputs:

```text
chroma_db/
function_lookup.pkl
```

---

### build_call_graph.py

Responsible for:

* Detecting function calls
* Building repository-wide dependency graphs
* Creating unique function nodes
* Saving graph data

Outputs:

```text
call_graph.pkl
```

---

### query.py

Responsible for:

* Accepting user questions
* Retrieving relevant functions from ChromaDB
* Expanding context using call graph dependencies
* Sending context to the LLM
* Returning repository-grounded answers

---

## Technologies Used

* Python
* Tree-Sitter
* Tree-Sitter C++
* ChromaDB
* SentenceTransformers
* Ollama
* Llama 3
* NetworkX
* Pickle

---

## Current Capabilities

✅ Function-level indexing

✅ Semantic repository search

✅ Repository-grounded question answering

✅ Function call graph generation

✅ Cross-file dependency analysis

✅ Graph-enhanced retrieval (Graph-RAG)

✅ Dependency-aware context expansion

---

## Future Enhancements

* Multi-hop graph traversal
* Cross-language support
* Class and method dependency graphs
* Import dependency analysis
* Frontend ↔ Backend dependency tracking
* API flow tracing
* Database interaction mapping
* Query classification and routing
* Interactive dependency visualization
* Agentic code exploration workflows

---

## Example Query

```text
What does connect_any_tracker do?
```

Retrieval Flow:

```text
connect_any_tracker
       ↓
connect_to_addr
```

The system retrieves both functions and uses them as context before generating an answer.

---

## Status

Current Version: Graph-RAG Prototype

The project successfully combines semantic retrieval with static dependency analysis, providing a foundation for an advanced repository understanding and code intelligence system.
