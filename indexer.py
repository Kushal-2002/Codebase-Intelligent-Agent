"""
Index a C/C++ repository: function chunks into ChromaDB, plus the call graph.

This is the logic of ingest.py and build_call_graph.py as reusable functions,
so the web app can index any cloned repository into its own folder.

Node ids look like "<repo folder name>/<path inside repo>::<function name>",
e.g. "repository/client.cpp::connect_to_addr" for the bundled sample.
"""

import os
import pickle
import re

import chromadb
import networkx as nx
import tree_sitter_cpp as tscpp
from tree_sitter import Language, Parser

SOURCE_EXTENSIONS = (".c", ".cpp", ".cc", ".cxx", ".h", ".hpp")

SKIP_DIRS = {".git", "node_modules", "build", "cmake-build-debug", "cmake-build-release"}

# files above this are usually generated or vendored (amalgamations, tables)
MAX_FILE_BYTES = 1_000_000

COLLECTION_NAME = "code_chunks"

parser = Parser(Language(tscpp.language()))

_embedding_model = None


def get_embedding_model():

    global _embedding_model

    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    return _embedding_model


# ---------------- parsing ----------------


def iter_source_files(repo_dir):

    # yields (node path, file path); node path keeps the repo folder name
    # as prefix so ids match the original "repository/..." layout

    prefix = os.path.basename(os.path.normpath(repo_dir))

    for root, dirs, files in os.walk(repo_dir):
        dirs[:] = sorted(d for d in dirs if d not in SKIP_DIRS)

        for file in sorted(files):
            if not file.endswith(SOURCE_EXTENSIONS):
                continue

            file_path = os.path.join(root, file)

            if os.path.getsize(file_path) > MAX_FILE_BYTES:
                continue

            rel = os.path.relpath(file_path, repo_dir)

            yield os.path.join(prefix, rel), file_path


def parse_file(file_path):

    with open(file_path, "r", errors="ignore") as f:
        return parser.parse(bytes(f.read(), "utf8")).root_node


def function_name(declarator):

    # the name inside a declarator, read from the syntax tree rather than by
    # splitting text: handles "foo (int x)", "*foo(...)", "&Foo::bar(...)"

    node = declarator

    while node is not None and node.type != "function_declarator":
        node = node.child_by_field_name("declarator")

    if node is None:
        text = declarator.text.decode("utf8").split("(")[0]
    else:
        name = node.child_by_field_name("declarator")
        text = name.text.decode("utf8") if name else ""

    # unknown macros (LUA_API, EXPORT, ...) can make the parser glue the return
    # type onto the name, e.g. "lua_CFunction lua_atpanic"; names have no spaces
    parts = text.split()

    return parts[-1] if parts else ""


def match_key(name):

    # what a call site can be matched on: "Foo::bar", "obj.bar", "p->bar<T>" -> "bar"

    return re.split(r"\.|->|::", name.split("<")[0])[-1].strip()


def iter_functions(node):

    # (full declarator text, short name, function node) for every definition

    if node.type == "function_definition":
        declarator = node.child_by_field_name("declarator")

        if declarator:
            short_name = function_name(declarator)

            if short_name:
                yield declarator.text.decode("utf8"), short_name, node

    for child in node.children:
        yield from iter_functions(child)


def find_calls(node):

    calls = []

    def dfs(n):

        if n.type == "call_expression":
            function_node = n.child_by_field_name("function")

            if function_node:
                calls.append(match_key(function_node.text.decode("utf8").split("(")[0]))

        for child in n.children:
            dfs(child)

    dfs(node)

    return calls


# ---------------- indexing ----------------


def index_repository(repo_dir, chroma_path, data_dir, progress=None):

    # progress(message) is called with short status updates for the UI

    def report(message):
        if progress:
            progress(message)

    os.makedirs(data_dir, exist_ok=True)

    report("Parsing source files")

    parsed = [(node_path, parse_file(file_path)) for node_path, file_path in iter_source_files(repo_dir)]

    function_lookup = {}
    chunks = []
    function_to_nodes = {}

    # pass 1: every function defined in the repo, so calls into libraries
    # (printf, socket, ...) can be told apart from repo functions
    for node_path, root in parsed:
        for full_name, short_name, fn_node in iter_functions(root):
            node_id = f"{node_path}::{short_name}"
            function_text = fn_node.text.decode("utf8")

            function_lookup[node_id] = function_text

            key = match_key(short_name)
            function_to_nodes.setdefault(key, [])

            if node_id not in function_to_nodes[key]:
                function_to_nodes[key].append(node_id)

            chunks.append(
                {
                    "id": f"{node_path}_{full_name}",
                    "text": function_text,
                    "embedding_text": full_name + "\n\n" + function_text,
                    "metadata": {
                        "path": node_path,
                        "function_name": full_name,
                        "node": node_id,
                        "type": "function",
                    },
                }
            )

    # pass 2: call edges, preferring a same-file definition when names collide
    report(f"Building call graph for {len(function_lookup)} functions")

    call_graph = {}

    for node_path, root in parsed:
        for _, short_name, fn_node in iter_functions(root):
            targets = set()

            for call in find_calls(fn_node):
                if call not in function_to_nodes:
                    continue

                local_node = f"{node_path}::{call}"

                if local_node in function_to_nodes[call]:
                    targets.add(local_node)
                else:
                    targets.update(function_to_nodes[call])

            call_graph[f"{node_path}::{short_name}"] = list(targets)

    G = nx.DiGraph()

    for func, calls in call_graph.items():
        G.add_node(func)

        for call in calls:
            G.add_edge(func, call)

    # embeddings: one batched encode instead of one call per function
    report(f"Embedding {len(chunks)} functions")

    client = chromadb.PersistentClient(path=chroma_path)

    if COLLECTION_NAME in [c.name for c in client.list_collections()]:
        client.delete_collection(name=COLLECTION_NAME)

    collection = client.get_or_create_collection(name=COLLECTION_NAME)

    # duplicate ids (e.g. overloads with identical declarators) would make
    # chroma reject the batch; keep the last one like the old upsert loop did
    unique = list({chunk["id"]: chunk for chunk in chunks}.values())

    if unique:
        embeddings = get_embedding_model().encode(
            [c["embedding_text"] for c in unique], batch_size=64, show_progress_bar=False
        )

        batch = 2000

        for start in range(0, len(unique), batch):
            part = unique[start : start + batch]

            collection.upsert(
                ids=[c["id"] for c in part],
                embeddings=[e.tolist() for e in embeddings[start : start + batch]],
                documents=[c["text"] for c in part],
                metadatas=[c["metadata"] for c in part],
            )

    with open(os.path.join(data_dir, "function_lookup.pkl"), "wb") as f:
        pickle.dump(function_lookup, f)

    with open(os.path.join(data_dir, "call_graph.pkl"), "wb") as f:
        pickle.dump(call_graph, f)

    with open(os.path.join(data_dir, "graph.pkl"), "wb") as f:
        pickle.dump(G, f)

    stats = {
        "files": len(parsed),
        "functions": len(function_lookup),
        "call_edges": G.number_of_edges(),
    }

    report(f"Indexed {stats['files']} files, {stats['functions']} functions, {stats['call_edges']} call edges")

    return stats
