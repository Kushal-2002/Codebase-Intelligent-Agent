"""
Index a C/C++ or Python repository: function chunks into ChromaDB, plus the
call graph.

This is the logic of ingest.py and build_call_graph.py as reusable functions,
so the web app can index any cloned repository into its own folder.

Node ids look like "<repo folder name>/<path inside repo>::<function name>",
e.g. "repository/client.cpp::connect_to_addr" for the bundled sample, or
"requests/src/requests/sessions.py::Session.request" for a Python method.
"""

import os
import pickle
import re

import chromadb
import networkx as nx
import tree_sitter_cpp as tscpp
import tree_sitter_python as tspython
from tree_sitter import Language, Parser

CPP_EXTENSIONS = (".c", ".cpp", ".cc", ".cxx", ".h", ".hpp")
PYTHON_EXTENSIONS = (".py",)
SOURCE_EXTENSIONS = CPP_EXTENSIONS + PYTHON_EXTENSIONS

SKIP_DIRS = {
    ".git",
    "node_modules",
    "build",
    "dist",
    "cmake-build-debug",
    "cmake-build-release",
    "__pycache__",
    ".venv",
    "venv",
    "env",
    ".tox",
    "site-packages",
}

# files above this are usually generated or vendored (amalgamations, tables)
MAX_FILE_BYTES = 1_000_000

COLLECTION_NAME = "code_chunks"

PARSERS = {
    "cpp": Parser(Language(tscpp.language())),
    "python": Parser(Language(tspython.language())),
}

_embedding_model = None


def get_embedding_model():

    global _embedding_model

    if _embedding_model is None:
        from sentence_transformers import SentenceTransformer

        _embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

    return _embedding_model


def language_of(file_path):

    return "python" if file_path.endswith(PYTHON_EXTENSIONS) else "cpp"


# ---------------- files ----------------


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
        return PARSERS[language_of(file_path)].parse(bytes(f.read(), "utf8")).root_node


def match_key(name):

    # what a call site can be matched on: "Foo::bar", "obj.bar", "p->bar<T>" -> "bar"

    return re.split(r"\.|->|::", name.split("<")[0])[-1].strip()


# ---------------- C / C++ ----------------


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


def iter_cpp_functions(node):

    if node.type == "function_definition":
        declarator = node.child_by_field_name("declarator")

        if declarator:
            short_name = function_name(declarator)

            if short_name:
                yield declarator.text.decode("utf8"), short_name, node

    for child in node.children:
        yield from iter_cpp_functions(child)


def find_cpp_calls(node):

    calls = []

    def dfs(n):

        if n.type == "call_expression":
            function_node = n.child_by_field_name("function")

            if function_node:
                key = match_key(function_node.text.decode("utf8").split("(")[0])
                calls.append((key, "direct"))

        for child in n.children:
            dfs(child)

    dfs(node)

    return calls


# ---------------- Python ----------------


def iter_python_functions(node, classes=()):

    # methods are named "Class.method" so same-named methods in different
    # classes stay separate nodes

    if node.type == "class_definition":
        name = node.child_by_field_name("name")

        if name:
            classes = classes + (name.text.decode("utf8"),)

    elif node.type == "function_definition":
        name = node.child_by_field_name("name")

        if name:
            short_name = ".".join(classes + (name.text.decode("utf8"),))
            params = node.child_by_field_name("parameters")
            params_text = params.text.decode("utf8") if params else "()"

            yield short_name + params_text, short_name, node

    for child in node.children:
        yield from iter_python_functions(child, classes)


def find_python_calls(node):

    # "name":      foo()                -> module-level functions only (a bare
    #                                        call can't reach a method; it is
    #                                        often a builtin like next())
    # "self":      self.foo(), cls.foo() -> methods
    # "attribute": obj.foo()            -> methods, only when foo is unambiguous
    #                                        in the repo, since obj could be a
    #                                        dict, a library object, ...

    calls = []

    def dfs(n):

        if n.type == "call":
            function_node = n.child_by_field_name("function")

            if function_node is not None:
                if function_node.type == "identifier":
                    calls.append((function_node.text.decode("utf8"), "name"))

                elif function_node.type == "attribute":
                    receiver = function_node.child_by_field_name("object")
                    attribute = function_node.child_by_field_name("attribute")

                    if attribute is not None:
                        kind = (
                            "self"
                            if receiver is not None and receiver.text in (b"self", b"cls")
                            else "attribute"
                        )
                        calls.append((attribute.text.decode("utf8"), kind))

        for child in n.children:
            dfs(child)

    dfs(node)

    return calls


def iter_functions(root, language):

    # (full signature text, short name, function node) for every definition

    if language == "python":
        return iter_python_functions(root)

    return iter_cpp_functions(root)


def find_calls(node, language):

    if language == "python":
        return find_python_calls(node)

    return find_cpp_calls(node)


# ---------------- indexing ----------------


# obj.<name>() with these names is almost always a dict/list/str/set/file
# method, so it is never linked to a repo method of the same name
BUILTIN_METHOD_NAMES = set(
    dir(dict) + dir(list) + dir(str) + dir(set) + dir(bytes)
    + ["read", "write", "close", "readline", "readlines", "flush", "seek"]
)


def is_method(node_id):

    # Python methods are "Class.method"; C++ names never contain "."
    return "." in node_id.split("::", 1)[1]


def python_candidates(nodes, kind):

    # which definitions a call of this kind may resolve to (see find_python_calls);
    # C/C++ calls ("direct") keep every candidate

    if kind == "name":
        return [n for n in nodes if not is_method(n)]

    if kind == "self":
        return [n for n in nodes if is_method(n)]

    if kind == "attribute":
        methods = [n for n in nodes if is_method(n)]
        name = nodes[0].rsplit(".", 1)[-1] if nodes else ""
        return methods if len(methods) == 1 and name not in BUILTIN_METHOD_NAMES else []

    return nodes


def index_repository(repo_dir, chroma_path, data_dir, progress=None):

    # progress(message) is called with short status updates for the UI

    def report(message):
        if progress:
            progress(message)

    os.makedirs(data_dir, exist_ok=True)

    report("Parsing source files")

    parsed = [
        (node_path, parse_file(file_path), language_of(file_path))
        for node_path, file_path in iter_source_files(repo_dir)
    ]

    function_lookup = {}
    chunks = []
    function_to_nodes = {}

    # pass 1: every function defined in the repo, so calls into libraries
    # (printf, socket, ...) can be told apart from repo functions
    for node_path, root, language in parsed:
        for full_name, short_name, fn_node in iter_functions(root, language):
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

    for node_path, root, language in parsed:
        for _, short_name, fn_node in iter_functions(root, language):
            targets = set()

            for call, kind in find_calls(fn_node, language):
                candidates = python_candidates(function_to_nodes.get(call, []), kind)

                if not candidates:
                    continue

                local = [c for c in candidates if c.startswith(f"{node_path}::")]

                # prefer a definition in the same file when names collide
                targets.update(local or candidates)

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
