import os
import pickle

import chromadb

from indexer import COLLECTION_NAME, get_embedding_model
from planner import decide_action

# the bundled sample repository; the web app switches to other repos
# with load_index()
DEFAULT_CHROMA_PATH = "./chroma_db"
DEFAULT_DATA_DIR = "data"

_index = None


class Index:

    def __init__(self, chroma_path, data_dir):

        self.key = (os.path.abspath(chroma_path), os.path.abspath(data_dir))

        client = chromadb.PersistentClient(path=chroma_path)
        self.collection = client.get_or_create_collection(name=COLLECTION_NAME)

        with open(os.path.join(data_dir, "call_graph.pkl"), "rb") as f:
            self.call_graph = pickle.load(f)

        with open(os.path.join(data_dir, "function_lookup.pkl"), "rb") as f:
            self.function_lookup = pickle.load(f)

        with open(os.path.join(data_dir, "graph.pkl"), "rb") as f:
            self.G = pickle.load(f)


def load_index(chroma_path=DEFAULT_CHROMA_PATH, data_dir=DEFAULT_DATA_DIR):

    # make every tool below work on this index (no-op if already active)

    global _index

    key = (os.path.abspath(chroma_path), os.path.abspath(data_dir))

    if _index is None or _index.key != key:
        _index = Index(chroma_path, data_dir)

    return _index


def reset_index():

    # forget the open index, e.g. after its folder was deleted and rebuilt;
    # chroma also caches one client per path, which would point at the old files

    global _index

    from chromadb.api.client import SharedSystemClient

    _index = None
    SharedSystemClient.clear_system_cache()


def active_index():

    return _index if _index is not None else load_index()


def build_context(nodes):

    context = ""

    for node in nodes:
        source = get_source(node)

        if source:
            context += "\nRELATED FUNCTION:\n"
            context += source
            context += "\n"

    return context


def search_code(question, n_results=3):

    question_embedding = get_embedding_model().encode(question)

    results = active_index().collection.query(
        query_embeddings=[question_embedding.tolist()], n_results=n_results
    )

    return results


def find_graph_nodes(function_name):

    short_name = function_name.split("(")[0]

    matches = []

    for node in active_index().call_graph:
        if node.endswith(f"::{short_name}"):
            matches.append(node)

    return matches


def get_dependencies(node):

    G = active_index().G

    # functions with no call edges were never added as graph nodes in older indexes
    if node is None or node not in G:
        return []

    return list(G.successors(node))


def get_callers(node):

    G = active_index().G

    if node is None or node not in G:
        return []

    return list(G.predecessors(node))


def get_source(node):

    return active_index().function_lookup.get(node, "")


def retrieve_context(question):

    tool_call = decide_action(question)

    print(tool_call)

    if tool_call.startswith("search_code("):
        function_name = tool_call[len("search_code(") : -1]
        action = "search"

    elif tool_call.startswith("get_callers("):
        function_name = tool_call[len("get_callers(") : -1]

        action = "callers"

    elif tool_call.startswith("get_dependencies("):
        function_name = tool_call[len("get_dependencies(") : -1]

        action = "dependencies"

    else:
        results = search_code(question)

        function_name = results["metadatas"][0][0]["function_name"]
        function_name = function_name.split("(")[0]

        action = "search"

    relevant_text = ""

    processed_nodes = set()

    matched_nodes = find_graph_nodes(function_name)

    for matched_node in matched_nodes:
        if matched_node in processed_nodes:
            continue

        processed_nodes.add(matched_node)

        if action == "callers":
            related_nodes = get_callers(matched_node)

        elif action == "dependencies":
            related_nodes = get_dependencies(matched_node)

        else:
            related_nodes = []

        relevant_text += "\nFUNCTION:\n"

        source = get_source(matched_node)

        if source:
            relevant_text += source

        relevant_text += build_context(related_nodes[:5])

    return relevant_text
