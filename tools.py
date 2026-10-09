from sentence_transformers import SentenceTransformer
import chromadb
import pickle
import networkx as nx
from planner import decide_action

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(name="code_chunks")

with open("data/call_graph.pkl", "rb") as f:
    call_graph = pickle.load(f)

with open("data/function_lookup.pkl", "rb") as f:
    function_lookup = pickle.load(f)

with open("data/graph.pkl", "rb") as f:
    G = pickle.load(f)


def build_context(nodes):

    context = ""

    for node in nodes:
        source = get_source(node)

        if source:
            context += "\nRELATED FUNCTION:\n"
            context += source
            context += "\n"

    return context


def search_code(question):

    question_embedding = embedding_model.encode(question)

    results = collection.query(
        query_embeddings=[question_embedding.tolist()], n_results=3
    )

    return results


def find_graph_nodes(function_name):

    short_name = function_name.split("(")[0]

    matches = []

    for node in call_graph:
        if node.endswith(f"::{short_name}"):
            matches.append(node)

    return matches


def get_dependencies(node):

    if node is None:
        return []

    return list(G.successors(node))


def get_callers(node):

    if node is None:
        return []

    return list(G.predecessors(node))


def get_source(node):

    return function_lookup.get(node, "")


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
