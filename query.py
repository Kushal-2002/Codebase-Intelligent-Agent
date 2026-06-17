from ollama import chat
from sentence_transformers import SentenceTransformer
import pickle
import networkx as nx

import chromadb

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(name="code_chunks")

with open("call_graph.pkl", "rb") as f:
    call_graph = pickle.load(f)

with open("function_lookup.pkl", "rb") as f:
    function_lookup = pickle.load(f)

with open("graph.pkl", "rb") as f:
    G = pickle.load(f)


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


question = input("Ask a question: ")

results = search_code(question)

relevant_text = ""

for i in range(len(results["documents"][0])):
    chunk_text = results["documents"][0][i]

    function_name = results["metadatas"][0][i]["function_name"]

    matched_nodes = find_graph_nodes(function_name)

    print("\nMATCHED NODES:")
    print(matched_nodes)

    print("\nMATCHED FUNCTION:")
    print(function_name)

    question_lower = question.lower()
    processed_nodes = set()

    for matched_node in matched_nodes:
        if matched_node in processed_nodes:
            continue

        processed_nodes.add(matched_node)

        print("\nGRAPH NODE:")
        print(matched_node)

        if "who calls" in question_lower or "where is" in question_lower:
            related_nodes = get_callers(matched_node)
            print("\nCALLERS:")
        else:
            related_nodes = get_dependencies(matched_node)
            print("\nDEPENDENCIES:")

        for related_node in related_nodes:
            print("   ->", related_node)

            source = get_source(related_node)

            if source:
                relevant_text += "\nRELATED FUNCTION:\n"
                relevant_text += source
                relevant_text += "\n"
print("\nCONTEXT LENGTH =", len(relevant_text))


response = chat(
    model="llama3",
    messages=[
        {
            "role": "user",
            "content": f"""
            You are a codebase assistant.

            Answer ONLY using the repository context.

            If the answer cannot be found in the context, say:
            "I could not find this information in the repository."

            Repository Context:
            {relevant_text}

            Question:
            {question}
            """,
        }
    ],
)

print(response.message.content)
