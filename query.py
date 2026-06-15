import os
from ollama import chat
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import pickle


import chromadb

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(name="code_chunks")

with open("call_graph.pkl", "rb") as f:
    call_graph = pickle.load(f)

with open("function_lookup.pkl", "rb") as f:
    function_lookup = pickle.load(f)

question = input("Ask a question: ")


question_embedding = embedding_model.encode(question)

results = collection.query(query_embeddings=[question_embedding.tolist()], n_results=1)

relevant_text = ""

for i in range(len(results["documents"][0])):
    chunk_text = results["documents"][0][i]

    path = results["metadatas"][0][i]["path"]

    function_name = results["metadatas"][0][i]["function_name"]
    short_name = function_name.split("(")[0]

    neighbors = []
    print("\nGRAPH NEIGHBORS:\n")

    for node in call_graph:
        if node.endswith(f"::{short_name}"):
            print("MATCHED NODE:", node)

            for neighbor in call_graph[node]:
                print("   ->", neighbor)
                neighbors.append(neighbor)

    print("\nDEPENDENCIES FOUND:")

    neighbors = list(set(neighbors))
    neighbors = list(set(neighbors))

    for neighbor in neighbors:
        print(neighbor)

        if neighbor in function_lookup:
            print("\nADDING DEPENDENCY SOURCE:")
            print(neighbor)
            relevant_text += "\nDEPENDENCY:\n"

            relevant_text += function_lookup[neighbor]

            relevant_text += "\n"

    function_name = results["metadatas"][0][i]["function_name"]

    # print("\nSOURCE CODE:")
    # print(chunk_text)

    print("-" * 80)

    relevant_text += f"\nFILE: {path}\n"

    relevant_text += chunk_text
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
