import os
from ollama import chat
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import chromadb

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(name="code_chunks")


question = input("Ask a question: ")


question_embedding = embedding_model.encode(question)

results = collection.query(query_embeddings=[question_embedding.tolist()], n_results=3)

relevant_text = ""

for i in range(len(results["documents"][0])):
    chunk_text = results["documents"][0][i]

    path = results["metadatas"][0][i]["path"]

    chunk_id = results["metadatas"][0][i]["chunk_id"]
    print(f"Retrieved: {path} | Chunk: {chunk_id}")

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
