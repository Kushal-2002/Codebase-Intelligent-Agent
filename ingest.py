import os
from ollama import chat
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import chromadb

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.create_collection(name="code_chunks")


for root, dirs, files in os.walk("repository"):
    if ".git" in root:
        continue

    for file in files:
        path = os.path.join(root, file)

        with open(path, "r", errors="ignore") as f:
            content = f.read()

            chunk_size = 1000

            for i in range(0, len(content), chunk_size):
                chunk_text = content[i : i + chunk_size]

                embedding = embedding_model.encode(chunk_text)

                collection.add(
                    ids=[f"{path}_{i}"],
                    embeddings=[embedding.tolist()],
                    documents=[chunk_text],
                    metadatas=[{"path": path, "chunk_id": i}],
                )

print("Ingestion complete")
