import os
from ollama import chat
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import chromadb
from tree_sitter import Language, Parser
import tree_sitter_cpp as tscpp
import pickle


embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
client = chromadb.PersistentClient(path="./chroma_db")

collection = client.get_or_create_collection(name="code_chunks")

function_lookup = {}

CPP_LANGUAGE = Language(tscpp.language())
parser = Parser(CPP_LANGUAGE)


def traverse(node):

    if node.type == "function_definition":
        function_text = node.text.decode("utf8")

        declarator = node.child_by_field_name("declarator")

        function_name = "unknown"

        if declarator:
            function_name = declarator.text.decode("utf8")
            short_name = function_name.split("(")[0]

            node_id = f"{path}::{short_name}"

            function_lookup[node_id] = function_text

        embedding_text = function_name + "\n\n" + function_text

        embedding = embedding_model.encode(embedding_text)

        collection.add(
            ids=[f"{path}_{function_name}"],
            embeddings=[embedding.tolist()],
            documents=[function_text],
            metadatas=[
                {
                    "path": path,
                    "function_name": function_name,
                    "type": "function",
                }
            ],
        )

    for child in node.children:
        traverse(child)


for root, dirs, files in os.walk("repository"):
    if ".git" in root:
        continue

    for file in files:
        path = os.path.join(root, file)

        with open(path, "r", errors="ignore") as f:
            content = f.read()

            tree = parser.parse(bytes(content, "utf8"))
            root_node = tree.root_node
            traverse(root_node)

with open("function_lookup.pkl", "wb") as f:
    pickle.dump(function_lookup, f)

print("Ingestion complete")
