"""
Index the bundled sample repository (repository/) into ./chroma_db and data/.

The indexing logic lives in indexer.py; build_call_graph.py runs the same
thing, since chunks and call graph are now built together in one pass.
To index any GitHub repository, use the web app: streamlit run app.py
"""

from indexer import index_repository
from tools import DEFAULT_CHROMA_PATH, DEFAULT_DATA_DIR

if __name__ == "__main__":
    index_repository("repository", DEFAULT_CHROMA_PATH, DEFAULT_DATA_DIR, progress=print)
