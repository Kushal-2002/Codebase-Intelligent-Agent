"""
One place that decides which LLM the whole project talks to.

    LLM_PROVIDER  ollama (default, local) | gemini (Google Gemini API)
    LLM_MODEL     model name; defaults to llama3.1 / gemini-3.5-flash

Gemini needs GOOGLE_API_KEY (or GEMINI_API_KEY). Put it in a .env file in the
project root (see .env.example); .env is git-ignored and loaded automatically.

    LLM_PROVIDER=gemini python agent_langgraph.py
    LLM_PROVIDER=ollama LLM_MODEL=qwen2.5 python eval/run_eval.py
"""

import os

from dotenv import load_dotenv

load_dotenv()

DEFAULT_MODELS = {
    # llama3.1 rather than llama3: the LangGraph agent needs native tool
    # calling, which Ollama does not support for the original llama3
    "ollama": "llama3.1",
    "gemini": "gemini-3.5-flash",
}

PROVIDER = os.environ.get("LLM_PROVIDER", "ollama").lower()

if PROVIDER not in DEFAULT_MODELS:
    raise ValueError(f"LLM_PROVIDER must be one of {sorted(DEFAULT_MODELS)}, got {PROVIDER!r}")

MODEL = os.environ.get("LLM_MODEL", DEFAULT_MODELS[PROVIDER])

_chat_model = None


def get_chat_model():

    # a LangChain chat model, so the LangGraph agent can bind_tools on it

    global _chat_model

    if _chat_model is None:
        if PROVIDER == "gemini":
            from langchain_google_genai import ChatGoogleGenerativeAI

            if not (os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")):
                raise RuntimeError(
                    "LLM_PROVIDER=gemini needs GOOGLE_API_KEY. Add it to .env (see .env.example)."
                )

            # retries with backoff on transient errors and rate limits during eval runs
            _chat_model = ChatGoogleGenerativeAI(model=MODEL, temperature=0, max_retries=6)

        else:
            from langchain_ollama import ChatOllama

            _chat_model = ChatOllama(model=MODEL, temperature=0)

    return _chat_model


def complete(prompt):

    # single-prompt text completion, used by the planner and the answer step

    return get_chat_model().invoke(prompt).text
