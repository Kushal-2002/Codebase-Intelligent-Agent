"""
Web interface: paste a GitHub repository link, then ask questions about it.

    streamlit run app.py
"""

import streamlit as st

import tools
from llm import MODEL, PROVIDER
from workspace import list_workspaces, prepare_repository
from tools import DEFAULT_CHROMA_PATH, DEFAULT_DATA_DIR

st.set_page_config(page_title="Codebase Intelligence Agent", page_icon="🔎", layout="centered")

SAMPLE = {
    "chroma": DEFAULT_CHROMA_PATH,
    "data": DEFAULT_DATA_DIR,
    "meta": {"name": "Bundled sample (repository/)", "url": None},
}

AGENTS = {
    "LangGraph agent": "langgraph",
    "Hand-written agent": "agent",
}


def workspace_key(ws):

    return ws["meta"]["name"]


# ---------------- state ----------------

if "workspace" not in st.session_state:
    st.session_state.workspace = SAMPLE

if "chats" not in st.session_state:
    st.session_state.chats = {}  # repo name -> list of messages


# ---------------- sidebar: choose a repository ----------------

with st.sidebar:
    st.header("Repository")

    with st.form("load_repo"):
        url = st.text_input("GitHub link", placeholder="https://github.com/owner/repo")
        refresh = st.checkbox("Re-clone and re-index", help="Pull the latest code even if this repo was indexed before")
        submitted = st.form_submit_button("Load repository", use_container_width=True)

    if submitted and url:
        try:
            with st.status("Preparing repository...", expanded=True) as status:
                ws = prepare_repository(url, refresh=refresh, progress=st.write)
                status.update(label=f"Ready: {ws['meta']['name']}", state="complete", expanded=False)

            st.session_state.workspace = ws

        except Exception as error:  # clone/parse errors are shown, not raised
            st.error(str(error))

    options = [SAMPLE] + list_workspaces()
    names = [workspace_key(ws) for ws in options]
    current = workspace_key(st.session_state.workspace)

    chosen = st.selectbox(
        "Indexed repositories",
        names,
        index=names.index(current) if current in names else 0,
    )

    if chosen != current:
        st.session_state.workspace = options[names.index(chosen)]
        st.rerun()

    st.divider()

    agent_label = st.radio("Agent", list(AGENTS))
    st.caption(f"LLM: {PROVIDER} / {MODEL}")

    if st.button("Clear chat", use_container_width=True):
        st.session_state.chats.pop(current, None)
        st.rerun()


# ---------------- main: chat ----------------

ws = st.session_state.workspace
meta = ws["meta"]
name = workspace_key(ws)

tools.load_index(ws["chroma"], ws["data"])

st.title("Codebase Intelligence Agent")

if meta.get("url"):
    st.markdown(f"Asking about **[{meta['name']}]({meta['url']})** at `{meta.get('commit', '?')}`")
else:
    st.markdown(f"Asking about **{meta['name']}**")

if "functions" in meta:
    cols = st.columns(3)
    cols[0].metric("Files", f"{meta['files']:,}")
    cols[1].metric("Functions", f"{meta['functions']:,}")
    cols[2].metric("Call edges", f"{meta['call_edges']:,}")

st.caption("C and C++ repositories are supported. Paste a GitHub link in the sidebar to index a new one.")

messages = st.session_state.chats.setdefault(name, [])


def render_details(message):

    if message.get("steps"):
        with st.expander(f"Agent steps ({len(message['steps'])})"):
            for step in message["steps"]:
                st.markdown(f"- {step}")

    if message.get("nodes"):
        with st.expander(f"Functions used ({len(message['nodes'])})"):
            for node in message["nodes"]:
                st.markdown(f"- `{node}`")


for message in messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_details(message)


question = st.chat_input("Ask a question about this repository")

if question:
    messages.append({"role": "user", "content": question})

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        steps = []

        try:
            with st.status("Thinking...", expanded=True) as status:

                def on_step(step, action, function_name):
                    target = f"(`{function_name}`)" if function_name else ""
                    text = f"step {step}: `{action}`{target}"
                    steps.append(text)
                    st.write(text)

                if AGENTS[agent_label] == "langgraph":
                    from agent_langgraph import run_langgraph_agent

                    result = run_langgraph_agent(question, on_step=on_step)
                else:
                    from agent import run_agent

                    result = run_agent(question, verbose=False, on_step=on_step)

                status.update(label=f"Done in {result['steps']} steps", state="complete", expanded=False)

            reply = {
                "role": "assistant",
                "content": result["answer"],
                "steps": steps,
                "nodes": result["nodes"],
            }

        except Exception as error:  # e.g. missing API key, model not pulled
            reply = {"role": "assistant", "content": f"**Error:** {error}"}

        st.markdown(reply["content"])
        render_details(reply)

    messages.append(reply)
