"""
Web interface: paste a GitHub repository link, then ask questions about it.

    streamlit run app.py
"""

import streamlit as st

import tools
from graph_view import MAX_NODES, answer_graph, neighbourhood, split_node, to_dot
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
    "Baseline (hand-written loop)": "agent",
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

index = tools.load_index(ws["chroma"], ws["data"])

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

st.caption("C, C++ and Python repositories are supported. Paste a GitHub link in the sidebar to index a new one.")

messages = st.session_state.chats.setdefault(name, [])

chat_tab, graph_tab = st.tabs(["Chat", "Call graph"])


def render_details(message):

    if message.get("steps"):
        with st.expander(f"Agent steps ({len(message['steps'])})"):
            for step in message["steps"]:
                st.markdown(f"- {step}")

    if message.get("nodes"):
        with st.expander(f"Functions used ({len(message['nodes'])})"):
            for node in message["nodes"]:
                st.markdown(f"- `{node}`")

        with st.expander("Call graph of these functions"):
            st.graphviz_chart(answer_graph(index.G, message["nodes"]))
            st.caption("Arrows point from caller to callee. Colours mark source files.")


# ---------------- tab: chat ----------------

with chat_tab:
    # messages go in a container above the input, so new ones don't land
    # below the input box (inside a tab it isn't pinned to the page bottom)
    history = st.container()
    question = st.chat_input("Ask a question about this repository")

with history:
    for message in messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            render_details(message)

if question:
    with history:
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


# ---------------- tab: call graph explorer ----------------

with graph_tab:
    G = index.G

    # call_graph has every function, including ones with no edges at all
    functions = sorted(index.call_graph, key=lambda n: split_node(n)[1].lower())

    if not functions:
        st.info("This repository has no indexed functions.")
    else:
        def is_test(node):
            path = split_node(node)[0].lower()
            return "test" in path or "conftest" in path

        # start on the most connected function, preferring non-test code
        busiest = max(functions, key=lambda n: (not is_test(n), G.degree(n) if n in G else 0))

        def describe(node):
            path, fn = split_node(node)
            return f"{fn}  ·  {path.split('/', 1)[-1]}"

        selected = st.selectbox(
            "Function",
            functions,
            index=functions.index(busiest),
            format_func=describe,
            key=f"graph_fn_{name}",
            help="Type to search. Starts on the most connected non-test function.",
        )

        col1, col2 = st.columns([2, 1])
        directions = {"Callers and callees": "both", "Callers": "callers", "Callees": "callees"}
        direction = directions[col1.radio("Show", list(directions), horizontal=True)]
        depth = col2.slider("Levels", 1, 3, 1)

        nodes, truncated = neighbourhood(G, selected, depth=depth, direction=direction)

        callers = len(list(G.predecessors(selected))) if selected in G else 0
        callees = len(list(G.successors(selected))) if selected in G else 0
        st.caption(f"`{selected}` is called by {callers} and calls {callees} functions in this repository.")

        st.graphviz_chart(to_dot(G, nodes, focus=[selected]))

        if truncated:
            st.caption(f"Showing the first {MAX_NODES} functions; use fewer levels or one direction to see less.")

        with st.expander("Source code"):
            language = "python" if split_node(selected)[0].endswith(".py") else "cpp"
            st.code(tools.get_source(selected) or "(source not available)", language=language)
