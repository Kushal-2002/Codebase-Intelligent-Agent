"""
The same codebase agent as agent.py, rebuilt as a LangGraph state machine.

    seed ──> planner ──(tool call)──> tools ──┐
                ^                             │
                └─────────────────────────────┘
             planner ──(no tool call / step limit)──> answer ──> END

Differences from the hand-written loop in agent.py:
    - state is a typed dict instead of one growing conversation string
    - the planner uses native tool calling (bind_tools) instead of regex parsing
    - tools run through LangGraph's prebuilt ToolNode

The tool logic (execute_tool) and the answer prompt (generate_answer) are
shared with agent.py, so the eval compares orchestration, not retrieval.

Works with Gemini or any Ollama model that supports tool calling (see llm.py):

    python agent_langgraph.py
    LLM_PROVIDER=gemini python agent_langgraph.py
    python agent_langgraph.py --diagram
"""

import sys
from typing import Annotated, TypedDict

from langchain_core.messages import (
    AIMessage,
    AnyMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from agent import execute_tool, generate_answer, semantic_context
from llm import get_chat_model


# ---------------- tools ----------------
# content_and_artifact: the model only sees the observation text, while the
# graph nodes it covers ride along on ToolMessage.artifact for the eval


@tool(response_format="content_and_artifact")
def search_code(function_name: str):
    """Show the source code of a function, to understand what it does.
    Falls back to semantic search if no function has exactly this name."""
    return execute_tool("search_code", function_name)


@tool(response_format="content_and_artifact")
def get_callers(function_name: str):
    """List the functions that call the given function, with their source."""
    return execute_tool("get_callers", function_name)


@tool(response_format="content_and_artifact")
def get_dependencies(function_name: str):
    """List the functions called by the given function, with their source."""
    return execute_tool("get_dependencies", function_name)


TOOLS = [search_code, get_callers, get_dependencies]

PLANNER_PROMPT = """You are an autonomous codebase agent exploring a C++ repository.

You are given the user's question and the closest functions found by semantic search.
Use the tools to gather the code needed to answer the question:

- search_code: read what a function does
- get_callers: find which functions call a function
- get_dependencies: find which functions a function calls

Rules:
- Call one tool at a time, using a function name that appears in the observations.
- Do not repeat a tool call you have already made.
- When the observations are enough to answer, reply with the single word DONE
  and no tool call. Another step writes the final answer."""

planner_llm = get_chat_model().bind_tools(TOOLS)


# ---------------- state ----------------


class AgentState(TypedDict):
    question: str
    messages: Annotated[list[AnyMessage], add_messages]
    seed_nodes: list[str]
    steps: int
    max_steps: int
    finished: bool
    answer: str
    conversation: str


# ---------------- graph nodes ----------------


def seed(state: AgentState):

    # same as agent.py: start with semantic search so the planner knows
    # which function names actually exist in the repository
    context, nodes = semantic_context(state["question"])

    return {
        "messages": [
            SystemMessage(PLANNER_PROMPT),
            HumanMessage(f"Question:\n{state['question']}\n\nObservation:\n{context}"),
        ],
        "seed_nodes": nodes,
        "steps": 0,
        "finished": False,
    }


def planner(state: AgentState, config: RunnableConfig):

    response = planner_llm.invoke(state["messages"])
    steps = state["steps"] + 1

    on_step = config.get("configurable", {}).get("on_step")

    if on_step:
        if response.tool_calls:
            call = response.tool_calls[0]
            on_step(steps, call["name"], call["args"].get("function_name"))
        else:
            on_step(steps, "finish", None)

    return {
        "messages": [response],
        "steps": steps,
        "finished": not response.tool_calls,
    }


def route_after_planner(state: AgentState):

    if state["finished"] or state["steps"] >= state["max_steps"]:
        return "answer"

    return "tools"


def render_conversation(messages):

    # flatten the message history into the same text layout agent.py builds,
    # so generate_answer receives an equivalent prompt

    conversation = ""

    for message in messages:
        if isinstance(message, HumanMessage):
            conversation += f"{message.content}\n"

        elif isinstance(message, AIMessage):
            for call in message.tool_calls:
                arg = call["args"].get("function_name")
                conversation += f"\nTool call: {call['name']}({arg})\n"

        elif isinstance(message, ToolMessage):
            conversation += f"\nObservation:\n{message.content}\n"

    return conversation


def answer(state: AgentState):

    conversation = render_conversation(state["messages"])

    return {
        "answer": generate_answer(state["question"], conversation),
        "conversation": conversation,
    }


def build_graph():

    graph = StateGraph(AgentState)

    graph.add_node("seed", seed)
    graph.add_node("planner", planner)
    graph.add_node("tools", ToolNode(TOOLS))
    graph.add_node("answer", answer)

    graph.add_edge(START, "seed")
    graph.add_edge("seed", "planner")
    graph.add_conditional_edges("planner", route_after_planner, ["tools", "answer"])
    graph.add_edge("tools", "planner")
    graph.add_edge("answer", END)

    return graph.compile()


app = build_graph()


# ---------------- entry point ----------------


def run_langgraph_agent(question, max_steps=5, on_step=None):

    # same return shape as agent.run_agent, so the eval can treat both alike

    state = app.invoke(
        {"question": question, "max_steps": max_steps},
        config={"configurable": {"on_step": on_step}},
    )

    nodes = list(state["seed_nodes"])

    for message in state["messages"]:
        if isinstance(message, ToolMessage) and message.artifact:
            nodes += message.artifact

    return {
        "answer": state["answer"],
        "nodes": list(dict.fromkeys(nodes)),
        "steps": state["steps"],
        "finished": state["finished"],
        "context_chars": len(state["conversation"]),
    }


if __name__ == "__main__":
    if "--diagram" in sys.argv:
        print(app.get_graph().draw_mermaid())
        sys.exit()

    question = input("Ask a question: ")

    def print_step(step, action, function_name):
        target = f"({function_name})" if function_name else ""
        print(f"step {step}: {action}{target}")

    result = run_langgraph_agent(question, on_step=print_step)

    print("\nFINAL ANSWER:\n")
    print(result["answer"])
