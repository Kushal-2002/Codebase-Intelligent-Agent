import re

from planner import decide_action, MODEL
from ollama import chat

from tools import (
    search_code,
    find_graph_nodes,
    get_callers,
    get_dependencies,
    get_source,
    build_context,
)


TOOL_PATTERN = re.compile(
    r"\b(search_code|get_callers|get_dependencies)\(\s*([^()\s]+)\s*\)"
)

FINISH_PATTERN = re.compile(r"\bfinish\(", re.IGNORECASE)


def parse_planner_output(response):

    # the LLM sometimes wraps the tool call in extra text, so search for it
    # anywhere in the response instead of relying on startswith

    match = TOOL_PATTERN.search(response)

    if match:
        return match.group(1), match.group(2).strip("'\"`")

    if FINISH_PATTERN.search(response):
        return "finish", None

    return None, None


def generate_answer(question, conversation):

    response = chat(
        model=MODEL,
        options={"temperature": 0},
        messages=[
            {
                "role": "user",
                "content": f"""
You are a codebase assistant.

Use ONLY the observations below to answer the user's question.

Question:
{question}

Conversation:
{conversation}

Give a clear answer.

If the observations are insufficient, say so.
""",
            }
        ],
    )

    return response.message.content


def semantic_context(query):

    results = search_code(query)

    context = ""
    nodes = []

    for document, metadata in zip(results["documents"][0], results["metadatas"][0]):
        context += f"\nFUNCTION ({metadata['path']} :: {metadata['function_name']}):\n"
        context += document
        context += "\n"

        nodes.append(f"{metadata['path']}::{metadata['function_name'].split('(')[0]}")

    return context, nodes


def execute_tool(action, function_name):

    # returns the observation text and the graph nodes whose source it contains

    if action == "search_code":
        matched_nodes = find_graph_nodes(function_name)

        context = ""
        nodes = []

        for node in matched_nodes:
            source = get_source(node)

            if source:
                context += f"\nFUNCTION ({node}):\n"
                context += source
                context += "\n"
                nodes.append(node)

        # no exact function match, fall back to semantic search
        if not context:
            context, nodes = semantic_context(function_name)
            context = "No exact function match. Closest functions by semantic search:\n" + context

        return context, nodes

    elif action == "get_callers":
        matched_nodes = find_graph_nodes(function_name)

        if not matched_nodes:
            return f"No function named {function_name} found in the call graph.", []

        context = ""
        nodes = []

        for node in matched_nodes:
            callers = get_callers(node)

            context += f"\nCALLERS OF {node}: {callers or 'none'}\n"
            context += build_context(callers)
            nodes += callers

        return context, nodes

    elif action == "get_dependencies":
        matched_nodes = find_graph_nodes(function_name)

        if not matched_nodes:
            return f"No function named {function_name} found in the call graph.", []

        context = ""
        nodes = []

        for node in matched_nodes:
            deps = get_dependencies(node)

            context += f"\nFUNCTIONS CALLED BY {node}: {deps or 'none'}\n"
            context += build_context(deps)
            nodes += deps

        return context, nodes

    return "No observation.", []


def run_agent(question, max_steps=5, verbose=True, on_step=None):

    conversation = f"Question:\n{question}\n"

    # seed the conversation with semantic search so the planner knows
    # which function names actually exist in the repository
    seed_context, retrieved_nodes = semantic_context(question)
    conversation += f"\nObservation:\n{seed_context}\n"

    steps = 0
    finished = False

    for step in range(max_steps):
        response = decide_action(conversation)
        steps += 1

        action, function_name = parse_planner_output(response)

        if on_step:
            on_step(steps, action, function_name)

        if action == "finish":
            finished = True
            if verbose:
                print("\nPlanner Finished\n")
            break

        if action is None:
            observation = (
                "Invalid planner output. Return exactly one tool call or finish()."
            )
        else:
            observation, nodes = execute_tool(action, function_name)
            retrieved_nodes += nodes

        if verbose:
            print("\nObservation:")
            print(observation)

        conversation += f"\nTool call: {action}({function_name})\n"
        conversation += f"\nObservation:\n{observation}\n"

    if not finished and verbose:
        print(f"\nReached the step limit ({max_steps}), answering with what was collected\n")

    answer = generate_answer(question, conversation)

    return {
        "answer": answer,
        "nodes": list(dict.fromkeys(retrieved_nodes)),
        "steps": steps,
        "finished": finished,
        "context_chars": len(conversation),
    }


if __name__ == "__main__":
    question = input("Ask a question: ")

    result = run_agent(question)

    print("\nFINAL ANSWER:\n")
    print(result["answer"])
