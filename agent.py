import re

from planner import decide_action
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
        model="llama3",
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

    for document, metadata in zip(results["documents"][0], results["metadatas"][0]):
        context += f"\nFUNCTION ({metadata['path']} :: {metadata['function_name']}):\n"
        context += document
        context += "\n"

    return context


def execute_tool(action, function_name):

    if action == "search_code":
        matched_nodes = find_graph_nodes(function_name)

        context = ""

        for node in matched_nodes:
            source = get_source(node)

            if source:
                context += f"\nFUNCTION ({node}):\n"
                context += source
                context += "\n"

        # no exact function match, fall back to semantic search
        if not context:
            context = "No exact function match. Closest functions by semantic search:\n"
            context += semantic_context(function_name)

        return context

    elif action == "get_callers":
        matched_nodes = find_graph_nodes(function_name)

        if not matched_nodes:
            return f"No function named {function_name} found in the call graph."

        context = ""

        for node in matched_nodes:
            callers = get_callers(node)

            context += f"\nCALLERS OF {node}: {callers or 'none'}\n"
            context += build_context(callers)

        return context

    elif action == "get_dependencies":
        matched_nodes = find_graph_nodes(function_name)

        if not matched_nodes:
            return f"No function named {function_name} found in the call graph."

        context = ""

        for node in matched_nodes:
            deps = get_dependencies(node)

            context += f"\nFUNCTIONS CALLED BY {node}: {deps or 'none'}\n"
            context += build_context(deps)

        return context

    return "No observation."


question = input("Ask a question: ")

conversation = f"Question:\n{question}\n"

# seed the conversation with semantic search so the planner knows
# which function names actually exist in the repository
conversation += f"\nObservation:\n{semantic_context(question)}\n"

MAX_STEPS = 5

for step in range(MAX_STEPS):
    response = decide_action(conversation)

    action, function_name = parse_planner_output(response)

    if action == "finish":
        print("\nPlanner Finished\n")
        break

    if action is None:
        observation = (
            "Invalid planner output. Return exactly one tool call or finish()."
        )
    else:
        observation = execute_tool(action, function_name)

    print("\nObservation:")
    print(observation)

    conversation += f"\nTool call: {action}({function_name})\n"
    conversation += f"\nObservation:\n{observation}\n"

else:
    print(f"\nReached the step limit ({MAX_STEPS}), answering with what was collected\n")


answer = generate_answer(question, conversation)

print("\nFINAL ANSWER:\n")
print(answer)
