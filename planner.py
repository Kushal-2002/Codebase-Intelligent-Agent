import os

from ollama import chat

# override with e.g. LLM_MODEL=llama3.2 for a lighter model
MODEL = os.environ.get("LLM_MODEL", "llama3")


def decide_action(conversation):

    response = chat(
        model=MODEL,
        options={"temperature": 0},
        messages=[
            {
                "role": "user",
                "content": f"""
You are an autonomous codebase agent.

You have three available tools.

search_code(function_name)
- Use to understand what a function does.

get_callers(function_name)
- Use to find which functions call another function.

get_dependencies(function_name)
- Use to find which functions are called by another function.

You will receive the entire conversation, including previous observations.

Your job is to either:

1. Return EXACTLY ONE tool call.

Examples:

search_code(connect_any_tracker)

get_callers(connect_to_addr)

get_dependencies(connect_any_tracker)

finish(answer)

OR

2. If the observations are enough to answer,
return

finish(<your answer>)

Rules:

- Never explain your reasoning.
- Never return more than one tool call.
- Never return multiple options.
- Never ask questions.
- If enough observations are available, return FINAL ANSWER instead of another tool.

Conversation:

{conversation}
""",
            }
        ],
    )

    response = response.message.content.strip()

    print("\nPLANNER OUTPUT:")
    print(response)

    return response
