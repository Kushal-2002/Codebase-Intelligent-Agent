from ollama import chat

from tools import retrieve_context


question = input("Ask a question: ")

relevant_text = retrieve_context(question)

print("\nCONTEXT LENGTH =", len(relevant_text))


response = chat(
    model="llama3",
    messages=[
        {
            "role": "user",
            "content": f"""
You are a codebase assistant.

Answer ONLY using the repository context.

If the answer cannot be found in the context, say:
"I could not find this information in the repository."

Repository Context:
{relevant_text}

Question:
{question}
""",
        }
    ],
)

print(response.message.content)
