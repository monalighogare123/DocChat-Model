from dotenv import load_dotenv
from groq import Groq

load_dotenv()
client = Groq()
MODEL = "openai/gpt-oss-120b"      # keep the model name that worked for you

SYSTEM = (
    "You are a helpful assistant that answers questions about a document. "
    "Use ONLY the context provided in the user's message. "
    "If the answer is not in the context, reply exactly: "
    "\"I couldn't find that in the document.\" "
    "When you use information, mention the page like (Page 3). "
    "Be clear and concise."
)

def ask_llm(prompt, system=SYSTEM):
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1000,
        temperature=0.2,
    )
    return response.choices[0].message.content

def stream_llm(prompt, system=SYSTEM):
    """Yield the answer piece by piece."""
    stream = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        max_tokens=1000,
        temperature=0.2,
        stream=True,
    )
    for chunk in stream:
        piece = chunk.choices[0].delta.content
        if piece:
            yield piece
            
            
REWRITE_SYSTEM = (
    "You rewrite follow-up questions into standalone questions. "
    "Using the conversation, rewrite the last question so it can be understood "
    "without the conversation: replace words like 'it', 'that', 'this' with what "
    "they refer to. If the question is already standalone, return it unchanged. "
    "Output ONLY the rewritten question, nothing else."
)


def rewrite_question(history, question):
    """Turn a follow-up question into a standalone one using the chat history."""
    if not history:
        return question                      # first question: nothing to fix
    convo = "\n".join(
        f"{m['role'].capitalize()}: {m['content'][:500]}" for m in history
    )
    prompt = (
        f"Conversation:\n{convo}\n\n"
        f"Last question: {question}\n\n"
        "Standalone question:"
    )
    rewritten = ask_llm(prompt, system=REWRITE_SYSTEM).strip()
    return rewritten or question             # fall back if the AI returned nothing

if __name__ == "__main__":
    print(ask_llm("Say hello in one friendly sentence.", system="You are a friendly assistant."))