from rag import build_index, retrieve, build_prompt
from llm import ask_llm

collection = build_index("sample.pdf")

questions = [
    "What is narrow AI?",
    "What does RAG stand for and how does it work?",
    "What is the capital of France?",
]

for q in questions:
    hits = retrieve(collection, q, k=4)
    prompt = build_prompt(q, hits)
    answer = ask_llm(prompt)

    print("\n" + "=" * 70)
    print("QUESTION:", q)
    print("ANSWER:", answer)
    print("SOURCE PAGES:", sorted({h["page"] for h in hits}))