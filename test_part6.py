from rag import build_index, retrieve

collection = build_index("sample.pdf")

questions = [
    "What is narrow AI?",
    "What does RAG stand for?",
    "What is the capital of France?",
]

for q in questions:
    print("\n" + "=" * 70)
    print("QUESTION:", q)
    for rank, hit in enumerate(retrieve(collection, q, k=3), start=1):
        print(f"\n#{rank} | Page {hit['page']} | distance {hit['distance']:.2f}")
        print(hit["text"][:250].replace("\n", " "))