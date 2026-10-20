from rag import build_index

collection = build_index("sample.pdf")

print("Chunks stored:", collection.count())

peek = collection.peek(2)
for doc, meta in zip(peek["documents"], peek["metadatas"]):
    print("\n--- Stored chunk ---")
    print("Page:", meta["page"])
    print(doc[:200])