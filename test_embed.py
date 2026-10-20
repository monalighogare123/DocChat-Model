from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("all-MiniLM-L6-v2")

sentences = [
    "How do I cancel my subscription?",
    "Steps to terminate your membership",
    "The weather is sunny today",
]

vectors = model.encode(sentences)

print("Number of vectors:", len(vectors))
print("Numbers in each vector:", len(vectors[0]))
print("First 5 numbers of vector 1:", vectors[0][:5])

print("\nSimilarity (1 = same meaning, 0 = unrelated):")
print("Cancel vs Terminate:", round(float(util.cos_sim(vectors[0], vectors[1])), 2))
print("Cancel vs Weather:  ", round(float(util.cos_sim(vectors[0], vectors[2])), 2))