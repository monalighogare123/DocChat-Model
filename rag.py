from sentence_transformers import SentenceTransformer
import chromadb

from pypdf import PdfReader

def extract_pages(file):
    """Read a PDF and return a list of (page_number, text) pairs."""
    reader = PdfReader(file)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        pages.append((i + 1, text))
    return pages

def chunk_text(text, size=800, overlap=150):
    """Split text into overlapping pieces of about `size` characters."""
    chunks = []
    start = 0
    while start < len(text):
        chunks.append(text[start:start + size])
        if start + size >= len(text):      # this chunk already reached the end
            break
        start += size - overlap
    return chunks

embedder = SentenceTransformer("all-MiniLM-L6-v2")
db = chromadb.Client()


def build_index(file, name="docs", size=800, overlap=150):
    """Read a PDF, chunk it, embed the chunks, and store them in the database."""
    try:
        db.delete_collection(name)        # remove any old index with this name
    except Exception:
        pass
    collection = db.create_collection(name)

    ids, documents, metadatas = [], [], []
    for page_num, text in extract_pages(file):
        for j, chunk in enumerate(chunk_text(text, size, overlap)):
            if chunk.strip():
                ids.append(f"p{page_num}_c{j}")
                documents.append(chunk)
                metadatas.append({"page": page_num})

    if not documents:
        raise ValueError("No text found. This may be a scanned PDF.")

    embeddings = embedder.encode(documents).tolist()
    collection.add(
        ids=ids,
        documents=documents,
        embeddings=embeddings,
        metadatas=metadatas,
    )
    return collection

def retrieve(collection, question, k=4):
    """Find the k chunks whose meaning is closest to the question."""
    q_vector = embedder.encode([question]).tolist()
    result = collection.query(
        query_embeddings=q_vector,
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )

    hits = []
    for doc, meta, dist in zip(
        result["documents"][0],
        result["metadatas"][0],
        result["distances"][0],
    ):
        hits.append({"text": doc, "page": meta["page"], "distance": dist})
    return hits

def build_prompt(question, hits, history=None):
    """Combine chat history, retrieved chunks, and the question into one prompt."""
    context = "\n\n".join(f"[Page {h['page']}]\n{h['text']}" for h in hits)

    convo = ""
    if history:
        lines = "\n".join(
            f"{m['role'].capitalize()}: {m['content'][:500]}" for m in history
        )
        convo = f"Conversation so far (for reference only, not a source of facts):\n{lines}\n\n"

    return f"{convo}Context:\n{context}\n\nQuestion: {question}"