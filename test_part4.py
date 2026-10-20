from rag import extract_pages, chunk_text

pages = extract_pages("sample.pdf")

all_chunks = []
for page_num, text in pages:
    for chunk in chunk_text(text):
        if chunk.strip():                      # skip empty chunks
            all_chunks.append((page_num, chunk))

print("Total pages:", len(pages))
print("Total chunks:", len(all_chunks))

print("\n--- Chunk 1 ---")
print("From page:", all_chunks[0][0])
print(all_chunks[0][1])

print("\n--- Chunk 2 ---")
print("From page:", all_chunks[1][0])
print(all_chunks[1][1])