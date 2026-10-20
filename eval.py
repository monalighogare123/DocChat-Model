import sys
import time
from rag import build_index, retrieve, build_prompt
from llm import ask_llm

PDF = "sample.pdf"
PAUSE = 4          # seconds between AI calls, to stay under the free rate limit

# (question, [acceptable keywords], page where the answer lives)
# keywords = None means the document has NO answer: the AI should say it couldn't find it.
TESTS = [
    ("What is narrow AI?", ["spam detection", "particular task"], 2),
    ("What is general AI?", ["hypothetical"], 2),
    ("Why might AI output be unreliable?", ["incorrect", "biased", "out of date"], 2),
    ("What does regression predict?", ["numeric value"], 3),
    ("What does unsupervised learning look for?", ["structure"], 3),
    ("What is test data used for?", ["not used", "performance"], 3),
    ("What are convolutional neural networks often used for?", ["image"], 4),
    ("How do recurrent neural networks process data?", ["step by step", "sequence"], 4),
    ("What does RAG stand for?", ["retrieval-augmented generation"], 7),
    # ADD 6 to 10 MORE of your own here, from pages 5, 6, 8 to 13  ->
    # ("your question", ["keyword"], page_number),
    ("What is the capital of France?", None, None),
    ("Who won the 2018 FIFA World Cup?", None, None),
]


def norm(text):
    """Lowercase and unify look-alike characters so matching is fair."""
    text = text.lower().replace("’", "'")
    for ch in "\u2010\u2011\u2012\u2013\u2014\u2212":     # fancy hyphens and dashes
        text = text.replace(ch, "-")
    return text


def ask_safe(prompt):
    """Call the AI, retrying if we hit a rate limit."""
    for _ in range(3):
        try:
            return ask_llm(prompt)
        except Exception as e:
            print("   (API problem, waiting 30s) ", str(e)[:80])
            time.sleep(30)
    return ""


def evaluate(collection, k, use_llm=True, verbose=True, retriever=None):
    retriever = retriever or retrieve
    positives = [t for t in TESTS if t[1]]
    hit = top1 = answer_ok = 0

    for question, keywords, page in TESTS:
        hits = retriever(collection, question, k)
        pages = [h["page"] for h in hits]

        if keywords:
            hit += page in pages          # right page somewhere in the results
            top1 += pages[0] == page      # right page ranked first

        if not use_llm:
            continue                      # retrieval-only runs make NO AI calls

        answer = norm(ask_safe(build_prompt(question, hits)))
        if keywords:
            ok = any(norm(w) in answer for w in keywords)
        else:
            ok = "couldn't find" in answer
        answer_ok += ok

        if verbose:
            print(f"{'✅' if ok else '❌'} {question}")
            if not ok:
                print(f"     expected page {page}, retrieved {sorted(set(pages))}")
                print(f"     answer: {answer[:150]!r}")
        time.sleep(PAUSE)

    answer_rate = answer_ok / len(TESTS) if use_llm else None
    return hit / len(positives), top1 / len(positives), answer_rate


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "full"

    if mode == "sweep":
        print("size   k   hit rate   top-1 rate")
        for size in (400, 800, 1200):
            collection = build_index(PDF, size=size, overlap=150)
            for k in (1, 2, 4):
                r, t, _ = evaluate(collection, k, use_llm=False)
                print(f"{size:<6} {k:<3} {r:>7.0%}   {t:>9.0%}")
    else:
        collection = build_index(PDF)
        r, t, a = evaluate(collection, k=4)
        print("\nBaseline (size 800, overlap 150, k=4)")
        print(f"Retrieval hit rate: {r:.0%}")
        print(f"Top-1 rate:         {t:.0%}")
        print(f"Answer accuracy:    {a:.0%}")