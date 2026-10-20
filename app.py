import os
import uuid
import streamlit as st

# On the hosted app the key lives in Streamlit "secrets", not in a .env file.
try:
    if "GROQ_API_KEY" in st.secrets:
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
except Exception:
    pass          # running locally with no secrets file: .env will be used

from rag import build_index, retrieve, build_prompt
from llm import stream_llm, rewrite_question

st.set_page_config(page_title="DocChat", page_icon="📄", layout="wide")

MEMORY_MESSAGES = 6        # NEW: how many recent messages the AI remembers (3 exchanges)

# ---------- Styling and header banner ----------
st.markdown("""
<style>
#MainMenu, footer {visibility: hidden;}

.hero {
    padding: 1.2rem 1.6rem;
    border-radius: 16px;
    background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 100%);
    margin-bottom: 1rem;
}
.hero h1 { margin: 0; color: white; font-size: 2rem; }
.hero p  { margin: .3rem 0 0; color: rgba(255,255,255,.85); }

[data-testid="stChatMessage"] {
    background: #1e293b;
    border-radius: 14px;
    padding: 12px 16px;
    margin-bottom: 10px;
}

.stButton > button {
    border-radius: 999px;
    border: 1px solid #38bdf8;
    background: transparent;
    color: #38bdf8;
}
.stButton > button:hover {
    background: #38bdf8;
    color: #0f172a;
}

.doc-card {
    padding: .8rem 1rem;
    border-radius: 12px;
    background: #0f172a;
    border: 1px solid #334155;
    margin-top: .5rem;
}
</style>

<div class="hero">
    <h1>📄 DocChat</h1>
    <p>Upload a PDF and ask questions. Every answer cites its source pages.</p>
</div>
""", unsafe_allow_html=True)

# ---------- Memory that survives reruns ----------
if "messages" not in st.session_state:
    st.session_state.messages = []
    
if "sid" not in st.session_state:
    st.session_state.sid = uuid.uuid4().hex[:8]     # unique id for this visitor


def show_sources(hits):
    with st.expander("📚 Sources"):
        for h in hits:
            st.markdown(f"**Page {h['page']}**")
            st.caption(h["text"][:400] + "...")


def friendly_error(e):                                 # NEW
    """Turn a technical error into a message a normal user understands."""
    text = str(e).lower()
    if "429" in text or "rate limit" in text:
        return "⏳ Too many requests right now. Please wait a minute and try again."
    if "401" in text or "api key" in text or "authentication" in text:
        return "🔑 There is a problem with the API key. Check GROQ_API_KEY in your .env file."
    if "connection" in text or "timeout" in text or "timed out" in text:
        return "🌐 Couldn't reach the AI service. Please check your internet connection."
    return f"Something went wrong: {e}"


# ---------- Sidebar ----------
with st.sidebar:
    st.header("Your document")
    pdf = st.file_uploader("Upload a PDF", type="pdf")
    st.caption("🔒 Demo app: please don't upload confidential documents.")
    k = st.slider("Passages to retrieve", 1, 8, 4)
    if st.button("🗑️ Clear chat"):
        st.session_state.messages = []
        st.rerun()

# ---------- Index the PDF once per upload ----------
if pdf and st.session_state.get("pdf_name") != pdf.name:
    with st.spinner("Reading and indexing your PDF..."):
        try:
            collection = build_index(pdf, name=f"docs_{st.session_state.sid}")
            metas = collection.get(include=["metadatas"])["metadatas"]
            st.session_state.collection = collection
            st.session_state.pdf_name = pdf.name
            st.session_state.info = {
                "pages": max(m["page"] for m in metas),
                "chunks": collection.count(),
            }
            st.session_state.messages = []
        except ValueError as e:
            st.sidebar.error(str(e))

# ---------- Document info card ----------
if "pdf_name" in st.session_state:
    info = st.session_state.info
    st.sidebar.markdown(
        f"""<div class="doc-card">
        ✅ <b>{st.session_state.pdf_name}</b><br>
        📑 {info['pages']} pages &nbsp;•&nbsp; 🧩 {info['chunks']} chunks
        </div>""",
        unsafe_allow_html=True,
    )

# ---------- Suggested questions ----------
SUGGESTIONS = [
    "Summarize this document",
    "What are the key points?",
    "Explain the main topic in simple words",
]


def set_pending(text):
    st.session_state.pending = text


if (
    "collection" in st.session_state
    and not st.session_state.messages
    and "pending" not in st.session_state
):
    st.markdown("**Try asking:**")
    cols = st.columns(len(SUGGESTIONS))
    for col, text in zip(cols, SUGGESTIONS):
        col.button(text, on_click=set_pending, args=(text,), use_container_width=True)

# ---------- Show the chat history ----------
for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.write(m["content"])
        if m.get("sources"):
            show_sources(m["sources"])

# ---------- Handle a new question ----------
typed = st.chat_input("Ask something about your document...")
question = typed or st.session_state.pop("pending", None)

if question:
    if "collection" not in st.session_state:
        st.warning("Please upload a PDF in the sidebar first.")
    else:
        # NEW: take the recent messages BEFORE adding the new question
        history = st.session_state.messages[-MEMORY_MESSAGES:]

        with st.chat_message("user"):
            st.write(question)

        with st.chat_message("assistant"):
            try:                                                       # NEW
                with st.spinner("Searching the document..."):
                    standalone = rewrite_question(history, question)   # NEW
                    hits = retrieve(st.session_state.collection, standalone, k)
                    prompt = build_prompt(question, hits, history)     # NEW: history
                if standalone != question:
                    st.caption(f"🔎 Searched for: *{standalone}*")     # NEW
                answer = st.write_stream(stream_llm(prompt))
            except Exception as e:                                     # NEW
                st.error(friendly_error(e))
            else:
                found = "couldn't find" not in answer.lower()
                if found:
                    show_sources(hits)
                # Save to history only when the answer succeeded
                st.session_state.messages.append(
                    {"role": "user", "content": question}
                )
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer,
                     "sources": hits if found else None}
                )