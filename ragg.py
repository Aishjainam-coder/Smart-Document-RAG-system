# ============================================================
# Smart Document Q&A — Rebuilt with LangChain
# ============================================================
#
# WHAT LANGCHAIN REPLACED AND WHY:
#
# 1. DocumentProcessor (PyPDF2 + python-docx manual code)
#    → LangChain Loaders: PyPDFLoader, Docx2txtLoader, TextLoader
#    WHY: Built-in page metadata, better text extraction,
#         consistent Document objects across all file types.
#
# 2. chunk_text() — manual word-count splitter
#    → RecursiveCharacterTextSplitter
#    WHY: Respects sentence/paragraph boundaries. Smarter splits
#         = better retrieval quality. Supports many strategies.
#
# 3. SimpleEmbedding (TF-IDF) + SentenceTransformer manual calls
#    → HuggingFaceEmbeddings / OpenAIEmbeddings
#    WHY: Standardized interface. Swap model in one line.
#         No manual encode(), no numpy juggling.
#
# 4. SimpleVectorDB + manual cosine similarity (np.argsort)
#    → Chroma via LangChain
#    WHY: Handles batching, persistence, filtering.
#         .as_retriever() wires it to the chain automatically.
#
# 5. build_messages() — f-string prompt builder
#    → ChatPromptTemplate
#    WHY: Reusable, testable, version-controlled prompt objects.
#         Variables are explicit, not buried in f-strings.
#
# 6. LLMClient (raw requests.post to OpenAI/Ollama)
#    → ChatOpenAI / ChatOllama
#    WHY: Standardized interface, built-in retry, streaming,
#         async support. Switch providers with one import swap.
#
# 7. enforce_strict_answer() — 80-line post-processing hack
#    → Removed. The chain + good prompt does this natively.
#    WHY: LangChain LCEL chains are composable and transparent.
#         Better to fix the prompt than to patch the output.
#
# 8. The entire RAGSystem class wire-up (200+ lines)
#    → One LCEL chain:
#      retriever | prompt | llm | StrOutputParser()
#    WHY: Readable, debuggable, and extensible in 4 lines.
#
# ============================================================
# INSTALL:
#   pip install streamlit langchain langchain-community
#               langchain-openai langchain-chroma
#               sentence-transformers chromadb
#               pypdf python-docx
# ============================================================

import os
import time
import tempfile
import streamlit as st

# ── LangChain: Document Loaders ──────────────────────────────
# Replaces: DocumentProcessor (PyPDF2, python-docx, manual txt)
# These loaders return LangChain Document objects with content
# + metadata (source, page number) automatically attached.
from langchain_community.document_loaders import (
    PyPDFLoader,        # PDF → one Document per page
    Docx2txtLoader,     # DOCX → full text as Document
    TextLoader,         # TXT → full text as Document
)

# ── LangChain: Text Splitter ─────────────────────────────────
# Replaces: chunk_text() — manual word-count split
# RecursiveCharacterTextSplitter tries to split on paragraphs,
# then sentences, then words — preserving semantic boundaries.
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ── LangChain: Embeddings ────────────────────────────────────
# Replaces: SimpleEmbedding (TF-IDF) + SentenceTransformer calls
# HuggingFaceEmbeddings wraps sentence-transformers with a
# standard interface. Swap to OpenAIEmbeddings in one line.
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings  # Optional upgrade

# ── LangChain: Vector Store ──────────────────────────────────
# Replaces: SimpleVectorDB + manual np.argsort cosine similarity
# Chroma integration handles add, query, dedup, batching.
# .as_retriever() returns a Retriever object the chain uses.
from langchain_chroma import Chroma

# ── LangChain: Prompt Template ───────────────────────────────
# Replaces: build_messages() — manual f-string prompt builder
# ChatPromptTemplate is a reusable, inspectable prompt object.
from langchain_core.prompts import ChatPromptTemplate

# ── LangChain: LLM / Chat Models ────────────────────────────
# Replaces: LLMClient (raw requests.post to OpenAI/Ollama)
# Standard interface — swap providers by changing the import.
from langchain_openai import ChatOpenAI
from langchain_community.llms import Ollama  # for local models

# ── LangChain: Output Parser ─────────────────────────────────
# Converts the ChatMessage object the LLM returns into a
# plain Python string. Part of the LCEL chain.
from langchain_core.output_parsers import StrOutputParser

# ── LangChain: Chain Primitives ──────────────────────────────
# RunnablePassthrough passes the question through unchanged
# so it arrives at the prompt alongside the retrieved context.
from langchain_core.runnables import RunnablePassthrough

# ── LangChain: Document formatting helper ────────────────────
from langchain_core.documents import Document


# ============================================================
# Streamlit page config & styles
# ============================================================
st.set_page_config(
    page_title="Smart Document Q&A — LangChain RAG",
    page_icon="🦜",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main-header { font-size:2.2rem; color:#1e3a8a; text-align:center;
                   margin-bottom:1.2rem; font-weight:800; }
    .lc-badge { display:inline-block; background:#f0f9ff; color:#0369a1;
                border:1px solid #bae6fd; border-radius:6px;
                padding:2px 8px; font-size:0.75rem; margin:2px; }
    .benefit-box { padding:0.8rem; background:#f0fdf4;
                   border-left:4px solid #16a34a; border-radius:8px; margin:0.5rem 0; }
    .replaced-box { padding:0.8rem; background:#fef9c3;
                    border-left:4px solid #ca8a04; border-radius:8px; margin:0.5rem 0; }
    .chat-q { padding:0.7rem; background:#f1f5f9;
              border-left:4px solid #3b82f6; border-radius:8px; margin:0.4rem 0; }
    .chat-a { padding:0.7rem; background:#f0fdf4;
              border-left:4px solid #22c55e; border-radius:8px; margin:0.4rem 0; }
    .chain-box { font-family:monospace; background:#1e293b; color:#7dd3fc;
                 padding:1rem; border-radius:10px; font-size:0.85rem; margin:0.5rem 0; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# The RAG Prompt — replaces build_messages()
# ============================================================
# WHY ChatPromptTemplate instead of f-strings:
#  - Variables ({context}, {question}) are declared explicitly
#  - Template is reusable and testable independently
#  - Easy to version, share, or load from LangChain Hub
#  - Works with any LLM backend without changes

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system",
     "You are a precise assistant that answers ONLY from the provided context. "
     "Always cite sources by referencing the document name. "
     "If the answer is not in the context, say so clearly. "
     "Be concise: max 5 bullet points or 5 short sentences."),
    ("human",
     "Context:\n{context}\n\n"
     "Question: {question}\n\n"
     "Answer using only the context above. Cite the source document inline."),
])


# ============================================================
# Helper: format retrieved docs for the prompt
# ============================================================
def format_docs(docs: list[Document]) -> str:
    """
    LangChain retrievers return Document objects with .page_content
    and .metadata. We format them with source tags so the LLM
    can cite them inline — same idea as your original [file#chunk]
    tags, but cleaner and built from LangChain metadata.
    """
    parts = []
    for doc in docs:
        source = doc.metadata.get("source", "unknown")
        page   = doc.metadata.get("page", "")
        tag    = f"[{os.path.basename(source)}" + (f" p.{page}]" if page != "" else "]")
        parts.append(f"{tag}\n{doc.page_content}")
    return "\n\n".join(parts)


# ============================================================
# Session state
# ============================================================
if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None   # Chroma instance
if "history"     not in st.session_state:
    st.session_state.history = []
if "last_docs"   not in st.session_state:
    st.session_state.last_docs = []


# ============================================================
# Sidebar — Upload + Settings
# ============================================================
with st.sidebar:
    st.header("📄 Upload Documents")

    uploaded = st.file_uploader(
        "PDF / DOCX / TXT",
        type=["pdf", "docx", "txt"],
        accept_multiple_files=True,
    )

    if uploaded and st.button("Process Documents", type="primary"):
        with st.spinner("Loading + chunking + embedding…"):

            all_docs = []

            for f in uploaded:
                # Save to a temp file so LangChain loaders can open it
                suffix = os.path.splitext(f.name)[1]
                with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                    tmp.write(f.read())
                    tmp_path = tmp.name

                # ── LangChain Loaders ──────────────────────────
                # BEFORE: if name.endswith(".pdf"): PyPDF2.PdfReader(f)...
                # AFTER:  one loader per file type, consistent interface
                # BENEFIT: metadata (source, page) attached automatically;
                #          handles encoding, errors, and multi-page docs
                try:
                    if f.name.lower().endswith(".pdf"):
                        loader = PyPDFLoader(tmp_path)
                    elif f.name.lower().endswith(".docx"):
                        loader = Docx2txtLoader(tmp_path)
                    else:
                        loader = TextLoader(tmp_path, encoding="utf-8")

                    docs = loader.load()

                    # Fix the source metadata to show the original filename
                    for doc in docs:
                        doc.metadata["source"] = f.name

                    all_docs.extend(docs)
                    st.success(f"✓ Loaded: {f.name} ({len(docs)} page/sections)")
                except Exception as e:
                    st.error(f"Error loading {f.name}: {e}")

            if all_docs:
                # ── LangChain Text Splitter ────────────────────
                # BEFORE: chunk_text() split on word count with
                #         manual overlap logic (fragile)
                # AFTER:  RecursiveCharacterTextSplitter — tries
                #         "\n\n", "\n", " ", "" in order
                # BENEFIT: Preserves paragraph/sentence boundaries
                #          = better retrieval context quality
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=600,         # chars (not words)
                    chunk_overlap=100,
                    add_start_index=True,   # adds char offset to metadata
                )
                chunks = splitter.split_documents(all_docs)
                st.info(f"Split into {len(chunks)} chunks")

                # ── LangChain Embeddings ───────────────────────
                # BEFORE: SimpleEmbedding (TF-IDF) or
                #         SentenceTransformer called manually
                # AFTER:  HuggingFaceEmbeddings — same model,
                #         standardized interface
                # BENEFIT: Swap to OpenAIEmbeddings with one line:
                #   embeddings = OpenAIEmbeddings(api_key=...)
                embeddings = HuggingFaceEmbeddings(
                    model_name="all-MiniLM-L6-v2",
                    model_kwargs={"device": "cpu"},
                )

                # ── LangChain Vector Store ─────────────────────
                # BEFORE: SimpleVectorDB.add() + manual np.argsort
                #         cosine similarity loop
                # AFTER:  Chroma.from_documents() — one call to
                #         embed + store all chunks
                # BENEFIT: Built-in similarity search, filtering,
                #          metadata queries, persistence option
                st.session_state.vectorstore = Chroma.from_documents(
                    documents=chunks,
                    embedding=embeddings,
                )
                st.markdown(
                    "<div class='benefit-box'>✅ <b>LangChain Chroma</b> — "
                    f"{len(chunks)} chunks embedded and indexed.</div>",
                    unsafe_allow_html=True,
                )

    st.header("🧹 Maintenance")
    if st.button("Reset"):
        st.session_state.vectorstore = None
        st.session_state.history = []
        st.session_state.last_docs = []
        st.success("Reset done.")

    st.header("⚙️ Retrieval Settings")
    top_k = st.slider("Retrieved Chunks (k)", 3, 15, 5)

    st.header("🧠 LLM Settings")
    provider = st.selectbox("Provider", ["OpenAI", "Ollama"])
    if provider == "OpenAI":
        api_key   = st.text_input("OpenAI API Key", type="password",
                                  value=os.getenv("OPENAI_API_KEY", ""))
        model     = st.text_input("Model", value="gpt-4o-mini")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    else:
        api_key     = ""
        model       = st.text_input("Ollama Model", value="llama3.1:8b")
        temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)

    if st.button("Clear Chat"):
        st.session_state.history = []


# ============================================================
# Main — Chat Interface
# ============================================================
st.markdown('<div class="main-header">🦜 Smart Document Q&A — LangChain RAG</div>',
            unsafe_allow_html=True)

# Status badges
c1, c2, c3, c4 = st.columns(4)
with c1: st.metric("Loader",    "LangChain")
with c2: st.metric("Splitter",  "Recursive")
with c3: st.metric("Embeddings","MiniLM-L6")
with c4: st.metric("Chain",     "LCEL")

left, right = st.columns([2, 1])

with left:
    st.subheader("💬 Ask Your Documents")

    if st.session_state.vectorstore:

        # Show chat history
        for q, a in st.session_state.history:
            st.markdown(f"<div class='chat-q'><b>You:</b> {q}</div>",
                        unsafe_allow_html=True)
            st.markdown(f"<div class='chat-a'><b>AI:</b> {a}</div>",
                        unsafe_allow_html=True)

        query = st.text_input("Ask a question…", key="q_input")

        if st.button("Ask", type="primary") and query:
            with st.spinner("Retrieving + Generating…"):
                t0 = time.time()

                # ── LangChain Retriever ────────────────────────
                # BEFORE: rag.retrieve() — manual embed query,
                #         cosine similarity loop, np.argsort
                # AFTER:  vectorstore.as_retriever() — one object
                #         that plugs directly into the chain
                # BENEFIT: Supports MMR, similarity threshold,
                #          metadata filtering — all via kwargs
                retriever = st.session_state.vectorstore.as_retriever(
                    search_type="similarity",
                    search_kwargs={"k": top_k},
                )

                # ── LangChain LLM ──────────────────────────────
                # BEFORE: raw requests.post() to OpenAI/Ollama
                #         with manual header/payload building
                # AFTER:  ChatOpenAI / Ollama — standardized,
                #         with built-in retry and streaming
                # BENEFIT: Swap providers by changing one import
                if provider == "OpenAI":
                    llm = ChatOpenAI(
                        model=model,
                        temperature=temperature,
                        api_key=api_key,
                    )
                else:
                    llm = Ollama(model=model, temperature=temperature)

                # ── LangChain LCEL Chain ───────────────────────
                # BEFORE: 200+ lines across RAGSystem methods:
                #   retrieve() → build_messages() → llm_answer()
                #   → enforce_strict_answer()
                #
                # AFTER: 4 lines using the pipe (|) operator:
                #
                #   retriever  → fetches relevant chunks
                #   format_docs → formats them for the prompt
                #   RAG_PROMPT → fills {context} + {question}
                #   llm        → generates the answer
                #   StrOutputParser → extracts the text string
                #
                # BENEFIT:
                #  - Each step is independently testable
                #  - Add streaming with .stream() instead of .invoke()
                #  - Add memory with ConversationBufferMemory
                #  - Add agents with .bind_tools()
                #  - Inspect any step with .steps attribute
                rag_chain = (
                    {
                        "context":  retriever | format_docs,
                        "question": RunnablePassthrough(),
                    }
                    | RAG_PROMPT
                    | llm
                    | StrOutputParser()
                )

                try:
                    answer = rag_chain.invoke(query)
                    # Also fetch docs so we can show them in the panel
                    st.session_state.last_docs = retriever.invoke(query)
                except Exception as e:
                    answer = f"Error: {e}"
                    st.session_state.last_docs = []

                elapsed = time.time() - t0
                st.session_state.history.append((query, answer))
                st.success(f"Done in {elapsed:.2f}s")
                st.rerun()

    else:
        st.markdown("""
        <div style='padding:1rem;background:#dbeafe;border-left:4px solid #3b82f6;border-radius:8px'>
            <b>Welcome!</b> Upload documents from the sidebar to start.<br><br>
            <b>What LangChain does here:</b>
            <ul>
                <li>📥 <b>Loaders</b> — read PDF/DOCX/TXT with one line each</li>
                <li>✂️ <b>Splitter</b> — sentence-aware chunking</li>
                <li>🔢 <b>Embeddings</b> — swap models in one line</li>
                <li>🗄️ <b>Chroma</b> — vector search without manual math</li>
                <li>🔗 <b>LCEL Chain</b> — entire pipeline in 4 lines</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

with right:
    st.subheader("📊 Retrieved Chunks")

    if st.session_state.last_docs:
        st.caption(f"{len(st.session_state.last_docs)} chunks used in last answer")
        for doc in st.session_state.last_docs:
            src  = doc.metadata.get("source", "unknown")
            page = doc.metadata.get("page", "")
            label = f"{os.path.basename(src)}" + (f" · p.{page}" if page != "" else "")
            with st.expander(label):
                st.code(doc.page_content[:1000] +
                        ("…" if len(doc.page_content) > 1000 else ""))
    else:
        st.info("Ask a question to see retrieved chunks here.")

    st.subheader("🔗 The LangChain Chain")
    st.markdown("""
    <div class='chain-box'>
retriever | format_docs<br>
&nbsp;&nbsp;→ RAG_PROMPT<br>
&nbsp;&nbsp;&nbsp;&nbsp;→ llm<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;→ StrOutputParser()
    </div>
    """, unsafe_allow_html=True)

    st.subheader("✨ LangChain Benefits")
    benefits = [
        ("🔄", "Swap LLM",        "ChatOpenAI → ChatOllama in 1 line"),
        ("🔄", "Swap embeddings", "HuggingFace → OpenAI in 1 line"),
        ("🔄", "Swap vector DB",  "Chroma → Pinecone in 1 line"),
        ("📡", "Streaming",       ".stream() instead of .invoke()"),
        ("🧠", "Memory",          "Add ConversationBufferMemory"),
        ("🤖", "Agents",          "Add tools with .bind_tools()"),
        ("🐛", "Debug",           "LangSmith traces every step"),
    ]
    for icon, title, detail in benefits:
        st.markdown(
            f"**{icon} {title}** — <span style='color:gray;font-size:0.85rem'>{detail}</span>",
            unsafe_allow_html=True,
        )
