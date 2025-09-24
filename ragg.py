# Smart Document Q&A System with RAG + LLM (Upgraded)
# - Keeps your original fallbacks (TF‑IDF + in‑memory DB)
# - Adds real LLM answer generation (OpenAI or Ollama)
# - Clean prompt construction with citations and safeguards
# - Minimal extra deps (uses requests for HTTP)

import streamlit as st
import os
import time
import hashlib
import re
from datetime import datetime
from typing import List, Dict, Any

import numpy as np

# Optional dependencies (same as your original design)
try:
    import chromadb
    CHROMADB_AVAILABLE = True
except Exception:
    CHROMADB_AVAILABLE = False

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except Exception:
    SENTENCE_TRANSFORMERS_AVAILABLE = False

import PyPDF2
from docx import Document
import requests

# -----------------------------
# Streamlit Config & Styles
# -----------------------------
st.set_page_config(
    page_title="Smart Document Q&A (RAG + LLM)",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    .main-header { font-size: 2.2rem; color: #1e3a8a; text-align: center; margin-bottom: 1.2rem; font-weight: 800; }
    .feature-box { padding: 0.8rem; border-radius: 12px; border: 1px solid #e5e7eb; margin: 0.8rem 0; background-color: #f9fafb; }
    .success-box { padding: 0.8rem; background-color: #dcfce7; border-left: 4px solid #16a34a; border-radius: 8px; margin: 0.6rem 0; }
    .info-box { padding: 0.8rem; background-color: #dbeafe; border-left: 4px solid #3b82f6; border-radius: 8px; margin: 0.6rem 0; }
    .chat-message { padding: 0.8rem; margin: 0.6rem 0; border-radius: 10px; background-color: #f8fafc; border-left: 4px solid #3b82f6; }
    .answer-box { padding: 0.8rem; margin: 0.6rem 0; border-radius: 10px; background-color: #f0fdf4; border-left: 4px solid #22c55e; }
    code { white-space: pre-wrap; }
</style>
""",
    unsafe_allow_html=True,
)

# -----------------------------
# Embedding (Fallback TF‑IDF)
# -----------------------------
class SimpleEmbedding:
    def __init__(self):
        self.vocab = {}
        self.idf = {}
        self.vocab_size = 0

    def fit(self, texts):
        word_counts = {}
        doc_counts = {}
        for text in texts:
            words = self._tok(text)
            uniq = set(words)
            for w in words:
                word_counts[w] = word_counts.get(w, 0) + 1
            for w in uniq:
                doc_counts[w] = doc_counts.get(w, 0) + 1
        self.vocab = {w: i for i, w in enumerate(word_counts.keys())}
        self.vocab_size = len(self.vocab)
        N = max(1, len(texts))
        for w, c in doc_counts.items():
            self.idf[w] = np.log(N / c)

    def encode(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        vecs = []
        for t in texts:
            v = np.zeros(self.vocab_size)
            words = self._tok(t)
            counts = {}
            for w in words:
                if w in self.vocab:
                    counts[w] = counts.get(w, 0) + 1
            for w, c in counts.items():
                if w in self.vocab:
                    tf = c / max(1, len(words))
                    idf = self.idf.get(w, 0)
                    v[self.vocab[w]] = tf * idf
            n = np.linalg.norm(v)
            if n > 0:
                v = v / n
            vecs.append(v)
        return np.array(vecs)

    def _tok(self, text):
        text = text.lower()
        text = re.sub(r"[^\w\s]", "", text)
        return text.split()

# -----------------------------
# Document I/O
# -----------------------------
class DocumentProcessor:
    @staticmethod
    def pdf(pdf_file):
        try:
            reader = PyPDF2.PdfReader(pdf_file)
            text = "\n".join([p.extract_text() or "" for p in reader.pages])
            return text
        except Exception as e:
            st.error(f"PDF read error: {e}")
            return ""

    @staticmethod
    def docx(docx_file):
        try:
            doc = Document(docx_file)
            return "\n".join([p.text for p in doc.paragraphs])
        except Exception as e:
            st.error(f"DOCX read error: {e}")
            return ""

    @staticmethod
    def txt(txt_file):
        try:
            data = txt_file.read()
            if isinstance(data, bytes):
                return data.decode("utf-8", errors="replace")
            return str(data)
        except Exception as e:
            st.error(f"TXT read error: {e}")
            return ""

# -----------------------------
# Simple Vector DB (fallback)
# -----------------------------
class SimpleVectorDB:
    def __init__(self):
        self.documents = []
        self.embeddings = []
        self.metadata = []
        self.ids = []
        self._id_index = {}

    def add(self, documents, embeddings, metadatas, ids):
        for doc, emb, meta, cid in zip(documents, embeddings, metadatas, ids):
            if cid in self._id_index:
                continue
            self.documents.append(doc)
            self.embeddings.append(emb)
            self.metadata.append(meta)
            self.ids.append(cid)
            self._id_index[cid] = len(self.ids) - 1

    def clear(self):
        self.documents = []
        self.embeddings = []
        self.metadata = []
        self.ids = []
        self._id_index = {}

    def set_all(self, documents, embeddings, metadatas, ids):
        self.documents = list(documents)
        self.embeddings = list(embeddings)
        self.metadata = list(metadatas)
        self.ids = list(ids)
        self._id_index = {cid: i for i, cid in enumerate(self.ids)}

    def query(self, query_embeddings, n_results=5):
        if not self.embeddings:
            return {"documents": [[]], "metadatas": [[]], "distances": [[]]}
        q = np.array(query_embeddings[0])
        sims = []
        for e in self.embeddings:
            e = np.array(e)
            denom = (np.linalg.norm(q) * np.linalg.norm(e))
            sim = float(np.dot(q, e) / denom) if denom else 0.0
            sims.append(1 - sim)  # smaller is better
        idx = np.argsort(sims)[:n_results]
        return {
            "documents": [[self.documents[i] for i in idx]],
            "metadatas": [[self.metadata[i] for i in idx]],
            "distances": [[float(sims[i]) for i in idx]],
        }

# -----------------------------
# LLM Adapters
# -----------------------------
class LLMClient:
    @staticmethod
    def call_openai(model: str, api_key: str, messages: List[Dict[str, str]], temperature: float = 0.2, max_tokens: int = 700) -> str:
        if not api_key:
            raise RuntimeError("OpenAI API key not provided")
        url = "https://api.openai.com/v1/chat/completions"
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        r = requests.post(url, headers=headers, json=payload, timeout=60)
        if r.status_code >= 400:
            raise RuntimeError(f"OpenAI error: {r.status_code} — {r.text[:300]}")
        data = r.json()
        return data["choices"][0]["message"]["content"].strip()

    @staticmethod
    def call_ollama(model: str, messages: List[Dict[str, str]], temperature: float = 0.2, max_tokens: int = 700) -> str:
        # Requires local Ollama server running on 11434
        url = "http://localhost:11434/api/chat"
        payload = {
            "model": model,
            "messages": messages,
            "options": {"temperature": temperature, "num_predict": max_tokens},
            "stream": False,
        }
        r = requests.post(url, json=payload, timeout=120)
        if r.status_code >= 400:
            raise RuntimeError(f"Ollama error: {r.status_code} — {r.text[:300]}")
        data = r.json()
        return data.get("message", {}).get("content", "").strip()

# -----------------------------
# RAG System (with LLM answers)
# -----------------------------
class RAGSystem:
    def __init__(self):
        self.embedding_model = None
        self.vector_db = None
        self.chunk_size = 200  # Reduced from 500 for more focused chunks
        self.chunk_overlap = 50
        self.use_simple_embedding = not SENTENCE_TRANSFORMERS_AVAILABLE

    # Embeddings
    def load_embedding_model(self):
        if SENTENCE_TRANSFORMERS_AVAILABLE:
            try:
                return SentenceTransformer("all-MiniLM-L6-v2")
            except Exception as e:
                st.warning(f"SentenceTransformer load failed, using TF-IDF: {e}")
                self.use_simple_embedding = True
        if self.use_simple_embedding:
            return SimpleEmbedding()

    # Vector DB
    def initialize_vector_db(self):
        if CHROMADB_AVAILABLE:
            try:
                client = chromadb.Client()
                col = client.get_or_create_collection(name="documents", metadata={"hnsw:space": "cosine"})
                return col
            except Exception as e:
                st.warning(f"Chroma init failed, using in-memory: {e}")
        return SimpleVectorDB()

    def reset_index(self):
        # Reset vector DB; keep sentence-transformer model if available
        self.vector_db = self.initialize_vector_db()
        if isinstance(self.embedding_model, SimpleEmbedding):
            # Force refit on next add
            self.embedding_model = None

    # Chunking
    def chunk_text(self, text: str) -> List[str]:
        words = text.split()
        chunks = []
        step = max(1, self.chunk_size - self.chunk_overlap)
        for i in range(0, len(words), step):
            chunk = " ".join(words[i : i + self.chunk_size])
            if chunk.strip():
                chunks.append(chunk.strip())
        return chunks

    # Ingest
    def add_documents(self, documents: List[str], filenames: List[str]):
        if not self.embedding_model:
            self.embedding_model = self.load_embedding_model()
        if not self.vector_db:
            self.vector_db = self.initialize_vector_db()

        all_chunks, all_metadata, all_ids = [], [], []
        for idx, (doc, fname) in enumerate(zip(documents, filenames)):
            chunks = self.chunk_text(doc)
            for j, ch in enumerate(chunks):
                cid = f"{fname}_{idx}_{j}_{hashlib.md5(ch.encode()).hexdigest()[:8]}"
                all_chunks.append(ch)
                all_metadata.append({"filename": fname, "chunk_index": j, "timestamp": datetime.now().isoformat()})
                all_ids.append(cid)

        # Create/store embeddings
        if isinstance(self.embedding_model, SimpleEmbedding) and isinstance(self.vector_db, SimpleVectorDB):
            # Refit TF-IDF on full corpus (existing + new), then replace DB to keep vector spaces aligned
            corpus_docs = list(self.vector_db.documents) + all_chunks
            corpus_meta = list(self.vector_db.metadata) + all_metadata
            corpus_ids = list(self.vector_db.ids) + all_ids
            self.embedding_model.fit(corpus_docs)
            corpus_embeds = self.embedding_model.encode(corpus_docs).tolist()
            self.vector_db.set_all(documents=corpus_docs, embeddings=corpus_embeds, metadatas=corpus_meta, ids=corpus_ids)
            return len(all_chunks)
        else:
            embeds = self.embedding_model.encode(all_chunks).tolist()
            self.vector_db.add(documents=all_chunks, embeddings=embeds, metadatas=all_metadata, ids=all_ids)
            return len(all_chunks)

    # Retrieve
    def retrieve(self, query: str, n_results: int = 5) -> List[Dict[str, Any]]:
        if not self.embedding_model or not self.vector_db:
            return []
        q_embed = self.embedding_model.encode([query])
        if isinstance(q_embed, np.ndarray):
            q_embed = q_embed.tolist()
        res = self.vector_db.query(query_embeddings=q_embed, n_results=n_results)
        out = []
        for i in range(len(res["documents"][0])):
            out.append({
                "content": res["documents"][0][i],
                "metadata": res["metadatas"][0][i],
                "distance": res["distances"][0][i] if "distances" in res else 0.0,
            })
        return out

    # Build LLM prompt
    @staticmethod
    def build_messages(query: str, chunks: List[Dict[str, Any]], max_context_chars: int = 8000) -> List[Dict[str, str]]:
        # Compile context with lightweight citation tags
        ctx_parts = []
        used = 0
        for idx, ch in enumerate(chunks):
            tag = f"[{ch['metadata']['filename']}#{ch['metadata']['chunk_index']}]"
            block = f"{tag}\n{ch['content']}"
            if used + len(block) > max_context_chars:
                break
            ctx_parts.append(block)
            used += len(block)
        context = "\n\n".join(ctx_parts) if ctx_parts else "(no context available)"

        system = (
            "You are a strict, precise assistant that answers ONLY the specific question using the provided context. "
            "Do NOT summarize unrelated information. "
            "Cite sources inline using their tags like [file#chunk]. "
            "If the answer is not in the context, say you don't have that information. "
            "Prefer concise, factual responses. "
            "If asked about certifications, only mention certifications and nothing else."
        )
        user = (
            f"Question: {query}\n\n"
            f"Context:\n{context}\n\n"
            "Instructions:\n"
            "- Only answer the specific question asked.\n"
            "- Do NOT include unrelated details.\n"
            "- Use bullet points or short paragraphs.\n"
            "- Include citations [file#chunk] after the sentences that use them.\n"
            "- If multiple chunks agree, you can cite multiple tags.\n"
        )
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    # Call LLM
    @staticmethod
    def llm_answer(provider: str, model: str, api_key: str, messages: List[Dict[str, str]], temperature: float, max_tokens: int) -> str:
        if provider == "OpenAI":
            return LLMClient.call_openai(model=model, api_key=api_key, messages=messages, temperature=temperature, max_tokens=max_tokens)
        elif provider == "Ollama":
            return LLMClient.call_ollama(model=model, messages=messages, temperature=temperature, max_tokens=max_tokens)
        else:
            raise RuntimeError("Unsupported LLM provider")

# -----------------------------
# App State
# -----------------------------
if "rag" not in st.session_state:
    st.session_state.rag = RAGSystem()
if "docs_loaded" not in st.session_state:
    st.session_state.docs_loaded = False
if "history" not in st.session_state:
    st.session_state.history = []
if "last_retrieved" not in st.session_state:
    st.session_state.last_retrieved = []

# -----------------------------
# Header & Status
# -----------------------------
st.markdown('<div class="main-header">🤖 Smart Document Q&A — RAG + LLM</div>', unsafe_allow_html=True)
colA, colB, colC, colD = st.columns(4)
with colA:
    st.metric("Embeddings", "Sentence-TX" if SENTENCE_TRANSFORMERS_AVAILABLE else "TF‑IDF Fallback")
with colB:
    st.metric("Vector DB", "Chroma" if CHROMADB_AVAILABLE else "In‑Memory")
with colC:
    st.metric("LLM", "On")
with colD:
    st.metric("Status", "🟢 Ready")

# -----------------------------
# Sidebar — Upload & Settings
# -----------------------------
with st.sidebar:
    st.header("📄 Upload Documents")
    files = st.file_uploader("Upload PDF / DOCX / TXT", type=["pdf", "docx", "txt"], accept_multiple_files=True)
    if files and st.button("Process Documents", type="primary"):
        with st.spinner("Processing…"):
            docs, fnames = [], []
            for f in files:
                name = f.name
                text = ""
                if name.lower().endswith(".pdf"):
                    text = DocumentProcessor.pdf(f)
                elif name.lower().endswith(".docx"):
                    text = DocumentProcessor.docx(f)
                elif name.lower().endswith(".txt"):
                    text = DocumentProcessor.txt(f)
                if text.strip():
                    docs.append(text)
                    fnames.append(name)
                    st.success(f"Processed: {name}")
                else:
                    st.warning(f"No text extracted: {name}")
            if docs:
                n_chunks = st.session_state.rag.add_documents(docs, fnames)
                st.session_state.docs_loaded = True
                st.markdown(f"<div class='success-box'><b>Ingested:</b> {len(docs)} docs → {n_chunks} chunks</div>", unsafe_allow_html=True)

    st.header("🧹 Maintenance")
    if st.button("Reset Index"):
        st.session_state.rag.reset_index()
        st.session_state.docs_loaded = False
        st.session_state.history = []
        st.session_state.last_retrieved = []
        st.success("Index, context and history reset.")

    st.header("⚙️ Retrieval Settings")
    top_k = st.slider("Retrieved Chunks", 3, 15, 5)
    max_ctx = st.slider("Max Context (chars)", 2000, 16000, 8000, step=1000)

    st.header("🧠 LLM Settings")
    provider = st.selectbox("Provider", ["OpenAI", "Ollama"], index=0)
    if provider == "OpenAI":
        default_model = "gpt-4o-mini"
        api_key = st.text_input("OPENAI_API_KEY", value=os.getenv("OPENAI_API_KEY", ""), type="password")
    else:
        default_model = "llama3.1:8b"
        api_key = ""  # Not used for Ollama
    model = st.text_input("Model", value=default_model)
    temperature = st.slider("Temperature", 0.0, 1.0, 0.2, 0.05)
    max_tokens = st.slider("Max Tokens", 256, 4096, 700, 64)

    if st.button("Clear Chat History"):
        st.session_state.history = []
        st.success("Chat history cleared.")

# -----------------------------
# Main — Chat
# -----------------------------
left, right = st.columns([2, 1])
with left:
    st.subheader("💬 Ask Questions")

    if st.session_state.docs_loaded:
        for q, a in st.session_state.history:
            st.markdown(f"<div class='chat-message'><b>You:</b> {q}</div>", unsafe_allow_html=True)
            st.markdown(f"<div class='answer-box'><b>AI:</b> {a}</div>", unsafe_allow_html=True)

        query = st.text_input("Type your question…", key="q_input", placeholder="e.g., Summarize the key findings and cite sources")
        if st.button("Ask", type="primary") and query:
            with st.spinner("Retrieving + Generating…"):
                t0 = time.time()
                chunks = st.session_state.rag.retrieve(query, n_results=top_k)
                st.session_state.last_retrieved = chunks
                messages = RAGSystem.build_messages(query, chunks, max_context_chars=max_ctx)
                try:
                    answer = RAGSystem.llm_answer(provider, model, api_key, messages, temperature, max_tokens)
                except Exception as e:
                    # Fallback: minimal extractive answer if LLM fails
                    # Take top sentences by keyword overlap
                    qw = set(query.lower().split())
                    sent_scores = []
                    for ch in chunks:
                        tag = f"[{ch['metadata']['filename']}#{ch['metadata']['chunk_index']}]"
                        for s in re.split(r"(?<=[.!?])\s+", ch["content"]):
                            if len(s) < 20:
                                continue
                            sc = len(qw.intersection(set(s.lower().split())))
                            if sc > 0:
                                sent_scores.append((sc, s.strip(), tag))
                    sent_scores.sort(key=lambda x: x[0], reverse=True)
                    tops = sent_scores[:4]
                    if tops:
                        answer = "\n".join([f"- {s} {tag}" for _, s, tag in tops])
                    else:
                        answer = "I couldn't find enough context to answer. Try rephrasing or uploading richer documents."
                    answer += "\n\n_(LLM failed; provided extractive fallback.)_"

                t1 = time.time()
                st.session_state.history.append((query, answer))
                st.success(f"Done in {t1 - t0:.2f}s")
                st.rerun()
    else:
        st.markdown(
            """
            <div class='info-box'>
                <b>Welcome!</b> Upload documents from the sidebar to start asking grounded questions. The app will:
                <ul>
                    <li>Chunk & embed your files</li>
                    <li>Retrieve top‑K relevant chunks</li>
                    <li>Ask an LLM to compose a concise, cited answer</li>
                </ul>
            </div>
            """,
            unsafe_allow_html=True,
        )

with right:
    st.subheader("📊 Analytics")
    if st.session_state.docs_loaded:
        st.metric("Queries", len(st.session_state.history))
        if st.session_state.history:
            st.write("**Last Question:**", st.session_state.history[-1][0][:80] + ("…" if len(st.session_state.history[-1][0]) > 80 else ""))
            with st.expander("View last retrieved chunks"):
                for i, ch in enumerate(st.session_state.last_retrieved or []):
                    tag = f"[{ch['metadata']['filename']}#{ch['metadata']['chunk_index']}]"
                    st.caption(tag)
                    st.code(ch["content"][:1200] + ("…" if len(ch["content"]) > 1200 else ""))
    else:
        st.info("Upload docs to view analytics")

    st.subheader("✨ Tips")
    st.markdown(
        "- Be specific in questions\n"
        "- Ask for summaries, comparisons, timelines\n"
        "- Check the citations like [file#chunk]\n"
        "- Increase Top‑K if recall seems low"
    )
