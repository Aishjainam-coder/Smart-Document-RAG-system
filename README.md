## Smart Document Q&A (RAG + LLM)

A simple, fast Streamlit app to ask grounded questions about your documents. It chunks, embeds, retrieves, and lets an LLM compose concise answers with inline citations.

### Features
- **Multi‑format ingest**: PDF, DOCX, TXT
- **RAG pipeline**: chunk → embed → retrieve → answer with citations
- **Embeddings**: Sentence-Transformers (preferred) or TF‑IDF fallback
- **Vector DB**: Chroma (if installed) or in‑memory fallback
- **LLMs**: OpenAI API or local Ollama
- **DX**: Reset index button, view last retrieved chunks, extractive fallback if LLM fails

### Requirements
- Python 3.9+
- pip

### Install
```bash
python -m venv .venv
".venv\\Scripts\\activate"  # Windows PowerShell

pip install --upgrade pip
pip install streamlit numpy requests PyPDF2 python-docx
# Optional (recommended)
pip install sentence-transformers chromadb
```

Alternatively, add a `requirements.txt` and install via `pip install -r requirements.txt`.

### Environment (optional)
- For OpenAI: set `OPENAI_API_KEY`
```powershell
$env:OPENAI_API_KEY = "sk-..."
```
- For Ollama: install and run Ollama locally (`http://localhost:11434`). Example model: `llama3.1:8b`.

### Run
```bash
streamlit run ragg.py
```

### Usage
1. Open the app URL printed by Streamlit.
2. In the sidebar, upload PDF/DOCX/TXT files, then click "Process Documents".
3. Adjust retrieval (Top‑K, Max Context) and LLM settings (provider, model, temperature, max tokens).
4. Ask a question. Answers include inline citations like `[file#chunk]`.
5. Use "Reset Index" to clear all vectors/history and start fresh.
6. See "View last retrieved chunks" in Analytics to inspect context used.

### LLM Options
- **OpenAI**: pick a chat model (e.g., `gpt-4o-mini`) and provide `OPENAI_API_KEY`.
- **Ollama**: choose a local model (e.g., `llama3.1:8b`); no API key required.

### Notes & Tips
- If you don’t install `sentence-transformers`, the app uses a TF‑IDF fallback.
- With TF‑IDF, embeddings are refit across the full corpus on each new ingest to keep vector spaces aligned.
- PDFs with no extractable text may require OCR (not included).

### Troubleshooting
- Blank answers or missing citations: increase Top‑K or Max Context.
- OpenAI errors: verify `OPENAI_API_KEY` and model name.
- Ollama errors: ensure the server is running (`ollama serve`) and the model is pulled.
- Windows path issues: run PowerShell as admin if needed and activate the venv before `streamlit run`.

### Push to GitHub
```powershell
cd C:\Users\hi\ragg
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/Aishjainam-coder/Smart-Document-RAG-system.git
git push -u origin main
```


