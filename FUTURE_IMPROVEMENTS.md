# Future Improvements — Document-Aware AI Assistant

> This document lives on the `meta` branch and is updated alongside every feature development on `dev`.
> It is never merged into `dev` or `main`.

---

## Branch Strategy

| Branch | Purpose |
|---|---|
| `main` | Production-ready, stable code only |
| `dev` | Active development, all features merged here first |
| `meta` | Documentation-only branch — tracks all improvements, never merged |

**Meta branch update rule:** Every time a new feature lands on `dev`, update this document on `meta` to reflect its status.

---

## Improvement Roadmap

### Status Legend
| Symbol | Meaning |
|---|---|
| `[ ]` | Not started |
| `[~]` | In progress |
| `[x]` | Completed |

---

## 1. LLM & AI Enhancements

### 1.1 Streaming Responses
- **Current state:** `ask_streaming()` in `chat_engine.py:137` is a stub — only `pass`
- **Improvement:** Implement real token-by-token streaming in the Streamlit UI using `st.write_stream()`
- **Impact:** Much better UX — user sees the answer being typed rather than waiting for full response
- **Status:** `[ ]`

### 1.2 Multi-LLM Support
- **Current state:** Hardcoded to `gpt-3.5-turbo` (OpenAI) or `flan-t5-base` (local)
- **Improvement:** Add support for:
  - Claude (Anthropic) — `claude-sonnet-4-6`, `claude-opus-4-6`
  - GPT-4o (OpenAI)
  - Ollama (local models — LLaMA 3, Mistral, Gemma)
  - Google Gemini
- **How:** Abstract LLM selection into a factory pattern with a UI dropdown
- **Status:** `[ ]`

### 1.3 Conversation Memory
- **Current state:** Each question is completely independent — the LLM has no memory of previous turns
- **Improvement:** Add `ConversationalRetrievalChain` with `ConversationBufferMemory` from LangChain
- **Impact:** Users can ask follow-up questions like "Can you elaborate on that?" or "What about point 2?"
- **Status:** `[ ]`

### 1.4 Semantic Chunking
- **Current state:** `semantic_chunking()` in `document_processor.py:103` is a placeholder
- **Improvement:** Implement real semantic chunking using sentence boundary detection instead of fixed character counts
- **How:** Use `langchain_experimental.text_splitter.SemanticChunker` with embedding-based splitting
- **Impact:** Better context preservation at chunk boundaries
- **Status:** `[ ]`

### 1.5 Configurable Chain Types
- **Current state:** Hardcoded to `chain_type="stuff"` in `chat_engine.py:91` — all chunks shoved into one prompt
- **Improvement:** Let user select chain strategy via UI:
  - `stuff` — current (best for small docs)
  - `map_reduce` — parallel processing (best for large docs)
  - `refine` — iterative refinement (best for nuanced answers)
  - `map_rerank` — pick best answer from multiple passes
- **Status:** `[ ]`

---

## 2. Document Support Expansion

### 2.1 New File Formats
- **Current state:** Only PDF, TXT, MD supported (`utils.py:52`)
- **Improvement:** Add support for:
  - `.docx` — Microsoft Word (via `python-docx`)
  - `.xlsx` / `.csv` — Spreadsheets (via `pandas`)
  - `.html` — Web pages (via `BeautifulSoup`)
  - `.pptx` — PowerPoint slides (via `python-pptx`)
  - `.json` — Structured data
- **Status:** `[ ]`

### 2.2 OCR for Scanned PDFs
- **Current state:** PyPDF fails on image-based/scanned PDFs with no extractable text
- **Improvement:** Integrate Tesseract OCR via `pytesseract` as a fallback when PDF text extraction returns empty
- **Status:** `[ ]`

### 2.3 URL / Web Page Ingestion
- **Improvement:** Allow users to paste a URL and scrape it as a document source
- **How:** Use `WebBaseLoader` from LangChain
- **Status:** `[ ]`

### 2.4 Duplicate Detection
- **Current state:** MD5 hash check exists in `utils.py` but `INSERT OR REPLACE` in `database.py:88` silently replaces
- **Improvement:** Warn user explicitly when a duplicate file is uploaded, with option to skip or re-process
- **Status:** `[ ]`

---

## 3. Retrieval Quality Improvements

### 3.1 Hybrid Search (Keyword + Semantic)
- **Current state:** Pure vector similarity search only
- **Improvement:** Combine BM25 keyword search with semantic search for better recall
- **How:** Use `EnsembleRetriever` from LangChain combining `BM25Retriever` + `ChromaDB`
- **Impact:** Catches exact keyword matches that semantic search sometimes misses
- **Status:** `[ ]`

### 3.2 Re-Ranking Retrieved Chunks
- **Current state:** Top-K chunks are passed to LLM in raw retrieval order
- **Improvement:** Add a re-ranker (e.g., `cross-encoder/ms-marco-MiniLM-L-6-v2`) to score and reorder chunks by actual relevance to the question
- **Impact:** Higher quality context passed to LLM = better answers
- **Status:** `[ ]`

### 3.3 Configurable K (Retrieval Count)
- **Current state:** `k=4` is hardcoded in `chat_engine.py:93`
- **Improvement:** Expose `k` as a UI slider (e.g., 2–10 chunks) so users can tune recall vs. speed
- **Status:** `[ ]`

### 3.4 Relevance Score Threshold
- **Current state:** All top-K results returned regardless of similarity score
- **Improvement:** Filter out chunks below a minimum similarity threshold (e.g., < 0.3 cosine similarity) to avoid irrelevant context being passed to LLM
- **Status:** `[ ]`

---

## 4. UI / UX Improvements

### 4.1 Fix Broken Sidebar Image
- **Current state:** `app.py:80` loads from `https://via.placeholder.com/...` — an external placeholder URL
- **Improvement:** Replace with a local logo image or remove it
- **Status:** `[ ]`

### 4.2 Fix Hardcoded Document ID
- **Current state:** `document_ids=[1]` hardcoded in `app.py:277` — chat records are not linked to actual documents
- **Improvement:** Track real document IDs returned from `db_manager.add_document_record()` and pass them correctly
- **Status:** `[ ]`

### 4.3 Multi-Session Support
- **Current state:** Only one active session per browser tab; no session history browsing
- **Improvement:** Add a session selector in the sidebar so users can switch between past conversations
- **Status:** `[ ]`

### 4.4 Document Management Panel
- **Improvement:** Add a tab to view all uploaded documents with options to:
  - See chunk counts and file sizes
  - Delete individual documents from the vector store
  - Re-process with different chunk settings
- **Status:** `[ ]`

### 4.5 Answer Confidence Indicator
- **Improvement:** Show a confidence/relevance score alongside each answer based on the similarity scores of retrieved chunks
- **Status:** `[ ]`

### 4.6 Dark Mode Support
- **Improvement:** Streamlit supports theming via `.streamlit/config.toml` — add a dark/light toggle
- **Status:** `[ ]`

---

## 5. Performance & Scalability

### 5.1 Model Caching
- **Current state:** LLM and embedding models reload every time the app restarts
- **Improvement:** Use `@st.cache_resource` decorator to cache model loading across sessions
- **Impact:** Dramatically faster startup after first load
- **Status:** `[ ]`

### 5.2 Async Document Processing
- **Current state:** Documents are processed synchronously, blocking the UI
- **Improvement:** Use background threads or `asyncio` with Streamlit's `st.status()` for non-blocking processing
- **Status:** `[ ]`

### 5.3 Batch Embedding
- **Current state:** Embeddings generated one chunk at a time
- **Improvement:** Use batch embedding API calls for significantly faster ingestion of large documents
- **Status:** `[ ]`

---

## 6. Testing

### 6.1 Unit Tests
- **Current state:** No tests exist
- **Improvement:** Add `pytest` test suite covering:
  - `document_processor.py` — chunking logic, file type handling
  - `database.py` — CRUD operations
  - `utils.py` — hash generation, file validation
- **Status:** `[ ]`

### 6.2 Integration Tests
- **Improvement:** End-to-end test: upload a document → ask a question → verify answer contains expected content
- **Status:** `[ ]`

### 6.3 CI Pipeline
- **Improvement:** Add GitHub Actions workflow (`.github/workflows/ci.yml`) to run tests on every PR to `dev`
- **Status:** `[ ]`

---

## 7. Security & Configuration

### 7.1 User Authentication
- **Improvement:** Add basic authentication using `streamlit-authenticator` to restrict access
- **Status:** `[ ]`

### 7.2 API Key Management
- **Current state:** API key read from `.env` — no validation or error messaging if missing/invalid
- **Improvement:** Validate API key on startup, show clear error in UI if invalid, support key input via UI
- **Status:** `[ ]`

### 7.3 File Size Limits
- **Improvement:** Enforce maximum file size on upload (e.g., 50MB) with a clear user message
- **Status:** `[ ]`

---

## 8. Deployment

### 8.1 Docker Support
- **Improvement:** Add `Dockerfile` and `docker-compose.yml` for one-command local setup
- **Status:** `[ ]`

### 8.2 Streamlit Cloud Deployment
- **Improvement:** Add `requirements.txt` adjustments and `secrets.toml` template for Streamlit Cloud deployment
- **Status:** `[ ]`

### 8.3 Environment Configuration
- **Improvement:** Add `.env.example` template so new developers know which variables to configure
- **Status:** `[ ]`

---

## Update Log

| Date | Feature Merged to Dev | Meta Updated By |
|---|---|---|
| 2026-02-19 | Initial project setup | B.M.G.Gimhana Mahela |

---

*This document is maintained on the `meta` branch only. Update this file whenever a feature moves from `[ ]` to `[~]` or `[x]`, and add a row to the Update Log above.*
