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

## Status Legend

| Symbol | Meaning |
|---|---|
| `[ ]` | Not started |
| `[~]` | In progress |
| `[x]` | Completed |

---

## Development Phases — Priority Order

```
Phase 1 → Fix broken foundation issues          (Quick wins, unblock everything)
Phase 2 → Core RAG quality                      (Heart of the product)
Phase 3 → LLM & retrieval expansion             (Power & flexibility)
Phase 4 → Document support expansion            (Wider input coverage)
Phase 5 → UI/UX polish                          (User experience)
Phase 6 → Testing & CI                          (Code confidence)
Phase 7 → Security & config                     (Production readiness)
Phase 8 → Deployment                            (Ship it)
```

---

## Phase 1 — Foundation Fixes
> **Goal:** Fix all existing broken/hardcoded things before building on top of them.
> These are bugs and gaps in the current code that will cause problems later.

| # | Improvement | File / Location | Effort | Status |
|---|---|---|---|---|
| 1.1 | Model caching with `@st.cache_resource` — models reload on every rerun | `chat_engine.py`, `vector_store.py` | Low | `[x]` |
| 1.2 | Fix hardcoded `document_ids=[1]` — chat records not linked to real documents | `app.py:277` | Low | `[x]` |
| 1.3 | Fix broken sidebar placeholder image URL | `app.py:80` | Low | `[x]` |
| 1.4 | Add `.env.example` — new developers have no template to configure from | root | Low | `[x]` |
| 1.5 | Explicit duplicate file warning — currently silently replaces on re-upload | `database.py:88`, `app.py` | Low | `[x]` |
| 1.6 | File size limit enforcement on upload with clear user message | `app.py` | Low | `[x]` |

---

## Phase 2 — Core RAG Quality
> **Goal:** Make the Q&A meaningfully better. These directly impact the quality of answers.

| # | Improvement | File / Location | Effort | Status |
|---|---|---|---|---|
| 2.1 | Conversation memory — LLM currently has no memory of previous turns | `chat_engine.py` | Medium | `[ ]` |
| 2.2 | Streaming responses — implement the `ask_streaming()` stub | `chat_engine.py:137`, `app.py` | Medium | `[ ]` |
| 2.3 | Relevance score threshold — filter out low-similarity chunks before passing to LLM | `vector_store.py`, `chat_engine.py` | Low | `[ ]` |
| 2.4 | Configurable K via UI slider — currently hardcoded `k=4` | `chat_engine.py:93`, `app.py` | Low | `[ ]` |
| 2.5 | Semantic chunking — implement the placeholder in `semantic_chunking()` | `document_processor.py:103` | Medium | `[ ]` |
| 2.6 | Configurable chain types (stuff / map_reduce / refine / map_rerank) via UI | `chat_engine.py:91` | Medium | `[ ]` |

---

## Phase 3 — LLM & Retrieval Expansion
> **Goal:** Give users flexibility in model choice and improve retrieval accuracy.

| # | Improvement | File / Location | Effort | Status |
|---|---|---|---|---|
| 3.1 | Multi-LLM support — Claude, GPT-4o, Ollama (LLaMA 3, Mistral), Gemini | `chat_engine.py` | High | `[ ]` |
| 3.2 | Hybrid search — combine BM25 keyword search with semantic vector search | `vector_store.py` | Medium | `[ ]` |
| 3.3 | Re-ranking — score and reorder retrieved chunks by true relevance before LLM call | `chat_engine.py` | Medium | `[ ]` |
| 3.4 | Answer confidence indicator — show similarity scores alongside answers | `app.py`, `vector_store.py` | Low | `[ ]` |

---

## Phase 4 — Document Support Expansion
> **Goal:** Support more file types so the assistant works with real-world document sets.

| # | Improvement | File / Location | Effort | Status |
|---|---|---|---|---|
| 4.1 | DOCX support — Microsoft Word files via `python-docx` | `document_processor.py` | Low | `[ ]` |
| 4.2 | CSV / XLSX support — spreadsheet ingestion via `pandas` | `document_processor.py` | Low | `[ ]` |
| 4.3 | HTML support — web page ingestion via `BeautifulSoup` | `document_processor.py` | Low | `[ ]` |
| 4.4 | PPTX support — PowerPoint slides via `python-pptx` | `document_processor.py` | Medium | `[ ]` |
| 4.5 | OCR fallback — Tesseract OCR for scanned/image-based PDFs | `document_processor.py` | Medium | `[ ]` |
| 4.6 | URL ingestion — paste a URL and scrape it as a document source | `document_processor.py`, `app.py` | Medium | `[ ]` |

---

## Phase 5 — UI / UX Polish
> **Goal:** Make the app feel complete and professional for real users.

| # | Improvement | File / Location | Effort | Status |
|---|---|---|---|---|
| 5.1 | Document management panel — view, delete, and re-process uploaded documents | `app.py`, `database.py` | Medium | `[ ]` |
| 5.2 | Multi-session support — browse and resume past conversations from sidebar | `app.py`, `database.py` | Medium | `[ ]` |
| 5.3 | Async document processing — non-blocking upload with `st.status()` | `app.py` | Medium | `[ ]` |
| 5.4 | Dark mode support — Streamlit theming via `.streamlit/config.toml` | `.streamlit/` | Low | `[ ]` |

---

## Phase 6 — Testing & CI
> **Goal:** Build confidence in the codebase before scaling further.

| # | Improvement | File / Location | Effort | Status |
|---|---|---|---|---|
| 6.1 | Unit tests — `pytest` suite for `document_processor`, `database`, `utils` | `tests/` | Medium | `[ ]` |
| 6.2 | Integration tests — upload doc → ask question → verify answer | `tests/` | Medium | `[ ]` |
| 6.3 | GitHub Actions CI — run tests automatically on every PR to `dev` | `.github/workflows/` | Low | `[ ]` |

---

## Phase 7 — Security & Configuration
> **Goal:** Harden the app for shared or production use.

| # | Improvement | File / Location | Effort | Status |
|---|---|---|---|---|
| 7.1 | API key validation — validate on startup, show clear UI error if missing/invalid | `chat_engine.py`, `app.py` | Low | `[ ]` |
| 7.2 | User authentication — restrict access via `streamlit-authenticator` | `app.py` | Medium | `[ ]` |
| 7.3 | Batch embedding — batch API calls for faster large-document ingestion | `vector_store.py` | Medium | `[ ]` |

---

## Phase 8 — Deployment
> **Goal:** Make the project easy to run anywhere and shareable publicly.

| # | Improvement | File / Location | Effort | Status |
|---|---|---|---|---|
| 8.1 | Docker support — `Dockerfile` + `docker-compose.yml` for one-command setup | root | Medium | `[ ]` |
| 8.2 | Streamlit Cloud deployment — `secrets.toml` template + deployment guide | root | Low | `[ ]` |

---

## Effort Reference

| Level | What it means |
|---|---|
| Low | A few hours — isolated change, single file |
| Medium | 1–2 days — touches multiple files, some design decisions |
| High | 3+ days — architectural change, significant testing needed |

---

## Update Log

| Date | Feature Merged to Dev | Phase | Status Changed | Updated By |
|---|---|---|---|---|
| 2026-02-19 | Initial project setup | — | — | B.M.G.Gimhana Mahela |
| 2026-02-19 | Phase 1 — all 6 foundation fixes | Phase 1 | All `[x]` | B.M.G.Gimhana Mahela |

---

*This document is maintained on the `meta` branch only. Update the Status column and Update Log whenever a feature progresses, and switch back to `dev` to continue development.*
