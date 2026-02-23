# Code Explanation — Document-Aware AI Assistant

> Full walkthrough of every file, every function, and the complete data flow —
> written in execution order so you can follow exactly what happens from
> `streamlit run app.py` to a answered question on screen.

---

## Table of Contents

1. [File Map — What Each File Does](#1-file-map)
2. [Execution Order Overview](#2-execution-order-overview)
3. [File 1 — `utils.py` (Shared Utilities)](#3-utils-py)
4. [File 2 — `database.py` (Data Persistence)](#4-database-py)
5. [File 3 — `document_processor.py` (Text Extraction & Chunking)](#5-document_processor-py)
6. [File 4 — `vector_store.py` (Embeddings & Search)](#6-vector_store-py)
7. [File 5 — `chat_engine.py` (RAG & LLM)](#7-chat_engine-py)
8. [File 6 — `app.py` (Entry Point & UI)](#8-app-py)
9. [Complete Flow A — Document Upload](#9-complete-flow-a--document-upload)
10. [Complete Flow B — Asking a Question](#10-complete-flow-b--asking-a-question)

---

## 1. File Map

```
document-ai-assistant/
│
├── app.py                  ← Entry point. Streamlit UI, ties everything together.
├── document_processor.py   ← Extracts text from files, splits into chunks.
├── vector_store.py         ← Converts chunks to vectors, stores & searches them.
├── chat_engine.py          ← Loads LLM, runs RAG chain (retrieve → answer).
├── database.py             ← SQLite persistence for documents & chat history.
├── utils.py                ← Shared helpers: logging, hashing, validation.
│
├── .env                    ← Your local secrets (never committed).
├── .env.example            ← Template showing all required keys.
├── .streamlit/
│   └── config.toml         ← Streamlit server config (file watcher fix for torch).
├── requirements.txt        ← All Python dependencies.
└── .github/
    └── workflows/
        └── ci.yml          ← GitHub Actions CI (runs on every PR).
```

**Dependency direction** (who imports who):
```
app.py
  ├── document_processor.py
  ├── vector_store.py
  │     └── (langchain Chroma → chromadb internally)
  ├── chat_engine.py
  │     └── (langchain RetrievalQA → vector_store internally)
  ├── database.py
  └── utils.py
```
`app.py` is the only file that imports from all others.
No other file imports from `app.py` — this is intentional (one-way dependency).

---

## 2. Execution Order Overview

```
$ streamlit run app.py
        │
        ▼
[1] app.py — os.environ set before any import
        │
        ▼
[2] utils.py — logging configured at import time
        │
        ▼
[3] app.py — st.set_page_config(), CSS injected
        │
        ▼
[4] app.py — main() called
        ├── setup_directories()          ← utils.py
        └── init_session_state()
              ├── DocumentProcessor()    ← document_processor.py
              ├── VectorStoreManager()   ← vector_store.py
              └── DatabaseManager()     ← database.py
        │
        ▼
[5] Streamlit renders UI
        ├── render_sidebar()             ← waiting for user upload
        └── render_chat_interface()      ← waiting for user question
```

---

## 3. `utils.py`

**Role:** Shared utilities loaded before anything else. Sets up logging, directory structure, and file helpers used by `app.py`.

---

### Logging setup (lines 13–21)

```python
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),   # writes to app.log on disk
        logging.StreamHandler()            # also prints to terminal
    ]
)
logger = logging.getLogger(__name__)
```

This runs **at import time** — the moment `app.py` does `from utils import logger`,
Python executes this code. So logging is configured before the app even starts.

- `logging.FileHandler('app.log')` — every log line is saved to `app.log`
- `logging.StreamHandler()` — same lines also appear in your terminal
- `__name__` resolves to `"utils"` — log lines show which module they came from

---

### `setup_directories()` (line 23)

```python
def setup_directories():
    directories = ['data/uploaded', 'data/chroma_db', 'logs']
    for directory in directories:
        os.makedirs(directory, exist_ok=True)
```

Called once on startup from `app.py → main()`.
`exist_ok=True` means it will not raise an error if the folder already exists —
safe to call every time the app starts.

| Directory | Purpose |
|---|---|
| `data/uploaded` | Reserved for saving uploaded files (not yet used) |
| `data/chroma_db` | ChromaDB persists vector data here |
| `logs` | Reserved for future structured logs |

---

### `generate_file_hash()` (line 30)

```python
def generate_file_hash(file_content: bytes) -> str:
    return hashlib.sha256(file_content).hexdigest()
```

Takes the raw bytes of a file and produces a unique 64-character hex string.

**Why SHA-256?**
Two identical files will always produce the same hash. This is used to detect
duplicate uploads — if a file with the same hash already exists in the database,
we warn the user and skip reprocessing it.

Example:
```
"Hello world" → "64ec88ca00b268e5ba1a35678a1b5316..."
"Hello world" → "64ec88ca00b268e5ba1a35678a1b5316..."  ← identical, duplicate detected
"Hello World" → "a591a6d40bf420404a011733cfb7b190..."  ← different content, different hash
```

---

### `validate_file_type()` (line 42)

```python
def validate_file_type(file_name: str) -> bool:
    supported_extensions = ['.pdf', '.txt', '.md']
    ext = os.path.splitext(file_name)[1].lower()
    return ext in supported_extensions
```

`os.path.splitext("report.PDF")` returns `("report", ".PDF")`.
`.lower()` normalises it so `.PDF` and `.pdf` both pass.

---

## 4. `database.py`

**Role:** All data persistence. Stores document records and chat history in a local
SQLite database file (`chat_history.db`). Uses the standard `sqlite3` library.

---

### `DatabaseManager.__init__()` (line 18)

```python
def __init__(self, db_path: str = "chat_history.db"):
    self.db_path = db_path
    self.init_database()
```

On creation, it immediately calls `init_database()` to create tables.
SQLite creates the `.db` file automatically if it doesn't exist.

---

### `get_connection()` — Context Manager (line 28)

```python
@contextmanager
def get_connection(self):
    conn = sqlite3.connect(self.db_path)
    conn.row_factory = sqlite3.Row   # rows behave like dicts
    try:
        yield conn
        conn.commit()       # auto-commit on success
    except Exception as e:
        conn.rollback()     # auto-rollback on failure
        raise e
    finally:
        conn.close()        # always close the connection
```

The `@contextmanager` decorator allows this to be used with `with`:
```python
with self.get_connection() as conn:
    cursor = conn.cursor()
    # ... do work
```
This pattern guarantees the connection is always closed and either committed
or rolled back — even if an exception occurs. It prevents data corruption.

`conn.row_factory = sqlite3.Row` means query results can be accessed by
column name (`row['file_name']`) instead of just by index (`row[0]`).

---

### `init_database()` (line 42) — Table Schema

**documents table:**
```sql
CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    file_name   TEXT NOT NULL,
    file_hash   TEXT UNIQUE,      ← prevents duplicate file storage
    file_size   INTEGER,
    upload_time TIMESTAMP,
    chunk_count INTEGER,
    metadata    TEXT              ← JSON string of extra info
)
```

**chat_history table:**
```sql
CREATE TABLE IF NOT EXISTS chat_history (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id   TEXT,            ← UUID linking to a browser session
    question     TEXT,
    answer       TEXT,
    timestamp    TIMESTAMP,
    document_ids TEXT,            ← JSON list of document IDs used
    metadata     TEXT             ← JSON string (e.g. source count)
)
```

**Indexes:**
```sql
CREATE INDEX idx_session   ON chat_history(session_id)
CREATE INDEX idx_timestamp ON chat_history(timestamp)
```
Indexes make lookups by `session_id` and `timestamp` much faster —
without them, SQLite would scan every row.

---

### `add_document_record()` (line 77)

```python
cursor.execute("INSERT OR REPLACE INTO documents ...")
return cursor.lastrowid
```

`INSERT OR REPLACE` — if a document with the same `file_hash` already exists
(the `UNIQUE` constraint), it replaces it rather than throwing an error.
`lastrowid` returns the auto-generated `id` of the inserted row —
this is the real document ID that gets stored with chat records.

---

### `document_exists()` (line added in Phase 1)

```python
def document_exists(self, file_hash: str) -> Optional[Dict]:
    cursor.execute(
        "SELECT id, file_name FROM documents WHERE file_hash = ?",
        (file_hash,)
    )
    row = cursor.fetchone()
    return dict(row) if row else None
```

Called before processing each uploaded file. If it returns a dict,
the file is a duplicate and we show a warning instead of reprocessing.

**Note the `?` placeholder** — never use f-strings in SQL queries.
`?` with a tuple is parameterised SQL which prevents SQL injection.

---

## 5. `document_processor.py`

**Role:** Takes raw uploaded file bytes, extracts text, and splits it into
overlapping chunks ready for embedding. This is the first step of the RAG pipeline.

---

### `DocumentProcessor.__init__()` (line 22)

```python
def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
    self.text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", " ", ""]
    )
    self.markdown_splitter = MarkdownTextSplitter(...)
```

Two splitters are initialised. `text_splitter` is used for all files.
`markdown_splitter` exists for future use with `.md` files.

**`RecursiveCharacterTextSplitter` — how it works:**

It tries to split on `"\n\n"` first (paragraph breaks). If a chunk is still
too large, it falls back to `"\n"` (line breaks), then `" "` (words),
then `""` (individual characters as last resort). This preserves natural
language boundaries wherever possible.

**Why overlap?**
```
Document text: "...revenue was $4.2M in Q3. Operating costs rose by 8%..."
                                         ↑ chunk boundary

Chunk 1 (chars 0–1000):   "...revenue was $4.2M in Q3."
Chunk 2 (chars 800–1800):  "...Q3. Operating costs rose by 8%..."
                              ↑ 200-char overlap — key sentence appears in both
```
If a key sentence falls at a boundary, overlap ensures it appears complete
in at least one chunk.

---

### `load_document()` (line 46)

```python
with tempfile.NamedTemporaryFile(delete=False, suffix=...) as tmp_file:
    tmp_file.write(file_content)
    tmp_path = tmp_file.name

try:
    if file_name.endswith('.pdf'):
        loader = PyPDFLoader(tmp_path)
    else:
        loader = TextLoader(tmp_path, encoding='utf-8')
    documents = loader.load()
finally:
    os.unlink(tmp_path)   # always delete temp file
```

**Why a temp file?** LangChain's loaders (`PyPDFLoader`, `TextLoader`) require
a file path on disk, not bytes in memory. Streamlit gives us bytes from the
browser upload. So we write to a temp file, let LangChain read it, then delete it.

`delete=False` is necessary because on some systems (especially Windows),
you cannot open a file while it's still held open by the `with` block.
We set `delete=False` and manually delete it in the `finally` block.

Each `Document` object returned has two parts:
- `page_content` — the raw text string
- `metadata` — dict with source info (we add `source_file` and `chunk_strategy`)

---

### `chunk_documents()` (line 84)

```python
chunked_docs = self.text_splitter.split_documents(documents)

for i, doc in enumerate(chunked_docs):
    doc.metadata['chunk_index'] = i
    doc.metadata['total_chunks'] = len(chunked_docs)
```

`split_documents()` takes the loaded pages and splits each one according
to `chunk_size` and `chunk_overlap`.

Metadata is enriched with `chunk_index` (position of this chunk in the full list)
and `total_chunks` (how many chunks total). This is shown in the "View Sources"
expander in the UI — so you know which part of a document the answer came from.

---

### `process_document()` (line 112) — The Pipeline Entry Point

```python
def process_document(self, file_content: bytes, file_name: str) -> Dict:
    documents = self.load_document(file_content, file_name)
    chunked_docs = self.chunk_documents(documents)
    return {
        'chunks': chunked_docs,
        'total_chunks': len(chunked_docs),
        'original_docs': len(documents),
        'metadata': { ... }
    }
```

This is the only method `app.py` calls directly. It orchestrates
`load_document()` → `chunk_documents()` and returns the result as a dict.

---

## 6. `vector_store.py`

**Role:** Converts text chunks into numerical vectors (embeddings), stores them
in ChromaDB, and provides similarity search. This is the memory of the AI.

---

### Module-level cached functions (Phase 1 addition)

```python
@st.cache_resource(show_spinner="Loading embedding model...")
def _load_openai_embeddings(api_key: str):
    return OpenAIEmbeddings(openai_api_key=api_key, model="text-embedding-ada-002")

@st.cache_resource(show_spinner="Loading embedding model...")
def _load_huggingface_embeddings():
    from langchain.embeddings import HuggingFaceEmbeddings
    return HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
```

`@st.cache_resource` is a Streamlit decorator that:
- Runs the function **once** on first call
- Stores the returned object in memory
- Returns the **same object** on every subsequent call (no reloading)

Without this, the 500MB embedding model would be downloaded and loaded
from disk on every single Streamlit rerun (every button click, every upload).
With it, the model loads once and stays in memory for the session.

---

### `VectorStoreManager.__init__()` (line 22)

```python
def __init__(self, persist_directory: str = "data/chroma_db"):
    self.embeddings = self._get_embeddings()

    self.chroma_client = chromadb.PersistentClient(
        path=persist_directory,
        settings=Settings(anonymized_telemetry=False)  ← bugfix
    )
    self.vector_store = None
```

Two things happen:
1. The embedding model is fetched (from cache after first load)
2. A ChromaDB persistent client is created — this is the low-level DB connection

`Settings(anonymized_telemetry=False)` — disables the telemetry that was
causing the PostHog API error on startup (bugfix PR #2).

---

### `_get_embeddings()` (line 44)

```python
def _get_embeddings(self):
    api_key = os.getenv('OPENAI_API_KEY')
    if api_key:
        return _load_openai_embeddings(api_key)
    return _load_huggingface_embeddings()
```

Checks `.env` for `OPENAI_API_KEY`. If found → uses OpenAI's paid
`text-embedding-ada-002` model (1536 dimensions, higher quality).
If not → uses HuggingFace's free `all-MiniLM-L6-v2` (384 dimensions, runs locally).

**What does an embedding model actually do?**
```
Input:  "The quarterly revenue was $4.2M"
Output: [0.021, -0.134, 0.087, 0.456, ..., -0.023]  ← 384 numbers
```
Texts with similar meaning produce vectors that are mathematically close.
This is how the AI finds relevant chunks — not by keyword matching,
but by meaning similarity.

---

### `create_vector_store()` (line 61)

```python
def create_vector_store(self, documents: List[Document], collection_name="documents"):
    self.vector_store = Chroma.from_documents(
        documents=documents,
        embedding=self.embeddings,
        persist_directory=self.persist_directory,
        collection_name=collection_name
    )
    self.vector_store.persist()
```

`Chroma.from_documents()` does three things internally:
1. Calls the embedding model on every chunk → produces vectors
2. Stores each (text, vector, metadata) triple in ChromaDB
3. Persists the database to `data/chroma_db/` on disk

`persist()` ensures the data survives app restarts.

---

### `similarity_search()` (line 89)

```python
def similarity_search(self, query: str, k: int = 4) -> List[Document]:
    return self.vector_store.similarity_search(query, k=k)
```

Takes a plain text query, converts it to a vector using the same
embedding model, then finds the `k` stored vectors closest to it
using cosine similarity. Returns those `k` Document objects with their original text.

**Cosine similarity:** Measures the angle between two vectors.
- Score 1.0 = identical meaning
- Score 0.0 = completely unrelated

---

## 7. `chat_engine.py`

**Role:** The brain of the system. Loads the LLM, builds the RAG chain
(retrieve relevant chunks → inject into prompt → generate answer),
and returns structured responses with source attribution.

---

### Module-level cached functions (Phase 1 addition)

```python
@st.cache_resource(show_spinner="Loading AI model...")
def _load_openai_llm(api_key: str):
    return ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7, streaming=True, ...)

@st.cache_resource(show_spinner="Loading local AI model...")
def _load_local_llm():
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, ...)
    return HuggingFacePipeline(pipeline=pipe)
```

Same `@st.cache_resource` pattern as the embedding model.
Without this, the 250MB Flan-T5 model would reload from disk on every user
interaction. With it, it loads once.

`temperature=0.7` — controls randomness. 0.0 = deterministic (same answer
every time), 1.0 = highly creative/random. 0.7 is a balance for Q&A.

---

### `RAGChatEngine.__init__()` (line 22)

```python
def __init__(self, vector_store):
    self.vector_store = vector_store
    self.llm = self._get_llm()
    self.qa_chain = None
    self._setup_chain()
```

On creation, immediately loads the LLM and builds the chain.
This is called from `app.py` only after documents are processed.

---

### `_setup_chain()` — The Prompt Template (line 63)

```python
template = """
You are a helpful AI assistant answering questions based on the provided context.

Context from documents:
{context}

Question: {question}

Instructions:
- Answer based ONLY on the provided context
- If the answer isn't in the context, say "I cannot find this information in the provided documents"
- Be concise but thorough

Answer:"""

prompt = PromptTemplate(template=template, input_variables=["context", "question"])
```

`{context}` and `{question}` are placeholders. LangChain fills them in at runtime:
- `{context}` ← the 4 retrieved text chunks joined together
- `{question}` ← the user's actual question

The instruction *"Answer based ONLY on the provided context"* is what prevents
hallucination. The LLM is told explicitly it cannot use outside knowledge.

---

### The RetrievalQA Chain (line 89)

```python
self.qa_chain = RetrievalQA.from_chain_type(
    llm=self.llm,
    chain_type="stuff",
    retriever=self.vector_store.vector_store.as_retriever(
        search_kwargs={"k": 4}
    ),
    chain_type_kwargs={"prompt": prompt, "verbose": True},
    return_source_documents=True
)
```

`chain_type="stuff"` — all 4 retrieved chunks are "stuffed" directly into
the prompt as the `{context}`. Simple and effective for short to medium documents.

`as_retriever(search_kwargs={"k": 4})` — when the chain needs context,
it calls the vector store to retrieve the 4 most similar chunks.

`return_source_documents=True` — the response includes the original
Document objects, not just the answer text. This powers the "View Sources"
expander in the UI.

---

### `ask()` (line 102)

```python
def ask(self, question: str) -> Dict[str, Any]:
    result = self.qa_chain({"query": question})
    return {
        "answer": result["result"],
        "sources": [
            {
                "content": doc.page_content[:200] + "...",
                "metadata": doc.metadata
            }
            for doc in result["source_documents"]
        ]
    }
```

The single method `app.py` calls. Internally, LangChain does:
1. Embeds `question` → vector
2. Searches ChromaDB for 4 nearest chunks
3. Fills `{context}` and `{question}` in the prompt template
4. Sends the complete prompt to the LLM
5. Returns `result["result"]` (answer text) + `result["source_documents"]`

Source content is truncated to 200 characters (`[:200] + "..."`) for display.
The full chunk was already used by the LLM — this is just for showing the user.

---

## 8. `app.py`

**Role:** Entry point. Runs the Streamlit web server, defines the UI layout,
and coordinates all other modules. Every user interaction routes through here.

---

### First Lines — Environment Variable (lines 6–10)

```python
import os
os.environ['ANONYMIZED_TELEMETRY'] = 'False'
import streamlit as st
```

`os.environ` must be set **before** any `import` that triggers ChromaDB loading.
Python executes import-time code immediately when a module is first imported.
If ChromaDB is imported before this line, the telemetry is already initialised
and setting the env var afterwards has no effect.

This suppresses telemetry on **all** ChromaDB clients — including the ones
LangChain creates internally inside `Chroma.from_documents()`.

---

### Page Configuration (lines 28–33)

```python
st.set_page_config(
    page_title="Document AI Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)
```

Must be the **first Streamlit call** in the script. Streamlit throws an error
if any other `st.*` call comes before it. `layout="wide"` uses the full
browser width instead of the default narrow centered layout.

---

### Custom CSS (lines 36–62)

```python
st.markdown("""<style> ... </style>""", unsafe_allow_html=True)
```

Injects raw CSS into the page. `unsafe_allow_html=True` is required for
any HTML/CSS injection. The classes defined here (`.main-header`, `.chat-message`,
etc.) are used by `st.markdown()` calls throughout the app.

---

### `init_session_state()` (line 65)

```python
def init_session_state():
    if 'session_id' not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())
    if 'processor' not in st.session_state:
        st.session_state.processor = DocumentProcessor()
    if 'vector_store' not in st.session_state:
        st.session_state.vector_store = VectorStoreManager()
    ...
```

**Why `if 'key' not in st.session_state`?**

Streamlit re-runs the entire script top-to-bottom on every user interaction
(every button click, slider move, text input). Without the `if` guard,
a new `VectorStoreManager()` and `DatabaseManager()` would be created on
every click — wiping out all uploaded documents and chat history.

The `if` guard means: create these objects only on the very first run.
After that, the existing objects in `session_state` are reused.

`uuid.uuid4()` generates a random unique ID (e.g. `"f47ac10b-58cc-4372-a567-..."`)
used to group all chat messages in this browser session together in the database.

---

### `render_sidebar()` (line 83)

Builds the left panel with:
- **Banner** — inline HTML/CSS gradient div (no external image needed)
- **File uploader** — `st.file_uploader()` returns a list of `UploadedFile` objects
- **Sliders** — chunk size and overlap, passed to `process_documents()`
- **Process button** — triggers `process_documents()` when clicked
- **Statistics** — shown only after documents are processed
- **Clear chat** — resets `st.session_state.messages` and calls `st.rerun()`

---

### `process_documents()` (line 157) — The Upload Pipeline

This is called when the user clicks "Process Documents". For each file:

```
1. file_content = uploaded_file.getvalue()     ← raw bytes from browser
2. Check file size vs MAX_FILE_SIZE_MB          ← skip if too large
3. validate_file_type(file_name)                ← check extension
4. generate_file_hash(file_content)             ← SHA-256 hash
5. db_manager.document_exists(file_hash)        ← check for duplicate
6. processor.process_document(...)              ← extract text + chunk
7. db_manager.add_document_record(...)          ← save to SQLite
8. session_state.processed_document_ids.append(doc_id)
```

After all files:
```
9. vector_store.create_vector_store(all_chunks) ← embed + store in ChromaDB
10. RAGChatEngine(vector_store)                 ← build the QA chain
11. session_state.documents_processed = True    ← unlocks the chat UI
```

---

### `render_chat_interface()` (line 221)

The main area of the app. Two modes:

**Before documents are processed:**
Shows 3 feature cards and an info message.

**After documents are processed:**
```python
# Replay all previous messages from session state
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Wait for new input
if prompt := st.chat_input("Ask a question..."):
    # 1. Add user message to session + display it
    # 2. Call chat_engine.ask(prompt) → get answer + sources
    # 3. Display answer + expandable sources
    # 4. Save both to session_state.messages
    # 5. Save to database
```

The walrus operator `:=` (Python 3.8+) assigns and checks in one expression.
`if prompt := st.chat_input(...)` means: wait for input, assign it to `prompt`,
and only enter the `if` block if the user actually typed something.

---

### `main()` (line 330) — Entry Point

```python
def main():
    setup_directories()     # ensure data/, logs/ exist
    init_session_state()    # create objects if first run

    tab1, tab2 = st.tabs(["💬 Chat", "📊 Analytics"])

    with tab1:
        render_sidebar()
        render_chat_interface()

    with tab2:
        render_analytics()

if __name__ == "__main__":
    main()
```

`if __name__ == "__main__"` — standard Python entry point guard. This means
`main()` only runs when the script is executed directly (`streamlit run app.py`),
not when it's imported by another module.

---

## 9. Complete Flow A — Document Upload

```
User drags "report.pdf" → Streamlit browser
        │
        ▼
app.py → process_documents()
        │
        ├─[1] uploaded_file.getvalue() → raw PDF bytes
        │
        ├─[2] Size check: 2.1MB < 50MB ✓
        │
        ├─[3] validate_file_type("report.pdf") → ".pdf" in supported ✓
        │
        ├─[4] generate_file_hash(bytes) → "a3f9c2..."
        │
        ├─[5] db_manager.document_exists("a3f9c2...") → None (not a duplicate)
        │
        ├─[6] processor.process_document(bytes, "report.pdf")
        │       │
        │       ├─ load_document()
        │       │     ├─ Write bytes to /tmp/xyz.pdf
        │       │     ├─ PyPDFLoader("/tmp/xyz.pdf").load()
        │       │     │     └─ Extracts text page by page
        │       │     └─ Delete /tmp/xyz.pdf
        │       │
        │       └─ chunk_documents()
        │             └─ RecursiveCharacterTextSplitter splits into N chunks
        │                   Each chunk: { page_content: "...", metadata: {...} }
        │
        ├─[7] db_manager.add_document_record() → doc_id = 1
        │       └─ INSERT INTO documents (file_name, file_hash, ...) → SQLite
        │
        ├─[8] session_state.processed_document_ids = [1]
        │
        └─[9] vector_store.create_vector_store(all_chunks)
                │
                ├─ For each chunk:
                │     HuggingFace all-MiniLM-L6-v2 → [0.021, -0.134, ...]
                │
                └─ ChromaDB stores: (chunk_text, vector, metadata)
                       Persisted to data/chroma_db/ on disk
        │
        ▼
session_state.chat_engine = RAGChatEngine(vector_store)
        └─ Loads Flan-T5, builds RetrievalQA chain

        ▼
UI unlocks → chat input appears
```

---

## 10. Complete Flow B — Asking a Question

```
User types: "What was the revenue in Q3?"
        │
        ▼
app.py → render_chat_interface()
        │
        ├─ session_state.messages.append({"role": "user", "content": question})
        ├─ st.chat_message("user") displays it on screen
        │
        ▼
chat_engine.ask("What was the revenue in Q3?")
        │
        ├─[1] RetrievalQA receives {"query": "What was the revenue in Q3?"}
        │
        ├─[2] Retriever calls vector_store.similarity_search(question, k=4)
        │       │
        │       ├─ Embed question → [0.019, -0.128, ...]
        │       └─ ChromaDB cosine similarity search
        │             Returns 4 closest chunks:
        │               Chunk A: "The quarterly revenue was $4.2M in Q3..."
        │               Chunk B: "Q3 performance exceeded targets by 12%..."
        │               Chunk C: "Revenue breakdown: Q1 $3.1M, Q2 $3.8M..."
        │               Chunk D: "The CFO reported strong Q3 results..."
        │
        ├─[3] PromptTemplate fills in:
        │       {context} ← Chunk A + B + C + D joined as text
        │       {question} ← "What was the revenue in Q3?"
        │
        ├─[4] Complete prompt sent to Flan-T5 (or GPT-3.5-turbo)
        │
        ├─[5] LLM generates: "The quarterly revenue in Q3 was $4.2M..."
        │
        └─[6] Returns:
                {
                  "answer": "The quarterly revenue in Q3 was $4.2M...",
                  "sources": [
                    { "content": "The quarterly revenue was...", "metadata": {...} },
                    ...
                  ]
                }
        │
        ▼
app.py displays:
        ├─ st.markdown(answer)
        ├─ st.expander("View Sources") → shows chunk previews
        ├─ session_state.messages.append({"role": "assistant", ...})
        └─ db_manager.add_chat_record(session_id, question, answer, [1])
                └─ INSERT INTO chat_history → SQLite
```

---

*This document is maintained on the `meta` branch. Update it whenever new files
or significant logic changes are merged into `dev`.*
