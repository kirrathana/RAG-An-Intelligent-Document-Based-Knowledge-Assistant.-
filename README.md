# Valentius RAG Knowledge Assistant

A production-grade, hybrid Retrieval-Augmented Generation (RAG) knowledge assistant built for the **Valentius Kryptix AI Engineering Internship**. The system combines **Google Gemini** for high-dimensional document and query embeddings, **ChromaDB** for persistent vector retrieval, and **Groq API** (`openai/gpt-oss-20b`) for ultra-low latency answer generation and plain LLM comparisons, complete with latency telemetry and inline source citations.

---

## Hybrid Architecture Overview

```text
User Query (Web UI / API)
        │
        ├──► [1] Query Embedding (Google Gemini: gemini-embedding-001, RETRIEVAL_QUERY)
        │         └─► Measures embed_time_ms
        │
        ├──► [2] ChromaDB Persistent Search (data/chroma_db, Cosine HNSW)
        │         └─► Measures retrieval_time_ms (Top-k relevant chunks)
        │
        ├──► [3] Prompt Builder & Strict Grounding Guardrails (<retrieved_context>)
        │
        └──► [4] Ultra-Fast Answer Generation (Groq API: openai/gpt-oss-20b)
                  └─► Measures generation_time_ms & total_time_ms
                  └─► Grounded answer with source & page citations
```

### Key Architectural Components

- **Generation Engine:** Groq API running `openai/gpt-oss-20b` (configurable via `GROQ_MODEL`) at `https://api.groq.com/openai/v1`.
- **Embedding Engine:** Google Gemini `gemini-embedding-001` via the official Google Gen AI SDK (`RETRIEVAL_DOCUMENT` and `RETRIEVAL_QUERY` task types).
- **Vector Database:** ChromaDB persistent local storage with cosine similarity search and content deduplication.
- **Latency Telemetry:** Granular execution-time breakdown surfaced across API responses and frontend badges:
  - `embed_time_ms`: Gemini query embedding latency
  - `retrieval_time_ms`: ChromaDB vector lookup latency
  - `generation_time_ms`: Groq LLM generation latency
  - `total_time_ms`: Total end-to-end response latency

### Project Structure

```text
Valentius-RAG-Assistant/
├── app.py                     # Flask backend & static frontend server
├── config.py                  # Environment configuration via python-dotenv
├── compare_rag.py             # Live ingestion & RAG vs. non-RAG comparison runner
├── requirements.txt           # Python dependencies
├── .env.example               # Safe environment variable template
├── .gitignore                 # Excludes .env, .venv, and generated chroma_db
├── README.md
├── data/
│   ├── documents/             # Knowledge dataset (PDF, MD, TXT)
│   └── chroma_db/             # Persistent ChromaDB vector store
├── src/
│   ├── document_loader.py     # PyPDF + Markdown/TXT extraction & SHA-256 hashing
│   ├── text_splitter.py       # Paragraph-aware chunking (~500 tokens, ~75 overlap)
│   ├── embeddings.py          # Google Gen AI SDK embedding wrapper
│   ├── vector_store.py        # Persistent ChromaDB collection & deduplication
│   ├── retriever.py           # Top-k semantic search coordinator
│   └── rag_pipeline.py        # Grounded RAG prompt + Plain LLM comparison
├── frontend/
│   ├── index.html             # Knowledge Chat, Source Inspector, and Comparison UI
│   ├── style.css              # Dark navy/black theme with violet & cyan accents
│   └── script.js              # Vanilla JS client for all backend endpoints
└── tests/
    ├── test_chunking.py       # Extraction, chunking, overlap & metadata unit tests
    ├── test_retrieval.py      # Mocked embeddings, ChromaDB persistence & API tests
    └── test_integration_live.py # Live Gemini + ChromaDB integration tests
```

---

## Technologies Used

- **Python 3.12**
- **Groq API SDK (`groq`)**
  - Production generation model: `openai/gpt-oss-20b` (configurable via `GROQ_MODEL`)
  - Base URL: `https://api.groq.com/openai/v1`
- **Google Gen AI SDK (`google-genai`)**
  - Embedding model: `gemini-embedding-001` at 768 MRL dimensions (configurable via `GEMINI_EMBEDDING_MODEL` and `EMBEDDING_DIMENSIONS`)
- **ChromaDB (`chromadb`)** — persistent local vector database using cosine similarity
- **PyPDF (`pypdf`)** — page-by-page PDF text extraction
- **Flask (`flask`)** — REST backend and static UI hosting
- **python-dotenv** — environment variable management
- **pytest** — automated unit and live integration testing
- **HTML5 / CSS3 / Vanilla JavaScript** — responsive frontend dashboard with telemetry badges

---

## Knowledge Dataset & Sources

The `data/documents/` corpus currently contains **10 documents** (28 page records, 258 chunks) covering all core domains:

| Filename | Topic | Source |
| :--- | :--- | :--- |
| `rag_lewis_2020_arxiv.pdf` | Retrieval-Augmented Generation (19-page paper) | Lewis et al. (2020), [arXiv:2005.11401](https://arxiv.org/pdf/2005.11401) |
| `google_genai_sdk_docs.md` | Google Gemini & Gen AI Python SDK | Official [googleapis/python-genai README](https://raw.githubusercontent.com/googleapis/python-genai/main/README.md) |
| `chromadb_official_docs.md` | ChromaDB Clients, Collections & Querying | Official [Chroma Docs](https://docs.trychroma.com/llms.txt) |
| `python_pep8_fundamentals.txt` | Python Fundamentals & Style Guide | Official [Python PEP 8](https://raw.githubusercontent.com/python/peps/main/peps/pep-0008.rst) |
| `large_language_models_reference.txt` | Large Language Models | [Wikipedia: Large language model](https://en.wikipedia.org/wiki/Large_language_model) |
| `generative_ai_reference.txt` | Generative AI | [Wikipedia: Generative artificial intelligence](https://en.wikipedia.org/wiki/Generative_artificial_intelligence) |
| `embeddings_reference.txt` | Word & Vector Embeddings | [Wikipedia: Word embedding](https://en.wikipedia.org/wiki/Word_embedding) |
| `vector_databases_reference.txt` | Vector Databases | [Wikipedia: Vector database](https://en.wikipedia.org/wiki/Vector_database) |
| `semantic_search_reference.txt` | Semantic Search | [Wikipedia: Semantic search](https://en.wikipedia.org/wiki/Semantic_search) |
| `valentius_kryptix_rag_engineering_standard.md` | Internal Internship RAG Spec (`VK-RAG-2026-v1`) | Internal Valentius Kryptix AI Internship specification |

---

## Installation & Setup

### 1. Create Virtual Environment & Install Dependencies

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure Environment Variables

Copy `.env.example` to `.env` and set your Gemini API key:

```powershell
Copy-Item .env.example .env
```

Edit `.env`:

```ini
GEMINI_API_KEY=your_actual_gemini_api_key
GEMINI_GENERATION_MODEL=gemini-3.8-flash
GEMINI_EMBEDDING_MODEL=gemini-embedding-001
EMBEDDING_DIMENSIONS=768
CHUNK_SIZE_TOKENS=500
CHUNK_OVERLAP_TOKENS=75
TOP_K_RESULTS=5
```

> **Security Note:** `.env` and `data/chroma_db/` are excluded via `.gitignore`. Never commit API keys to version control.

---

## Chunking, Embedding & Retrieval Pipeline

1. **Document Ingestion (`src/document_loader.py`):**
   - Extracts PDFs page-by-page using `pypdf.PdfReader`, preserving 1-indexed `page` numbers.
   - Extracts `.md` and `.txt` files with UTF-8 / fallback decoding.
   - Computes SHA-256 `file_hash` per document and gracefully skips empty or unreadable files.
2. **Paragraph-Aware Chunking (`src/text_splitter.py`):**
   - Targets **500 tokens** per chunk with **75 tokens** of overlap (configurable).
   - Splits along `\n\n` paragraph boundaries first, falling back to single lines, sentences, and words only when a paragraph exceeds the token limit.
   - Provides `inspect_sample_chunks()` and `/api/chunks/sample` for chunk inspection.
3. **Embeddings & Persistent Vector Storage (`src/embeddings.py`, `src/vector_store.py`):**
   - Document chunks are embedded using `gemini-embedding-001` with `task_type="RETRIEVAL_DOCUMENT"` and `output_dimensionality=768`.
   - User queries are embedded using `task_type="RETRIEVAL_QUERY"` and the same `768` dimensionality.
   - Vectors, text, and metadata are stored in ChromaDB (`PersistentClient` at `data/chroma_db/`). Duplicate `file_hash` documents are automatically skipped.
4. **Grounded Generation (`src/rag_pipeline.py`):**
   - Encloses retrieved chunks inside `<retrieved_context>` tags with strict prompt-injection isolation.
   - Requires inline citations (`[Source: filename, Page N]`) and explicit abstention when context is missing.

---

## API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/` | Serves the web frontend (`frontend/index.html`) |
| `GET` | `/api/health` | Reports config and vector store status (never exposes secrets) |
| `GET` | `/api/documents` | Lists indexed documents, chunk counts, and token totals |
| `GET` | `/api/chunks/sample` | Inspects stored sample chunks (`?limit=5&source=...`) |
| `POST` | `/api/upload` | Uploads and ingests `.pdf`, `.md`, or `.txt` files |
| `POST` | `/api/ingest` | Scans `data/documents/` and indexes unindexed files |
| `POST` | `/api/retrieve` | Performs top-k semantic search and returns raw chunks + distances |
| `POST` | `/api/chat` | Executes full RAG retrieval + Gemini grounded generation |
| `POST` | `/api/compare` | Runs identical question through Plain Gemini vs. RAG + ChromaDB |

---

## Running the Application

```powershell
.\.venv\Scripts\python.exe app.py
```

Open `http://127.0.0.1:5000` in your browser.

---

## Testing

Run the automated unit tests (offline, mocked Gemini client + real ChromaDB & PyPDF):

```powershell
.\.venv\Scripts\pytest.exe -v
```

Run live integration + RAG vs. non-RAG comparison (requires `GEMINI_API_KEY` in `.env`):

```powershell
.\.venv\Scripts\python.exe compare_rag.py
.\.venv\Scripts\pytest.exe -v tests/test_integration_live.py
```

*(Live integration test outputs and verified RAG vs. non-RAG comparison results will be recorded here once executed against a configured `GEMINI_API_KEY`.)*

---

## Limitations & Future Improvements

- **OCR Support:** Scanned image-only PDFs without embedded text streams cannot be extracted by `pypdf` without adding OCR (e.g., Tesseract or Gemini multimodal ingestion).
- **Hybrid Retrieval:** Adding BM25 lexical matching + Reciprocal Rank Fusion (RRF) alongside dense Gemini embeddings would improve exact acronym/code identifier recall.
- **Conversation Memory:** Currently each `/api/chat` request is stateless single-turn RAG; multi-turn query rewriting can be added for follow-up questions.
- **Streaming Responses:** Server-Sent Events (SSE) can be added to stream Gemini tokens to the UI in real time.

