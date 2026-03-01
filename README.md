# 🏛️ Sanskrit RAG System — Architecture Document

## Overview

This document describes the complete technical architecture of the **Sanskrit Retrieval-Augmented Generation (RAG) System** — an end-to-end pipeline for ingesting Sanskrit documents and answering natural language queries from their content using CPU-based inference.

---

## System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        USER INTERFACE (Colab)                       │
│                   Input: Sanskrit / English / IAST query            │
└───────────────────────────────┬─────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     DOCUMENT INGESTION LAYER                        │
│                                                                     │
│   ┌──────────────┐    ┌──────────────┐    ┌──────────────────────┐  │
│   │  TextLoader  │    │ PyPDFLoader  │    │  UTF-8 Encoding      │  │
│   │  (.txt files)│    │ (.pdf files) │    │  (Devanagari support)│  │
│   └──────┬───────┘    └──────┬───────┘    └──────────────────────┘  │
│          └──────────┬────────┘                                      │
│                     ▼                                               │
│        ┌────────────────────────┐                                   │
│        │ RecursiveCharacter     │  chunk_size=500                   │
│        │    TextSplitter        │  chunk_overlap=50                 │
│        │ Separators: । \n\n \n │  (Sanskrit-aware)                 │
│        └────────────┬───────────┘                                   │
└─────────────────────┼───────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      EMBEDDING LAYER                                │
│                                                                     │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │     HuggingFaceEmbeddings                                    │  │
│   │     Model: paraphrase-multilingual-MiniLM-L12-v2            │  │
│   │     Device: CPU  │  Output: 384-dimensional vectors          │  │
│   └──────────────────────────────┬───────────────────────────────┘  │
│                                  │                                  │
│                    Text → [0.23, -0.81, 0.45 ...]                  │
└──────────────────────────────────┼──────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       VECTOR STORE (FAISS)                          │
│                                                                     │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │  FAISS Index (CPU)                                           │  │
│   │  - Stores all chunk embeddings                               │  │
│   │  - Similarity search: cosine / L2 distance                  │  │
│   │  - Persisted locally: sanskrit_faiss_index/                 │  │
│   └──────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
              ┌────────────────────┘
              │   At Query Time: top-k=3 similar chunks retrieved
              │
              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        RETRIEVAL LAYER                              │
│                                                                     │
│   Query → Embed → FAISS Search → Top-3 Relevant Chunks             │
│                                                                     │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │  Prompt Construction                                         │  │
│   │                                                              │  │
│   │  "Answer the question based on context below.               │  │
│   │   Context: {retrieved_chunks}                               │  │
│   │   Question: {user_query}                                    │  │
│   │   Answer:"                                                  │  │
│   └──────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     GENERATION LAYER (LLM)                          │
│                                                                     │
│   ┌──────────────────────────────────────────────────────────────┐  │
│   │  Model: google/flan-t5-base                                  │  │
│   │  Task: text2text-generation                                  │  │
│   │  Device: CPU  │  max_new_tokens: 200                         │  │
│   └──────────────────────────────────────────────────────────────┘  │
│                                                                     │
│                     Generated Answer (text)                         │
└──────────────────────────────────┬──────────────────────────────────┘
                                   │
                                   ▼
                        ┌──────────────────┐
                        │   Final Answer   │
                        │  Displayed to    │
                        │     User         │
                        └──────────────────┘
```

---

## Component Breakdown

### 1. Document Ingestion Layer

| Component | Library | Purpose |
|---|---|---|
| `TextLoader` | `langchain-community` | Loads `.txt` files with UTF-8 encoding for Devanagari |
| `PyPDFLoader` | `langchain-community` + `pypdf` | Extracts text from PDF documents |
| `RecursiveCharacterTextSplitter` | `langchain-text-splitters` | Splits documents into overlapping chunks |

**Chunking Strategy:**
- Chunk size: `500` characters
- Overlap: `50` characters
- Separators (in priority order): `।` → `\n\n` → `\n` → ` `
- The `।` (daṇḍa) separator ensures Sanskrit sentences are not split mid-meaning

---

### 2. Embedding Layer

| Component | Model | Dimensions | Languages |
|---|---|---|---|
| `HuggingFaceEmbeddings` | `paraphrase-multilingual-MiniLM-L12-v2` | 384 | 50+ including Sanskrit, Hindi, English |

- Converts each text chunk into a 384-dimensional numerical vector
- Similar meaning → similar vectors (cosine similarity)
- Runs entirely on CPU with no GPU requirement

---

### 3. Vector Store

| Component | Library | Storage |
|---|---|---|
| `FAISS` | `langchain-community` + `faiss-cpu` | Local disk (`sanskrit_faiss_index/`) |

- Facebook AI Similarity Search — optimized for fast nearest-neighbour lookup
- At query time, retrieves top-`k` (default: 3) most semantically similar chunks
- Index is saved to disk and can be reloaded without reprocessing documents

---

### 4. Generation Layer (LLM)

| Component | Model | Task | Device |
|---|---|---|---|
| `pipeline` | `google/flan-t5-base` | `text2text-generation` | CPU |

- Flan-T5 is instruction-tuned by Google — good at following directives
- Receives: system instructions + retrieved context + user question
- Outputs: natural language answer grounded in the retrieved Sanskrit text

**Upgrade Path:**

```
flan-t5-base (fast, ~250MB)
      ↓
flan-t5-large (better quality, ~800MB)
      ↓
flan-t5-xl (best quality, ~3GB, still CPU-compatible)
```

---

## Data Flow Summary

```
Document Upload
      ↓
Load (TextLoader / PyPDFLoader)
      ↓
Split into Chunks (RecursiveCharacterTextSplitter)
      ↓
Embed Chunks → Vectors (HuggingFaceEmbeddings)
      ↓
Store in FAISS Index
      ↓
[Query Time]
User Query → Embed → Search FAISS → Retrieve Top-3 Chunks
      ↓
Build Prompt (Instructions + Context + Question)
      ↓
Generate Answer (Flan-T5)
      ↓
Display Answer to User
```

---

## Technology Stack

| Layer | Technology | Version |
|---|---|---|
| Framework | LangChain | latest |
| Document Loaders | langchain-community | latest |
| Text Splitting | langchain-text-splitters | latest |
| Embeddings | langchain-huggingface | latest |
| Vector Store | FAISS (CPU) | faiss-cpu |
| Embedding Model | sentence-transformers | paraphrase-multilingual-MiniLM-L12-v2 |
| LLM | HuggingFace Transformers | google/flan-t5-base |
| PDF Support | pypdf | latest |
| Environment | Google Colab | Python 3.10+ |

---

## Design Decisions

**Why CPU-only?**
The system is designed to run without GPU access, making it accessible on free Google Colab tiers and standard laptops.

**Why multilingual embeddings?**
Sanskrit text in Devanagari script and IAST transliteration both need to be understood semantically. The `paraphrase-multilingual-MiniLM-L12-v2` model handles both.

**Why Flan-T5?**
It is instruction-tuned (follows directions well), lightweight enough for CPU, free to use, and produces coherent answers when given structured prompts.

**Why FAISS over other vector stores?**
FAISS is local, requires no server setup, no API keys, and no internet connection at query time — ideal for a self-contained Colab notebook.

**Why `।` as a separator?**
The daṇḍa (`।`) is the Sanskrit sentence-ending punctuation equivalent to a period. Using it as a primary split boundary preserves grammatical and semantic units in the chunks.

# 🪷 Sanskrit RAG System

> A Retrieval-Augmented Generation (RAG) pipeline for querying Sanskrit documents using CPU-based inference — no GPU required.

---

## 📌 What Is This?

This project lets you **upload Sanskrit documents** (`.txt` or `.pdf`) and **ask questions** about them in English, Sanskrit, or transliterated text. The system finds the most relevant passages from your documents and uses an AI model to generate a coherent answer.

**Example:**
- You upload a Sanskrit story document
- You ask: *"What mistake did Shankhanaada make?"*
- The system finds the relevant paragraph and answers: *"Shankhanaada carried sugar in a torn cloth, causing it to spill on the way."*

---

## ✅ Requirements

- Google Account (for Google Colab)
- Internet connection (for first-time model download only)
- No GPU needed — runs fully on CPU
- No local installation needed

---

## 🚀 Quick Start

### Step 1 — Open Google Colab
Go to [https://colab.research.google.com](https://colab.research.google.com) and create a new notebook.

### Step 2 — Copy & Run Each Cell

Copy the cells below **one by one** into Colab and run them in order using `Shift + Enter`.

---


## 📁 Project Structure

```
sanskrit-rag/
│
├── README.md                  ← This file
├── ARCHITECTURE.md            ← Technical architecture details
│
├── your_document.txt          ← Your uploaded Sanskrit text file
├── your_document.pdf          ← Or a PDF
│
└── sanskrit_faiss_index/      ← Auto-created after Cell 4
    ├── index.faiss
    └── index.pkl
```

---

## 🧩 How It Works (Plain English)

```
1. You upload a Sanskrit document
2. It gets cut into small 500-character chunks
3. Each chunk is converted into numbers (vectors) that capture its meaning
4. All vectors are stored in a searchable database (FAISS)
5. When you ask a question, it is also converted to numbers
6. FAISS finds the 3 chunks with the most similar numbers (= most relevant text)
7. Those chunks + your question are sent to Flan-T5 AI
8. Flan-T5 reads them and writes an answer
```

---

## ⚙️ Configuration Options

| Parameter | Default | Description |
|---|---|---|
| `chunk_size` | `500` | Max characters per chunk |
| `chunk_overlap` | `50` | Shared characters between chunks |
| `top_k` | `3` | Number of chunks retrieved per query |
| `max_new_tokens` | `200` | Max length of generated answer |
| `model` | `flan-t5-base` | LLM used for generation |

---

## 🔧 Troubleshooting

| Error | Fix |
|---|---|
| `No module named 'langchain.document_loaders'` | Use `from langchain_community.document_loaders import ...` |
| `No module named 'langchain.text_splitter'` | Use `from langchain_text_splitters import ...` |
| `cannot import HuggingFaceEmbeddings` | Use `from langchain_huggingface import HuggingFaceEmbeddings` |
| `requests version conflict` warning | Safe to ignore — it's a warning, not an error |
| Answer seems wrong or vague | Upgrade to `flan-t5-large` in Cell 5 for better quality |
| Sanskrit characters show as `???` | Ensure your `.txt` file is saved with **UTF-8 encoding** |

---

## 🚀 Upgrade Options

### Better Answer Quality
In Cell 5, replace `flan-t5-base` with:
```python
model="google/flan-t5-large"   # Better quality, ~800MB, still CPU-compatible
```

### Reload Existing Index (Skip Reprocessing)
If you've already run Cell 4, you can reload the saved index:
```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={"device": "cpu"}
)
vectorstore = FAISS.load_local(
    "sanskrit_faiss_index",
    embeddings,
    allow_dangerous_deserialization=True
)
print("✅ Index loaded!")
```

---

## 📦 Package Reference

| Package | Version | Purpose |
|---|---|---|
| `langchain` | latest | Core RAG framework |
| `langchain-community` | latest | Document loaders, FAISS integration |
| `langchain-text-splitters` | latest | Text chunking |
| `langchain-huggingface` | latest | HuggingFace embeddings integration |
| `sentence-transformers` | latest | Multilingual embedding model |
| `faiss-cpu` | latest | Vector similarity search |
| `transformers` | latest | Flan-T5 LLM |
| `pypdf` | latest | PDF text extraction |
| `accelerate` | latest | Optimized CPU inference |

---

## 📜 Supported Document Types

- `.txt` — Plain text files (must be UTF-8 encoded for Devanagari)
- `.pdf` — PDF documents with selectable text

---

## 👤 Author Notes

- Designed for **Google Colab free tier** (CPU only)
- Supports queries in **English**, **Sanskrit (Devanagari)**, and **IAST transliteration**
- The embedding model handles multilingual text natively — no translation needed
- All processing is local — your documents are not sent to any external server

---

## 📄 License

This project is open for academic and educational use.

---

*Built with LangChain · HuggingFace · FAISS · Google Flan-T5*
