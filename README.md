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

## 📓 Full Notebook — Cell by Cell

### Cell 1 — Install Dependencies
```python
!pip install -q langchain langchain-community langchain-text-splitters \
             langchain-huggingface sentence-transformers faiss-cpu \
             transformers pypdf accelerate
```

### Cell 2 — Upload Your Sanskrit Documents
```python
from google.colab import files

print("Upload your .txt or .pdf Sanskrit documents")
uploaded = files.upload()

doc_files = list(uploaded.keys())
print(f"Uploaded: {doc_files}")
```

### Cell 3 — Load & Preprocess Documents
```python
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

docs = []

for fname in doc_files:
    if fname.endswith(".pdf"):
        loader = PyPDFLoader(fname)
    else:
        loader = TextLoader(fname, encoding="utf-8")
    docs.extend(loader.load())

print(f"Loaded {len(docs)} document(s)")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    separators=["।", "\n\n", "\n", " "]
)
chunks = splitter.split_documents(docs)
print(f"Total chunks: {len(chunks)}")
```

### Cell 4 — Create Embeddings & FAISS Index
```python
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

print("Loading embedding model (CPU)...")
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    model_kwargs={"device": "cpu"}
)

print("Building FAISS index...")
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local("sanskrit_faiss_index")
print("✅ Index saved!")
```

### Cell 5 — Load CPU-based LLM
```python
from transformers import pipeline

print("Loading LLM (this may take a minute)...")
llm_pipeline = pipeline(
    "text2text-generation",
    model="google/flan-t5-base",
    device=-1
)
print("✅ LLM ready!")
```

### Cell 6 — Build the RAG Query Function
```python
def retrieve_and_answer(query: str, top_k: int = 3):
    print(f"\n🔍 Query: {query}\n")

    retriever = vectorstore.as_retriever(search_kwargs={"k": top_k})
    relevant_docs = retriever.invoke(query)

    context = "\n\n".join([doc.page_content for doc in relevant_docs])
    print("📄 Retrieved Context:\n", context[:600], "...\n")

    prompt = f"""Answer the question based on the Sanskrit text context below.

Context:
{context}

Question: {query}
Answer:"""

    result = llm_pipeline(prompt, max_new_tokens=200, truncation=True)
    answer = result[0]["generated_text"]

    print("💬 Answer:\n", answer)
    return answer
```

### Cell 7 — Run Sample Queries
```python
retrieve_and_answer("Who is Shankhanaada and what mistake did he make?")
```
```python
retrieve_and_answer("What is the moral of the story about the devotee?")
```
```python
retrieve_and_answer("What did kAlidAsa do when the foreign scholar arrived?")
```

### Cell 8 — Interactive Query Loop
```python
print("🪷 Sanskrit RAG System Ready! Type 'quit' to exit.\n")

while True:
    query = input("Enter your query (Sanskrit / English / Transliterated): ")
    if query.lower() == "quit":
        break
    retrieve_and_answer(query)
```

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
