```markdown
# Semantic Document Search (RAG Retrieval) — Streamlit + FAISS + LangChain

I built this project to solve a simple (but common) problem: **searching and extracting the right information from long PDFs/TXT documents quickly**, without manually scanning pages.

This app lets you upload a document, type a query, and it retrieves the most relevant chunks using **embeddings + FAISS vector search + MMR retrieval**.  
- It focuses on **fast and accurate retrieval** (no LLM-generated answers by default).

---

## What this app does

- Upload **PDF** or **TXT** documents
- Split text into meaningful **chunks**
- Create **embeddings** using `sentence-transformers` (`all-MiniLM-L6-v2`)
- Store/search vectors using **FAISS**
- Retrieve top relevant chunks using **MMR (Maximal Marginal Relevance)**  
  (helps reduce duplicate results and improves diversity)
- Show the retrieved context with **query term highlighting**

---

## Why I built it

Most “document Q&A” tools look impressive, but under the hood the quality depends heavily on the retrieval step.  
So I built this project to focus on the core retrieval pipeline:

**Chunking → Embeddings → FAISS Index → MMR Retrieval → Relevant Context**

This makes it a solid foundation for full RAG systems, and it’s also useful standalone for fast semantic search.

---

## Tech Stack

- **Python**
- **Streamlit** (UI)
- **LangChain** (text splitting + retrieval abstraction)
- **SentenceTransformers** (`all-MiniLM-L6-v2`) for embeddings
- **FAISS** for vector indexing/search
- **PyMuPDF** for PDF extraction

---

## Project Structure (typical)

```

src/
app.py
rag_pipeline.py
utils.py

````


## How to run locally

### 1) Create environment & install dependencies
```bash
pip install -r src/requirements.txt
````

### 2) Run the Streamlit app

```bash
streamlit run src/app.py
```

---

## How it works (high level)

1. **Extract text** from PDF/TXT
2. **Chunking** using `RecursiveCharacterTextSplitter`
3. Convert chunks into **embeddings**
4. Store in **FAISS**
5. On query:

   * run similarity search with **MMR**
   * return top chunks as relevant context

---

## Example Use Cases

* Searching inside long policies / SOPs
* Finding answers in technical documentation
* Quick semantic lookup for internal knowledge docs
* Base retrieval component for full RAG applications

---

## Future Improvements

Things I plan to add next:

* Persist FAISS index (avoid re-embedding every run)
* Add citations like **page numbers** for PDFs
* Add reranking for better relevance on harder queries

---

## Author

**Anshu Kumar**
If you're working on RAG / semantic search / GenAI systems and think this is useful, feel free to connect with me on LinkedIn.

```
