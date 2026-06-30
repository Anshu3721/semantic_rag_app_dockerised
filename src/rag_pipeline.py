"""
rag_pipeline.py

Handles document loading, chunking, embedding, retrieval (with MMR).
No LLM: Only retrieves and returns relevant document chunks.
"""

import os
from typing import List, Tuple

from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# ------- CONFIGURATION -------- #
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
EMBED_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# ------- DOCUMENT LOADING -------- #
def load_document(file_path: str):
    """
    Loads a document using LangChain's loaders based on file extension.
    """
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".txt":
        loader = TextLoader(file_path)
    elif ext == ".pdf":
        loader = PyPDFLoader(file_path)
    else:
        raise ValueError("Unsupported file type. Please upload a .txt or .pdf file.")
    return loader.load()

# ------- CHUNKING -------- #
def chunk_documents(documents: List, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """
    Splits documents into chunks for embedding & retrieval.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(documents)

# ------- EMBEDDING MODEL -------- #
def get_embedder():
    """
    Returns LangChain embedding model.
    """
    return HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)

# ------- VECTORSTORE SETUP (FAISS) -------- #
def create_vectorstore(chunks, embedder):
    """
    Stores chunk embeddings for retrieval.
    """
    return FAISS.from_documents(chunks, embedder)

# ------- MMR RETRIEVER -------- #
def get_mmr_retriever(vectorstore, k=2):
    """
    Uses vectorstore's retriever with MMR (Maximal Marginal Relevance).
    """
    return vectorstore.as_retriever(search_type="mmr", search_kwargs={"k": k, "fetch_k": 6})

# ------- RAG PIPELINE MAIN -------- #
class RAGPipeline:
    """
    End-to-end RAG pipeline (document -> chunks -> embed -> retrieve)
    """

    def __init__(self):
        self.embedder = get_embedder()
        self.vectorstore = None
        self.mmr_retriever = None
        self.chunks = None

    def process_file(self, file_path: str):
        docs = load_document(file_path)
        self.chunks = chunk_documents(docs)
        self.vectorstore = create_vectorstore(self.chunks, self.embedder)
        self.mmr_retriever = get_mmr_retriever(self.vectorstore)

    def answer_question(self, question: str) -> List[str]:
        """
        Returns only the most relevant document chunks (as plain text).
        """
        relevant_docs = self.mmr_retriever.invoke(question)
        chunk_texts = [doc.page_content for doc in relevant_docs]
        return chunk_texts
