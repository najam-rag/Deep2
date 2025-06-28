# Unified Ultra-RAG App with Clause Extraction and Optional JSONL Input
import streamlit as st
import os
import hashlib
import tempfile
import re
import json
from typing import List
from concurrent.futures import ThreadPoolExecutor

from PyPDF2 import PdfReader
from pdf2image import convert_from_path
import pytesseract

from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import BM25Retriever

from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

# === Configuration ===
st.set_page_config(page_title="⚡ Clause Finder RAG App", layout="wide")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4-turbo-preview"
OCR_THREADS = 4

# === Clause Extractor ===
def extract_clause(text: str) -> str:
    patterns = [
        r"(Clause\s*\d{1,2}\.\d{1,2}(?:\.\d{1,2})?)",
        r"(\d{1,2}\.\d{1,2}(?:\.\d{1,2})?)",
        r"(Clause\s*\d{1,2}\.\d{1,2}[a-zA-Z]?)",
        r"(Clause\s*\d{1,2})",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
    return "Clause not found"

# === Document Processor ===
class DocumentProcessor:
    def __init__(self):
        self.ocr_fallback = False
        self.processed_pages = 0

    def _parallel_ocr(self, images: List) -> List[str]:
        with ThreadPoolExecutor(max_workers=OCR_THREADS) as executor:
            return list(executor.map(pytesseract.image_to_string, images))

    def process(self, file_path: str) -> List[Document]:
        try:
            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                if self._validate(docs):
                    return docs
            except: pass

            try:
                loader = UnstructuredPDFLoader(file_path, mode="elements", strategy="fast")
                docs = loader.load()
                if self._validate(docs):
                    return docs
            except: pass

            self.ocr_fallback = True
            images = convert_from_path(file_path, thread_count=OCR_THREADS)
            texts = self._parallel_ocr(images)
            return [Document(page_content=t, metadata={"page": i+1}) for i, t in enumerate(texts) if t.strip()]

        except Exception as e:
            st.error(f"❌ Processing failed: {str(e)}")
            st.stop()

    def _validate(self, docs):
        return bool(docs) and any(len(doc.page_content.strip()) > 50 for doc in docs)

# === UI ===
st.sidebar.header("🔐 Login")
password = st.sidebar.text_input("Enter password", type="password")
if password != "Password":
    st.warning("🚫 Access denied")
    st.stop()

st.title("⚡ Clause-Smart Code Assistant")

# === Upload Mode Selection ===
input_method = st.sidebar.radio("📤 Select Input Type", ["Upload PDF", "Use Pre-chunked File"])
chunks = None

if input_method == "Upload PDF":
    uploaded_file = st.file_uploader("📎 Upload a PDF Code", type="pdf")
    if uploaded_file:
        pdf_bytes = uploaded_file.read()
        pdf_hash = hashlib.md5(pdf_bytes).hexdigest()
        vectorstore_dir = f"vectorstores/{pdf_hash}"

        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model=EMBEDDING_MODEL)

        if os.path.exists(vectorstore_dir):
            try:
                retriever = FAISS.load_local(vectorstore_dir, embeddings).as_retriever()
            except:
                os.system(f"rm -rf {vectorstore_dir}")
                retriever = None
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(pdf_bytes)
                tmp_path = tmp.name

            processor = DocumentProcessor()
            docs = processor.process(tmp_path)
            splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = splitter.split_documents(docs)

            db = FAISS.from_documents(chunks, embeddings)
            db.save_local(vectorstore_dir)
            retriever = db.as_retriever()
            llm = ChatOpenAI(model=LLM_MODEL, temperature=0.2, openai_api_key=OPENAI_API_KEY)
            qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

elif input_method == "Use Pre-chunked File":
    jsonl_file = st.file_uploader("📂 Upload your JSONL chunk file", type="jsonl")
    if jsonl_file:
        docs = []
        for line in jsonl_file:
            obj = json.loads(line)
            doc = Document(page_content=obj["content"], metadata=obj.get("metadata", {}))
            docs.append(doc)

        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = splitter.split_documents(docs)

        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model=EMBEDDING_MODEL)
        db = FAISS.from_documents(chunks, embeddings)
        retriever = db.as_retriever()
        llm = ChatOpenAI(model=LLM_MODEL, temperature=0.2, openai_api_key=OPENAI_API_KEY)
        qa = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

if not chunks and not input_method == "Use Pre-chunked File":
    st.info("Please upload a document")
    st.stop()

query = st.text_input("💬 Ask your question:")
if query:
    result = qa({"query": query})

    st.subheader("🔍 Answer")
    st.success(result["result"])

    st.subheader("📚 Source Snippets")
    for i, doc in enumerate(result["source_documents"][:3]):
        page = doc.metadata.get("page", "N/A")
        preview = doc.page_content.strip().replace("\n", " ")[:500]
        clause_info = extract_clause(preview)

        with st.expander(f"Source {i+1} — Page {page}, {clause_info}"):
            st.code(preview, language="text")
