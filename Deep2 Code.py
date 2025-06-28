# ✅ Smart RAG App with Weighted JSONL and PDF Support
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
import requests

from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import BM25Retriever
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.vectorstores.utils import DistanceStrategy

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

# === UI Login ===
st.sidebar.header("🔐 Login")
password = st.sidebar.text_input("Enter password", type="password")
if password != "Password":
    st.warning("🚫 Access denied")
    st.stop()

st.title("⚡ Clause-Smart Code Assistant")

# === Code Selection Dropdown ===
code_option = st.sidebar.selectbox("📘 Select Code Standard", ["AS3000", "AS3017", "AS3003"])
code_to_jsonl = {
    "AS3000": "https://raw.githubusercontent.com/najam-rag/Deep2/main/as3000_chunks_by_clause.jsonl",
    "AS3017": "https://raw.githubusercontent.com/YOUR_REPO/main/as3017_chunks.jsonl",
    "AS3003": "https://raw.githubusercontent.com/YOUR_REPO/main/as3003_chunks.jsonl",
}
selected_jsonl_url = code_to_jsonl.get(code_option)

# === PDF Upload ===
uploaded_file = st.file_uploader("📌 Upload Your Code PDF", type="pdf")
if not uploaded_file:
    st.info("📎 Please upload a code PDF to begin.")
    st.stop()

# === Load JSONL from GitHub (SAFE)
jsonl_response = requests.get(selected_jsonl_url)
if jsonl_response.status_code != 200:
    st.error("❌ Failed to fetch standard chunks from GitHub.")
    st.stop()

jsonl_chunks = []
for line in jsonl_response.text.strip().splitlines():
    line = line.strip()
    if not line:
        continue
    try:
        obj = json.loads(line)
        doc = Document(
            page_content=obj["content"],
            metadata={
                "clause": obj.get("metadata", {}).get("clause", "N/A"),
                "page": obj.get("metadata", {}).get("page", "N/A"),
                "source": "JSONL"
            }
        )
        jsonl_chunks.append(doc)
    except json.JSONDecodeError:
        st.warning("⚠️ Skipping malformed JSONL line.")

# === Process Uploaded PDF ===
with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
    tmp.write(uploaded_file.read())
    tmp_path = tmp.name

processor = DocumentProcessor()
pdf_docs = processor.process(tmp_path)

splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
jsonl_split = splitter.split_documents(jsonl_chunks)
pdf_split = splitter.split_documents(pdf_docs)

# === Combine with Weight: JSONL gets more weight via duplication ===
weighted_chunks = jsonl_split * 3 + pdf_split  # Weight: JSONL 3x stronger

# === Embeddings & QA ===
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model=EMBEDDING_MODEL)
db = FAISS.from_documents(weighted_chunks, embeddings)
retriever = db.as_retriever()
llm = ChatOpenAI(model=LLM_MODEL, temperature=0.2, openai_api_key=OPENAI_API_KEY)
qa = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

# === Question Input ===
query = st.text_input("💬 Ask your question:")
if query:
    result = qa({"question": query})

    st.subheader("🔍 Answer")
    st.success(result["answer"])

    st.subheader("📚 Source Snippets")
    for i, doc in enumerate(result["source_documents"][:3]):
        page = doc.metadata.get("page", "N/A")
        clause_info = doc.metadata.get("clause", extract_clause(doc.page_content))
        source = doc.metadata.get("source", "uploaded PDF")
        preview = doc.page_content.strip().replace("\n", " ")[:500]

        with st.expander(f"Source {i+1} — Clause {clause_info}, Page {page} ({source})"):
            st.code(preview, language="text")
