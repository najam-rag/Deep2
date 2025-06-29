# ✅ Smart RAG App with One-Time Vectorizing, QA Memory, Verified Override, Feedback Correction
import streamlit as st
import os
import hashlib
import tempfile
import re
import json
import base64
import time
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

# === Configuration ===
st.set_page_config(page_title="⚡ Clause Finder RAG App", layout="wide")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
GITHUB_TOKEN = st.secrets.get("GITHUB_TOKEN")
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-3.5-turbo-1106"
OCR_THREADS = 4
GITHUB_QA_FILE_URL = "https://api.github.com/repos/najam-rag/Deep2/contents/qa_memory.jsonl"

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

# === Grouping by Clause ===
def group_by_clause(docs: List[Document]) -> List[Document]:
    grouped_docs = []
    current_clause = None
    current_text = ""
    current_page = None

    for doc in docs:
        lines = doc.page_content.splitlines()
        for line in lines:
            clause_match = re.match(r"(?:Clause\s*)?(\d{1,2}(?:\.\d{1,2}){1,2})", line.strip())
            if clause_match:
                if current_text and current_clause:
                    grouped_docs.append(Document(
                        page_content=current_text.strip(),
                        metadata={"clause": current_clause, "page": current_page, "source": "PDF"}
                    ))
                current_clause = clause_match.group(1)
                current_text = line + "\n"
                current_page = doc.metadata.get("page", None)
            else:
                current_text += line + "\n"

    if current_text and current_clause:
        grouped_docs.append(Document(
            page_content=current_text.strip(),
            metadata={"clause": current_clause, "page": current_page, "source": "PDF"}
        ))

    return grouped_docs

# === Load JSONL Chunks ===
@st.cache_data(show_spinner=False)
def load_jsonl_chunks_from_url(url: str):
    response = requests.get(url)
    if response.status_code != 200:
        return []
    chunks = []
    for line in response.text.strip().splitlines():
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
            chunks.append(doc)
        except json.JSONDecodeError:
            continue
    return chunks

# === QA Memory ===
def load_qa_memory_jsonl():
    url = "https://raw.githubusercontent.com/najam-rag/Deep2/main/qa_memory.jsonl"
    response = requests.get(url)
    qa_docs = []
    if response.status_code == 200:
        for line in response.text.strip().splitlines():
            try:
                record = json.loads(line)
                qa_docs.append(Document(
                    page_content=record["answer"],
                    metadata={"question": record["query"], "source": "qa_memory"}
                ))
            except: continue
    return qa_docs

def get_qa_vectorstore():
    now = time.time()
    if "qa_vectorstore" not in st.session_state:
        st.session_state.qa_vectorstore = None
        st.session_state.qa_embed_time = 0

    if (now - st.session_state.qa_embed_time) > 300 or st.session_state.qa_vectorstore is None:
        docs = load_qa_memory_jsonl()
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model=EMBEDDING_MODEL)
        st.session_state.qa_vectorstore = FAISS.from_documents(docs, embeddings)
        st.session_state.qa_embed_time = now
        st.toast("🔁 QA memory re-embedded.")
    return st.session_state.qa_vectorstore

# === Initialize Vectorstore ===
def initialize_vectorstore_once(file_hash, pdf_path, jsonl_chunks):
    if "active_vectorstore" in st.session_state and st.session_state.get("vectorstore_hash") == file_hash:
        return st.session_state.active_vectorstore

    processor = DocumentProcessor()
    pdf_docs = processor.process(pdf_path)
    grouped_pdf_docs = group_by_clause_with_notes(pdf_docs)

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    jsonl_split = splitter.split_documents(jsonl_chunks)
    pdf_split = splitter.split_documents(grouped_pdf_docs)
    weighted_chunks = jsonl_split * 3 + pdf_split if jsonl_chunks else pdf_split

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY, model=EMBEDDING_MODEL)
    db = FAISS.from_documents(weighted_chunks, embeddings)

    st.session_state.active_vectorstore = db
    st.session_state.vectorstore_hash = file_hash
    return db

# === GitHub Correction Pusher ===
def push_to_github(record):
    headers = {"Authorization": f"Bearer {GITHUB_TOKEN}"}
    get_res = requests.get(GITHUB_QA_FILE_URL, headers=headers)
    if get_res.status_code != 200:
        return False

    sha = get_res.json()["sha"]
    old_content = base64.b64decode(get_res.json()["content"]).decode("utf-8")
    updated = old_content + json.dumps(record) + "\n"

    payload = {
        "message": f"Add correction: {record['query']}",
        "content": base64.b64encode(updated.encode()).decode(),
        "sha": sha
    }
    put_res = requests.put(GITHUB_QA_FILE_URL, headers=headers, json=payload)
    return put_res.status_code in [200, 201]

# === UI Setup ===
st.sidebar.header("🔐 Login")
if st.sidebar.text_input("Enter password", type="password") != "Password":
    st.warning("🚫 Access denied")
    st.stop()

st.title("⚡ Clause-Smart Code Assistant")

code_option = st.sidebar.selectbox("📘 Select Code Standard", ["None", "AS3000", "AS3017", "AS3003"])
code_to_jsonl = {
    "AS3000": "https://raw.githubusercontent.com/najam-rag/Deep2/main/as3000_chunks_by_clause.jsonl",
    "AS3017": "https://raw.githubusercontent.com/YOUR_REPO/main/as3017_chunks.jsonl",
    "AS3003": "https://raw.githubusercontent.com/YOUR_REPO/main/as3003_chunks.jsonl",
}
selected_jsonl_url = code_to_jsonl.get(code_option)

uploaded_file = st.file_uploader("📌 Upload Your Code PDF", type="pdf")
if not uploaded_file:
    st.info("📌 Please upload a code PDF to begin.")
    st.stop()

with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
    tmp.write(uploaded_file.read())
    tmp_path = tmp.name
    file_hash = hashlib.md5(open(tmp_path, 'rb').read()).hexdigest()

jsonl_chunks = load_jsonl_chunks_from_url(selected_jsonl_url) if selected_jsonl_url and code_option != "None" else []
db = initialize_vectorstore_once(file_hash, tmp_path, jsonl_chunks)
retriever = db.as_retriever()
llm = ChatOpenAI(model=LLM_MODEL, temperature=0.2, openai_api_key=OPENAI_API_KEY)
qa = RetrievalQAWithSourcesChain.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)

qa_vectorstore = get_qa_vectorstore()
qa_retriever = qa_vectorstore.as_retriever()

def group_by_clause_with_notes(docs: List[Document]) -> List[Document]:
    grouped_docs = []
    current_clause = None
    current_text = ""
    current_page = None
    current_type = "Clause"

    def save_current():
        if current_text and current_clause:
            grouped_docs.append(Document(
                page_content=current_text.strip(),
                metadata={
                    "clause": current_clause,
                    "page": current_page,
                    "type": current_type,
                    "source": "PDF"
                }
            ))

    for doc in docs:
        lines = doc.page_content.splitlines()
        for line in lines:
            stripped = line.strip()
            clause_match = re.match(r"(?:Clause\s*)?(\d{1,2}(?:\.\d{1,2}){1,2})", stripped)
            note_match = re.match(r"(NOTE(?:\s*\d*)?:?|Notes\s*:)", stripped, re.IGNORECASE)
            exception_match = re.match(r"(EXCEPTION(?:\s*\d*)?:?|Exceptions\s*:)", stripped, re.IGNORECASE)

            if clause_match:
                save_current()
                current_clause = clause_match.group(1)
                current_text = line + "\n"
                current_page = doc.metadata.get("page", None)
                current_type = "Clause"
            elif note_match:
                save_current()
                current_type = "Note"
                current_text = line + "\n"
            elif exception_match:
                save_current()
                current_type = "Exception"
                current_text = line + "\n"
            else:
                current_text += line + "\n"

    save_current()
    return grouped_docs


query = st.text_input("💬 Ask your question:")
if query:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity

    def extract_keywords(text):
        return re.findall(r"[a-zA-Z]{3,}", text.lower())

    def get_best_qa_match(query, qa_docs):
        query_keywords = extract_keywords(query)
        if not query_keywords:
            return []
    
        candidates = []
        for doc in qa_docs:
            q = doc.metadata.get("question", "").lower()
            score = sum(1 for word in query_keywords if word in q)
            if score > 0:
                candidates.append((score, doc))
    
        candidates.sort(reverse=True, key=lambda x: x[0])
        return [c[1] for c in candidates[:3]] if candidates else []

    qa_docs_all = load_qa_memory_jsonl()
    qa_docs = get_best_qa_match(query, qa_docs_all)
    doc_docs = retriever.get_relevant_documents(query)

    def deduplicate_by_content(docs):
        seen = set()
        unique_docs = []
        for d in docs:
            snippet = d.page_content.strip()[:100]
            if snippet not in seen:
                seen.add(snippet)
                unique_docs.append(d)
        return unique_docs

    merged_docs = deduplicate_by_content(qa_docs + doc_docs)
    result = qa.combine_documents_chain.run(input_documents=merged_docs, question=query)

    st.subheader("🔍 Answer")
    st.success(result)

    st.subheader("📚 Source Snippets")
    for i, doc in enumerate(merged_docs[:3]):
        page = doc.metadata.get("page", "N/A")
        clause_info = doc.metadata.get("clause", extract_clause(doc.page_content))
        source = doc.metadata.get("source", "uploaded PDF")
        preview = doc.page_content.strip().replace("\n", " ")[:500]
        with st.expander(f"Source {i+1} — Clause {clause_info}, Page {page} ({source})"):
            st.code(preview, language="text")

    st.markdown("---")
    st.subheader("🧠 Was this answer correct?")
    feedback_col1, feedback_col2 = st.columns([1, 3])
    with feedback_col1:
        is_correct = st.radio("Feedback", ["Yes", "No"], horizontal=True)

    if is_correct == "No":
        corrected = st.text_area("✍️ Enter the correct answer below:", height=150)
        if st.button("✅ Submit Correction"):
            if corrected.strip():
                record = {"query": query.strip(), "answer": corrected.strip()}
                success = push_to_github(record)
                if success:
                    st.success("✅ Correction saved to GitHub!")
                else:
                    st.error("❌ Failed to save correction to GitHub.")
            else:
                st.warning("Please enter a corrected answer before submitting.")
