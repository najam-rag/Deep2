import streamlit as st
import os
import hashlib
import tempfile
import re
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor

# Document Processing
from PyPDF2 import PdfReader, PdfWriter
from pdf2image import convert_from_path
import pytesseract
from langchain_community.document_loaders import PyPDFLoader, UnstructuredPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.schema import Document

# Vectorstores and Retrieval
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import BM25Retriever

# LLM Components
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# Configuration
st.set_page_config(page_title="⚡ Ultra-RAG Assistant", layout="wide")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")

# Constants
MAX_FILE_SIZE_MB = 200
MAX_PAGES_TO_PROCESS = 100
OCR_THREADS = 4
EMBEDDING_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4-turbo-preview"

# ======================
# Core Processing Classes
# ======================

class DocumentProcessor:
    """Optimized PDF processor with parallel OCR"""

    def __init__(self):
        self.ocr_fallback = False
        self.processed_pages = 0

    def _parallel_ocr(self, images: List) -> List[str]:
        """Process images in parallel"""
        with ThreadPoolExecutor(max_workers=OCR_THREADS) as executor:
            return list(executor.map(pytesseract.image_to_string, images))

    def process(self, file_path: str) -> List[Document]:
        """Process PDF with automatic fallback to OCR"""
        try:
            try:
                loader = PyPDFLoader(file_path)
                docs = loader.load()
                if self._validate_docs(docs):
                    self.processed_pages = len(docs)
                    return self._enhance_metadata(docs)
            except Exception as e:
                st.warning(f"Structured extraction failed: {str(e)}")

            try:
                loader = UnstructuredPDFLoader(file_path, mode="elements", strategy="fast")
                docs = loader.load()
                if self._validate_docs(docs):
                    self.processed_pages = len(docs)
                    return self._enhance_metadata(docs)
            except Exception as e:
                st.warning(f"Unstructured extraction failed: {str(e)}")

            self.ocr_fallback = True
            images = convert_from_path(file_path, thread_count=OCR_THREADS)
            texts = self._parallel_ocr(images)
            docs = [
                Document(page_content=t, metadata={"page": i+1, "source": "OCR"})
                for i, t in enumerate(texts) if t.strip()
            ]
            self.processed_pages = len(docs)
            return self._enhance_metadata(docs)

        except Exception as e:
            st.error(f"❌ Critical processing error: {str(e)}")
            st.stop()

    def _validate_docs(self, docs: List[Document]) -> bool:
        return bool(docs) and any(len(doc.page_content.strip()) > 50 for doc in docs)

    def _enhance_metadata(self, docs: List[Document]) -> List[Document]:
        for doc in docs:
            if clause_match := re.search(r"(?i)(clause|section|part)\s*(\d+(?:\.\d+)*)", doc.page_content):
                doc.metadata["clause"] = f"{clause_match.group(1)} {clause_match.group(2)}"
            if any(x in doc.page_content[:200].lower() for x in ["table", "figure", "diagram"]):
                doc.metadata["content_type"] = "visual"
        return docs

class VectorStoreManager:
    def __init__(self):
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY,
            model=EMBEDDING_MODEL,
            chunk_size=500
        )
        self.vectorstore = None
        self.bm25_retriever = None

    def initialize_from_documents(self, chunks: List[Document]):
        try:
            self.vectorstore = FAISS.from_documents(
                documents=chunks[:200],
                embedding=self.embeddings
            )
            self.bm25_retriever = BM25Retriever.from_documents(chunks)
            self.bm25_retriever.k = 5
            return True
        except Exception as e:
            st.error(f"Vector store initialization failed: {str(e)}")
            return False

    def query(self, question: str) -> List[Document]:
        if not self.vectorstore:
            return self.bm25_retriever.invoke(question)
        try:
            if re.search(r"(?i)(clause|section|table)\s*[\d\.]+", question):
                return self.bm25_retriever.invoke(question)
            vector_results = self.vectorstore.similarity_search(question, k=5)
            keyword_results = self.bm25_retriever.invoke(question)
            combined = {
                doc.metadata.get("page", ""): doc
                for doc in vector_results + keyword_results
            }
            return list(combined.values())[:5]
        except Exception:
            return self.bm25_retriever.invoke(question)

# ======================
# Streamlit Application
# ======================

def main():
    if "authenticated" not in st.session_state:
        st.sidebar.header("🔐 Secure Login")
        password = st.sidebar.text_input("Enter access code", type="password")
        if password == st.secrets.get("APP_PASSWORD", "default_pass"):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.warning("Access denied")
            st.stop()

    st.title("⚡ Ultra-RAG Assistant")
    st.markdown("""
    **Advanced document analysis with hybrid retrieval**  
    *Upload technical documents for AI-powered Q&A*
    """)

    uploaded_file = st.file_uploader(
        "📎 Upload document (PDF, max 200MB)",
        type="pdf",
        accept_multiple_files=False
    )

    if not uploaded_file:
        st.info("Please upload a document to begin")
        st.stop()

    if uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
        st.error(f"File exceeds maximum size of {MAX_FILE_SIZE_MB}MB")
        st.stop()

    if "vectorstore" not in st.session_state:
        with st.spinner("🚀 Initializing document processing..."):
            try:
                with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                    tmp_file.write(uploaded_file.read())
                    temp_path = tmp_file.name

                processor = DocumentProcessor()
                docs = processor.process(temp_path)

                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=1000,
                    chunk_overlap=200,
                    separators=["\n\n", "\nClause", "\nSection", "(?<=\\. )"]
                )
                chunks = splitter.split_documents(docs)

                vs_manager = VectorStoreManager()
                if not vs_manager.initialize_from_documents(chunks):
                    st.stop()

                st.session_state.vectorstore = vs_manager.vectorstore.as_retriever()

                st.success(f"""
                ✅ Document loaded successfully  
                Pages processed: {processor.processed_pages}  
                Content type: {'OCR' if processor.ocr_fallback else 'Native text'}
                """)

            finally:
                if 'temp_path' in locals():
                    try:
                        os.unlink(temp_path)
                    except:
                        pass

    query = st.text_input("💬 Ask about the document:")
    if query:
        with st.spinner("🔍 Retrieving information..."):
            try:
                llm = ChatOpenAI(
                    model=LLM_MODEL,
                    temperature=0.2,
                    openai_api_key=OPENAI_API_KEY,
                    max_tokens=2000
                )

                qa = RetrievalQA.from_chain_type(
                    llm=llm,
                    retriever=st.session_state.vectorstore,
                    chain_type="stuff",
                    return_source_documents=True
                )

                result = qa.invoke({"query": query})

                st.subheader("🔍 Answer")
                st.markdown(result["result"])

                st.subheader("📌 Key Sources")
                for i, doc in enumerate(result["source_documents"][:3]):
                    with st.expander(f"Source {i+1} (Page {doc.metadata.get('page', 'N/A')})"):
                        st.caption(f"**Relevance Score:** {doc.metadata.get('score', 0):.2f}")
                        st.code(doc.page_content.strip(), language="text")
            except Exception as e:
                st.error(f"Query failed: {str(e)}")

if __name__ == "__main__":
    main()
