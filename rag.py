
from uuid import uuid4
from dotenv import load_dotenv
from pathlib import Path
import os

from prompt import PROMPT  # Make sure this imports your ChatPromptTemplate 

from langchain_unstructured import UnstructuredLoader        
from langchain_community.document_loaders import PyPDFLoader # For PDF files
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain

# ----------- .env Loading -------------
load_dotenv()

# ----------- Constants ----------------
CHUNK_SIZE = 1000
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
VECTORSTORE_DIR = Path("resources/vectorstore")
COLLECTION_NAME = "real_estate"

# ----------- Globals ------------------
llm = None
vector_store = None

# ----------- Initialization -----------
def initialize_components():
    global llm, vector_store

    if llm is None:
        llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.9, max_tokens=500)

    if vector_store is None:
        ef = HuggingFaceEmbeddings(
            model_name=EMBEDDING_MODEL,
            model_kwargs={"trust_remote_code": True}
        )
        vector_store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=ef,
            persist_directory=str(VECTORSTORE_DIR)
        )

# ----------- Document Processing -------
def process_documents(file_paths):
    initialize_components()

    # Example: Load documents (you need to implement actual loading logic)
    docs = []
    for file_path in file_paths:
        ext = Path(file_path).suffix.lower()
        if ext == ".pdf":
            loader = PyPDFLoader(file_path)
        else:
            loader = UnstructuredLoader(file_path)
        docs.extend(loader.load())

    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=200)
    split_docs = text_splitter.split_documents(docs)

    # Sanitize metadata: convert any list metadata with one string item to just that string,
    # and convert other non-primitive values to strings (or remove as needed)
    for doc in split_docs:
        cleaned_metadata = {}
        for k, v in doc.metadata.items():
            if isinstance(v, (str, int, float, bool)) or v is None:
                cleaned_metadata[k] = v
            elif isinstance(v, list):
                if len(v) == 1 and isinstance(v[0], str):
                    cleaned_metadata[k] = v[0]  # Single-item list → string
                else:
                    cleaned_metadata[k] = str(v)  # Convert list or other complex to string
            else:
                cleaned_metadata[k] = str(v)  # Convert other types to string
        doc.metadata = cleaned_metadata

    uuids = [str(uuid4()) for _ in split_docs]
    vector_store.add_documents(split_docs, ids=uuids)


# ----------- RAG QA Chain --------------
def generate_answer(query):
    initialize_components()
    if not vector_store:
        raise RuntimeError("Vector database is not initialized")

    # Make sure PROMPT uses context and input (see below)
    stuff_chain = create_stuff_documents_chain(llm, PROMPT)

    retriever = vector_store.as_retriever()
    rag_chain = create_retrieval_chain(retriever, stuff_chain)

    result = rag_chain.invoke({"input": query})

    answer = result.get("answer", "")

    sources_docs = [
        doc.metadata.get("source", "unknown")
        for doc in result.get("context", [])
    ]

    return answer, sources_docs

# ---------- MAIN TEST -----------
if __name__ == "__main__":
    local_files = [
        r"C:\Users\prasa\Downloads\Realtor_Housing_Forecast_2025.docx",
        # Add more local docx/txt/pdf/etc paths as needed
    ]
    process_documents(local_files)
    

    # Test query:
    question = "What is the Inventory Growth?"
    answer, sources = generate_answer(question)
    print(f"\nAnswer: {answer}")
    print(f"Sources: {sources}")










