# 🏠 Real Estate Research Assistant (RAG + Streamlit)

An AI-powered **Real Estate Research Assistant** that uses **Retrieval-Augmented Generation (RAG)** to answer questions about uploaded documents (PDF, DOCX, TXT).  
Built with **LangChain**, **Chroma Vector Database**, **Groq LLaMA 3.3-70B**, and **HuggingFace embeddings**, this tool processes, stores, and retrieves relevant information to answer user queries with supporting sources.

---

## 🚀 Features
- 📄 **Upload** PDF, DOCX, and TXT documents via a Streamlit interface.
- 🔍 **Extract & Chunk** text for semantic search.
- 🗄 **Persist** embeddings in Chroma DB for fast reloading without reprocessing.
- 🤖 **Answer questions** using Groq LLaMA 3.3 with retrieved document context.
- 📌 **Show sources** for every answer.

---

## 📂 Project Structure
├── main.py # Streamlit UI for uploading documents and asking questions
├── rag.py # RAG pipeline for document processing and query answering
├── prompt.py # Custom prompt templates for the LLM
├── requirements.txt # Python dependencies
├── .env # Environment variables (API keys)
└── resources/
└── vectorstore/ # Chroma persistent vector DB (auto-created)


---

## 🔑 Requirements
- **Python 3.10+**
- Groq API key (`GROQ_API_KEY`)
- HuggingFace transformers for embeddings
- LangChain & Chroma
- Streamlit for the UI

python -m venv venv
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate
🛠 How It Works
1. Document Loading
PDFs → PyPDFLoader

DOCX/TXT → UnstructuredFileLoader

2. Text Chunking
RecursiveCharacterTextSplitter

Chunk size: 1000 characters

Overlap: 200 characters

3. Embeddings & Vector Store
Embedding model: sentence-transformers/all-MiniLM-L6-v2

Vector DB: Chroma (stored in resources/vectorstore/)

4. Retrieval & Answering
Retriever: vector_store.as_retriever()

LLM: Groq LLaMA 3.3-70B via ChatGroq

Custom prompt from prompt.py

📜 Example Prompt
python
Copy
Edit
from langchain_core.prompts import ChatPromptTemplate

PROMPT = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a helpful assistant for RealEstate research. "
     "Use the given context to answer the question. "
     "If you don't know the answer, say you don't know. "
     "Limit your answer to three sentences.\n\nContext:\n{context}"),
    ("human", "{input}")
])
✅ Example Run
User Query:

What is the inventory growth?

AI Answer:

The inventory of homes for sale grew by 14% year-over-year in 2025.

Sources:
Realtor_Housing_Forecast_2025.docx

📌 Notes
To reset, delete resources/vectorstore and reprocess your docs.

Ensure your Groq API key has sufficient quota.

All document metadata is cleaned to avoid vector store serialization errors.

📋 requirements.txt
txt
Copy
Edit
streamlit
python-dotenv
langchain
langchain-community
langchain-chroma
langchain-groq
langchain-huggingface
sentence-transformers
pypdf
unstructured
