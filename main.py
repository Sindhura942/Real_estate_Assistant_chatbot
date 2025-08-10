import os
import streamlit as st
from rag import process_documents, generate_answer

st.title("Real Estate Research Assistant")

# Sidebar for uploading documents
uploaded_files = st.sidebar.file_uploader(
    "Upload documents (PDF, DOCX, TXT)", 
    type=["pdf", "docx", "txt"], 
    accept_multiple_files=True
)

process_button = st.sidebar.button("Process Documents")
status_placeholder = st.empty()

if process_button:
    temp_dir = "uploaded_docs"
    os.makedirs(temp_dir, exist_ok=True)
    file_paths = []
    if uploaded_files:
        for file in uploaded_files:
            save_path = os.path.join(temp_dir, file.name)
            with open(save_path, "wb") as f:
                f.write(file.read())
            file_paths.append(save_path)
        # Process the uploaded documents
        process_documents(file_paths)
        status_placeholder.success("Documents processed and added to the knowledge base.")
    else:
        status_placeholder.error("Please upload at least one document.")

# Input for user question
query = st.text_input("Ask a question about the uploaded documents:")
if query:
    try:
        answer, sources = generate_answer(query)
        st.header("Answer")
        st.write(answer)
        if sources:
            st.subheader("Sources")
            for source in sources:
                st.write(os.path.basename(source))
    except Exception as e:
        st.error(f"Error: {e}")
