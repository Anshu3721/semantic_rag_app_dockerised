"""
app.py

Streamlit UI for semantic document Q&A:
- Upload document
- Ask question
- View relevant context (highlighted)
"""

import streamlit as st
import tempfile
from rag_pipeline import RAGPipeline
from utils import highlight_text, show_error

st.set_page_config(page_title="Semantic Document Q&A", layout="wide")
st.title("Semantic Document Q&A App")

if 'rag' not in st.session_state:
    st.session_state['rag'] = RAGPipeline()

uploaded_file = st.file_uploader(
    "Upload a .txt or .pdf file",
    type=["txt", "pdf"],
    help="Only .txt and .pdf files supported"
)

if uploaded_file:
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix="." + uploaded_file.name.split('.')[-1]) as tmp:
            tmp.write(uploaded_file.read())
            file_path = tmp.name
        st.session_state['rag'].process_file(file_path)
        st.success("Document processed! Enter your question below.")
    except Exception as e:
        show_error(f"Error processing file: {e}")

if st.session_state['rag'].vectorstore:
    st.subheader("Ask a question about your document")
    user_question = st.text_input("Question:")
    if st.button("Get Answer") and user_question:
        try:
            with st.spinner("Retrieving relevant context..."):
                chunk_texts = st.session_state['rag'].answer_question(user_question)
            st.markdown("### Relevant Context")
            for i, chunk_text in enumerate(chunk_texts):
                st.markdown(f"**Chunk {i+1}:**")
                st.markdown(
                    highlight_text(chunk_text, [user_question]), unsafe_allow_html=True
                )
        except Exception as e:
            show_error(f"Error retrieving context: {e}")
else:
    st.info("Please upload a document to start.")
# added one line