from __future__ import annotations

import streamlit as st

from src.rag.rag_pipeline import RAGPipeline


st.set_page_config(page_title="Resume RAG", page_icon="📄", layout="wide")
st.title("Resume RAG Assistant")

if "pipeline" not in st.session_state:
    st.session_state.pipeline = RAGPipeline(data_dir="data")

pipeline = st.session_state.pipeline

with st.sidebar:
    st.header("Options")
    resume_name = st.text_input("Resume filename (optional)", value="")
    if st.button("Build index"):
        with st.spinner("Indexing resumes..."):
            indexed = pipeline.build_index(data_dir="data")
        st.success(f"Indexed {len(indexed)} chunks.")

question = st.text_area("Ask a question about the resume", height=120)
if st.button("Ask") and question.strip():
    with st.spinner("Searching the resume..."):
        result = pipeline.answer_question(question, resume_name=resume_name or None)
    st.subheader("Answer")
    st.write(result["answer"])
    if result.get("retrieved_context"):
        with st.expander("Retrieved context"):
            st.write(result["retrieved_context"])
