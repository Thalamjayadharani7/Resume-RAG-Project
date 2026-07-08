from __future__ import annotations

import streamlit as st

from src.rag.rag_pipeline import RAGPipeline


def main() -> None:
    st.set_page_config(
        page_title="Resume RAG",
        page_icon="📄",
        layout="wide",
    )

    st.title("📄 Resume RAG Assistant")
    st.markdown("Ask questions about one or more resumes using Retrieval-Augmented Generation (RAG).")

    # Initialize pipeline only once
    if "pipeline" not in st.session_state:
        st.session_state.pipeline = RAGPipeline(data_dir="data")

    pipeline = st.session_state.pipeline

    # Sidebar
    with st.sidebar:
        st.header("Options")

        resume_name = st.text_input(
            "Resume filename (optional)",
            placeholder="e.g. jaya.pdf",
        )

        if st.button("Build Index"):
            with st.spinner("Building vector index..."):
                indexed = pipeline.build_index(data_dir="data")

            st.success(f"Indexed {len(indexed)} chunks successfully!")

    st.divider()

    # Question Input
    question = st.text_area(
        "Ask a question",
        placeholder="Example: What are the candidate's skills?",
        height=120,
    )

    if st.button("Generate Answer"):

        if not question.strip():
            st.warning("Please enter a question.")
            st.stop()

        with st.spinner("Searching resumes..."):

            result = pipeline.answer_question(
                question=question,
                resume_name=resume_name.strip() or None,
            )

        st.subheader("Answer")
        st.write(result["answer"])

        if result.get("retrieved_context"):
            with st.expander("Retrieved Context"):
                st.write(result["retrieved_context"])


if __name__ == "__main__":
    main()