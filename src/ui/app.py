import streamlit as st

from src.rag.rag_pipeline import run_rag_pipeline


def main():

    st.set_page_config(
        page_title="Resume RAG",
        page_icon="📄",
        layout="centered",
    )

    st.title("Resume RAG System")

    uploaded_files = st.file_uploader(
        "Upload Resume PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )

    question = st.text_input("Ask a question")

    if st.button("Generate Answer"):

        if not uploaded_files:
            st.error("Please upload a resume.")
            return

        if not question.strip():
            st.error("Please enter a question.")
            return

        with st.spinner("Generating answer..."):

            answer = run_rag_pipeline(
                question=question,
                uploaded_files=uploaded_files,
            )

        st.subheader("Generated Response")
        st.write(answer)


if __name__ == "__main__":
    main()