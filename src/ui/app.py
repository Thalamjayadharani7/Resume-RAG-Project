import streamlit as st

from src.rag.rag_pipeline import run_rag_pipeline


def main() -> None:
    st.set_page_config(page_title="Resume RAG System", page_icon="📄", layout="centered")
    st.title("Resume RAG System")
    st.caption("Upload one or more resume PDFs and ask a question about them.")

    uploaded_files = st.file_uploader(
        "Upload resume PDFs",
        type=["pdf"],
        accept_multiple_files=True,
    )
    question = st.text_input("Ask a question about the resume")

    if st.button("Generate Answer", type="primary"):
        if not uploaded_files:
            st.error("Please upload a resume document first.")
        elif not question.strip():
            st.error("Please enter a question.")
        else:
            with st.spinner("Analyzing the uploaded resume(s)..."):
                answer = run_rag_pipeline(question=question, uploaded_files=uploaded_files)

            st.subheader("Generated Response")
            st.write(answer)


if __name__ == "__main__":
    main()
