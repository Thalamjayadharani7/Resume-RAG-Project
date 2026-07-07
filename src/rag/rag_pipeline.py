from __future__ import annotations

import logging
from typing import Any, Optional

from src.data_processing.chunking import TextChunker
from src.data_processing.pdf_loader import PDFLoader
from src.rag.llm import GeminiClient
from src.rag.prompt import PromptBuilder
from src.rag.retriever import Retriever

logger = logging.getLogger(__name__)


class RAGPipeline:
    """Coordinate the Resume RAG flow from PDF ingestion to answer generation."""

    def __init__(
        self,
        retriever: Optional[Retriever] = None,
        prompt_builder: Optional[PromptBuilder] = None,
        llm_client: Optional[GeminiClient] = None,
        data_dir: Optional[str] = None,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        top_k: int = 5,
    ) -> None:
        self.data_dir = data_dir
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.top_k = top_k

        self.retriever = retriever or Retriever(top_k=top_k)
        self.prompt_builder = prompt_builder or PromptBuilder()
        self.llm_client = llm_client

    def list_resume_files(self, data_dir: Optional[str] = None) -> list[str]:
        """Return the available PDF filenames in the configured data directory."""
        loader = PDFLoader(data_dir=data_dir or self.data_dir)
        return [pdf_path.name for pdf_path in loader.list_pdf_files()]

    def build_index(self, data_dir: Optional[str] = None, pattern: str = "*.pdf") -> list[str]:
        """Load PDFs, chunk them, embed them, and store them in the vector database."""
        loader = PDFLoader(data_dir=data_dir or self.data_dir)
        documents = loader.load_documents(pattern=pattern)
        if not documents:
            logger.warning("No resume documents were loaded for indexing.")
            return []

        chunker = TextChunker(chunk_size=self.chunk_size, chunk_overlap=self.chunk_overlap)
        chunks = chunker.chunk_documents(documents)
        if not chunks:
            logger.warning("No chunks were created from the loaded resumes.")
            return []

        return self.retriever.index_documents(chunks)

    def _get_llm_client(self) -> GeminiClient:
        if self.llm_client is None:
            self.llm_client = GeminiClient()
        return self.llm_client

    def answer_question(self, question: str, resume_name: Optional[str] = None) -> dict[str, Any]:
        """Retrieve relevant context, build a prompt, and return the generated answer."""
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a non-empty string")

        retrieved_context = self.retriever.retrieve_context(question, resume_name=resume_name)
        if not retrieved_context.strip():
            answer = "I couldn't find that information in the provided resume."
        else:
            prompt = self.prompt_builder.build_prompt(retrieved_context, question)
            answer = self._get_llm_client().generate_response(prompt)

        return {
            "question": question,
            "retrieved_context": retrieved_context,
            "answer": answer,
        }


def run_rag_pipeline(question: str, uploaded_files: Optional[list[object]] = None) -> str:
    """Compatibility wrapper for the indexed resume pipeline."""
    if not question or not str(question).strip():
        return "Please enter a question."

    pipeline = RAGPipeline(data_dir="data")
    result = pipeline.answer_question(question)
    return result["answer"]
