from __future__ import annotations

import logging
from pathlib import Path

from src.rag.rag_pipeline import RAGPipeline


def configure_logging() -> None:
    """Configure application-wide logging for console and file output."""
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(log_dir / "resume_rag.log", encoding="utf-8"),
        ],
    )


def _prompt_user(prompt: str) -> str | None:
    """Read a line from user input while handling EOF and Ctrl+C gracefully."""
    try:
        return input(prompt).strip()
    except EOFError:
        print("No input received. Exiting.")
        return None
    except KeyboardInterrupt:
        print("\nInterrupted. Exiting.")
        return None


def main() -> None:
    """Run the Resume RAG flow end to end from the command line."""
    configure_logging()
    logger = logging.getLogger(__name__)

    try:
        pipeline = RAGPipeline(data_dir="data")
        indexed_files = pipeline.build_index(data_dir="data")
        if indexed_files:
            logger.info(
                "Indexed %d chunks from %d file(s).",
                len(indexed_files),
                len(pipeline.list_resume_files(data_dir="data")),
            )
        else:
            logger.warning("No PDF files were indexed. Place resume PDFs in the data directory and try again.")

        available_resumes = pipeline.list_resume_files(data_dir="data")
        while True:
            question = _prompt_user("Ask a question about the resume: ")
            if question is None:
                break
            if not question:
                continue
            if question.lower() in {"exit", "quit"}:
                break

            resume_name = None
            if len(available_resumes) > 1:
                resume_name = _prompt_user(
                    "Multiple resumes found. Enter the PDF filename to use or press Enter to search all: "
                )
                if resume_name is None:
                    break
                if resume_name and resume_name not in available_resumes:
                    print("Resume not found. Please choose one of the available PDFs.")
                    continue
                if not resume_name:
                    resume_name = None

            result = pipeline.answer_question(question, resume_name=resume_name)
            print("\nAnswer:")
            print(result["answer"])
            if result.get("retrieved_context"):
                print("\nRetrieved context:")
                print(result["retrieved_context"])
    except Exception as exc:  # pragma: no cover - CLI safety net
        logger.exception("Application failed unexpectedly.")
        print(f"Sorry, something went wrong: {exc}")


if __name__ == "__main__":
    main()
