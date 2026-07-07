from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)

try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - handled at runtime
    PdfReader = None


@dataclass
class PDFDocument:
    """Represents a single PDF document with its extracted text."""

    filename: str
    text: str


class PDFLoader:
    """Load and extract text from PDF files in a configurable directory."""

    def __init__(self, data_dir: Optional[Union[str, Path]] = None) -> None:
        default_dir = Path(__file__).resolve().parents[2] / "data"
        self.data_dir = Path(data_dir or default_dir).resolve()
        logger.info("Initialized PDFLoader with data directory: %s", self.data_dir)

    def load_documents(self, pattern: str = "*.pdf") -> list[PDFDocument]:
        """Load all PDF files from the configured directory and extract text."""
        if PdfReader is None:
            raise ImportError("pypdf is required to read PDF files.")

        if not self.data_dir.exists():
            logger.warning("Data directory does not exist: %s", self.data_dir)
            return []

        pdf_files = sorted(self.data_dir.rglob(pattern))
        if not pdf_files:
            logger.warning("No PDF files found in %s", self.data_dir)
            return []

        documents: list[PDFDocument] = []
        for pdf_path in pdf_files:
            if not pdf_path.is_file():
                continue

            try:
                reader = PdfReader(str(pdf_path))
                text_parts: list[str] = []
                for page in reader.pages:
                    page_text = page.extract_text() or ""
                    if page_text:
                        text_parts.append(page_text)

                text = "\n".join(text_parts).strip()
                if not text:
                    logger.warning("Skipping empty or unreadable PDF: %s", pdf_path.name)
                    continue

                documents.append(PDFDocument(filename=pdf_path.name, text=text))
                logger.info("Loaded %s (%d characters)", pdf_path.name, len(text))
            except (FileNotFoundError, PermissionError, ValueError, RuntimeError) as exc:
                logger.warning("Skipped unreadable PDF %s: %s", pdf_path.name, exc)
            except Exception as exc:  # pragma: no cover - logging branch
                logger.exception("Failed to extract text from %s: %s", pdf_path.name, exc)

        return documents

    def list_pdf_files(self, pattern: str = "*.pdf") -> list[Path]:
        """Return the available PDF file paths for the configured data directory."""
        if not self.data_dir.exists():
            return []
        return sorted(path for path in self.data_dir.rglob(pattern) if path.is_file())
