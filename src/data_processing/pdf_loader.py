from __future__ import annotations

import logging
import re
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
    page_texts: list[tuple[int, str]]
    candidate_name: Optional[str] = None


class PDFLoader:
    """Load and extract text from PDF files in a configurable directory."""

    def __init__(self, data_dir: Optional[Union[str, Path]] = None) -> None:
        default_dir = Path(__file__).resolve().parents[2] / "data"
        self.data_dir = Path(data_dir or default_dir).resolve()
        logger.info("Initialized PDFLoader with data directory: %s", self.data_dir)

    @staticmethod
    def _extract_candidate_name(text: str, filename: str) -> Optional[str]:
        """Infer a likely candidate name from the document text or filename."""
        if not text:
            return None

        for line in text.splitlines():
            candidate = re.sub(r"\s+", " ", line.strip(" -\t")).strip()
            if not candidate:
                continue

            lowered = candidate.lower()
            if any(token in lowered for token in ["professional summary", "education", "skills", "experience", "contact", "email", "phone", "linkedin", "github", "resume", "profile", "cv"]):
                continue
            if len(candidate.split()) > 4:
                continue
            if re.fullmatch(r"[A-Za-z][A-Za-z .'-]{1,40}", candidate):
                return candidate.title()

        stem = Path(filename).stem
        fallback = re.sub(r"[_\-\.]+", " ", stem).strip()
        if fallback and fallback.lower() not in {"resume", "profile", "cv"}:
            return re.sub(r"\s+", " ", fallback).title()
        return None

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
                page_texts: list[tuple[int, str]] = []
                text_parts: list[str] = []
                for page_number, page in enumerate(reader.pages, start=1):
                    page_text = page.extract_text() or ""
                    if page_text:
                        page_texts.append((page_number, page_text))
                        text_parts.append(page_text)

                text = "\n".join(page_text for _, page_text in page_texts).strip()
                if not text:
                    logger.warning("Skipping empty or unreadable PDF: %s", pdf_path.name)
                    continue

                candidate_name = self._extract_candidate_name(text, pdf_path.name)
                documents.append(
                    PDFDocument(
                        filename=pdf_path.name,
                        text=text,
                        page_texts=page_texts,
                        candidate_name=candidate_name,
                    )
                )
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
