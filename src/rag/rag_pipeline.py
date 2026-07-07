from io import BytesIO
from typing import List, Optional

try:
    from pypdf import PdfReader
except ImportError:
    PdfReader = None

from src.rag.llm import generate_answer
from src.rag.prompt import create_prompt
from src.rag.retriever import retrieve_relevant_text


def _extract_uploaded_bytes(uploaded_file) -> bytes:
    if hasattr(uploaded_file, "getvalue"):
        value = uploaded_file.getvalue()
        if isinstance(value, (bytes, bytearray)):
            return bytes(value)

    if hasattr(uploaded_file, "read"):
        try:
            uploaded_file.seek(0)
        except Exception:
            pass

        data = uploaded_file.read()
        if isinstance(data, (bytes, bytearray)):
            return bytes(data)

    return b""


def _extract_text_from_bytes(file_bytes: bytes) -> str:
    if not file_bytes:
        return ""

    try:
        if PdfReader is not None and b"%PDF" in file_bytes[:8]:
            reader = PdfReader(BytesIO(file_bytes))
            pages = [page.extract_text() or "" for page in reader.pages]
            return "\n".join(pages).strip()
    except Exception as e:
        print("PDF Extraction Error:", e)

    try:
        return file_bytes.decode("utf-8", errors="ignore")
    except Exception:
        return ""


def run_rag_pipeline(question: str, uploaded_files: Optional[List[object]] = None) -> str:

    if not uploaded_files:
        return "Please upload a resume document first."

    if not question.strip():
        return "Please enter a question."

    documents = []

    for uploaded_file in uploaded_files:
        text = _extract_text_from_bytes(
            _extract_uploaded_bytes(uploaded_file)
        )

        if text.strip():
            documents.append(text)

    if not documents:
        return "No text could be extracted from the uploaded PDF."

    context = retrieve_relevant_text(question, documents)

    print("=" * 50)
    print("QUESTION")
    print(question)
    print("=" * 50)

    print("=" * 50)
    print("CONTEXT")
    print(context)
    print("=" * 50)

    prompt = create_prompt(question, context)

    answer = generate_answer(
        prompt=prompt,
        question=question,
        context=context,
    )

    return answer