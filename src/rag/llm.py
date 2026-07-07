from __future__ import annotations

import logging
import os
from typing import Any

from dotenv import load_dotenv

try:
    import google.generativeai as genai
except ImportError as exc:
    genai = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

logger = logging.getLogger(__name__)


class GeminiClientError(Exception):
    pass


class GeminiAuthenticationError(GeminiClientError):
    pass


class GeminiAPIError(GeminiClientError):
    pass


class GeminiNetworkError(GeminiClientError):
    pass


class GeminiClient:
    def __init__(self, model_name: str = "gemini-1.5-flash", env_path: str | None = None):
        self.model_name = model_name
        self.env_path = env_path
        self.api_key = self._load_api_key()
        self._model: Any | None = None
        self._initialize_model()

    def _load_api_key(self) -> str:
        if self.env_path:
            load_dotenv(dotenv_path=self.env_path)
        else:
            load_dotenv()

        api_key = os.getenv("GOOGLE_API_KEY", "").strip()

        if not api_key:
            raise GeminiAuthenticationError(
                "GOOGLE_API_KEY is not set. Please configure it in the .env file."
            )

        return api_key

    def _initialize_model(self):
        if genai is None:
            raise GeminiAPIError(
                "google-generativeai is not installed."
            ) from _IMPORT_ERROR

        genai.configure(api_key=self.api_key)
        self._model = genai.GenerativeModel(self.model_name)

    def generate_response(self, prompt: str) -> str:
        response = self._model.generate_content(prompt)
        return self._extract_response_text(response)

    def _extract_response_text(self, response: Any) -> str:
        text = getattr(response, "text", None)

        if text and str(text).strip():
            return str(text).strip()

        if hasattr(response, "candidates") and response.candidates:
            candidate = response.candidates[0]

            if hasattr(candidate, "content") and hasattr(candidate.content, "parts"):
                parts = [
                    part.text
                    for part in candidate.content.parts
                    if hasattr(part, "text") and part.text
                ]

                if parts:
                    return "".join(parts).strip()

        return str(response).strip()


def heuristic_answer(question: str, context: str) -> str:
    question_lower = question.lower()
    lines = [line.strip() for line in context.splitlines() if line.strip()]

    if any(k in question_lower for k in ["skill", "skills"]):
        for line in lines:
            if "skill" in line.lower():
                return f"Based on the resume, the relevant skills mentioned are: {line}"

    if any(k in question_lower for k in ["experience", "worked", "work"]):
        for line in lines:
            if "experience" in line.lower() or "work" in line.lower():
                return f"The resume mentions: {line}"

    if any(k in question_lower for k in ["education", "degree", "college", "university"]):
        for line in lines:
            if "education" in line.lower() or "degree" in line.lower():
                return f"The resume mentions: {line}"

    if any(k in question_lower for k in ["name", "email", "phone", "contact"]):
        for line in lines:
            if "@" in line or any(
                token in line.lower()
                for token in ["name", "phone", "contact"]
            ):
                return f"The resume mentions: {line}"

    return "The requested information is not available in the provided document."


def generate_answer(prompt: str, question: str, context: str) -> str:
    try:
        client = GeminiClient(
            model_name=os.getenv("MODEL_NAME", "gemini-1.5-flash")
        )
        return client.generate_response(prompt)
    except Exception:
        return heuristic_answer(question, context)