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
    """Base exception for Gemini client."""


class GeminiAuthenticationError(GeminiClientError):
    """Raised when API key is invalid or missing."""


class GeminiAPIError(GeminiClientError):
    """Raised for Gemini API errors."""


class GeminiNetworkError(GeminiClientError):
    """Raised for network-related errors."""


class GeminiClient:

    def __init__(
        self,
        model_name: str = "gemini-1.5-flash",
        env_path: str | None = None,
    ):
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
                "GOOGLE_API_KEY is not set in the .env file."
            )

        return api_key

    def _initialize_model(self):

        if genai is None:
            raise GeminiAPIError(
                "google-generativeai package is not installed."
            ) from _IMPORT_ERROR

        try:
            genai.configure(api_key=self.api_key)
            self._model = genai.GenerativeModel(self.model_name)

        except Exception as e:
            raise GeminiAPIError(f"Unable to initialize Gemini model: {e}")

    def generate_response(self, prompt: str) -> str:

        try:
            response = self._model.generate_content(prompt)
            return self._extract_response_text(response)

        except Exception as e:
            raise GeminiAPIError(str(e))

    def _extract_response_text(self, response: Any) -> str:

        text = getattr(response, "text", None)

        if text and str(text).strip():
            return str(text).strip()

        if hasattr(response, "candidates") and response.candidates:

            candidate = response.candidates[0]

            if (
                hasattr(candidate, "content")
                and hasattr(candidate.content, "parts")
            ):

                parts = []

                for part in candidate.content.parts:

                    if hasattr(part, "text") and part.text:
                        parts.append(part.text)

                if parts:
                    return "".join(parts).strip()

        return str(response)


def heuristic_answer(question: str, context: str) -> str:

    question = question.lower()

    lines = [line.strip() for line in context.splitlines() if line.strip()]

    if "skill" in question:
        for line in lines:
            if "skill" in line.lower():
                return line

    if any(word in question for word in ["experience", "worked", "work"]):
        for line in lines:
            if "experience" in line.lower():
                return line

    if any(word in question for word in ["education", "college", "degree"]):
        for line in lines:
            if (
                "education" in line.lower()
                or "college" in line.lower()
                or "degree" in line.lower()
            ):
                return line

    if any(word in question for word in ["email", "phone", "contact"]):
        for line in lines:
            if (
                "@"
                in line
                or "phone" in line.lower()
                or "contact" in line.lower()
            ):
                return line

    return "The requested information is not available in the provided document."


def generate_answer(
    prompt: str,
    question: str,
    context: str,
) -> str:

    try:

        print("=" * 80)
        print("PROMPT SENT TO GEMINI")
        print("=" * 80)
        print(prompt)

        client = GeminiClient(
            model_name=os.getenv(
                "MODEL_NAME",
                "gemini-1.5-flash",
            )
        )

        answer = client.generate_response(prompt)

        print("=" * 80)
        print("GEMINI RESPONSE")
        print("=" * 80)
        print(answer)

        return answer

    except Exception as e:

        print("=" * 80)
        print("GEMINI ERROR")
        print("=" * 80)
        print(type(e).__name__)
        print(e)

        print("=" * 80)
        print("USING HEURISTIC FALLBACK")
        print("=" * 80)

        return heuristic_answer(question, context)