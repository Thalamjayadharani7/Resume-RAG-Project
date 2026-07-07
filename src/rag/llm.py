<<<<<<< Updated upstream
from __future__ import annotations

import logging
import os
from typing import Any

from dotenv import load_dotenv

try:
    import google.generativeai as genai
except ImportError as exc:  # pragma: no cover - environment-specific
    genai = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

logger = logging.getLogger(__name__)


class GeminiClientError(Exception):
    """Base exception for Gemini client failures."""


class GeminiAuthenticationError(GeminiClientError):
    """Raised when the provided API key is invalid."""


class GeminiAPIError(GeminiClientError):
    """Raised when the Gemini API request fails."""


class GeminiNetworkError(GeminiClientError):
    """Raised when a network issue prevents a Gemini request."""


class GeminiClient:
    """Client for communicating with Google's Gemini language model."""

    def __init__(self, model_name: str = "gemini-1.5-flash", env_path: str | None = None) -> None:
        """Initialize the Gemini client and load credentials from the environment."""
        self.model_name = model_name
        self.env_path = env_path
        self.api_key = self._load_api_key()
        self._model: Any | None = None
        self._initialize_model()

    def _load_api_key(self) -> str:
        """Load the Google API key from a .env file or the environment."""
        if self.env_path:
            load_dotenv(dotenv_path=self.env_path)
        else:
            load_dotenv()

        api_key = os.getenv("GOOGLE_API_KEY", "").strip()
        if not api_key:
            logger.error("GOOGLE_API_KEY is missing.")
            raise GeminiAuthenticationError(
                "GOOGLE_API_KEY is not set. Please configure it in the .env file."
            )
        return api_key

    def _initialize_model(self) -> None:
        """Configure the Gemini SDK and initialize the model."""
        if genai is None:
            raise GeminiAPIError(
                "google-generativeai is not installed. Install the dependency to use Gemini."
            ) from _IMPORT_ERROR

        try:
            genai.configure(api_key=self.api_key)
            self._model = genai.GenerativeModel(self.model_name)
        except Exception as exc:
            logger.exception("Failed to initialize Gemini model.")
            raise GeminiAPIError(f"Failed to initialize Gemini model: {exc}") from exc

    def generate_response(self, prompt: str) -> str:
        """Send a prompt to Gemini and return the generated text response."""
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("prompt must be a non-empty string.")

        if self._model is None:
            raise GeminiAPIError("Gemini model is not initialized.")

        try:
            response = self._model.generate_content(prompt)
        except Exception as exc:
            error_message = str(exc).lower()
            if "api key" in error_message or "invalid" in error_message:
                logger.error("Invalid Gemini API key.")
                raise GeminiAuthenticationError("Invalid Google Gemini API key.") from exc
            if any(token in error_message for token in ["network", "connection", "timeout", "temporarily"]):
                logger.error("Network error while calling Gemini.")
                raise GeminiNetworkError("Network error while communicating with Gemini.") from exc

            logger.exception("Gemini API request failed.")
            raise GeminiAPIError(f"Gemini API request failed: {exc}") from exc

        return self._extract_response_text(response)

    def _extract_response_text(self, response: Any) -> str:
        """Extract readable text from the Gemini response object."""
        try:
            text = getattr(response, "text", None)
            if isinstance(text, str) and text.strip():
                return text.strip()
        except Exception:
            logger.debug("Unable to read response.text attribute.", exc_info=True)

        try:
            if hasattr(response, "candidates") and response.candidates:
                first_candidate = response.candidates[0]
                if hasattr(first_candidate, "content") and hasattr(first_candidate.content, "parts"):
                    parts = []
                    for part in first_candidate.content.parts:
                        if hasattr(part, "text") and part.text:
                            parts.append(part.text)
                    if parts:
                        return "".join(parts).strip()
        except Exception:
            logger.debug("Unable to parse Gemini response candidates.", exc_info=True)

        if response is None:
            raise GeminiAPIError("Gemini returned an empty response.")

        return str(response).strip()
=======
import os


def generate_answer(prompt: str, question: str, context: str) -> str:
    try:
        import google.generativeai as genai
    except Exception:
        return heuristic_answer(question, context)

    api_key = os.getenv("GOOGLE_API_KEY")
    if not api_key:
        return heuristic_answer(question, context)

    try:
        genai.configure(api_key=api_key)
        model_name = os.getenv("MODEL_NAME", "gemini-1.5-flash")
        model = genai.GenerativeModel(model_name)
        response = model.generate_content(prompt)
        answer = getattr(response, "text", None)
        if answer and str(answer).strip():
            return str(answer).strip()
    except Exception:
        pass

    return heuristic_answer(question, context)


def heuristic_answer(question: str, context: str) -> str:
    question_lower = question.lower()
    lines = [line.strip() for line in context.splitlines() if line.strip()]

    if any(keyword in question_lower for keyword in ["skill", "skills"]):
        for line in lines:
            if "skill" in line.lower():
                return f"Based on the resume, the relevant skills mentioned are: {line}"

    if any(keyword in question_lower for keyword in ["experience", "worked", "work"]):
        for line in lines:
            if "experience" in line.lower() or "work" in line.lower():
                return f"The resume mentions: {line}"

    if any(keyword in question_lower for keyword in ["education", "degree", "college", "university"]):
        for line in lines:
            if "education" in line.lower() or "degree" in line.lower():
                return f"The resume mentions: {line}"

    if any(keyword in question_lower for keyword in ["name", "email", "phone", "contact"]):
        for line in lines:
            if "@" in line or any(token in line.lower() for token in ["name", "phone", "contact"]):
                return f"The resume mentions: {line}"

    return "The requested information is not available in the provided document."
>>>>>>> Stashed changes
