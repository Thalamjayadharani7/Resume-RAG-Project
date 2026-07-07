from __future__ import annotations

import logging
import os
from typing import Any

from dotenv import load_dotenv

try:
    from google import genai
    from google.genai import errors as genai_errors
except ImportError as exc:  # pragma: no cover - environment-specific
    genai = None
    genai_errors = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None

logger = logging.getLogger(__name__)


class GeminiClientError(Exception):
    """Base exception for Gemini client failures."""


class GeminiAuthenticationError(GeminiClientError):
    """Raised when the provided API key is missing or invalid."""


class GeminiAPIError(GeminiClientError):
    """Raised when the Gemini API request fails."""


class GeminiNetworkError(GeminiClientError):
    """Raised when a network issue prevents a Gemini request."""


class GeminiRateLimitError(GeminiClientError):
    """Raised when the Gemini API rate limit is exceeded."""


class GeminiClient:
    """Client for communicating with Google's Gemini language model."""

    def __init__(self, model_name: str | None = None, env_path: str | None = None, timeout_seconds: int = 30) -> None:
        """Initialize the Gemini client and load credentials from the environment."""
        self.model_name = (model_name or os.getenv("GEMINI_MODEL_NAME", os.getenv("MODEL_NAME", "gemini-2.5-flash"))).strip()
        self.env_path = env_path
        self.timeout_seconds = timeout_seconds
        self.api_key = self._load_api_key()
        self._client: Any | None = None
        if self.api_key:
            self._initialize_model()
        else:
            logger.warning("Gemini client initialized without an API key; requests will return a friendly fallback message.")

    def _load_api_key(self) -> str:
        if self.env_path:
            load_dotenv(dotenv_path=self.env_path)
        else:
            load_dotenv()

        api_key = os.getenv("GOOGLE_API_KEY", "").strip() or os.getenv("GEMINI_API_KEY", "").strip()
        if not api_key:
            logger.error("GOOGLE_API_KEY is missing.")
            return ""
        return api_key

    def _initialize_model(self) -> None:
        """Configure the Gemini SDK and initialize the client."""
        if genai is None:
            raise GeminiAPIError("The google-genai package is not installed. Install the dependency to use Gemini.") from _IMPORT_ERROR

        try:
            self._client = genai.Client(api_key=self.api_key)
            logger.info("Initialized Gemini client for model '%s'", self.model_name)
        except Exception as exc:
            logger.exception("Failed to initialize Gemini client.")
            raise GeminiAPIError(f"Failed to initialize Gemini client: {exc}") from exc

    def generate_response(self, prompt: str) -> str:
        """Send a prompt to Gemini and return the generated text response."""
        if not isinstance(prompt, str) or not prompt.strip():
            logger.warning("Received an empty prompt for Gemini.")
            return "I couldn't generate an answer because the prompt was empty."

        if not self.api_key or self._client is None:
            logger.error("Gemini client is not initialized because no valid API key is configured.")
            return "The Gemini service is unavailable right now because the API key is missing or invalid."

        try:
            response = self._client.models.generate_content(model=self.model_name, contents=prompt)
        except Exception as exc:  # pragma: no cover - depends on SDK runtime
            return self._handle_generation_error(exc)

        return self._extract_response_text(response)

    def _handle_generation_error(self, exc: Exception) -> str:
        """Translate Gemini SDK exceptions into friendly user-facing messages."""
        error_message = str(exc).lower()
        if isinstance(exc, GeminiClientError):
            raise exc

        if genai_errors is not None and isinstance(exc, genai_errors.APIError):
            if "quota" in error_message or "resource exhausted" in error_message:
                logger.warning("Gemini quota exceeded: %s", exc)
                return "The Gemini service quota has been exceeded. Please try again later."
            if "rate limit" in error_message or "429" in error_message:
                logger.warning("Gemini rate limit exceeded: %s", exc)
                return "The Gemini service is currently rate-limiting requests. Please try again shortly."
            if "invalid model" in error_message or ("model" in error_message and "not found" in error_message):
                logger.warning("Gemini model is invalid: %s", exc)
                return "The configured Gemini model is not available. Please update the model setting."
            logger.error("Gemini API request failed: %s", exc)
            return "The Gemini service could not answer the question right now. Please try again later."

        if genai_errors is not None and isinstance(exc, genai_errors.ClientError):
            logger.error("Gemini client error: %s", exc)
            return "The Gemini service rejected the request. Please verify your configuration."

        if any(token in error_message for token in ["api key", "authentication", "forbidden", "permission"]):
            logger.error("Invalid Gemini API key: %s", exc)
            return "The Gemini API key is invalid or missing. Please verify your configuration."

        if any(token in error_message for token in ["network", "connection", "timeout", "temporarily", "unavailable"]):
            logger.warning("Gemini network or timeout issue: %s", exc)
            return "The Gemini service is currently unreachable. Please check your network connection and try again."

        logger.exception("Unexpected Gemini error.")
        return "The Gemini service could not answer the question right now. Please try again later."

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
                    parts: list[str] = []
                    for part in first_candidate.content.parts:
                        if hasattr(part, "text") and part.text:
                            parts.append(part.text)
                    if parts:
                        return "".join(parts).strip()
        except Exception:
            logger.debug("Unable to parse Gemini response candidates.", exc_info=True)

        if response is None:
            logger.warning("Gemini returned an empty response.")
            return "I couldn't generate a reliable answer from the provided resume context."

        return str(response).strip() or "I couldn't generate a reliable answer from the provided resume context."


def heuristic_answer(question: str, context: str) -> str:
    if not context.strip():
        return "I couldn't find that information in the provided resume."

    question_lower = question.lower()
    lines = [line.strip() for line in context.splitlines() if line.strip()]

    for line in lines:
        line_lower = line.lower()
        if any(keyword in question_lower for keyword in ["skill", "skills"]) and "skill" in line_lower:
            return f"Based on the resume, the relevant skills mentioned are: {line}"
        if any(keyword in question_lower for keyword in ["experience", "worked", "work"]) and ("experience" in line_lower or "work" in line_lower):
            return f"The resume mentions: {line}"
        if any(keyword in question_lower for keyword in ["education", "degree", "college", "university"]) and ("education" in line_lower or "degree" in line_lower):
            return f"The resume mentions: {line}"
        if any(keyword in question_lower for keyword in ["name", "email", "phone", "contact"]) and ("@" in line or any(token in line_lower for token in ["name", "phone", "contact"])):
            return f"The resume mentions: {line}"

    return "I couldn't find that information in the provided resume."


def generate_answer(prompt: str, question: str, context: str) -> str:
    try:
        client = GeminiClient(model_name=os.getenv("GEMINI_MODEL_NAME", os.getenv("MODEL_NAME", "gemini-2.5-flash")))
        return client.generate_response(prompt)
    except Exception:
        return heuristic_answer(question, context)