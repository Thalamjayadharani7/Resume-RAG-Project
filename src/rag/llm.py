from __future__ import annotations

import logging
import os
from typing import Any

from dotenv import load_dotenv

try:
   
    from openai import OpenAI
except ImportError as exc:
    raise ImportError(
        "Please install openai using: pip install openai"
    ) from exc
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
    """Client for communicating with OpenRouter language models."""
    def __init__(
        self,
        model_name: str | None = None,
        env_path: str | None = None,
        timeout_seconds: int = 30) -> None:
        """Initialize the Gemini client and load credentials from the environment."""
        self.model_name = (
            model_name or os.getenv(
                "MODEL_NAME",
                "meta-llama/llama-3.3-70b-instruct"
            )
        ).strip()
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

        api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        if not api_key:
            logger.error("OPENROUTER_API_KEY is missing.")
            return ""
        return api_key

    def _initialize_model(self):
        try:
            self._client = OpenAI(
                api_key=self.api_key,
                base_url="https://openrouter.ai/api/v1"
            )
            logger.info(
                "Initialized OpenRouter model '%s'",
                self.model_name,
            )
        except Exception as exc:
            logger.exception("Failed to initialize OpenRouter")
            raise GeminiAPIError(str(exc))
        
    def generate_response(self, prompt: str) -> str:
        if not prompt.strip():
            return "Prompt is empty."
        try:
            response = self._client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                temperature=0,
            )
            return response.choices[0].message.content.strip()
        except Exception as exc:
            return self._handle_generation_error(exc)
    def _handle_generation_error(self, exc: Exception) -> str:
        """Handle OpenRouter API errors."""
        error_message = str(exc).lower()
        if "401" in error_message or "invalid api key" in error_message:
            return "Invalid OpenRouter API Key."

        if "429" in error_message:
            return "OpenRouter rate limit exceeded. Please try again later."

        if "404" in error_message:
            return "Model not found."
   
        logger.exception("OpenRouter Error")
        return f"OpenRouter Error: {exc}"  
    
      
    def _extract_response_text(self, response: Any) -> str:
        ''''''


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


def generate_answer(
    prompt: str,
    question: str,
    context: str,
) -> str:

    try:
        client = GeminiClient(
     model_name=os.getenv("MODEL_NAME", "meta-llama/llama-3.3-70b-instruct")
)
        return client.generate_response(prompt)
    except Exception:
        return heuristic_answer(question, context)