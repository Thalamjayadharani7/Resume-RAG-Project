from __future__ import annotations


class PromptBuilder:
    """Build structured prompts for resume-based retrieval augmented generation."""

    def __init__(self, system_instructions: str | None = None) -> None:
        """Initialize the prompt builder with reusable system instructions."""
        self.system_instructions = system_instructions or (
            "You are a precise assistant that answers questions using only the "
            "provided retrieved context."
        )

    def build_prompt(self, retrieved_context: str, user_question: str) -> str:
        """Create a complete prompt for the language model."""
        if not isinstance(retrieved_context, str):
            raise TypeError("retrieved_context must be a string.")
        if not isinstance(user_question, str):
            raise TypeError("user_question must be a string.")

        normalized_context = retrieved_context.strip()
        if not normalized_context:
            return (
                "Answer the user's question using the retrieved context only. "
                "If the answer is not available, return exactly: "
                '"I couldn\'t find that information in the provided resume."'
            )

        return (
            "System Instructions\n"
            f"{self.system_instructions}\n\n"
            "Retrieved Context\n"
            f"{normalized_context}\n\n"
            "User Question\n"
            f"{user_question.strip()}\n\n"
            "Clear Answering Rules\n"
            "- Answer ONLY using the retrieved resume context.\n"
            "- Never use outside knowledge.\n"
            "- Never hallucinate.\n"
            "- Never guess.\n"
            "- If the answer is not available in the retrieved context, return exactly: "
            '"I couldn\'t find that information in the provided resume."'
        )
