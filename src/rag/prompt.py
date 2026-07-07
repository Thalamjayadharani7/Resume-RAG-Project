from __future__ import annotations


class PromptBuilder:
    def __init__(self, system_instructions: str | None = None):
        self.system_instructions = system_instructions or (
            "You are an assistant that answers questions about resume content."
        )

    def build_prompt(self, retrieved_context: str, user_question: str) -> str:
        safe_context = retrieved_context.strip() or "No relevant context available."

        return (
            f"{self.system_instructions}\n\n"
            "Use only the provided context.\n"
            "If the answer is not present, return exactly:\n"
            '"The requested information is not available in the provided document."\n\n'
            f"Question:\n{user_question}\n\n"
            f"Resume Context:\n{safe_context}\n\n"
            "Answer:"
        )


def create_prompt(question: str, context: str) -> str:
    return PromptBuilder().build_prompt(context, question)