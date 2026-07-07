from __future__ import annotations


class PromptBuilder:
    def __init__(self, system_instructions: str | None = None):
        self.system_instructions = system_instructions or (
            "You are an assistant that answers questions about resume content."
        )

    def build_prompt(self, retrieved_context: str, user_question: str) -> str:
        normalized_context = retrieved_context.strip()
        if not normalized_context:
            return (
                "Answer the user's question using the retrieved context only. "
                "If the answer is not available, return exactly: "
                '"I couldn\'t find that information in the provided resume."'
            )

        return (
            f"{self.system_instructions}\n\n"
            "Answer the question using only the retrieved resume context.\n"
            "You may use information from any section of the resume, including professional summary, skills, education, projects, certifications, experience, achievements, and interests.\n"
            "Do not require exact wording matches; paraphrase or infer from the surrounding context when appropriate.\n"
            "Never use outside knowledge or guess.\n"
            "If the answer is not present in the retrieved context, return exactly: "
            '"I couldn\'t find that information in the provided resume."\n\n'
            f"User Question:\n{user_question.strip()}\n\n"
            f"Retrieved Context:\n{normalized_context}\n\n"
            "Answer:"
        )


def create_prompt(question: str, context: str) -> str:
    return PromptBuilder().build_prompt(context, question)