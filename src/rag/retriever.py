import re
from typing import List


def retrieve_relevant_text(question: str, documents: List[str]) -> str:
    if not documents:
        return ""

    question_terms = [term for term in re.findall(r"[a-zA-Z0-9]+", question.lower()) if len(term) > 2]
    if not question_terms:
        return documents[0]

    best_document = documents[0]
    best_score = -1

    for document in documents:
        doc_lower = document.lower()
        score = sum(1 for term in question_terms if term in doc_lower)
        if score > best_score:
            best_score = score
            best_document = document

    if best_score <= 0:
        return documents[0]

    lines = [line.strip() for line in best_document.splitlines() if line.strip()]
    relevant_lines = []
    for line in lines:
        line_lower = line.lower()
        if any(term in line_lower for term in question_terms):
            relevant_lines.append(line)

    if relevant_lines:
        return "\n".join(relevant_lines[:5])

    return "\n".join(lines[:5])
