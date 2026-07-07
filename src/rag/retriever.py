import re
from typing import List


def retrieve_relevant_text(question: str, documents: List[str]) -> str:

    if not documents:
        return ""

    question_terms = [
        word
        for word in re.findall(r"\w+", question.lower())
        if len(word) > 2
    ]

    best_doc = documents[0]
    best_score = -1

    for document in documents:

        score = 0

        doc_lower = document.lower()

        for term in question_terms:
            score += doc_lower.count(term)

        if score > best_score:
            best_score = score
            best_doc = document

    if best_score <= 0:
        return best_doc

    lines = best_doc.splitlines()

    matched = []

    for line in lines:

        lower = line.lower()

        if any(term in lower for term in question_terms):
            matched.append(line.strip())

    if matched:
        return "\n".join(matched[:20])

    return best_doc