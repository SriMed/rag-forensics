"""Prompts for query-corpus fit analysis (Issue #8)."""


def build_question_generation_prompt(chunk_texts: str, original_question: str) -> str:
    """Build the question generation prompt.

    Uses f-string concatenation rather than str.format() so that brace characters
    in chunk_texts or original_question are never interpreted as format placeholders.
    """
    return (
        "You are analyzing retrieved document chunks from a knowledge base.\n\n"
        f"Retrieved chunks:\n{chunk_texts}\n\n"
        f'The user asked: "{original_question}"\n\n'
        "Generate 3-5 specific questions that these chunks would answer well.\n"
        "Each question should be:\n"
        "- Directly answerable from the chunk content above\n"
        "- Specific, not generic\n"
        "- Phrased the way a real user would ask it\n\n"
        "Return only a JSON array of question strings. No explanation, no preamble.\n"
        'Example format: ["What is X?", "How does Y work?", "When was Z introduced?"]'
    )
