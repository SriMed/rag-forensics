"""Prompts for query-corpus fit analysis (Issue #8)."""

QUESTION_GENERATION_PROMPT = """\
You are analyzing retrieved document chunks from a knowledge base.

Retrieved chunks:
{chunk_texts}

The user asked: "{original_question}"

Generate 3-5 specific questions that these chunks would answer well.
Each question should be:
- Directly answerable from the chunk content above
- Specific, not generic
- Phrased the way a real user would ask it

Return only a JSON array of question strings. No explanation, no preamble.
Example format: ["What is X?", "How does Y work?", "When was Z introduced?"]"""
