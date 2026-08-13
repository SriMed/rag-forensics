from models import RetrievedChunk

GENERATION_SYSTEM_PROMPT = (
    "You are a precise answer generator for a RAG system. "
    "Answer the question using ONLY information from the provided context chunks. "
    "Do not add any information that is not present in the chunks. "
    "A chunk marked truncated ends before its source thought is complete. For such a chunk, "
    "do not supply or guess the missing continuation, even when it seems obvious from prior "
    "knowledge. State only what the visible text supports and disclose when truncation prevents "
    "a complete answer. A chunk marked unknown has unavailable completeness metadata; do not "
    "describe it as truncated. "
    "Be concise and factual."
)


def build_generation_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    context = "\n\n".join(
        f"[Chunk {i + 1}; completeness={chunk.completeness}; "
        f"completeness_source={chunk.completeness_source}]: {chunk.text}"
        for i, chunk in enumerate(chunks)
    )
    return (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer based solely on the context above:"
    )
