from models import RetrievedChunk

GENERATION_SYSTEM_PROMPT = (
    "You are a precise answer generator for a RAG system. "
    "Answer the question using ONLY information from the provided context chunks. "
    "Do not add any information that is not present in the chunks. "
    "Be concise and factual."
)


def build_generation_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    context = "\n\n".join(
        f"[Chunk {i + 1}]: {chunk.text}" for i, chunk in enumerate(chunks)
    )
    return (
        f"Context:\n{context}\n\n"
        f"Question: {question}\n\n"
        "Answer based solely on the context above:"
    )
