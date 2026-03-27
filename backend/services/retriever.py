"""Retriever: wraps ChromaDB collections for the RAG Forensics demo."""
import random
import chromadb
from models import StoredExample, RetrievedChunk

_CHROMA_PATH = "./data/chroma"
_client: chromadb.PersistentClient | None = None


def _get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=_CHROMA_PATH)
    return _client


def _get_collection(domain: str) -> chromadb.Collection:
    return _get_client().get_collection(name=domain)


def get_random_example(domain: str) -> StoredExample:
    """Return a random example from the given domain collection."""
    collection = _get_collection(domain)
    result = collection.get(include=["metadatas", "documents"])
    metadatas = result["metadatas"]
    ids = result["ids"]
    documents = result["documents"]

    idx = random.randrange(len(metadatas))
    meta = metadatas[idx]
    example_id = meta["example_id"]
    question = meta["question"]
    text = documents[idx] if documents else ""
    context_preview = text[:300]

    return StoredExample(
        example_id=example_id,
        question=question,
        context_preview=context_preview,
    )


def retrieve(example_id: str, domain: str, top_k: int = 5) -> list[RetrievedChunk]:
    """Return up to top_k chunks most similar to the example's question, ordered by score desc."""
    collection = _get_collection(domain)

    meta_result = collection.get(include=["metadatas"])
    metadatas = meta_result["metadatas"]

    question = None
    for meta in metadatas:
        if meta.get("example_id") == example_id:
            question = meta.get("question")
            break

    if question is None:
        return []

    query_result = collection.query(
        query_texts=[question],
        n_results=top_k,
        include=["documents", "distances"],
    )

    documents = query_result["documents"][0]
    distances = query_result["distances"][0]
    ids = query_result["ids"][0]

    if not documents:
        return []

    chunks = []
    for chunk_id, text, distance in zip(ids, documents, distances):
        # Convert L2 distance to a similarity score in [0, 1].
        score = float(max(0.0, 1.0 - distance))
        chunks.append(RetrievedChunk(chunk_id=chunk_id, text=text, score=score))

    chunks.sort(key=lambda c: c.score, reverse=True)
    return chunks


def get_reference_answer(example_id: str, domain: str) -> str:
    """Return the ground-truth answer for the given example_id."""
    collection = _get_collection(domain)
    result = collection.get(include=["metadatas"])
    for meta in result["metadatas"]:
        if meta.get("example_id") == example_id:
            return meta.get("answer", "")
    return ""
