"""Retriever: wraps ChromaDB collections for the RAG Forensics demo."""
import logging
import random
import chromadb
from models import StoredExample, RetrievedChunk

logger = logging.getLogger(__name__)

_CHROMA_PATH = "./data/chroma"
_client: chromadb.PersistentClient | None = None


def _get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=_CHROMA_PATH)
    return _client


def _get_collection(domain: str) -> chromadb.Collection:
    return _get_client().get_collection(name=domain)


def _find_metadata(metadatas: list, example_id: str) -> dict | None:
    for meta in metadatas:
        if meta.get("example_id") == example_id:
            return meta
    return None


def get_random_example(domain: str) -> StoredExample:
    """Return a random example from the given domain collection."""
    collection = _get_collection(domain)
    result = collection.get(include=["metadatas", "documents"])
    metadatas = result["metadatas"]
    documents = result["documents"]

    idx = random.randrange(len(metadatas))
    meta = metadatas[idx]
    text = documents[idx] if documents else ""

    return StoredExample(
        example_id=meta["example_id"],
        question=meta["question"],
        context_preview=text[:300],
    )


def _retrieve_chunks(question: str, collection: chromadb.Collection, top_k: int) -> list[RetrievedChunk]:
    query_result = collection.query(
        query_texts=[question],
        n_results=top_k,
        include=["documents", "distances"],
    )
    documents = query_result["documents"][0]
    distances = query_result["distances"][0]
    chunk_ids = query_result["ids"][0]

    if not documents:
        return []

    chunks = [
        RetrievedChunk(
            chunk_id=chunk_id,
            text=text,
            score=float(max(0.0, 1.0 - distance)),
        )
        for chunk_id, text, distance in zip(chunk_ids, documents, distances)
    ]
    chunks.sort(key=lambda c: c.score, reverse=True)
    return chunks


def retrieve(example_id: str, domain: str, top_k: int = 5) -> list[RetrievedChunk]:
    """Return up to top_k chunks most similar to the example's question, ordered by score desc."""
    collection = _get_collection(domain)
    metadatas = collection.get(include=["metadatas"])["metadatas"]
    meta = _find_metadata(metadatas, example_id)
    if meta is None:
        return []
    return _retrieve_chunks(meta["question"], collection, top_k)


_DOMAINS = ["techqa", "finqa", "covidqa"]


def retrieve_for_example(example_id: str) -> tuple[str, list[RetrievedChunk]]:
    """Return (question, chunks) for the given example_id, searching all domains."""
    for domain in _DOMAINS:
        try:
            collection = _get_collection(domain)
            metadatas = collection.get(include=["metadatas"])["metadatas"]
            meta = _find_metadata(metadatas, example_id)
            if meta is None:
                logger.debug("example_id=%s not found in domain=%s", example_id, domain)
                continue
            question = meta.get("question", "")
            chunks = _retrieve_chunks(question, collection, top_k=5)
            if chunks:
                logger.debug("found example_id=%s in domain=%s, scores=%s",
                             example_id, domain, [round(c.score, 3) for c in chunks])
                return question, chunks
        except Exception:
            logger.debug("error searching domain=%s for example_id=%s", domain, example_id, exc_info=True)
            continue
    logger.warning("example_id=%s not found in any domain", example_id)
    return "", []


def get_reference_answer(example_id: str, domain: str) -> str:
    """Return the ground-truth answer for the given example_id."""
    collection = _get_collection(domain)
    metadatas = collection.get(include=["metadatas"])["metadatas"]
    meta = _find_metadata(metadatas, example_id)
    return meta.get("answer", "") if meta else ""
