"""Bootstrap ChromaDB with RAGBench data.

Run from backend/:
    poetry run python scripts/bootstrap_data.py

Loads techqa, finqa, covidqa splits from rungalileo/ragbench, embeds with
sentence-transformers/all-MiniLM-L6-v2, and stores in ./data/chroma.
"""
import hashlib
import logging
import os
import sys
from uuid import uuid4

# Allow imports from backend root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import chromadb
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "./data/chroma"
DOMAINS = ["techqa", "finqa", "covidqa"]
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 256
logger = logging.getLogger(__name__)


def bootstrap():
    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    for domain in DOMAINS:
        print(f"\n--- {domain} ---")
        dataset = load_dataset("rungalileo/ragbench", domain, split="train")

        all_ids = []
        all_texts = []
        all_metadatas = []

        for row in dataset:
            question = row["question"]
            example_id = row.get("id")
            if example_id is None or example_id == "":
                example_id = row.get("example_id")
            if example_id is None or example_id == "":
                example_id = f"{domain}-{hashlib.sha256(question.encode('utf-8')).hexdigest()}"
            answer = row.get("answer") or row.get("response") or ""

            # Documents are stored as a list of context chunks
            documents = row.get("documents") or []
            if isinstance(documents, str):
                documents = [documents]

            for chunk_idx, chunk_text in enumerate(documents):
                chunk_id = f"{example_id}_chunk_{chunk_idx}"
                all_ids.append(chunk_id)
                all_texts.append(chunk_text)
                all_metadatas.append(
                    {
                        "example_id": str(example_id),
                        "question": question,
                        "answer": answer,
                        "domain": domain,
                        # RAGBench supplies already-formed document strings without
                        # source-boundary metadata. Do not infer completeness from punctuation.
                        "chunk_completeness": "unknown",
                        "chunk_completeness_source": "unavailable",
                    }
                )

        if not all_texts:
            raise ValueError(f"No chunks found for {domain}; existing corpus was not changed")
        print(f"  Embedding {len(all_texts)} chunks...")
        embeddings = model.encode(all_texts, batch_size=BATCH_SIZE, show_progress_bar=True).tolist()

        staging_name = f"{domain}-staging-{uuid4().hex}"
        collection = client.create_collection(name=staging_name)

        try:
            for start in range(0, len(all_ids), BATCH_SIZE):
                end = start + BATCH_SIZE
                collection.add(
                    ids=all_ids[start:end],
                    documents=all_texts[start:end],
                    embeddings=embeddings[start:end],
                    metadatas=all_metadatas[start:end],
                )
            _promote_collection(client, collection, domain)
        except Exception:
            # Only discard the incomplete staging collection, never the public or backup data.
            try:
                client.delete_collection(name=staging_name)
            except chromadb.errors.NotFoundError:
                pass
            except Exception:
                logger.warning("Could not remove staging collection %s", staging_name, exc_info=True)
            raise

        sample_idx = 0
        sample_question = all_metadatas[sample_idx]["question"] if all_metadatas else "N/A"
        print(f"  Indexed: {len(all_ids)} chunks")
        print(f"  Sample question: {sample_question[:120]}")

    print("\nBootstrap complete.")


def _promote_collection(client, collection, domain):
    """Keep the previous collection until the staged replacement has its public name."""
    try:
        previous = client.get_collection(name=domain)
    except chromadb.errors.NotFoundError:
        previous = None
    backup_name = f"{domain}-backup-{uuid4().hex}"
    if previous is not None:
        previous.modify(name=backup_name)
    try:
        collection.modify(name=domain)
    except Exception:
        # Roll back any failed promotion, then propagate it; never hide database errors.
        if previous is not None:
            previous.modify(name=domain)
        raise
    if previous is not None:
        client.delete_collection(name=backup_name)


if __name__ == "__main__":
    bootstrap()
