"""Bootstrap ChromaDB with RAGBench data.

Run from backend/:
    poetry run python scripts/bootstrap_data.py

Loads techqa, finqa, covidqa splits from rungalileo/ragbench, embeds with
sentence-transformers/all-MiniLM-L6-v2, and stores in ./data/chroma.
"""
import sys
import os

# Allow imports from backend root
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import chromadb
from datasets import load_dataset
from sentence_transformers import SentenceTransformer

CHROMA_PATH = "./data/chroma"
DOMAINS = ["techqa", "finqa", "covidqa"]
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 256


def bootstrap():
    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    client = chromadb.PersistentClient(path=CHROMA_PATH)

    for domain in DOMAINS:
        print(f"\n--- {domain} ---")
        dataset = load_dataset("rungalileo/ragbench", domain, split="train")

        # Delete existing collection so re-runs are idempotent
        try:
            client.delete_collection(name=domain)
        except Exception:
            pass
        collection = client.create_collection(name=domain)

        all_ids = []
        all_texts = []
        all_metadatas = []

        for row in dataset:
            example_id = row.get("id") or row.get("example_id") or str(hash(row["question"]))
            question = row["question"]
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
                    }
                )

        print(f"  Embedding {len(all_texts)} chunks...")
        embeddings = model.encode(all_texts, batch_size=BATCH_SIZE, show_progress_bar=True).tolist()

        # Insert in batches to avoid memory spikes
        for start in range(0, len(all_ids), BATCH_SIZE):
            end = start + BATCH_SIZE
            collection.add(
                ids=all_ids[start:end],
                documents=all_texts[start:end],
                embeddings=embeddings[start:end],
                metadatas=all_metadatas[start:end],
            )

        sample_idx = 0
        sample_question = all_metadatas[sample_idx]["question"] if all_metadatas else "N/A"
        print(f"  Indexed: {len(all_ids)} chunks")
        print(f"  Sample question: {sample_question[:120]}")

    print("\nBootstrap complete.")


if __name__ == "__main__":
    bootstrap()
