"""Calibrate signal weight normalization constants against RAGBench data.

Runs the three pure forensic modules (no LLM) across RAGBench examples and
computes empirical 95th percentiles for the metrics used in rank_signals().
"""
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import chromadb
from sentence_transformers import SentenceTransformer

from services.forensics.retrieval_distribution import analyze_retrieval_distribution
from services.forensics.embedding_analysis import analyze_embedding_space
from services.forensics.chunk_attribution import analyze_chunk_attribution
from services.retriever import retrieve_for_example, get_embedding_model, _CHROMA_PATH

_DOMAINS = ["techqa", "finqa", "covidqa"]
_MAX_PER_DOMAIN = 100  # cap to keep runtime reasonable


def get_example_ids(domain: str, max_n: int) -> list[str]:
    client = chromadb.PersistentClient(path=_CHROMA_PATH)
    col = client.get_collection(name=domain)
    result = col.get(include=["metadatas"])
    seen = {}
    for meta in result["metadatas"]:
        eid = meta.get("example_id")
        if eid and eid not in seen:
            seen[eid] = True
            if len(seen) >= max_n:
                break
    return list(seen.keys())


def main():
    score_entropies = []
    query_isolations = []
    tail_masses = []
    weak_match_fractions = []

    model = get_embedding_model()

    for domain in _DOMAINS:
        example_ids = get_example_ids(domain, _MAX_PER_DOMAIN)
        print(f"{domain}: {len(example_ids)} examples")

        for i, eid in enumerate(example_ids):
            try:
                question, result = retrieve_for_example(eid)
                chunks = result.chunks
                query_emb = np.array(result.query_embedding)
                chunk_embs = np.array(result.chunk_embeddings)

                chunk_ids = [c.chunk_id for c in chunks]
                dist = analyze_retrieval_distribution(chunks)
                emb = analyze_embedding_space(query_emb, chunk_embs, chunk_ids)
                attr = analyze_chunk_attribution("placeholder answer text", chunks, list(result.chunk_embeddings))

                score_entropies.append(dist.score_entropy)
                query_isolations.append(emb.query_isolation)
                tail_masses.append(dist.tail_mass)
                weak_match_fractions.append(attr.weak_match_fraction)

            except Exception as e:
                print(f"  skip {eid}: {e}")

            if (i + 1) % 20 == 0:
                print(f"  processed {i + 1}/{len(example_ids)}")

    entropies = np.array(score_entropies)
    isolations = np.array(query_isolations)
    tails = np.array(tail_masses)
    weak_fracs = np.array(weak_match_fractions)

    print("\n=== Calibration Results ===")
    print(f"n={len(entropies)} examples")
    print(f"\nscore_entropy")
    print(f"  mean={entropies.mean():.3f}  p50={np.percentile(entropies, 50):.3f}  "
          f"p75={np.percentile(entropies, 75):.3f}  p95={np.percentile(entropies, 95):.3f}  "
          f"max={entropies.max():.3f}")
    print(f"\nquery_isolation")
    print(f"  mean={isolations.mean():.3f}  p50={np.percentile(isolations, 50):.3f}  "
          f"p75={np.percentile(isolations, 75):.3f}  p95={np.percentile(isolations, 95):.3f}  "
          f"max={isolations.max():.3f}")
    print(f"\ntail_mass")
    print(f"  mean={tails.mean():.3f}  p50={np.percentile(tails, 50):.3f}  "
          f"p75={np.percentile(tails, 75):.3f}  p95={np.percentile(tails, 95):.3f}  "
          f"max={tails.max():.3f}")
    print(f"\nweak_match_fraction")
    print(f"  mean={weak_fracs.mean():.3f}  p50={np.percentile(weak_fracs, 50):.3f}  "
          f"p75={np.percentile(weak_fracs, 75):.3f}  p95={np.percentile(weak_fracs, 95):.3f}  "
          f"max={weak_fracs.max():.3f}")

    print("\n=== Suggested constants for signal_weights.py ===")
    print(f"ENTROPY_P95 = {np.percentile(entropies, 95):.3f}")
    # isolation concern threshold: 1.0 is the neutral point; p95 of (isolation - 1.0) for isolation > 1.0
    excess = isolations[isolations > 1.0] - 1.0
    if len(excess) > 0:
        iso_p95_excess = np.percentile(excess, 95)
        print(f"ISOLATION_EXCESS_P95 = {iso_p95_excess:.3f}  "
              f"({len(excess)}/{len(isolations)} examples had isolation > 1.0)")
    else:
        iso_p95_excess = 1.5
        print(f"ISOLATION_EXCESS_P95 = {iso_p95_excess:.3f}  (no examples with isolation > 1.0, using default)")
    print(f"TAIL_MASS_P95 = {np.percentile(tails, 95):.3f}")


if __name__ == "__main__":
    main()
