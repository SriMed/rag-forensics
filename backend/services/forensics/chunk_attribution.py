"""Chunk attribution forensics — sentence-level grounding analysis."""
import numpy as np
import nltk
from sklearn.metrics.pairwise import cosine_similarity

from models import AttributionEntry, ChunkAttributionMetrics, RetrievedChunk
from services.retriever import get_embedding_model

nltk.download("punkt_tab", quiet=True)


def analyze_chunk_attribution(
    answer: str,
    chunks: list[RetrievedChunk],
    chunk_embeddings: list[list[float]],
) -> ChunkAttributionMetrics:
    """Compute sentence-level attribution metrics for a generated answer.

    Accepts pre-computed chunk embeddings — only sentences are embedded here.
    """
    sentences = nltk.sent_tokenize(answer)
    if not sentences:
        return ChunkAttributionMetrics(
            unattributed_fraction=0.0,
            mean_attribution_score=0.0,
            weak_match_fraction=0.0,
            attribution_map=[],
        )

    model = get_embedding_model()
    sentence_embeddings = model.encode(sentences)  # shape: (n_sentences, dim)
    chunk_emb_matrix = np.array(chunk_embeddings, dtype=float)  # shape: (n_chunks, dim)

    attribution_map: list[AttributionEntry] = []
    best_scores: list[float] = []

    for i, sentence in enumerate(sentences):
        sims = cosine_similarity(
            sentence_embeddings[i].reshape(1, -1),
            chunk_emb_matrix,
        ).flatten()

        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        best_scores.append(best_score)

        if best_score > 0.75:
            strength = "strong"
            chunk_id: str | None = chunks[best_idx].chunk_id
        elif best_score > 0.4:
            strength = "weak"
            chunk_id = chunks[best_idx].chunk_id
        else:
            strength = "unattributed"
            chunk_id = None

        attribution_map.append(
            AttributionEntry(
                sentence=sentence,
                chunk_id=chunk_id,
                similarity_score=best_score,
                attribution_strength=strength,
            )
        )

    n = len(sentences)
    unattributed_fraction = sum(
        1 for e in attribution_map if e.attribution_strength == "unattributed"
    ) / n
    mean_attribution_score = float(np.mean(best_scores))
    weak_match_fraction = sum(
        1 for e in attribution_map if e.attribution_strength == "weak"
    ) / n

    return ChunkAttributionMetrics(
        unattributed_fraction=unattributed_fraction,
        mean_attribution_score=mean_attribution_score,
        weak_match_fraction=weak_match_fraction,
        attribution_map=attribution_map,
    )
