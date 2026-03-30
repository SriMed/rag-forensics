"""Embedding space analysis — geometric coherence of query vs retrieved chunks."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_distances

from models import EmbeddingPoint, EmbeddingSpaceMetrics


def analyze_embedding_space(
    query_embedding: np.ndarray,
    chunk_embeddings: list[np.ndarray],
    chunk_ids: list[str],
) -> EmbeddingSpaceMetrics:
    """Compute geometric coherence metrics for query vs retrieved chunks.

    Accepts pre-computed embeddings — makes no model calls.
    """
    Q = np.array(query_embedding, dtype=float).reshape(1, -1)
    C = np.array(chunk_embeddings, dtype=float)
    n = len(C)

    centroid = C.mean(axis=0, keepdims=True)
    centroid_distance = float(cosine_distances(Q, centroid)[0][0])

    if n > 1:
        pairwise = cosine_distances(C, C)
        upper_triangle = pairwise[np.triu_indices(n, k=1)]
        chunk_spread = float(upper_triangle.mean())
    else:
        chunk_spread = 0.0

    chunk_to_centroid_dists = cosine_distances(C, centroid).flatten()
    mean_chunk_centroid_dist = float(chunk_to_centroid_dists.mean()) if n > 0 else 1.0
    query_isolation = (
        centroid_distance / mean_chunk_centroid_dist
        if mean_chunk_centroid_dist > 0
        else 1.0
    )

    all_embeddings = np.vstack([Q, C])
    n_components = min(2, all_embeddings.shape[0], all_embeddings.shape[1])
    pca = PCA(n_components=n_components)
    projected_partial = pca.fit_transform(all_embeddings)
    # Pad to 2 columns if fewer than 2 components were possible
    if projected_partial.shape[1] < 2:
        pad = np.zeros((projected_partial.shape[0], 2 - projected_partial.shape[1]))
        projected = np.hstack([projected_partial, pad])
    else:
        projected = projected_partial

    points = [
        EmbeddingPoint(
            label="query",
            x=float(projected[0, 0]),
            y=float(projected[0, 1]),
            is_query=True,
        )
    ]
    for i, chunk_id in enumerate(chunk_ids):
        points.append(
            EmbeddingPoint(
                label=chunk_id,
                x=float(projected[i + 1, 0]),
                y=float(projected[i + 1, 1]),
                is_query=False,
            )
        )

    return EmbeddingSpaceMetrics(
        centroid_distance=centroid_distance,
        chunk_spread=chunk_spread,
        query_isolation=query_isolation,
        projection=points,
    )
