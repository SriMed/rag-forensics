import numpy as np
from scipy.optimize import curve_fit
from models import RetrievedChunk, RetrievalDistributionMetrics


def analyze_retrieval_distribution(chunks: list[RetrievedChunk]) -> RetrievalDistributionMetrics:
    scores = np.array([c.score for c in chunks], dtype=float)
    scores = np.sort(scores)[::-1]  # descending by score
    n = len(scores)

    score_gap = float(scores[0] - scores[1]) if n > 1 else 0.0

    total = scores.sum()
    if total > 0:
        normalized = scores / total
        score_entropy = float(max(0.0, -np.sum(normalized * np.log(normalized + 1e-9))))
    else:
        score_entropy = 0.0
    max_entropy = float(np.log(n)) if n > 1 else 0.0
    normalized_entropy = score_entropy / max_entropy if max_entropy > 0 else 0.0

    ranks = np.arange(n, dtype=float)
    decay_rate: float | None = None
    if n >= 3:
        try:
            popt, _ = curve_fit(
                lambda x, a, b: a * np.exp(-b * x),
                ranks,
                scores,
                p0=[1.0, 0.1],
                maxfev=1000,
            )
            # Negative decay means scores increase with rank — nonsensical for sorted-descending
            # data; treat as a fit failure rather than silently pass 0.
            decay_rate = float(popt[1]) if popt[1] >= 0 else None
        except (RuntimeError, TypeError, ValueError):
            decay_rate = None

    tail_mass = float(scores[2:].sum() / total) if (n > 2 and total > 0) else 0.0

    return RetrievalDistributionMetrics(
        score_gap=score_gap,
        score_entropy=score_entropy,
        decay_rate=decay_rate,
        tail_mass=tail_mass,
        top_score=float(scores[0]),
        n_chunks=n,
        normalized_entropy=normalized_entropy,
    )
