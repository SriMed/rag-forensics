import numpy as np
from scipy.optimize import curve_fit
from models import RetrievedChunk, RetrievalDistributionMetrics


def analyze_retrieval_distribution(chunks: list[RetrievedChunk]) -> RetrievalDistributionMetrics:
    scores = np.array([c.score for c in chunks], dtype=float)
    scores = np.sort(scores)[::-1]  # descending by score
    n = len(scores)

    score_gap = float(scores[0] - scores[1]) if n > 1 else 0.0

    normalized = scores / scores.sum()
    score_entropy = float(max(0.0, -np.sum(normalized * np.log(normalized + 1e-9))))

    ranks = np.arange(n, dtype=float)
    if n < 3:
        decay_rate = 0.0
    else:
        try:
            popt, _ = curve_fit(
                lambda x, a, b: a * np.exp(-b * x),
                ranks,
                scores,
                p0=[1.0, 0.1],
                maxfev=1000,
            )
            decay_rate = float(popt[1])
        except (RuntimeError, TypeError, ValueError):
            decay_rate = 0.0

    tail_mass = float(scores[2:].sum() / scores.sum()) if n > 2 else 0.0

    return RetrievalDistributionMetrics(
        score_gap=score_gap,
        score_entropy=score_entropy,
        decay_rate=decay_rate,
        tail_mass=tail_mass,
        top_score=float(scores[0]),
        n_chunks=n,
    )
