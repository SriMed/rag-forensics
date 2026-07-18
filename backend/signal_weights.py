"""Signal weight configuration for ranked-signals verdict generation.

Every constant is one of:
  [calibrated] — derived from empirical 95th percentile across RAGBench corpus
  [priority]   — relative scaling requiring human judgment, not a normalization issue
  [threshold]  — below this, a signal is suppressed entirely

Calibration: scripts/calibrate_signal_weights.py ran the three pure forensic modules
(retrieval_distribution, embedding_analysis, chunk_attribution) across 100 examples
per domain (300 total; techqa / finqa / covidqa) and computed empirical percentiles.

Corpus stats (n=300):
  score_entropy:   mean=1.17  p50=1.59  p75=1.61  p95=1.61  max=1.61
  query_isolation: mean=21.5  p50=2.6   p75=5.3   p95=35.1  max=1277
  tail_mass:       mean=0.38  p50=0.53  p75=0.59  p95=0.60  max=0.60
"""
from dataclasses import dataclass


@dataclass
class SignalWeights:
    # --- Normalization denominators (calibrated against RAGBench) -----------------

    # 95th-percentile score_entropy across RAGBench = ln(5) ≈ 1.609, the theoretical
    # maximum entropy for 5 equally-scored chunks. Dividing raw entropy by this value
    # maps [0, max] → [0, 1].  The old ad-hoc constant was 2.5, which compressed every
    # signal into the bottom 64% of the concern scale.
    entropy_p95: float = 1.61  # [calibrated]

    # (isolation_p95 − isolation_threshold) = 35.1 − 1.0 = 34.1.  Used as the
    # denominator for (isolation − threshold) so that p95 maps to concern = 1.0.
    # The old constant was 1.5, which mapped isolation = 2.5 to full concern — but
    # 93% of corpus examples have isolation > 2.5, making the signal nearly useless.
    isolation_excess_range: float = 34.1  # [calibrated]

    # 95th-percentile tail_mass across RAGBench.  Normalises tail_mass → [0, 1]
    # before applying tail_mass_weight.  The old ad-hoc constant happened to match
    # (0.6 ≈ p95 = 0.598), so only the threshold below was adjusted.
    tail_mass_p95: float = 0.60  # [calibrated]

    # --- Priority weights (human-judged relative scaling) ------------------------

    # Underconfidence is less actionable than overconfidence: the system hedged when
    # it didn't need to, but the answer is still factually grounded.
    underconfidence_weight: float = 0.7  # [priority]

    # Weak matches indicate partial grounding — better than no support at all, so
    # scaled below unattributed_fraction (which is always weighted 1.0×).
    weak_match_weight: float = 0.6  # [priority]

    # Tail-mass noise is a retrieval quality issue, but the generator may still
    # produce correct answers despite noisy context, so it ranks below direct
    # attribution failures.
    tail_mass_weight: float = 0.65  # [priority]

    # --- Thresholds (below this, signal is suppressed) ---------------------------

    # Minimum entropy concern before adding the ambiguous_retrieval signal.
    # Filters very concentrated retrievals where one chunk clearly dominates.
    entropy_min_concern: float = 0.1  # [threshold]

    # Query isolation must exceed this ratio before flagging.  Ratio = 1.0 is the
    # geometric neutral point (query equidistant from centroid as the average chunk).
    # 93% of corpus examples exceed 1.0, so the calibrated isolation_excess_range
    # provides the actual scaling; this threshold only suppresses the degenerate case
    # where isolation ≤ 1.0.
    isolation_threshold: float = 1.0  # [threshold]

    # Tail mass below this fraction is not flagged.  Set to the corpus mean (0.38)
    # so only above-average-noise retrievals trigger the signal.  The old threshold
    # was 0.2, which fired on ~80% of corpus examples.
    tail_mass_threshold: float = 0.38  # [threshold, calibrated as corpus mean]


DEFAULT_WEIGHTS = SignalWeights()
