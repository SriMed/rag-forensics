"""Prompts for the verdict generator (Issue #9)."""

RECOMMENDATION_RENDER_PROMPT = """
You are writing one sentence for a RAG diagnostic report.

The RAG system has been analyzed. Here are the key signals:
- Retrieval score entropy: {score_entropy:.2f} (higher = more ambiguous retrieval)
- Score gap (top vs second chunk): {score_gap:.2f} (higher = more decisive retrieval)
- Decay rate: {decay_rate:.2f} (higher = steeper relevance drop-off)
- Tail mass: {tail_mass:.2f} (higher = more low-relevance content reaching generator)
- Centroid distance: {centroid_distance:.2f} (higher = query geometrically far from retrieved content)
- Chunk spread: {chunk_spread:.2f} (higher = retrieved chunks from different semantic regions)
- Query isolation: {query_isolation:.2f} (> 1.0 = query more isolated than chunks are from each other)
- Answer faithfulness: {faithfulness_score:.2f}
- Unattributed fraction: {unattributed_fraction:.2f} (fraction of answer not traceable to any chunk)
- Weak match fraction: {weak_match_fraction:.2f} (fraction of answer loosely but not strongly grounded)
- Overconfident claim fraction: {overconfident_fraction:.2f}
- Underconfident claim fraction: {underconfident_fraction:.2f}

Root cause identified: {root_cause}
Pipeline component to address: {pipeline_component}
Recommended action: {action}
Emphasis: {render_hint}

Write exactly one sentence that:
1. Names what the signals show is happening in the pipeline
2. Names the specific component to fix
3. States the specific action to take

Do not use hedging language. Do not say "may" or "might". Be direct and specific.
Maximum 50 words.
Example of the right register: "Your retrieval is decisive but generation is going beyond retrieved content — increase chunk size or use overlapping windows to give the model more grounding material."
"""

DIMENSION_EXPLANATION_PROMPT = """
Write one plain-English sentence explaining this RAG evaluation signal to a non-technical stakeholder.

Signal: {dimension_name}
Value: {metric_value}
What it measures: {what_it_measures}

Do not use ML jargon. Do not mention scores or numbers unless essential.
Maximum 30 words.
"""
