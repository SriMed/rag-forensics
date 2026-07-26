"""Prompts for the verdict generator."""

RANKED_SIGNALS_PROMPT = """You are writing a diagnostic summary for a RAG (Retrieval-Augmented Generation) pipeline developer.

The following signals have been ranked by concern level from a forensic analysis of the system's response:

{signals_text}

Additional context:
- Faithfulness score: {faithfulness_score:.2f} (1.0 = answer fully grounded in retrieved content)
- Retrieval relevance score: {retrieval_relevance_score:.2f} (1.0 = retrieved chunks highly relevant to the question)
- Unattributed answer fraction: {unattributed_fraction:.0%}
- Overconfident claim fraction: {overconfident_fraction:.0%}

Write 2–3 sentences for a developer who needs to know what to investigate next. Treat the signals as diagnostic hypotheses, not proof of root cause. Focus on the highest-ranked signals, name a plausible pipeline component, and propose a concrete test that could confirm or falsify the hypothesis. Do not repeat the numbers verbatim and do not claim that an intervention will work before it is tested."""

DIMENSION_EXPLANATION_PROMPT = """Write one plain-English sentence explaining this RAG evaluation signal to a non-technical stakeholder.

Signal: {dimension_name}
Value: {metric_value}
What it measures: {what_it_measures}

Do not use ML jargon. Do not mention scores or numbers unless essential.
Maximum 30 words."""
