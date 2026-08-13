"""Prompts for the verdict generator."""

RANKED_SIGNALS_PROMPT = """You are wording an already-constructed RAG diagnostic for a developer.

Structured diagnostic:
{reasoning_json}

Write 2–3 concise sentences. Preserve the observations as observations, keep materially different
hypotheses separate, name only the supplied component and test, and include the supplied
outcome-dependent interpretations. Reliability must bound the language: never turn an unvalidated,
partially calibrated, or model-judged observation into proof. Do not add causes, components, tests,
outcomes, or facts absent from the structure. Do not repeat numeric values unless needed to identify
an observation."""

DIMENSION_EXPLANATION_PROMPT = """Write one plain-English sentence explaining this RAG evaluation signal to a non-technical stakeholder.

Signal: {dimension_name}
Value: {metric_value}
What it measures: {what_it_measures}

Do not use ML jargon. Do not mention scores or numbers unless essential.
Maximum 30 words."""
