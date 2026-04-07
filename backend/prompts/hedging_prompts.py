CLAIM_EXTRACTION_PROMPT = """Extract all factual claims from the following answer. A factual claim is a specific, \
assertable statement about the world that could be verified.

Answer: {answer}

Return only a JSON array of claim strings. Include the original hedging language if present.
Example: ["The policy was enacted in 2019.", "It may apply to contracts signed after March.", "The fee is approximately $50."]
If there are no factual claims, return an empty array: []"""

ENTAILMENT_PROMPT = """Does the following retrieved context directly support this claim?

Context: {chunk_text}
Claim: {claim}

Answer only: supported or not_supported"""
