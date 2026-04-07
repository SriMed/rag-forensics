from typing import Literal
from pydantic import BaseModel


class StoredExample(BaseModel):
    example_id: str
    question: str
    context_preview: str


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    score: float


class ExampleRequest(BaseModel):
    domain: Literal["techqa", "finqa", "covidqa"]


class ExampleResponse(BaseModel):
    example_id: str
    question: str
    context_preview: str  # first 300 chars of top chunk


class DimensionResult(BaseModel):
    verdict: Literal["pass", "warn", "fail"]
    explanation: str
    evidence: list[str]


class AttributionEntry(BaseModel):
    sentence: str
    chunk_id: str | None
    similarity_score: float


class RetrievalDistributionMetrics(BaseModel):
    score_gap: float
    score_entropy: float
    decay_rate: float
    tail_mass: float
    top_score: float
    n_chunks: int


class EmbeddingPoint(BaseModel):
    label: str
    x: float
    y: float
    is_query: bool


class EmbeddingSpaceMetrics(BaseModel):
    centroid_distance: float
    chunk_spread: float
    query_isolation: float
    projection: list[EmbeddingPoint]


class RetrievalResult(BaseModel):
    chunks: list[RetrievedChunk]
    query_embedding: list[float]
    chunk_embeddings: list[list[float]]


class ClaimEntry(BaseModel):
    claim: str
    confidence_class: Literal["definitive", "hedged", "uncertain"]
    supported: bool
    mismatch_type: Literal["overconfident", "underconfident", "matched"]
    source_chunk_id: str | None


class HedgingMismatchMetrics(BaseModel):
    overconfident_fraction: float
    underconfident_fraction: float
    total_claims: int
    claim_breakdown: list[ClaimEntry]


class RAGASMetrics(BaseModel):
    retrieval_relevance_score: float
    faithfulness_score: float
    relevance_evidence: list[str]
    faithfulness_evidence: list[str]


class AnalyzeRequest(BaseModel):
    example_id: str


class AnalyzeResponse(BaseModel):
    question: str
    generated_answer: str
    retrieved_chunks: list[str]
    ragas: RAGASMetrics
    retrieval_score_distribution: DimensionResult
    hedging_mismatch: HedgingMismatchMetrics
    chunk_attribution: DimensionResult
    confidence_calibration: DimensionResult
    attribution_map: list[AttributionEntry]
    retrieval_distribution: RetrievalDistributionMetrics
    embedding_space: EmbeddingSpaceMetrics
