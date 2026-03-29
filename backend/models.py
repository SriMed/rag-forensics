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


class AnalyzeRequest(BaseModel):
    example_id: str


class AnalyzeResponse(BaseModel):
    question: str
    generated_answer: str
    retrieved_chunks: list[str]
    retrieval_relevance: DimensionResult
    answer_faithfulness: DimensionResult
    retrieval_score_distribution: DimensionResult
    hedging_verification_mismatch: DimensionResult
    chunk_attribution: DimensionResult
    confidence_calibration: DimensionResult
    attribution_map: list[AttributionEntry]
    retrieval_distribution: RetrievalDistributionMetrics
