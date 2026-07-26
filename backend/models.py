from typing import Literal
from pydantic import BaseModel, field_validator


class SuggestedQuestion(BaseModel):
    question: str
    source_chunk_ids: list[str]
    relevance_to_original: float


class QueryCorpusFitMetrics(BaseModel):
    triggered: bool
    trigger_reason: Literal["query_isolation", "retrieval_relevance", "entropy_faithfulness"] | None = None
    mismatch_type: Literal["query_mismatch", "coverage_gap", "ambiguous"] | None
    suggested_questions: list[SuggestedQuestion]
    mean_question_similarity: float | None


class StoredExample(BaseModel):
    example_id: str
    question: str
    context_preview: str


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    score: float


# CustomChunk is structurally identical to RetrievedChunk — alias to avoid duplication.
CustomChunk = RetrievedChunk


class ExampleRequest(BaseModel):
    domain: Literal["techqa", "finqa", "covidqa"]


class ExampleResponse(BaseModel):
    example_id: str
    question: str
    context_preview: str  # first 300 chars of top chunk


class AttributionEntry(BaseModel):
    sentence: str
    chunk_id: str | None
    similarity_score: float
    attribution_strength: Literal["strong", "weak", "unattributed"]


class ChunkAttributionMetrics(BaseModel):
    unattributed_fraction: float
    mean_attribution_score: float
    weak_match_fraction: float
    attribution_map: list[AttributionEntry]


class RetrievalDistributionMetrics(BaseModel):
    score_gap: float
    score_entropy: float
    decay_rate: float | None  # None when exponential fit fails
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


class CustomAnalyzeRequest(BaseModel):
    question: str
    answer: str
    chunks: list[RetrievedChunk]

    @field_validator("chunks")
    @classmethod
    def chunks_not_empty(cls, v):
        if not v:
            raise ValueError("chunks must not be empty")
        return v


class VerdictSignal(BaseModel):
    name: str
    concern_score: float
    description: str


class AnalyzeResponse(BaseModel):
    question: str
    generated_answer: str
    retrieved_chunks: list[str]
    ragas: RAGASMetrics
    hedging_mismatch: HedgingMismatchMetrics
    chunk_attribution: ChunkAttributionMetrics
    retrieval_distribution: RetrievalDistributionMetrics
    embedding_space: EmbeddingSpaceMetrics
    query_corpus_fit: QueryCorpusFitMetrics
    verdict_signals: list[VerdictSignal]  # all ranked signals, descending by concern_score
    recommendation: str
