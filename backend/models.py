from typing import Literal
from pydantic import BaseModel, Field, field_validator, model_validator


class SuggestedQuestion(BaseModel):
    question: str
    source_chunk_ids: list[str]
    relevance_to_original: float


class QueryCorpusFitMetrics(BaseModel):
    triggered: bool
    trigger_reason: Literal["query_isolation", "context_utilization", "entropy_faithfulness"] | None = None
    observed_fit: Literal["retrieved_context_near_miss", "retrieved_context_topic_gap", "ambiguous"] | None = None
    suggested_questions: list[SuggestedQuestion]
    mean_question_similarity: float | None
    status: Literal["ok", "not_run", "error"] = "ok"
    error: str | None = None


class StoredExample(BaseModel):
    example_id: str
    question: str
    context_preview: str


class RetrievedChunk(BaseModel):
    chunk_id: str
    text: str
    score: float = Field(ge=0.0, le=1.0)


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
    method: Literal["semantic_similarity"] = "semantic_similarity"
    caveat: str = (
        "Similarity identifies a semantically close source candidate; it does not prove entailment "
        "or establish that unsupported text is a hallucination."
    )


class RetrievalDistributionMetrics(BaseModel):
    score_gap: float
    score_entropy: float
    decay_rate: float | None  # None when exponential fit fails
    tail_mass: float
    top_score: float
    n_chunks: int
    normalized_entropy: float = 0.0
    interpretation: str = (
        "Distribution shape must be interpreted jointly with absolute relevance and score semantics."
    )


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
    status: Literal["ok", "error"] = "ok"
    error: str | None = None
    evaluated_chunk_count: int = 0


class RAGASMetricResult(BaseModel):
    score: float | None
    status: Literal["ok", "unavailable"]
    error: Literal["evaluation_failed", "non_finite_score"] | None = None

    @model_validator(mode="after")
    def status_matches_value(self):
        if self.status == "ok" and (self.score is None or self.error is not None):
            raise ValueError("ok RAGAS results require a score and no error")
        if self.status == "unavailable" and (self.score is not None or self.error is None):
            raise ValueError("unavailable RAGAS results require an error and no score")
        return self


class RAGASMetrics(BaseModel):
    context_utilization: RAGASMetricResult
    faithfulness: RAGASMetricResult
    utilization_context_excerpts: list[str]
    faithfulness_context_excerpts: list[str]
    excerpt_caveat: str = (
        "These are context excerpts for inspection, not evidence explaining the evaluator's score."
    )


class AnalyzeRequest(BaseModel):
    example_id: str


class CustomAnalyzeRequest(BaseModel):
    question: str
    answer: str
    chunks: list[RetrievedChunk]
    score_semantics: Literal["normalized_similarity"]

    @field_validator("chunks")
    @classmethod
    def chunks_not_empty(cls, v):
        if not v:
            raise ValueError("chunks must not be empty")
        return v


class VerdictSignal(BaseModel):
    name: str
    priority_score: float
    description: str
    score_kind: Literal["heuristic_priority"] = "heuristic_priority"
    reliability: Literal["unvalidated", "partially_calibrated", "model_judged"]


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
    verdict_signals: list[VerdictSignal]  # all ranked signals, descending by heuristic priority
    recommendation: str


class BenchmarkSentence(BaseModel):
    key: str
    text: str


class BenchmarkDocumentSentence(BenchmarkSentence):
    document_id: str


class BenchmarkSentenceSupport(BaseModel):
    response_sentence_key: str
    fully_supported: bool
    supporting_sentence_keys: list[str]
    explanation: str = ""


class RAGBenchEvaluationRecord(BaseModel):
    example_id: str
    domain: str
    question: str
    response: str
    chunks: list[RetrievedChunk]
    response_sentences: list[BenchmarkSentence]
    document_sentences: list[BenchmarkDocumentSentence]
    document_sentence_keys: set[str]
    unsupported_response_sentence_keys: set[str]
    sentence_support: dict[str, BenchmarkSentenceSupport]
    adherence_score: bool | None = None
    relevance_score: float | None = None
    utilization_score: float | None = None
    completeness_score: float | None = None


class UnsupportedSentencePrediction(BaseModel):
    sentence_key: str
    sentence: str
    gold_unsupported: bool
    predicted_unsupported: bool
    similarity_score: float
    source_chunk_id: str | None
    ragbench_fully_supported: bool | None
    ragbench_supporting_sentence_keys: list[str]


class UnsupportedDetectionMetrics(BaseModel):
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    precision: float
    recall: float
    f1: float
    coverage: float
    evaluated_sentences: int
    total_sentences: int


class RAGBenchExampleResult(BaseModel):
    example_id: str
    domain: str
    predictions: list[UnsupportedSentencePrediction]
    adherence_score: bool | None
    relevance_score: float | None
    utilization_score: float | None
    completeness_score: float | None


class RAGBenchBenchmarkMetadata(BaseModel):
    dataset: str
    dataset_config: str
    split: str
    seed: int
    requested_limit: int | None
    sample_count: int
    skipped_count: int
    skipped_rows: list[str]
    embedding_model: str
    unattributed_threshold: float
    timestamp: str
    llm_calls_enabled: Literal[False] = False


class RAGBenchBenchmarkReport(BaseModel):
    metadata: RAGBenchBenchmarkMetadata
    metrics: UnsupportedDetectionMetrics
    examples: list[RAGBenchExampleResult]


class AtomicClaim(BaseModel):
    claim_id: str
    parent_sentence_key: str
    text: str


class EvidenceCandidate(BaseModel):
    sentence_key: str
    document_id: str
    text: str
    selection_score: float


class NLIVerifierScores(BaseModel):
    entailment: float
    neutral: float
    contradiction: float
    label: Literal["entailment", "neutral", "contradiction"]


class ClaimVerification(BaseModel):
    claim: AtomicClaim
    parent_sentence_key: str
    evidence: EvidenceCandidate | None
    support_score: float | None
    predicted_supported: bool | None
    status: Literal["ok", "no_evidence", "verifier_error"]
    verifier_label: str | None = None
    nli_scores: NLIVerifierScores | None = None
    error: str | None = None


class GroundingSentencePrediction(BaseModel):
    example_id: str
    domain: str
    sentence_key: str
    sentence: str
    gold_unsupported: bool
    predicted_unsupported: bool | None
    unsupported_score: float | None
    claims: list[ClaimVerification]
    error_categories: list[str] = Field(default_factory=list)


class BinaryClassificationMetrics(BaseModel):
    true_positive: int
    false_positive: int
    true_negative: int
    false_negative: int
    precision: float
    recall: float
    f1: float
    auroc: float | None
    auprc: float | None
    prevalence: float
    coverage: float
    evaluated: int
    total: int


class CalibrationResult(BaseModel):
    threshold: float
    objective: Literal["f1"] = "f1"
    objective_value: float
    partition: Literal["calibration"] = "calibration"


class ConfidenceInterval(BaseModel):
    point_estimate: float
    lower: float
    upper: float
    confidence: float = 0.95
    iterations: int
    seed: int


class GroundingRunMetadata(BaseModel):
    dataset: str
    dataset_revision: str
    split_strategy: str
    calibration_split: str
    evaluation_split: str
    seed: int
    embedding_model: str
    embedding_model_revision: str
    entailment_model: str
    entailment_model_revision: str
    claim_decomposer: str
    claim_decomposer_version: str
    similarity_threshold: float
    entailment_threshold: float
    code_commit: str
    calibration_sample_count: int = 0
    evaluation_sample_count: int = 0
    skipped_rows: list[str] = Field(default_factory=list)


class GroundingMethodReport(BaseModel):
    method: str
    threshold: float | None
    per_domain: dict[str, BinaryClassificationMetrics]
    pooled: BinaryClassificationMetrics
    macro_f1: float
    macro_auprc: float | None
    confidence_intervals: dict[str, ConfidenceInterval]
    predictions: list[GroundingSentencePrediction]


class GroundingExperimentReport(BaseModel):
    metadata: GroundingRunMetadata
    calibration: dict[str, CalibrationResult]
    methods: dict[str, GroundingMethodReport]
    paired_b3_vs_b1: dict[str, ConfidenceInterval]


class OracleEvidenceEligibility(BaseModel):
    total_fully_supported: int
    eligible: int
    excluded: dict[str, int] = Field(default_factory=dict)


class OracleEvidenceSentenceResult(BaseModel):
    example_id: str
    domain: str
    sentence_key: str
    sentence: str
    annotated_evidence_keys: list[str]
    selected: GroundingSentencePrediction
    oracle: GroundingSentencePrediction
    oracle_pairs: list[ClaimVerification]
    selected_evidence_hit_at_1: bool


class OracleEvidenceRunMetadata(BaseModel):
    dataset: str
    dataset_revision: str
    evaluation_split: str
    seed: int
    embedding_model: str
    embedding_model_revision: str
    entailment_model: str
    entailment_model_revision: str
    claim_decomposer: str
    claim_decomposer_version: str
    entailment_threshold: float
    code_commit: str
    sample_count: int
    skipped_rows: list[str] = Field(default_factory=list)


class OracleEvidenceStratumMetrics(BaseModel):
    sentences: int
    selected_evaluated: int
    oracle_evaluated: int
    paired_evaluated: int
    selected_false_unsupported_rate: float | None
    oracle_false_unsupported_rate: float | None
    paired_difference: float | None


class OracleEvidenceDiagnosticReport(BaseModel):
    eligibility: OracleEvidenceEligibility
    selected_false_unsupported_rate: float | None
    oracle_false_unsupported_rate: float | None
    selected_evidence_hit_at_1: float | None
    paired_false_unsupported_difference: ConfidenceInterval | None
    selected_evaluated: int = 0
    oracle_evaluated: int = 0
    paired_evaluated: int = 0
    per_domain: dict[str, OracleEvidenceStratumMetrics] = Field(default_factory=dict)
    by_source_count: dict[str, OracleEvidenceStratumMetrics] = Field(default_factory=dict)
    predictions: list[OracleEvidenceSentenceResult]
    metadata: OracleEvidenceRunMetadata | None = None
