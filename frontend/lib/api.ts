export interface ExampleResult {
  exampleId: string;
  question: string;
  context: string;
}

export interface DimensionResult {
  verdict: "pass" | "warn" | "fail";
  explanation: string;
  evidence: string[];
}

export interface AttributionEntry {
  sentence: string;
  chunk_id: string | null;
  similarity_score: number;
  attribution_strength: "strong" | "weak" | "unattributed";
}

export interface ChunkAttributionMetrics {
  unattributed_fraction: number;
  mean_attribution_score: number;
  weak_match_fraction: number;
  attribution_map: AttributionEntry[];
}

export interface ClaimEntry {
  claim: string;
  confidence_class: "definitive" | "hedged" | "uncertain";
  supported: boolean;
  mismatch_type: "overconfident" | "underconfident" | "matched";
  source_chunk_id: string | null;
}

export interface HedgingMismatchMetrics {
  overconfident_fraction: number;
  underconfident_fraction: number;
  total_claims: number;
  claim_breakdown: ClaimEntry[];
}

export interface RAGASMetrics {
  retrieval_relevance_score: number;
  faithfulness_score: number;
  relevance_evidence: string[];
  faithfulness_evidence: string[];
}

export interface RetrievalDistributionMetrics {
  score_gap: number;
  score_entropy: number;
  decay_rate: number;
  tail_mass: number;
  top_score: number;
  n_chunks: number;
}

export interface EmbeddingPoint {
  label: string;
  x: number;
  y: number;
  is_query: boolean;
}

export interface EmbeddingSpaceMetrics {
  centroid_distance: number;
  chunk_spread: number;
  query_isolation: number;
  projection: EmbeddingPoint[];
}

export interface SuggestedQuestion {
  question: string;
  source_chunk_ids: string[];
  relevance_to_original: number;
}

export interface QueryCorpusFitMetrics {
  triggered: boolean;
  mismatch_type: "query_mismatch" | "coverage_gap" | "ambiguous" | null;
  suggested_questions: SuggestedQuestion[];
  mean_question_similarity: number | null;
}

export interface AnalyzeResponse {
  question: string;
  generated_answer: string;
  retrieved_chunks: string[];
  ragas: RAGASMetrics;
  retrieval_score_distribution: DimensionResult;
  hedging_mismatch: HedgingMismatchMetrics;
  chunk_attribution: ChunkAttributionMetrics;
  confidence_calibration: DimensionResult;
  retrieval_distribution: RetrievalDistributionMetrics;
  embedding_space: EmbeddingSpaceMetrics;
  query_corpus_fit: QueryCorpusFitMetrics;
  recommendation: string;
  rule_id: string;
}

export async function loadExample(domain: string): Promise<ExampleResult> {
  // Stub: returns mock data. Real implementation will call POST /example.
  return {
    exampleId: `${domain}-example-001`,
    question: `Sample question for domain: ${domain}`,
    context:
      "This is a sample context passage retrieved from the RAGBench dataset. " +
      "It contains information relevant to the question above and is used to " +
      "demonstrate how the RAG forensics system evaluates retrieval quality.",
  };
}

export async function analyzeExample(exampleId: string): Promise<void> {
  // Stub: no-op. Real implementation will call POST /analyze with exampleId.
  void exampleId;
}
