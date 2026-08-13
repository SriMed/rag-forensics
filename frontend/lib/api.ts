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
  method: "semantic_similarity";
  caveat: string;
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
  status: "ok" | "error";
  error:
    | "claim_extraction_failed"
    | "claim_extraction_parse_failed"
    | "claim_extraction_schema_failed"
    | null;
  evaluated_chunk_count: number;
}

export interface RAGASMetrics {
  context_utilization: RAGASMetricResult;
  faithfulness: RAGASMetricResult;
  utilization_context_excerpts: string[];
  faithfulness_context_excerpts: string[];
  excerpt_caveat: string;
}

export interface RAGASMetricResult {
  score: number | null;
  status: "ok" | "unavailable";
  error: "evaluation_failed" | "non_finite_score" | null;
}

export interface RetrievalDistributionMetrics {
  score_gap: number;
  score_entropy: number;
  decay_rate: number | null;
  tail_mass: number;
  top_score: number;
  n_chunks: number;
  normalized_entropy: number;
  interpretation: string;
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
  observed_fit: "retrieved_context_near_miss" | "retrieved_context_topic_gap" | "ambiguous" | null;
  suggested_questions: SuggestedQuestion[];
  mean_question_similarity: number | null;
  status: "ok" | "not_run" | "error";
  error: string | null;
}

export interface VerdictSignal {
  name: string;
  priority_score: number;
  description: string;
  score_kind: "heuristic_priority";
  reliability: "unvalidated" | "partially_calibrated" | "model_judged";
}

export interface AnalyzeResponse {
  question: string;
  generated_answer: string;
  retrieved_chunks: string[];
  ragas: RAGASMetrics;
  hedging_mismatch: HedgingMismatchMetrics;
  chunk_attribution: ChunkAttributionMetrics;
  retrieval_distribution: RetrievalDistributionMetrics;
  embedding_space: EmbeddingSpaceMetrics;
  query_corpus_fit: QueryCorpusFitMetrics;
  verdict_signals: VerdictSignal[];
  recommendation: string;
}

const API_URL = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

export async function loadExample(domain: string): Promise<ExampleResult> {
  const resp = await fetch(`${API_URL}/example`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ domain }),
  });
  if (!resp.ok) throw new Error(`loadExample failed: ${resp.status}`);
  const data = await resp.json();
  return {
    exampleId: data.example_id,
    question: data.question,
    context: data.context_preview,
  };
}

export async function analyzeExample(exampleId: string): Promise<AnalyzeResponse> {
  const resp = await fetch(`${API_URL}/analyze`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ example_id: exampleId }),
  });
  if (!resp.ok) throw new Error(`analyzeExample failed: ${resp.status}`);
  return resp.json();
}
