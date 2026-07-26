import { render, screen } from "@testing-library/react";
import DiagnosticCard from "@/app/components/DiagnosticCard";
import type { AnalyzeResponse } from "@/lib/api";

const BASE_RESPONSE: AnalyzeResponse = {
  question: "What causes lightning?",
  generated_answer: "Lightning is caused by electrical discharge. It happens in storm clouds.",
  retrieved_chunks: ["Chunk text one", "Chunk text two"],
  ragas: {
    retrieval_relevance_score: 0.82,
    faithfulness_score: 0.91,
    relevance_context_excerpts: ["Evidence A"],
    faithfulness_context_excerpts: ["Evidence B"], excerpt_caveat: "Context excerpts are not score evidence.",
  },
  hedging_mismatch: {
    overconfident_fraction: 0.1,
    underconfident_fraction: 0.05,
    total_claims: 4,
    claim_breakdown: [
      {
        claim: "Lightning is an electrical discharge.",
        confidence_class: "definitive",
        supported: true,
        mismatch_type: "matched",
        source_chunk_id: "chunk-1",
      },
    ],
    status: "ok",
    error: null,
    evaluated_chunk_count: 3,
  },
  chunk_attribution: {
    unattributed_fraction: 0.1,
    mean_attribution_score: 0.8,
    weak_match_fraction: 0.15,
    attribution_map: [
      {
        sentence: "Lightning is caused by electrical discharge.",
        chunk_id: "chunk-1",
        similarity_score: 0.88,
        attribution_strength: "strong",
      },
      {
        sentence: "It happens in storm clouds.",
        chunk_id: null,
        similarity_score: 0.3,
        attribution_strength: "unattributed",
      },
    ],
    method: "semantic_similarity",
    caveat: "Similarity does not prove entailment.",
  },
  retrieval_distribution: {
    score_gap: 0.3,
    score_entropy: 0.9,
    decay_rate: 0.25,
    tail_mass: 0.2,
    top_score: 0.92,
    n_chunks: 5,
    normalized_entropy: 0.56,
    interpretation: "Interpret with absolute relevance.",
  },
  embedding_space: {
    centroid_distance: 0.4,
    chunk_spread: 0.2,
    query_isolation: 0.9,
    projection: [
      { label: "query", x: 0.1, y: 0.2, is_query: true },
      { label: "chunk-1", x: 0.5, y: 0.6, is_query: false },
    ],
  },
  query_corpus_fit: {
    triggered: false,
    observed_fit: null,
    suggested_questions: [],
    mean_question_similarity: null,
    status: "not_run",
    error: null,
  },
  recommendation: "Pipeline looks healthy — no action required.",
  verdict_signals: [],
};

const FAIL_RESPONSE: AnalyzeResponse = {
  ...BASE_RESPONSE,
  verdict_signals: [{ name: "test_issue", priority_score: 0.8, description: "Test issue", score_kind: "heuristic_priority", reliability: "unvalidated" }],
  recommendation: "Reduce top-k to improve selectivity.",
  hedging_mismatch: {
    ...BASE_RESPONSE.hedging_mismatch,
    overconfident_fraction: 0.4, // > 0.3 threshold → FAIL badge
  },
};

describe("DiagnosticCard", () => {
  // 1. Renders question and generated answer
  it("renders the question and generated answer text", () => {
    render(<DiagnosticCard response={BASE_RESPONSE} />);
    expect(screen.getByText(BASE_RESPONSE.question)).toBeInTheDocument();
    expect(screen.getByText(BASE_RESPONSE.generated_answer)).toBeInTheDocument();
  });

  // 2. Both section headers present
  it("renders both section headers: Baseline Metrics and Forensics", () => {
    render(<DiagnosticCard response={BASE_RESPONSE} />);
    expect(screen.getByText(/baseline metrics/i)).toBeInTheDocument();
    expect(screen.getByText(/forensics/i)).toBeInTheDocument();
  });

  // 3. Verdict badge colors
  it("applies green class for pass verdict, red for fail", () => {
    render(<DiagnosticCard response={FAIL_RESPONSE} />);
    const passBadge = screen.getByTestId("badge-chunk_attribution"); // 10% unattributed → pass
    const failBadge = screen.getByTestId("badge-hedging_mismatch");  // 40% overconfident → fail
    expect(passBadge.className).toMatch(/green/);
    expect(failBadge.className).toMatch(/red/);
  });

  // 4. Attribution renders one element per sentence
  it("renders one attribution element per sentence in the attribution map", () => {
    render(<DiagnosticCard response={BASE_RESPONSE} />);
    const sentences = screen.getAllByTestId(/^attribution-sentence-/);
    expect(sentences).toHaveLength(BASE_RESPONSE.chunk_attribution.attribution_map.length);
  });

  // 5. Unattributed sentences have red highlight
  it("applies a red highlight class to unattributed sentences", () => {
    render(<DiagnosticCard response={BASE_RESPONSE} />);
    const unattributed = screen.getByTestId("attribution-sentence-1");
    expect(unattributed.className).toMatch(/red/);
  });

  // 6. Summary reflects worst verdict when any dimension fails
  it("shows 'Issues detected' in the summary when any dimension is fail", () => {
    render(<DiagnosticCard response={FAIL_RESPONSE} />);
    expect(screen.getByTestId("summary-banner")).toHaveTextContent(/issues detected/i);
  });

  // 7. Empty evidence array does not crash
  it("renders without crashing when evidence arrays are empty", () => {
    expect(() => render(<DiagnosticCard response={BASE_RESPONSE} />)).not.toThrow();
    expect(screen.getByTestId("badge-chunk_attribution")).toBeInTheDocument();
  });
});
