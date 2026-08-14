import { render, screen } from "@testing-library/react";
import DiagnosticCard from "@/app/components/DiagnosticCard";
import type { AnalyzeResponse } from "@/lib/api";

const BASE_RESPONSE: AnalyzeResponse = {
  question: "What causes lightning?",
  generated_answer: "Lightning is caused by electrical discharge. It happens in storm clouds.",
  retrieved_chunks: ["Chunk text one", "Chunk text two"],
  retrieved_chunk_details: [
    { chunk_id: "chunk-1", text: "Complete source text.", score: 0.88, completeness: "complete", completeness_source: "source" },
    { chunk_id: "chunk-2", text: "Boundary metadata unavailable", score: 0.61, completeness: "unknown", completeness_source: "unavailable" },
  ],
  ragas: {
    context_utilization: { score: 0.82, status: "ok", error: null },
    faithfulness: { score: 0.91, status: "ok", error: null },
    utilization_context_excerpts: ["Evidence A"],
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
        entailment_checks: [],
      },
    ],
    status: "ok",
    error: null,
    evaluated_chunk_count: 3,
    evaluated_claim_count: 4,
    unavailable_claim_count: 0,
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
    rejected_questions: [],
    mean_question_similarity: null,
    status: "not_run",
    error: null,
  },
  recommendation: "Review the structured record before choosing the next experiment.",
  verdict_signals: [],
  verdict_reasoning: {
    observations: [
      { signal_name: "retrieval_distribution", description: "Retrieval scores are relatively flat.", reliability: "unvalidated" },
      { signal_name: "faithfulness_unavailable", description: "Faithfulness evaluation is unavailable.", reliability: "model_judged" },
    ],
    hypotheses: [
      { hypothesis_id: "H1", statement: "Retrieval returned weakly relevant context." },
      { hypothesis_id: "H2", statement: "Generation did not use relevant retrieved evidence." },
    ],
    test: {
      component: "retriever",
      action: "Rerun with a retrieval configuration that raises absolute relevance.",
      interpretations: [
        { outcome: "Answer grounding improves", supports_hypothesis_ids: ["H1"] },
        { outcome: "Answer grounding does not improve", supports_hypothesis_ids: ["H2"] },
      ],
    },
  },
};

test("renders unavailable RAGAS metrics explicitly", () => {
  render(
    <DiagnosticCard
      response={{
        ...BASE_RESPONSE,
        ragas: {
          ...BASE_RESPONSE.ragas,
          context_utilization: {
            score: null,
            status: "unavailable",
            error: "evaluation_failed",
          },
        },
      }}
    />
  );
  expect(screen.getByText("Unavailable")).toBeInTheDocument();
  expect(screen.getByText("evaluation_failed")).toBeInTheDocument();
});

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

  it("uses neutral investigation-priority language for heuristic scores", () => {
    render(<DiagnosticCard response={FAIL_RESPONSE} />);
    const banner = screen.getByTestId("summary-banner");
    expect(banner).toHaveTextContent(/higher investigation priority/i);
    expect(banner).toHaveTextContent(/not calibrated health or severity/i);
    expect(banner).not.toHaveTextContent(/healthy|issues detected/i);
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

  it("renders observations with reliability, including unavailable evaluator state", () => {
    render(<DiagnosticCard response={BASE_RESPONSE} />);
    expect(screen.getByText("Faithfulness evaluation is unavailable.")).toBeInTheDocument();
    expect(screen.getByText(/reliability: model judged/i)).toBeInTheDocument();
    expect(screen.getByText(/reliability: unvalidated/i)).toBeInTheDocument();
  });

  it("renders competing retrieval and generation hypotheses as hypotheses", () => {
    render(<DiagnosticCard response={BASE_RESPONSE} />);
    expect(screen.getByText(/hypothesis: retrieval returned weakly relevant context/i)).toBeInTheDocument();
    expect(screen.getByText(/hypothesis: generation did not use relevant retrieved evidence/i)).toBeInTheDocument();
  });

  it("renders the named component, action, and outcome-dependent hypothesis support", () => {
    render(<DiagnosticCard response={BASE_RESPONSE} />);
    const test = screen.getByTestId("discriminating-test");
    expect(test).toHaveTextContent(/component: retriever/i);
    expect(test).toHaveTextContent(/rerun with a retrieval configuration/i);
    expect(test).toHaveTextContent(/answer grounding improves supports: h1/i);
    expect(test).toHaveTextContent(/answer grounding does not improve supports: h2/i);
  });

  it("renders chunk completeness and provenance, preserving unavailable metadata", () => {
    render(<DiagnosticCard response={BASE_RESPONSE} />);
    expect(screen.getByText(/completeness: complete/i)).toBeInTheDocument();
    expect(screen.getByText(/provenance: source/i)).toBeInTheDocument();
    expect(screen.getByText(/completeness: unknown/i)).toBeInTheDocument();
    expect(screen.getByText(/provenance: unavailable/i)).toBeInTheDocument();
  });
});
