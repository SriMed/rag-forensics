import { act, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import ExampleBrowser from "@/app/components/ExampleBrowser";
import type { AnalyzeResponse } from "@/lib/api";

const ANALYZE_FIXTURE: AnalyzeResponse = {
  question: "What is X?",
  generated_answer: "X is Y.",
  retrieved_chunks: [],
  retrieved_chunk_details: [],
  ragas: { context_utilization: { score: 0.8, status: "ok", error: null }, faithfulness: { score: 0.9, status: "ok", error: null }, utilization_context_excerpts: [], faithfulness_context_excerpts: [], excerpt_caveat: "Context excerpts are not score evidence." },
  hedging_mismatch: { overconfident_fraction: 0, underconfident_fraction: 0, total_claims: 0, claim_breakdown: [], status: "ok", error: null, evaluated_chunk_count: 0, evaluated_claim_count: 0, unavailable_claim_count: 0 },
  chunk_attribution: { unattributed_fraction: 0, mean_attribution_score: 0.9, weak_match_fraction: 0, attribution_map: [], method: "semantic_similarity", caveat: "Similarity does not prove entailment." },
  retrieval_distribution: { score_gap: 0.3, score_entropy: 0.9, decay_rate: 0.4, tail_mass: 0.1, top_score: 0.9, n_chunks: 3, normalized_entropy: 0.5, interpretation: "Interpret with absolute relevance." },
  embedding_space: { centroid_distance: 0.3, chunk_spread: 0.2, query_isolation: 0.8, projection: [] },
  query_corpus_fit: { triggered: false, observed_fit: null, suggested_questions: [], rejected_questions: [], mean_question_similarity: null, status: "not_run", error: null },
  recommendation: "Rerun the evaluator before interpreting answer quality.",
  verdict_signals: [],
  verdict_reasoning: {
    observations: [{ signal_name: "faithfulness_unavailable", description: "Faithfulness evaluation is unavailable.", reliability: "model_judged" }],
    hypotheses: [{ hypothesis_id: "H1", statement: "The evaluator failed independently of answer quality." }],
    test: null,
  },
};

// Deferred promise helper: gives tests explicit control over when a mock resolves,
// so the in-flight (loading) state is observable before resolution.
function deferred<T>(): { promise: Promise<T>; resolve: (v: T) => void } {
  let resolve!: (v: T) => void;
  const promise = new Promise<T>((r) => {
    resolve = r;
  });
  return { promise, resolve };
}

const EXAMPLE_RESULT = {
  exampleId: "ex-001",
  question: "What is the capital of France?",
  context:
    "France is a country in Western Europe. Its capital city is Paris, which is known for the Eiffel Tower and its rich cultural history.",
};

describe("ExampleBrowser", () => {
  let loadExample: jest.Mock;
  let analyzeExample: jest.Mock;

  beforeEach(() => {
    const { promise: lp, resolve: lr } = deferred<typeof EXAMPLE_RESULT>();
    loadExample = jest.fn(() => lp);
    // Resolve synchronously for tests that don't need to observe in-flight state.
    lr(EXAMPLE_RESULT);

    const { promise: ap, resolve: ar } = deferred<AnalyzeResponse>();
    analyzeExample = jest.fn(() => ap);
    ar(ANALYZE_FIXTURE);
  });

  // 1. Domain selector renders correct options
  it("renders domain selector with techqa, finqa, covidqa options", () => {
    render(<ExampleBrowser loadExample={loadExample} analyzeExample={analyzeExample} />);
    const select = screen.getByRole("combobox");
    const options = Array.from(select.querySelectorAll("option")).map((o) => o.value);
    expect(options).toEqual(["techqa", "finqa", "covidqa"]);
  });

  // 2. Default domain is techqa
  it("defaults to techqa as the selected domain", () => {
    render(<ExampleBrowser loadExample={loadExample} analyzeExample={analyzeExample} />);
    expect(screen.getByRole("combobox")).toHaveValue("techqa");
  });

  // 3. Preview card hidden before load
  it("hides the preview card before any example is loaded", () => {
    render(<ExampleBrowser loadExample={loadExample} analyzeExample={analyzeExample} />);
    expect(screen.queryByTestId("preview-card")).not.toBeInTheDocument();
  });

  // 4. 'Load Example' calls loadExample with selected domain
  it("calls loadExample with the selected domain when Load Example is clicked", async () => {
    const user = userEvent.setup();
    render(<ExampleBrowser loadExample={loadExample} analyzeExample={analyzeExample} />);
    await user.selectOptions(screen.getByRole("combobox"), "finqa");
    await user.click(screen.getByRole("button", { name: /load example/i }));
    expect(loadExample).toHaveBeenCalledWith("finqa");
  });

  // 5. Preview card renders question text after load
  it("shows the preview card with question text after load resolves", async () => {
    const user = userEvent.setup();
    render(<ExampleBrowser loadExample={loadExample} analyzeExample={analyzeExample} />);
    await user.click(screen.getByRole("button", { name: /load example/i }));
    await waitFor(() =>
      expect(screen.getByTestId("preview-card")).toBeInTheDocument()
    );
    expect(screen.getByTestId("preview-card")).toHaveTextContent(EXAMPLE_RESULT.question);
  });

  // 6. Preview card truncates long context; does not truncate short context
  it("truncates context longer than 300 chars to exactly 300 chars in the preview card", async () => {
    // 400 'A's — unambiguously longer than the 300-char limit.
    const longContext = "A".repeat(400);
    const { promise, resolve } = deferred<typeof EXAMPLE_RESULT>();
    loadExample = jest.fn(() => promise);
    resolve({ ...EXAMPLE_RESULT, context: longContext });

    const user = userEvent.setup();
    render(<ExampleBrowser loadExample={loadExample} analyzeExample={analyzeExample} />);
    await user.click(screen.getByRole("button", { name: /load example/i }));
    await waitFor(() => screen.getByTestId("preview-card"));

    const displayedContext = screen.getByTestId("preview-context").textContent ?? "";
    expect(displayedContext.length).toBeLessThanOrEqual(300);
    // The ellipsis must be present, proving truncation actually ran.
    expect(displayedContext).toMatch(/…$/);
  });

  it("does not truncate context that is 300 chars or shorter", async () => {
    const shortContext = "B".repeat(300);
    const { promise, resolve } = deferred<typeof EXAMPLE_RESULT>();
    loadExample = jest.fn(() => promise);
    resolve({ ...EXAMPLE_RESULT, context: shortContext });

    const user = userEvent.setup();
    render(<ExampleBrowser loadExample={loadExample} analyzeExample={analyzeExample} />);
    await user.click(screen.getByRole("button", { name: /load example/i }));
    await waitFor(() => screen.getByTestId("preview-card"));

    const displayedContext = screen.getByTestId("preview-context").textContent ?? "";
    expect(displayedContext).toBe(shortContext);
    expect(displayedContext).not.toMatch(/…$/);
  });

  // 7. 'Analyze' disabled before example loaded
  it("disables the Analyze button before an example is loaded", () => {
    render(<ExampleBrowser loadExample={loadExample} analyzeExample={analyzeExample} />);
    expect(screen.getByRole("button", { name: /analyze/i })).toBeDisabled();
  });

  // 8. 'Analyze' enabled after example loaded
  it("enables the Analyze button after an example is loaded", async () => {
    const user = userEvent.setup();
    render(<ExampleBrowser loadExample={loadExample} analyzeExample={analyzeExample} />);
    await user.click(screen.getByRole("button", { name: /load example/i }));
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /analyze/i })).not.toBeDisabled()
    );
  });

  // 9. 'Analyze' calls analyzeExample with exampleId
  it("calls analyzeExample with the correct exampleId when Analyze is clicked", async () => {
    const user = userEvent.setup();
    render(<ExampleBrowser loadExample={loadExample} analyzeExample={analyzeExample} />);
    await user.click(screen.getByRole("button", { name: /load example/i }));
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /analyze/i })).not.toBeDisabled()
    );
    await user.click(screen.getByRole("button", { name: /analyze/i }));
    await waitFor(() => expect(analyzeExample).toHaveBeenCalledWith(EXAMPLE_RESULT.exampleId));
  });

  // 10. Spinner on Load Example while in flight
  it("shows a loading spinner on Load Example while the call is in flight", async () => {
    const { promise, resolve } = deferred<typeof EXAMPLE_RESULT>();
    loadExample = jest.fn(() => promise);
    const user = userEvent.setup();
    render(<ExampleBrowser loadExample={loadExample} analyzeExample={analyzeExample} />);

    const clickPromise = user.click(screen.getByRole("button", { name: /load example/i }));

    await waitFor(() => expect(screen.getByTestId("load-spinner")).toBeInTheDocument());

    await act(async () => { resolve(EXAMPLE_RESULT); });
    await clickPromise;
    await waitFor(() => expect(screen.queryByTestId("load-spinner")).not.toBeInTheDocument());
  });

  // 11. Spinner on Analyze while in flight
  it("shows a loading spinner on Analyze while the call is in flight", async () => {
    const { promise, resolve } = deferred<AnalyzeResponse>();
    analyzeExample = jest.fn(() => promise);
    const user = userEvent.setup();
    render(<ExampleBrowser loadExample={loadExample} analyzeExample={analyzeExample} />);
    await user.click(screen.getByRole("button", { name: /load example/i }));
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /analyze/i })).not.toBeDisabled()
    );

    const analyzePromise = user.click(screen.getByRole("button", { name: /analyze/i }));

    await waitFor(() => expect(screen.getByTestId("analyze-spinner")).toBeInTheDocument());

    await act(async () => { resolve(ANALYZE_FIXTURE); });
    await analyzePromise;
    await waitFor(() =>
      expect(screen.queryByTestId("analyze-spinner")).not.toBeInTheDocument()
    );
  });

  // 12. DiagnosticCard renders after Analyze resolves
  it("renders DiagnosticCard with the recommendation after Analyze resolves", async () => {
    const user = userEvent.setup();
    render(<ExampleBrowser loadExample={loadExample} analyzeExample={analyzeExample} />);
    await user.click(screen.getByRole("button", { name: /load example/i }));
    await waitFor(() => screen.getByTestId("preview-card"));
    await user.click(screen.getByRole("button", { name: /analyze/i }));
    await waitFor(() =>
      expect(screen.getByTestId("summary-banner")).toBeInTheDocument()
    );
    expect(screen.getByText(ANALYZE_FIXTURE.recommendation)).toBeInTheDocument();
    expect(screen.getByTestId("summary-banner")).not.toHaveTextContent(ANALYZE_FIXTURE.recommendation);
  });

  // 13. Error message shown on analyzeExample rejection
  it("shows an error message when analyzeExample rejects", async () => {
    analyzeExample = jest.fn(() => Promise.reject(new Error("Network error")));
    const user = userEvent.setup();
    render(<ExampleBrowser loadExample={loadExample} analyzeExample={analyzeExample} />);
    await user.click(screen.getByRole("button", { name: /load example/i }));
    await waitFor(() => screen.getByTestId("preview-card"));
    await user.click(screen.getByRole("button", { name: /analyze/i }));
    await waitFor(() =>
      expect(screen.getByTestId("analyze-error")).toBeInTheDocument()
    );
  });

  // 14. Re-clicking Load Example replaces preview and resets Analyze
  it("re-loading replaces the preview card and disables Analyze until new load resolves", async () => {
    const { promise: promise2, resolve: resolve2 } = deferred<typeof EXAMPLE_RESULT>();
    let callCount = 0;
    loadExample = jest.fn(() => {
      callCount += 1;
      if (callCount === 1) return Promise.resolve(EXAMPLE_RESULT);
      return promise2;
    });
    const user = userEvent.setup();
    render(<ExampleBrowser loadExample={loadExample} analyzeExample={analyzeExample} />);

    // First load
    await user.click(screen.getByRole("button", { name: /load example/i }));
    await waitFor(() => screen.getByTestId("preview-card"));
    expect(screen.getByRole("button", { name: /analyze/i })).not.toBeDisabled();

    // Second load — Analyze must be disabled while in flight
    const secondClick = user.click(screen.getByRole("button", { name: /load example/i }));
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /analyze/i })).toBeDisabled()
    );

    // Resolve second load — Analyze re-enables
    await act(async () => { resolve2(EXAMPLE_RESULT); });
    await secondClick;
    await waitFor(() =>
      expect(screen.getByRole("button", { name: /analyze/i })).not.toBeDisabled()
    );
  });
});
