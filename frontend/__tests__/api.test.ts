import { loadExample, analyzeExample } from "@/lib/api";
import type { AnalyzeResponse } from "@/lib/api";

// Minimal AnalyzeResponse fixture — only fields the tests assert on
const ANALYZE_FIXTURE: AnalyzeResponse = {
  question: "What is X?",
  generated_answer: "X is Y.",
  retrieved_chunks: [],
  ragas: { context_utilization: { score: 0.8, status: "ok", error: null }, faithfulness: { score: 0.9, status: "ok", error: null }, utilization_context_excerpts: [], faithfulness_context_excerpts: [], excerpt_caveat: "Context excerpts are not score evidence." },
  hedging_mismatch: { overconfident_fraction: 0, underconfident_fraction: 0, total_claims: 0, claim_breakdown: [], status: "ok", error: null, evaluated_chunk_count: 0 },
  chunk_attribution: { unattributed_fraction: 0, mean_attribution_score: 0.9, weak_match_fraction: 0, attribution_map: [], method: "semantic_similarity", caveat: "Similarity does not prove entailment." },
  retrieval_distribution: { score_gap: 0.3, score_entropy: 0.9, decay_rate: 0.4, tail_mass: 0.1, top_score: 0.9, n_chunks: 3, normalized_entropy: 0.5, interpretation: "Interpret with absolute relevance." },
  embedding_space: { centroid_distance: 0.3, chunk_spread: 0.2, query_isolation: 0.8, projection: [] },
  query_corpus_fit: { triggered: false, observed_fit: null, suggested_questions: [], mean_question_similarity: null, status: "not_run", error: null },
  recommendation: "All good.",
  verdict_signals: [],
};

beforeEach(() => {
  global.fetch = jest.fn();
});

afterEach(() => {
  jest.resetAllMocks();
});

// --- loadExample ---

describe("loadExample", () => {
  // 1. Posts to /example with correct domain
  it("posts to /example with the selected domain", async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => ({ example_id: "techqa_001", question: "Q?", context_preview: "ctx" }),
    });
    await loadExample("techqa");
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining("/example"),
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ domain: "techqa" }),
      })
    );
  });

  // 2. Returns mapped ExampleResult on 200
  it("maps backend snake_case to frontend camelCase", async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => ({ example_id: "techqa_001", question: "Q?", context_preview: "ctx" }),
    });
    const result = await loadExample("techqa");
    expect(result.exampleId).toBe("techqa_001");
    expect(result.question).toBe("Q?");
    expect(result.context).toBe("ctx");
  });

  // 3. Throws on non-200
  it("throws on a non-200 response", async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({ ok: false, status: 500 });
    await expect(loadExample("techqa")).rejects.toThrow();
  });
});

// --- analyzeExample ---

describe("analyzeExample", () => {
  // 4. Posts to /analyze with correct example_id
  it("posts to /analyze with the example_id", async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => ANALYZE_FIXTURE,
    });
    await analyzeExample("techqa_001");
    expect(global.fetch).toHaveBeenCalledWith(
      expect.stringContaining("/analyze"),
      expect.objectContaining({
        method: "POST",
        body: JSON.stringify({ example_id: "techqa_001" }),
      })
    );
  });

  // 5. Returns parsed AnalyzeResponse on 200
  it("returns the parsed AnalyzeResponse on 200", async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({
      ok: true,
      json: async () => ANALYZE_FIXTURE,
    });
    const result = await analyzeExample("techqa_001");
    expect(result.verdict_signals).toEqual([]);
    expect(result.recommendation).toBe("All good.");
  });

  // 6. Throws on non-200
  it("throws on a non-200 response", async () => {
    (global.fetch as jest.Mock).mockResolvedValueOnce({ ok: false, status: 422 });
    await expect(analyzeExample("techqa_001")).rejects.toThrow();
  });
});
