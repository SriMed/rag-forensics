"use client";

import type { AnalyzeResponse } from "@/lib/api";

// Deterministic palette for chunk_id → color class. Cycles through 6 hues.
const CHUNK_COLORS = [
  "bg-blue-100 text-blue-800",
  "bg-purple-100 text-purple-800",
  "bg-green-100 text-green-800",
  "bg-orange-100 text-orange-800",
  "bg-teal-100 text-teal-800",
  "bg-pink-100 text-pink-800",
];

function chunkColor(chunkId: string, index: Map<string, number>): string {
  if (!index.has(chunkId)) {
    index.set(chunkId, index.size % CHUNK_COLORS.length);
  }
  return CHUNK_COLORS[index.get(chunkId)!];
}

type Verdict = "pass" | "warn" | "fail";

function verdictClass(verdict: Verdict): string {
  if (verdict === "pass") return "bg-green-100 text-green-800";
  if (verdict === "warn") return "bg-amber-100 text-amber-800";
  return "bg-red-100 text-red-800";
}

function VerdictBadge({ verdict, testId }: { verdict: Verdict; testId: string }) {
  return (
    <span
      data-testid={testId}
      className={`inline-block rounded px-2 py-0.5 text-xs font-semibold uppercase tracking-wide ${verdictClass(verdict)}`}
    >
      {verdict}
    </span>
  );
}

interface Props {
  response: AnalyzeResponse;
}

export default function DiagnosticCard({ response }: Props) {
  const {
    question,
    generated_answer,
    ragas,
    chunk_attribution,
    hedging_mismatch,
    retrieval_distribution,
    embedding_space,
    query_corpus_fit,
    recommendation,
    rule_id,
  } = response;

  // Derive synthetic DimensionResults for continuous-metric modules
  // so they render consistently in the forensics section.
  const hedgingVerdict: Verdict =
    hedging_mismatch.overconfident_fraction > 0.3
      ? "fail"
      : hedging_mismatch.overconfident_fraction > 0.15
      ? "warn"
      : "pass";

  const attributionVerdict: Verdict =
    chunk_attribution.unattributed_fraction > 0.4
      ? "fail"
      : chunk_attribution.unattributed_fraction > 0.2
      ? "warn"
      : "pass";

  // Banner uses rule_id as the authoritative verdict — it is the backend's
  // synthesised signal across all modules. Local per-dimension badges are
  // heuristic approximations for display only and can diverge.
  const overall: Verdict =
    rule_id === "R07" ? "pass" : "fail";

  const chunkColorIndex = new Map<string, number>();

  return (
    <div className="mx-auto max-w-3xl space-y-6 p-4 min-w-[320px]">
      {/* Summary banner */}
      <div
        data-testid="summary-banner"
        className={`rounded-lg px-4 py-3 text-sm font-medium ${
          overall === "pass"
            ? "bg-green-50 text-green-800 border border-green-200"
            : overall === "warn"
            ? "bg-amber-50 text-amber-800 border border-amber-200"
            : "bg-red-50 text-red-800 border border-red-200"
        }`}
      >
        {overall === "pass"
          ? "Pipeline looks healthy"
          : overall === "warn"
          ? "Potential issues detected"
          : "Issues detected"}{" "}
        — {recommendation}{" "}
        <span className="text-current/70 font-normal">({rule_id})</span>
      </div>

      {/* Question + Answer */}
      <div className="rounded-lg border border-gray-200 bg-white p-5 shadow-sm space-y-3">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wide text-gray-600 mb-1">
            Question
          </p>
          <p className="text-base font-medium text-gray-900">{question}</p>
        </div>
        <div>
          <p className="text-xs font-semibold uppercase tracking-wide text-gray-600 mb-1">
            Generated Answer
          </p>
          <p className="text-sm text-gray-700 leading-relaxed">{generated_answer}</p>
        </div>
      </div>

      {/* Baseline Metrics */}
      <section className="rounded-lg border border-gray-200 bg-white p-5 shadow-sm space-y-4">
        <h2 className="text-sm font-bold uppercase tracking-wide text-gray-500">
          Baseline Metrics
        </h2>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <p className="text-xs text-gray-600 mb-0.5">Retrieval Relevance</p>
            <p className="text-2xl font-bold text-gray-800">
              {(ragas.retrieval_relevance_score * 100).toFixed(0)}
              <span className="text-sm font-normal text-gray-600">%</span>
            </p>
          </div>
          <div>
            <p className="text-xs text-gray-600 mb-0.5">Faithfulness</p>
            <p className="text-2xl font-bold text-gray-800">
              {(ragas.faithfulness_score * 100).toFixed(0)}
              <span className="text-sm font-normal text-gray-600">%</span>
            </p>
          </div>
        </div>
        {ragas.relevance_evidence.length > 0 && (
          <div>
            <p className="text-xs text-gray-600 mb-1">Relevance Evidence</p>
            <ul className="space-y-1 pl-3 border-l-2 border-gray-200">
              {ragas.relevance_evidence.map((e, i) => (
                <li key={i} className="text-xs text-gray-500 italic">
                  {e}
                </li>
              ))}
            </ul>
          </div>
        )}
      </section>

      {/* Forensics */}
      <section className="rounded-lg border border-gray-200 bg-white p-5 shadow-sm space-y-5">
        <h2 className="text-sm font-bold uppercase tracking-wide text-gray-500">
          Forensics
        </h2>

        {/* Hedging Mismatch — continuous metrics */}
        <div className="space-y-1">
          <div className="flex items-center gap-3">
            <span className="text-sm font-medium text-gray-700">
              Hedging Mismatch
            </span>
            <VerdictBadge verdict={hedgingVerdict} testId="badge-hedging_mismatch" />
          </div>
          <p className="text-xs text-gray-500">
            Overconfident: {(hedging_mismatch.overconfident_fraction * 100).toFixed(0)}% ·
            Underconfident: {(hedging_mismatch.underconfident_fraction * 100).toFixed(0)}% ·
            {hedging_mismatch.total_claims} claims
          </p>
        </div>

        {/* Chunk Attribution — annotated answer */}
        <div className="space-y-2">
          <div className="flex items-center gap-3">
            <span className="text-sm font-medium text-gray-700">
              Chunk Attribution
            </span>
            <VerdictBadge verdict={attributionVerdict} testId="badge-chunk_attribution" />
          </div>
          <p className="text-xs text-gray-500">
            Unattributed: {(chunk_attribution.unattributed_fraction * 100).toFixed(0)}% ·
            Mean score: {chunk_attribution.mean_attribution_score.toFixed(2)}
          </p>
          <div className="flex flex-wrap gap-1 leading-loose">
            {chunk_attribution.attribution_map.map((entry, i) => {
              const isUnattributed = entry.attribution_strength === "unattributed";
              const colorClass = isUnattributed
                ? "bg-red-100 text-red-800"
                : chunkColor(entry.chunk_id ?? "__none__", chunkColorIndex);
              const tooltip = isUnattributed
                ? `Unattributed (score: ${entry.similarity_score.toFixed(2)})`
                : `${entry.chunk_id} · score: ${entry.similarity_score.toFixed(2)}`;
              return (
                <span
                  key={i}
                  data-testid={`attribution-sentence-${i}`}
                  title={tooltip}
                  className={`rounded px-1.5 py-0.5 text-sm cursor-default ${colorClass}`}
                >
                  {entry.sentence}
                </span>
              );
            })}
          </div>
        </div>

        {/* Retrieval Distribution — numeric summary */}
        <div className="space-y-1">
          <span className="text-sm font-medium text-gray-700">
            Retrieval Distribution
          </span>
          <div className="grid grid-cols-3 gap-2 text-xs text-gray-500">
            <span>Entropy: {retrieval_distribution.score_entropy.toFixed(2)}</span>
            <span>Gap: {retrieval_distribution.score_gap.toFixed(2)}</span>
            <span>Tail mass: {retrieval_distribution.tail_mass.toFixed(2)}</span>
            <span>Decay: {retrieval_distribution.decay_rate.toFixed(2)}</span>
            <span>Top: {retrieval_distribution.top_score.toFixed(2)}</span>
            <span>Chunks: {retrieval_distribution.n_chunks}</span>
          </div>
        </div>

        {/* Embedding Space — numeric summary */}
        <div className="space-y-1">
          <span className="text-sm font-medium text-gray-700">Embedding Space</span>
          <div className="grid grid-cols-3 gap-2 text-xs text-gray-500">
            <span>Centroid dist: {embedding_space.centroid_distance.toFixed(2)}</span>
            <span>Chunk spread: {embedding_space.chunk_spread.toFixed(2)}</span>
            <span>Query isolation: {embedding_space.query_isolation.toFixed(2)}</span>
          </div>
        </div>

        {/* Query-Corpus Fit — only shown when triggered */}
        {query_corpus_fit.triggered && (
          <div className="space-y-2">
            <div className="flex items-center gap-3">
              <span className="text-sm font-medium text-gray-700">
                Query-Corpus Fit
              </span>
              <span className="inline-block rounded px-2 py-0.5 text-xs font-semibold uppercase tracking-wide bg-red-100 text-red-800">
                {query_corpus_fit.mismatch_type ?? "triggered"}
              </span>
            </div>
            {query_corpus_fit.suggested_questions.length > 0 && (
              <div>
                <p className="text-xs text-gray-600 mb-1">Questions these chunks answer well:</p>
                <ul className="space-y-1 pl-3 border-l-2 border-gray-200">
                  {query_corpus_fit.suggested_questions.map((q, i) => (
                    <li key={i} className="text-xs text-gray-600">
                      {q.question}
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}
      </section>
    </div>
  );
}
