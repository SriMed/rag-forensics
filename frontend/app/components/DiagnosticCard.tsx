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
    verdict_signals,
    verdict_reasoning,
    retrieved_chunk_details,
    recommendation,
  } = response;

  const topPriority = verdict_signals[0]?.priority_score ?? 0;
  const priorityLabel =
    topPriority > 0.5
      ? "Higher investigation priority"
      : topPriority > 0.2
      ? "Moderate investigation priority"
      : "Lower investigation priority";

  const chunkColorIndex = new Map<string, number>();

  return (
    <div className="mx-auto max-w-3xl space-y-6 p-4 min-w-[320px]">
      {/* Priority banner: the score is a heuristic ordering index, not health or severity. */}
      <div
        data-testid="summary-banner"
        className="rounded-lg border border-slate-200 bg-slate-50 px-4 py-3 text-sm text-slate-800"
      >
        <span className="font-semibold">{priorityLabel}</span>
        <span className="text-slate-600"> · Heuristic ordering, not calibrated health or severity.</span>
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

      {/* Inspectable diagnostic record */}
      <section
        aria-labelledby="diagnostic-reasoning-heading"
        className="rounded-lg border border-gray-200 bg-white p-5 shadow-sm space-y-5"
      >
        <div>
          <h2 id="diagnostic-reasoning-heading" className="text-sm font-bold uppercase tracking-wide text-gray-500">
            Diagnostic Reasoning
          </h2>
          <p className="mt-1 text-xs text-gray-500">
            Observations and competing hypotheses to investigate; these do not establish a cause.
          </p>
        </div>

        <div>
          <h3 className="text-sm font-semibold text-gray-800">Observations</h3>
          <ul className="mt-2 space-y-2">
            {verdict_reasoning.observations.map((observation) => (
              <li key={observation.signal_name} className="rounded border border-gray-100 bg-gray-50 p-3">
                <div className="flex flex-wrap items-center gap-2">
                  <span className="text-sm font-medium text-gray-800">{observation.signal_name.replaceAll("_", " ")}</span>
                  <span className="rounded bg-slate-200 px-2 py-0.5 text-[11px] font-medium text-slate-700">
                    Reliability: {observation.reliability.replaceAll("_", " ")}
                  </span>
                </div>
                <p className="mt-1 text-sm text-gray-600">{observation.description}</p>
              </li>
            ))}
          </ul>
        </div>

        <div>
          <h3 className="text-sm font-semibold text-gray-800">Competing hypotheses</h3>
          <ul className="mt-2 space-y-2">
            {verdict_reasoning.hypotheses.map((hypothesis) => (
              <li key={hypothesis.hypothesis_id} className="text-sm text-gray-700">
                <span className="font-mono text-xs font-semibold text-gray-500">{hypothesis.hypothesis_id}</span>{" "}
                <span>Hypothesis: {hypothesis.statement}</span>
              </li>
            ))}
          </ul>
        </div>

        {verdict_reasoning.test && (
          <div data-testid="discriminating-test" className="rounded border border-blue-100 bg-blue-50 p-4 space-y-3">
            <h3 className="text-sm font-semibold text-blue-900">Discriminating test</h3>
            <p className="text-sm text-blue-900"><span className="font-semibold">Component:</span> {verdict_reasoning.test.component}</p>
            <p className="text-sm text-blue-900"><span className="font-semibold">Action:</span> {verdict_reasoning.test.action}</p>
            <ul className="space-y-2">
              {verdict_reasoning.test.interpretations.map((interpretation, index) => (
                <li key={`${interpretation.outcome}-${index}`} className="text-sm text-blue-900">
                  <span className="font-semibold">Outcome:</span> {interpretation.outcome}{" "}
                  <span className="text-blue-700">Supports: {interpretation.supports_hypothesis_ids.join(", ") || "none specified"}</span>
                </li>
              ))}
            </ul>
          </div>
        )}

        <div className="border-t border-gray-100 pt-4">
          <h3 className="text-xs font-semibold uppercase tracking-wide text-gray-500">Generated recommendation</h3>
          <p className="mt-1 text-sm text-gray-600">{recommendation}</p>
        </div>
      </section>

      {/* Retrieved evidence with source-boundary metadata */}
      <section className="rounded-lg border border-gray-200 bg-white p-5 shadow-sm space-y-3">
        <h2 className="text-sm font-bold uppercase tracking-wide text-gray-500">Retrieved Evidence</h2>
        {retrieved_chunk_details.length === 0 ? (
          <p className="text-sm text-gray-500">No retrieved chunk details available.</p>
        ) : (
          <ul className="space-y-3">
            {retrieved_chunk_details.map((chunk) => (
              <li key={chunk.chunk_id} className="rounded border border-gray-100 p-3">
                <div className="flex flex-wrap gap-x-3 gap-y-1 text-xs text-gray-500">
                  <span className="font-mono font-semibold text-gray-700">{chunk.chunk_id}</span>
                  <span>Score: {chunk.score.toFixed(2)}</span>
                  <span>Completeness: {chunk.completeness}</span>
                  <span>Provenance: {chunk.completeness_source}</span>
                </div>
                <p className="mt-2 text-sm text-gray-700">{chunk.text}</p>
              </li>
            ))}
          </ul>
        )}
      </section>

      {/* Baseline Metrics */}
      <section className="rounded-lg border border-gray-200 bg-white p-5 shadow-sm space-y-4">
        <h2 className="text-sm font-bold uppercase tracking-wide text-gray-500">
          Baseline Metrics
        </h2>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <p className="text-xs text-gray-600 mb-0.5">Context Utilization</p>
            <p className="text-2xl font-bold text-gray-800">
              {ragas.context_utilization.score === null
                ? "Unavailable"
                : `${(ragas.context_utilization.score * 100).toFixed(0)}%`}
            </p>
            {ragas.context_utilization.error && (
              <p className="text-xs text-gray-500">{ragas.context_utilization.error}</p>
            )}
          </div>
          <div>
            <p className="text-xs text-gray-600 mb-0.5">Faithfulness</p>
            <p className="text-2xl font-bold text-gray-800">
              {ragas.faithfulness.score === null
                ? "Unavailable"
                : `${(ragas.faithfulness.score * 100).toFixed(0)}%`}
            </p>
            {ragas.faithfulness.error && (
              <p className="text-xs text-gray-500">{ragas.faithfulness.error}</p>
            )}
          </div>
        </div>
        {ragas.utilization_context_excerpts.length > 0 && (
          <div>
            <p className="text-xs text-gray-600 mb-1">Context excerpts</p>
            <ul className="space-y-1 pl-3 border-l-2 border-gray-200">
              {ragas.utilization_context_excerpts.map((e, i) => (
                <li key={i} className="text-xs text-gray-500 italic">
                  {e}
                </li>
              ))}
            </ul>
            <p className="mt-1 text-[11px] text-gray-500">{ragas.excerpt_caveat}</p>
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
          <span className="text-sm font-medium text-gray-700">Hedging Mismatch</span>
          <p className="text-xs text-gray-500">
            {hedging_mismatch.status === "error" ? (
              <>Analysis unavailable: {hedging_mismatch.error}</>
            ) : <>Overconfident: {(hedging_mismatch.overconfident_fraction * 100).toFixed(0)}% ·
            {hedging_mismatch.evaluated_claim_count}/{hedging_mismatch.total_claims} claims evaluated</>}
          </p>
        </div>

        {/* Chunk Attribution — annotated answer */}
        <div className="space-y-2">
          <span className="text-sm font-medium text-gray-700">Chunk Attribution</span>
          <p className="text-xs text-gray-500">
            No close semantic source: {(chunk_attribution.unattributed_fraction * 100).toFixed(0)}% ·
            Mean score: {chunk_attribution.mean_attribution_score.toFixed(2)}
          </p>
          <p className="text-[11px] text-gray-500">{chunk_attribution.caveat}</p>
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
            <span>Normalized entropy: {retrieval_distribution.normalized_entropy.toFixed(2)}</span>
            <span>Gap: {retrieval_distribution.score_gap.toFixed(2)}</span>
            <span>Tail mass: {retrieval_distribution.tail_mass.toFixed(2)}</span>
            <span>Decay: {retrieval_distribution.decay_rate?.toFixed(2) ?? "unavailable"}</span>
            <span>Top: {retrieval_distribution.top_score.toFixed(2)}</span>
            <span>Chunks: {retrieval_distribution.n_chunks}</span>
          </div>
          <p className="text-[11px] text-gray-500">{retrieval_distribution.interpretation}</p>
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
                Retrieved-Context Fit
              </span>
              <span className="inline-block rounded px-2 py-0.5 text-xs font-semibold uppercase tracking-wide bg-red-100 text-red-800">
                {query_corpus_fit.observed_fit ?? query_corpus_fit.status}
              </span>
            </div>
            {query_corpus_fit.suggested_questions.length > 0 && (
              <div>
                <p className="text-xs text-gray-600 mb-1">Questions these chunks answer well:</p>
                <ul className="space-y-1 pl-3 border-l-2 border-gray-200">
                  {query_corpus_fit.suggested_questions.map((q, i) => (
                    <li key={i} className="text-xs text-gray-600">
                      {q.question} <span className="text-gray-400">({q.source_chunk_ids.join(", ")})</span>
                    </li>
                  ))}
                </ul>
              </div>
            )}
            {query_corpus_fit.status === "error" && query_corpus_fit.error && (
              <p className="text-xs text-red-700">
                Classification unavailable: {query_corpus_fit.error.replaceAll("_", " ")}.
              </p>
            )}
          </div>
        )}
      </section>
    </div>
  );
}
