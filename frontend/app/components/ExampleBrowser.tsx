"use client";

import { useState } from "react";
import type { ExampleResult } from "@/lib/api";

const CONTEXT_TRUNCATE_LENGTH = 300;

const DOMAINS = [
  { value: "techqa", label: "TechQA" },
  { value: "finqa", label: "FinQA" },
  { value: "covidqa", label: "CovidQA" },
] as const;

type Domain = (typeof DOMAINS)[number]["value"];

interface Props {
  loadExample: (domain: string) => Promise<ExampleResult>;
  analyzeExample: (exampleId: string) => Promise<void>;
}

export default function ExampleBrowser({ loadExample, analyzeExample }: Props) {
  const [domain, setDomain] = useState<Domain>("techqa");
  const [example, setExample] = useState<ExampleResult | null>(null);
  const [loadingExample, setLoadingExample] = useState(false);
  const [loadingAnalyze, setLoadingAnalyze] = useState(false);

  async function handleLoadExample() {
    setExample(null);
    setLoadingExample(true);
    try {
      const result = await loadExample(domain);
      setExample(result);
    } finally {
      setLoadingExample(false);
    }
  }

  async function handleAnalyze() {
    if (!example) return;
    setLoadingAnalyze(true);
    try {
      await analyzeExample(example.exampleId);
    } finally {
      setLoadingAnalyze(false);
    }
  }

  const truncatedContext = example
    ? example.context.length > CONTEXT_TRUNCATE_LENGTH
      ? example.context.slice(0, CONTEXT_TRUNCATE_LENGTH - 1) + "…"
      : example.context
    : null;

  return (
    <div className="mx-auto max-w-2xl space-y-6 p-6">
      <h1 className="text-2xl font-bold">RAG Forensics</h1>

      {/* Controls */}
      <div className="flex items-center gap-4">
        <select
          value={domain}
          onChange={(e) => setDomain(e.target.value as Domain)}
          className="rounded border border-gray-300 bg-white px-3 py-2 text-sm focus:outline-none focus:ring-2 focus:ring-blue-500"
        >
          {DOMAINS.map((d) => (
            <option key={d.value} value={d.value}>
              {d.label}
            </option>
          ))}
        </select>

        <button
          onClick={handleLoadExample}
          disabled={loadingExample}
          className="flex items-center gap-2 rounded bg-blue-600 px-4 py-2 text-sm font-medium text-white hover:bg-blue-700 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {loadingExample && (
            <span
              data-testid="load-spinner"
              className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent"
              aria-hidden="true"
            />
          )}
          Load Example
        </button>

        <button
          onClick={handleAnalyze}
          disabled={!example || loadingAnalyze || loadingExample}
          className="flex items-center gap-2 rounded bg-green-600 px-4 py-2 text-sm font-medium text-white hover:bg-green-700 disabled:cursor-not-allowed disabled:opacity-60"
        >
          {loadingAnalyze && (
            <span
              data-testid="analyze-spinner"
              className="h-4 w-4 animate-spin rounded-full border-2 border-white border-t-transparent"
              aria-hidden="true"
            />
          )}
          Analyze
        </button>
      </div>

      {/* Preview card */}
      {example && !loadingExample && (
        <div
          data-testid="preview-card"
          className="rounded-lg border border-gray-200 bg-white p-5 shadow-sm"
        >
          <p className="mb-3 text-sm font-semibold text-gray-500 uppercase tracking-wide">
            Question
          </p>
          <p className="mb-4 text-base font-medium text-gray-900">{example.question}</p>
          <p className="mb-2 text-sm font-semibold text-gray-500 uppercase tracking-wide">
            Context
          </p>
          <p data-testid="preview-context" className="text-sm text-gray-700 leading-relaxed">
            {truncatedContext}
          </p>
        </div>
      )}
    </div>
  );
}
