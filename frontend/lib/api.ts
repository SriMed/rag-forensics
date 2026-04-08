export interface ExampleResult {
  exampleId: string;
  question: string;
  context: string;
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
