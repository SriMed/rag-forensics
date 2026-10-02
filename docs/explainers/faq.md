# Frequently asked questions

Short answers to common questions about RAG Forensics, with links to the pages that hold the details.

## What is RAG Forensics?

A tool for investigating one RAG answer that may have gone wrong. Given a question, the answer, and the chunks that were retrieved, it records what it observed, a ranked set of possible explanations with reliability labels, and follow-up tests that could tell those explanations apart. It narrows an investigation; it does not prove a root cause. See [How RAG Forensics investigates an answer](how-rag-forensics-works.md).

## Why was it built?

Two problems came up repeatedly in a production RAG system.

First, evaluation scores such as faithfulness or relevance can show that an answer deserves attention, but they rarely show why. A weak answer may reflect retrieval, missing evidence, contradiction, changed qualifiers, generation, or a failed evaluator. Treating one score as proof of one cause hides the others.

Second, those scores carry meaning that is easiest to act on with data-science training. A mostly SWE-heavy team with little data-science support. A faithfulness score of 0.62 is a real measurement, but going from that number to "what do I change next" requires reasoning about cause and effect that the score does not spell out. RAG Forensics turns the scores and other signals into a record an engineer can act on: what was observed, which explanations remain open, and which experiment would distinguish them.

No proprietary incidents, outputs, or user research are included in this repository; its empirical claims come from public datasets.

## Why doesn't it run the follow-up tests itself?

Because it usually doesn't control the pipeline that produced the answer. The main entry point, [`POST /analyze/custom`](../reference/api-integration.md), receives a completed question, answer, and chunks from the caller's own system. Re-running retrieval with a different query or regenerating the answer would need access to that caller's retriever, index, and generator, so the tool records the proposed test and what each outcome would mean instead.

Running interventions directly is a different and already-studied design: RAGGY, for example, provides an interactive interface for changing pipeline stages and re-running them. See [Related work](../reference/related-work.md). Executing tests from RAG Forensics would be a possible extension, but it is not currently planned.

## What do terms like retriever, chunk, and faithfulness mean?

See [RAG terms used in this project](rag-terms.md) for the general vocabulary, and the [domain language](../../CONTEXT.md) for terms specific to this project's grounding experiments.

## How is it different from RAGAS, RAGChecker, or RAGVUE?

RAGAS and similar suites mainly produce quality scores. RAGChecker and RAGVUE provide finer-grained diagnostics and overlap substantially with this project. RAG Forensics organizes its output as an investigation record that keeps observations separate from hypotheses and ends with a discriminating test. It does not claim better diagnostic accuracy than those tools. See [Related work](../reference/related-work.md).
