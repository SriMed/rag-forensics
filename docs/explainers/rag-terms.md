# RAG terms used in this project

This page defines the retrieval-augmented generation (RAG) vocabulary used across the README and
documentation, for readers who build software but have not worked with RAG evaluation. Project-
specific terms for grounding experiments (claim, evidence, oracle-evidence condition, and so on)
are defined in the [domain language](../../CONTEXT.md) instead.

## The pipeline

**RAG (retrieval-augmented generation)**: A system that answers a question by first looking up
relevant text and then asking a language model to write an answer from that text. The lookup is
meant to keep the answer tied to known sources instead of the model's memory.

**Corpus**: The full collection of documents the system can search. In this project's demo it is
the RAGBench dataset stored in a local ChromaDB index.

**Chunk**: A piece of a document small enough to search and to fit in a model prompt, often a
paragraph or a few hundred words. Documents are split into chunks before they are indexed.

**Embedding**: A list of numbers that represents a piece of text so that texts with similar meaning
have nearby numbers. Chunks and questions are both converted to embeddings so they can be compared.

**Vector store / index**: The database that holds chunk embeddings and finds the ones closest to a
question's embedding. This project uses ChromaDB.

**Retriever**: The component that takes the question and returns the most similar chunks from the
index, usually the top few (the "top-k").

**Retrieval score**: The number the retriever assigns to each returned chunk to say how close it is
to the question. Scores from different retrievers are not on a shared scale; see
[Methods and architecture](../reference/methods.md).

**Retrieved context**: The chunks the retriever returned for one question. It is what the
generator is allowed to use.

**Generator**: The language model that reads the question and the retrieved context and writes
the answer.

## Evaluating answers

**Grounded / supported**: An answer statement is grounded when the retrieved context actually
states or directly implies it.

**Hallucination**: A statement in the answer that the retrieved context does not support, whether
it is invented, wrong, or comes from the model's own background knowledge.

**Faithfulness**: A score for how much of the answer is supported by the retrieved context.
RAGAS computes it by splitting the answer into statements and asking a model judge whether each is
supported.

**Context utilization**: A RAGAS score for whether the retrieved chunks were useful for producing
the answer that was given. It is not a direct measure of retriever quality; see
[ADR-033](../../ADR.md#adr-033-ragas-context-utilization-replaces-ambiguous-retrieval-relevance).

**Model judge (LLM-as-judge)**: A language model asked to grade something, such as whether a
statement is supported. Its judgments are useful but can themselves be wrong or fail to return.

**Entailment**: The relationship where one text logically supports another. "The meeting moved to
Tuesday" entails "The meeting is not on Monday." Similar wording does not guarantee entailment.

**Reference answer / ground truth**: A known-correct answer written in advance. Some evaluation
metrics require one; RAG Forensics' live analysis does not.

**Reference-free evaluation**: Evaluation that judges an answer using only the question, answer,
and retrieved context, without a reference answer.
