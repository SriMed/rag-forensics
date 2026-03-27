"""Tests for services/retriever.py — all ChromaDB calls are mocked."""
import pytest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_chroma_query_result(texts, scores, ids):
    """Build a ChromaDB-style query result dict."""
    return {
        "documents": [texts],
        "distances": [scores],
        "ids": [ids],
        "metadatas": [[{} for _ in texts]],
    }


def _make_chroma_get_result(ids, metadatas):
    """Build a ChromaDB-style get() result dict."""
    return {
        "ids": ids,
        "metadatas": metadatas,
        "documents": ["doc" for _ in ids],
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_METADATA = [
    {
        "example_id": "techqa-001",
        "question": "How does TCP work?",
        "answer": "TCP is a reliable, connection-oriented protocol.",
        "domain": "techqa",
    },
    {
        "example_id": "techqa-002",
        "question": "What is DNS?",
        "answer": "DNS stands for Domain Name System.",
        "domain": "techqa",
    },
]

SAMPLE_CHUNKS_TEXT = [
    "TCP establishes a connection via a three-way handshake.",
    "TCP guarantees delivery by using acknowledgements.",
    "Packets are retransmitted if no acknowledgement arrives.",
    "Flow control prevents the sender from overwhelming the receiver.",
    "TCP uses sequence numbers to reorder out-of-order segments.",
]

SAMPLE_CHUNK_IDS = [f"techqa-001_chunk_{i}" for i in range(5)]
# Scores from ChromaDB are *distances* (lower = more similar).
# retriever must convert to similarity floats in [0, 1].
SAMPLE_DISTANCES = [0.1, 0.25, 0.4, 0.6, 0.9]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestGetRandomExample:
    def test_returns_required_fields(self):
        """get_random_example() must return an object with the three required fields."""
        from services.retriever import get_random_example

        mock_collection = MagicMock()
        mock_collection.get.return_value = _make_chroma_get_result(
            ids=["techqa-001_chunk_0", "techqa-001_chunk_1"],
            metadatas=SAMPLE_METADATA[:2],
        )

        with patch("services.retriever._get_collection", return_value=mock_collection):
            result = get_random_example("techqa")

        assert hasattr(result, "example_id")
        assert hasattr(result, "question")
        assert hasattr(result, "context_preview")

    def test_example_id_is_non_empty_string(self):
        from services.retriever import get_random_example

        mock_collection = MagicMock()
        mock_collection.get.return_value = _make_chroma_get_result(
            ids=["techqa-001_chunk_0"],
            metadatas=[SAMPLE_METADATA[0]],
        )

        with patch("services.retriever._get_collection", return_value=mock_collection):
            result = get_random_example("techqa")

        assert isinstance(result.example_id, str) and result.example_id


class TestRetrieve:
    def _mock_collection(self, texts=None, distances=None, ids=None):
        texts = texts or SAMPLE_CHUNKS_TEXT
        distances = distances or SAMPLE_DISTANCES
        ids = ids or SAMPLE_CHUNK_IDS
        mock_collection = MagicMock()
        mock_collection.query.return_value = _make_chroma_query_result(texts, distances, ids)
        return mock_collection

    def test_returns_at_most_top_k_items(self):
        from services.retriever import retrieve

        mock_collection = self._mock_collection(
            texts=SAMPLE_CHUNKS_TEXT[:3],
            distances=SAMPLE_DISTANCES[:3],
            ids=SAMPLE_CHUNK_IDS[:3],
        )

        with patch("services.retriever._get_collection", return_value=mock_collection):
            results = retrieve("techqa-001", domain="techqa", top_k=3)

        assert len(results) <= 3

    def test_each_chunk_has_required_fields(self):
        from services.retriever import retrieve

        mock_collection = self._mock_collection()

        with patch("services.retriever._get_collection", return_value=mock_collection):
            results = retrieve("techqa-001", domain="techqa", top_k=5)

        for chunk in results:
            assert hasattr(chunk, "text"), "chunk missing 'text'"
            assert hasattr(chunk, "score"), "chunk missing 'score'"
            assert hasattr(chunk, "chunk_id"), "chunk missing 'chunk_id'"

    def test_scores_are_floats_between_0_and_1(self):
        from services.retriever import retrieve

        mock_collection = self._mock_collection()

        with patch("services.retriever._get_collection", return_value=mock_collection):
            results = retrieve("techqa-001", domain="techqa", top_k=5)

        for chunk in results:
            assert isinstance(chunk.score, float), f"score {chunk.score!r} is not float"
            assert 0.0 <= chunk.score <= 1.0, f"score {chunk.score} out of range"

    def test_results_ordered_descending_by_score(self):
        from services.retriever import retrieve

        mock_collection = self._mock_collection()

        with patch("services.retriever._get_collection", return_value=mock_collection):
            results = retrieve("techqa-001", domain="techqa", top_k=5)

        scores = [c.score for c in results]
        assert scores == sorted(scores, reverse=True), f"Not sorted descending: {scores}"

    def test_empty_collection_returns_empty_list(self):
        from services.retriever import retrieve

        mock_collection = MagicMock()
        mock_collection.query.return_value = _make_chroma_query_result([], [], [])

        with patch("services.retriever._get_collection", return_value=mock_collection):
            results = retrieve("techqa-001", domain="techqa", top_k=5)

        assert results == []


class TestGetReferenceAnswer:
    def test_returns_non_empty_string(self):
        from services.retriever import get_reference_answer

        mock_collection = MagicMock()
        mock_collection.get.return_value = _make_chroma_get_result(
            ids=["techqa-001_chunk_0"],
            metadatas=[SAMPLE_METADATA[0]],
        )

        with patch("services.retriever._get_collection", return_value=mock_collection):
            answer = get_reference_answer("techqa-001", domain="techqa")

        assert isinstance(answer, str) and answer
