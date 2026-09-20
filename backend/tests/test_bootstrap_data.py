"""Bootstrap against temporary Chroma storage; downloads and embeddings stay offline."""
import chromadb
import numpy as np
import pytest

from scripts import bootstrap_data


@pytest.fixture
def store(tmp_path, mocker):
    path = str(tmp_path / "chroma")
    mocker.patch.object(bootstrap_data, "CHROMA_PATH", path)
    mocker.patch.object(bootstrap_data, "DOMAINS", ["techqa"])
    mocker.patch.object(bootstrap_data, "load_dataset", return_value=[
        {"question": "What is TCP?", "answer": "A protocol.", "documents": ["TCP is a protocol."]},
    ])
    encoder = mocker.patch.object(bootstrap_data, "SentenceTransformer").return_value
    encoder.encode.return_value = np.array([[1.0, 0.0, 0.0]])
    client = chromadb.PersistentClient(path=path)
    collection = client.create_collection("techqa")
    collection.add(ids=["old"], documents=["Existing corpus"], embeddings=[[0.0, 1.0, 0.0]])
    return client, encoder


def test_embedding_failure_preserves_existing_corpus(store):
    client, encoder = store
    encoder.encode.side_effect = RuntimeError("embedding failed")

    with pytest.raises(RuntimeError, match="embedding failed"):
        bootstrap_data.bootstrap()

    assert client.get_collection("techqa").get()["documents"] == ["Existing corpus"]


def test_insertion_failure_preserves_existing_corpus(store, mocker):
    from chromadb.api.models.Collection import Collection

    client, _ = store
    mocker.patch.object(Collection, "add", side_effect=OSError("disk full"))

    with pytest.raises(OSError, match="disk full"):
        bootstrap_data.bootstrap()

    assert client.get_collection("techqa").get()["documents"] == ["Existing corpus"]


def test_promotion_failure_restores_existing_corpus(store, mocker):
    from chromadb.api.models.Collection import Collection

    client, _ = store
    modify = Collection.modify

    def fail_promotion(collection, name=None, **kwargs):
        if "-staging-" in collection.name and name == "techqa":
            raise OSError("promotion failed")
        return modify(collection, name=name, **kwargs)

    mocker.patch.object(Collection, "modify", fail_promotion)
    with pytest.raises(OSError, match="promotion failed"):
        bootstrap_data.bootstrap()

    assert client.get_collection("techqa").get()["documents"] == ["Existing corpus"]


def test_failed_insertion_leaves_no_staging_collection(store, mocker):
    from chromadb.api.models.Collection import Collection

    client, _ = store
    mocker.patch.object(Collection, "add", side_effect=OSError("disk full"))
    with pytest.raises(OSError, match="disk full"):
        bootstrap_data.bootstrap()

    assert [collection.name for collection in client.list_collections()] == ["techqa"]


def test_repeat_bootstrap_has_stable_ids_and_replaces_without_duplicates(store):
    client, _ = store
    bootstrap_data.bootstrap()
    first = client.get_collection("techqa").get()
    bootstrap_data.bootstrap()
    second = client.get_collection("techqa").get()

    assert first["ids"] == second["ids"] == [
        "techqa-5604c8ed66798ef77c74531a4dc47ba7bba03d06a9f20c1f00244acf385e0066_chunk_0"
    ]
    assert second["documents"] == ["TCP is a protocol."]
    assert [collection.name for collection in client.list_collections()] == ["techqa"]


def test_empty_dataset_cannot_replace_usable_corpus(store, mocker):
    client, _ = store
    mocker.patch.object(bootstrap_data, "load_dataset", return_value=[])

    with pytest.raises(ValueError, match="No chunks"):
        bootstrap_data.bootstrap()

    assert client.get_collection("techqa").get()["documents"] == ["Existing corpus"]


def test_numeric_zero_dataset_id_is_preserved(store, mocker):
    client, _ = store
    mocker.patch.object(bootstrap_data, "load_dataset", return_value=[
        {"id": 0, "question": "What is TCP?", "documents": ["TCP is a protocol."]},
    ])
    bootstrap_data.bootstrap()
    assert client.get_collection("techqa").get()["ids"] == ["0_chunk_0"]
