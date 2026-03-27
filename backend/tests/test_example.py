import pytest
from fastapi.testclient import TestClient
from main import app

client = TestClient(app)


def test_post_example_valid_domain_returns_200():
    response = client.post("/example", json={"domain": "techqa"})
    assert response.status_code == 200


def test_post_example_unknown_domain_returns_422():
    response = client.post("/example", json={"domain": "unknown_domain"})
    assert response.status_code == 422


def test_post_example_finqa_returns_200():
    response = client.post("/example", json={"domain": "finqa"})
    assert response.status_code == 200


def test_post_example_covidqa_returns_200():
    response = client.post("/example", json={"domain": "covidqa"})
    assert response.status_code == 200


def test_example_response_has_required_fields():
    response = client.post("/example", json={"domain": "techqa"})
    assert response.status_code == 200
    body = response.json()
    assert "example_id" in body
    assert "question" in body
    assert "context_preview" in body


def test_example_response_example_id_is_string():
    response = client.post("/example", json={"domain": "techqa"})
    body = response.json()
    assert isinstance(body["example_id"], str)


def test_example_response_context_preview_is_string():
    response = client.post("/example", json={"domain": "techqa"})
    body = response.json()
    assert isinstance(body["context_preview"], str)
