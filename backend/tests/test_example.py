import pytest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient
from main import app
from models import StoredExample

client = TestClient(app)

_FAKE_EXAMPLE = StoredExample(
    example_id="techqa-001",
    question="How does TCP work?",
    context_preview="TCP establishes a connection via a three-way handshake.",
)


def test_post_example_valid_domain_returns_200():
    with patch("routers.example.get_random_example", return_value=_FAKE_EXAMPLE):
        response = client.post("/example", json={"domain": "techqa"})
    assert response.status_code == 200


def test_post_example_unknown_domain_returns_422():
    response = client.post("/example", json={"domain": "unknown_domain"})
    assert response.status_code == 422


def test_post_example_finqa_returns_200():
    fake = StoredExample(
        example_id="finqa-001",
        question="What is EBITDA?",
        context_preview="EBITDA stands for Earnings Before Interest, Taxes, Depreciation, and Amortization.",
    )
    with patch("routers.example.get_random_example", return_value=fake):
        response = client.post("/example", json={"domain": "finqa"})
    assert response.status_code == 200


def test_post_example_covidqa_returns_200():
    fake = StoredExample(
        example_id="covidqa-001",
        question="How does COVID-19 spread?",
        context_preview="COVID-19 spreads primarily through respiratory droplets.",
    )
    with patch("routers.example.get_random_example", return_value=fake):
        response = client.post("/example", json={"domain": "covidqa"})
    assert response.status_code == 200


def test_example_response_has_required_fields():
    with patch("routers.example.get_random_example", return_value=_FAKE_EXAMPLE):
        response = client.post("/example", json={"domain": "techqa"})
    assert response.status_code == 200
    body = response.json()
    assert "example_id" in body
    assert "question" in body
    assert "context_preview" in body


def test_example_response_example_id_is_string():
    with patch("routers.example.get_random_example", return_value=_FAKE_EXAMPLE):
        response = client.post("/example", json={"domain": "techqa"})
    body = response.json()
    assert isinstance(body["example_id"], str)


def test_example_response_context_preview_is_string():
    with patch("routers.example.get_random_example", return_value=_FAKE_EXAMPLE):
        response = client.post("/example", json={"domain": "techqa"})
    body = response.json()
    assert isinstance(body["context_preview"], str)
