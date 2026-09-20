"""Tests for /health and /ready — no external calls, no secrets in responses."""
from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def _forbid_llm(mocker):
    return mocker.patch("services.llm.anthropic.Anthropic", side_effect=AssertionError("LLM must not be called"))


def test_health_is_ok_and_cheap(mocker):
    _forbid_llm(mocker)
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ready_when_key_and_corpus_present(mocker, monkeypatch):
    _forbid_llm(mocker)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-value")
    mocker.patch("routers.health.available_domains", return_value=["finqa", "techqa"])
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json() == {
        "status": "ready",
        "checks": {"anthropic_api_key": True, "bundled_corpus": True},
        "bundled_corpus_domains": ["finqa", "techqa"],
    }


def test_ready_reports_missing_api_key_as_503(mocker, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    mocker.patch("routers.health.available_domains", return_value=["techqa"])
    response = client.get("/ready")
    assert response.status_code == 503
    body = response.json()
    assert body["status"] == "not_ready"
    assert body["checks"]["anthropic_api_key"] is False


def test_missing_corpus_does_not_block_custom_analysis_readiness(mocker, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-value")
    mocker.patch("routers.health.available_domains", return_value=[])
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["checks"] == {"anthropic_api_key": True, "bundled_corpus": False}


def test_ready_never_echoes_the_api_key(mocker, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-super-secret")
    mocker.patch("routers.health.available_domains", return_value=[])
    assert "sk-ant-super-secret" not in client.get("/ready").text


def test_ready_treats_blank_api_key_as_missing(mocker, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "   ")
    mocker.patch("routers.health.available_domains", return_value=[])
    assert client.get("/ready").status_code == 503


def test_ready_survives_corpus_probe_failure(mocker, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test-value")
    mocker.patch("routers.health.available_domains", side_effect=OSError("disk error at /Users/x"))
    response = client.get("/ready")
    assert response.status_code == 200
    assert response.json()["checks"]["bundled_corpus"] is False
    assert "/Users/x" not in response.text
