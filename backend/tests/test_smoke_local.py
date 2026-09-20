"""The documented smoke client uses HTTP only; tests never call model providers."""
import httpx
import pytest


def test_health_only_smoke_does_not_submit_analysis():
    from scripts.smoke_local import run_smoke

    def respond(request):
        assert request.method == "GET"
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        assert request.url.path == "/ready"
        return httpx.Response(200, json={
            "status": "ready", "checks": {"anthropic_api_key": True, "bundled_corpus": False},
            "bundled_corpus_domains": [],
        })

    with httpx.Client(base_url="http://localhost:8000", transport=httpx.MockTransport(respond)) as client:
        result = run_smoke(client)

    assert result == {"health": "ok", "ready": True, "bundled_corpus_domains": [], "analysis": "not_requested"}


@pytest.fixture
def backend_client(mocker, monkeypatch):
    from fastapi.testclient import TestClient

    from main import app
    from tests.test_analyze_custom import _patch_services

    _patch_services(mocker)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")
    mocker.patch("routers.health.available_domains", return_value=[])
    with TestClient(app) as server:
        transport = httpx.MockTransport(lambda request: server.request(
            request.method, request.url.path, content=request.content, headers=request.headers,
        ))
        with httpx.Client(base_url="http://localhost:8000", transport=transport) as client:
            yield client


def test_analysis_smoke_uses_custom_endpoint_without_corpus(backend_client):
    from scripts.smoke_local import run_smoke

    result = run_smoke(backend_client, analyze=True)
    assert result["analysis"] == "ok"
    assert result["bundled_corpus_domains"] == []


def test_analysis_smoke_rejects_unavailable_evaluation(mocker, backend_client):
    from models import RAGASMetricResult
    from scripts.smoke_local import run_smoke
    mocker.patch("routers.analyze.score_answer_faithfulness", return_value=(
        RAGASMetricResult(score=None, status="unavailable", error="evaluation_failed"), [],
    ))
    with pytest.raises(ValueError, match="unavailable"):
        run_smoke(backend_client, analyze=True)


@pytest.mark.parametrize("failure", ["claim_extraction", "claim_judgments", "query_fit"])
def test_analysis_smoke_rejects_failed_forensics(mocker, backend_client, failure):
    from scripts.smoke_local import run_smoke
    from tests.test_analyze_custom import _STUB_HEDGING, _STUB_QUERY_FIT

    if failure == "query_fit":
        mocker.patch("routers.analyze.analyze_query_corpus_fit", return_value=_STUB_QUERY_FIT.model_copy(
            update={"triggered": True, "status": "error", "error": "fit_computation_failed"},
        ))
    else:
        updates = (
            {"status": "error", "error": "claim_extraction_failed"} if failure == "claim_extraction"
            else {"total_claims": 1, "unavailable_claim_count": 1}
        )
        mocker.patch("routers.analyze.analyze_hedging_mismatch", return_value=_STUB_HEDGING.model_copy(update=updates))
    with pytest.raises(ValueError, match="Forensics unavailable"):
        run_smoke(backend_client, analyze=True)


def test_missing_key_is_actionable_and_never_submits_analysis():
    from scripts.smoke_local import run_smoke

    def respond(request):
        assert request.method == "GET"
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(503, json={
            "status": "not_ready", "checks": {"anthropic_api_key": False, "bundled_corpus": False},
            "bundled_corpus_domains": [],
        })

    with httpx.Client(base_url="http://localhost:8000", transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            run_smoke(client, analyze=True)
