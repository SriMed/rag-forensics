import logging
import os

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from services.retriever import available_domains

logger = logging.getLogger(__name__)
router = APIRouter()


@router.get("/health")
def health() -> dict[str, str]:
    """Liveness: the process is up. Touches no data and calls no models."""
    return {"status": "ok"}


@router.get("/ready")
def ready() -> JSONResponse:
    """Readiness for /analyze/custom: the LLM key is configured. Reports the bundled corpus without requiring it."""
    api_key_configured = bool(os.environ.get("ANTHROPIC_API_KEY", "").strip())
    try:
        domains = available_domains()
    except (OSError, RuntimeError, ValueError):
        logger.warning("bundled corpus probe failed", exc_info=True)
        domains = []
    body = {
        "status": "ready" if api_key_configured else "not_ready",
        "checks": {"anthropic_api_key": api_key_configured, "bundled_corpus": bool(domains)},
        "bundled_corpus_domains": domains,
    }
    return JSONResponse(body, status_code=200 if api_key_configured else 503)
