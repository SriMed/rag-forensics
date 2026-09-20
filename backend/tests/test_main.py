"""Application startup logging, isolated from pytest's own logging configuration."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

_SCRIPT = r'''
import logging
import sys

if sys.argv[1] == "debug":
    logging.basicConfig(level=logging.DEBUG)

import main
import anthropic
import httpx

def respond(request):
    assert b"synthetic-caller-content" in request.content
    return httpx.Response(200, json={
        "id": "msg_test", "type": "message", "role": "assistant", "model": "test-model",
        "content": [{"type": "text", "text": "ok"}],
        "stop_reason": "end_turn", "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    })

with anthropic.Anthropic(
    api_key="synthetic-provider-key",
    http_client=httpx.Client(transport=httpx.MockTransport(respond)),
) as client:
    response = client.messages.create(
        model="test-model", max_tokens=1,
        messages=[{"role": "user", "content": "synthetic-caller-content"}],
    )
    assert response.content[0].text == "ok"

logging.getLogger("rag_forensics").info("synthetic-operation-completed")
logging.getLogger("rag_forensics").debug("synthetic-debug-detail")
logging.getLogger("anthropic").warning("synthetic-provider-warning")
'''


def _run(root_logging: str, **extra_env: str) -> str:
    env = {k: v for k, v in os.environ.items() if k not in {"LOG_LEVEL", "RAG_FORENSICS_LOG_ERROR_DETAILS"}}
    result = subprocess.run(
        [sys.executable, "-c", _SCRIPT, root_logging],
        cwd=Path(__file__).resolve().parents[1],
        env={
            **env,
            "PYTHON_DOTENV_DISABLED": "1",
            "ANTHROPIC_API_KEY": "synthetic-provider-key",
            "ANTHROPIC_LOG": "",
            "HF_HUB_OFFLINE": "1",
            "RAGAS_DO_NOT_TRACK": "true",
            "ANONYMIZED_TELEMETRY": "False",
            **extra_env,
        },
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout + result.stderr


@pytest.mark.parametrize("root_logging", ["default", "debug"])
def test_app_logs_operations_without_sdk_request_content(root_logging):
    output = _run(root_logging)
    assert "synthetic-operation-completed" in output
    assert "synthetic-provider-warning" in output
    assert "synthetic-caller-content" not in output
    assert "synthetic-provider-key" not in output
    if root_logging == "default":
        assert "synthetic-debug-detail" not in output


def test_log_level_env_enables_app_debug_but_never_sdk_payloads():
    output = _run("default", LOG_LEVEL="DEBUG")
    assert "synthetic-debug-detail" in output
    assert "synthetic-caller-content" not in output
    assert "synthetic-provider-key" not in output


def test_log_level_env_is_case_insensitive_and_can_quiet_info():
    output = _run("default", LOG_LEVEL="warning")
    assert "synthetic-operation-completed" not in output
    assert "synthetic-provider-warning" in output


def test_invalid_log_level_falls_back_to_info():
    output = _run("default", LOG_LEVEL="not-a-level")
    assert "synthetic-operation-completed" in output
    assert "synthetic-debug-detail" not in output
