"""Application startup logging, isolated from pytest's own logging configuration."""
import os
import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("root_logging", ["default", "debug"])
def test_app_logs_operations_without_sdk_request_content(root_logging):
    script = r'''
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
    result = subprocess.run(
        [sys.executable, "-c", script, root_logging],
        cwd=Path(__file__).resolve().parents[1],
        env={
            **os.environ,
            "PYTHON_DOTENV_DISABLED": "1",
            "ANTHROPIC_API_KEY": "synthetic-provider-key",
            "ANTHROPIC_LOG": "",
            "HF_HUB_OFFLINE": "1",
            "RAGAS_DO_NOT_TRACK": "true",
            "ANONYMIZED_TELEMETRY": "False",
        },
        capture_output=True, text=True, timeout=60,
    )
    assert result.returncode == 0, result.stderr
    output = result.stdout + result.stderr
    assert "synthetic-operation-completed" in output
    assert "synthetic-provider-warning" in output
    assert "synthetic-caller-content" not in output
    assert "synthetic-provider-key" not in output
    if root_logging == "default":
        assert "synthetic-debug-detail" not in output
