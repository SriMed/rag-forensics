from unittest.mock import MagicMock

import anthropic
import httpx
import pytest

from services import llm


def _response(*texts):
    message = MagicMock()
    message.content = [MagicMock(text=t) for t in texts]
    return message


def _patch_client(mocker, response=None, error=None):
    client = MagicMock()
    if error is not None:
        client.messages.create.side_effect = error
    else:
        client.messages.create.return_value = response
    factory = mocker.patch("services.llm.anthropic.Anthropic", return_value=client)
    return factory, client


def test_complete_returns_first_text_block(mocker):
    _patch_client(mocker, _response("hello"))
    assert llm.complete("prompt", model="m", max_tokens=10) == "hello"


def test_complete_forwards_request_parameters(mocker):
    _, client = _patch_client(mocker, _response("ok"))
    llm.complete("prompt", model="model-x", max_tokens=42, system="sys", output_config={"format": {}})
    client.messages.create.assert_called_once_with(
        model="model-x",
        max_tokens=42,
        messages=[{"role": "user", "content": "prompt"}],
        system="sys",
        output_config={"format": {}},
    )


def test_complete_omits_system_when_not_given(mocker):
    _, client = _patch_client(mocker, _response("ok"))
    llm.complete("prompt", model="m", max_tokens=1)
    assert "system" not in client.messages.create.call_args.kwargs


def test_client_has_bounded_timeout_and_retries(mocker):
    factory, _ = _patch_client(mocker, _response("ok"))
    llm.complete("prompt", model="m", max_tokens=1)
    kwargs = factory.call_args.kwargs
    assert 0 < kwargs["timeout"] <= 120
    assert 0 <= kwargs["max_retries"] <= 3


def test_complete_wraps_sdk_errors_in_llm_error(mocker):
    error = anthropic.APIConnectionError(request=httpx.Request("POST", "https://api.anthropic.com"))
    _patch_client(mocker, error=error)
    with pytest.raises(llm.LLMError) as excinfo:
        llm.complete("prompt", model="m", max_tokens=1)
    assert excinfo.value.__cause__ is error


def test_complete_raises_llm_error_without_text_block(mocker):
    message = MagicMock()
    message.content = [MagicMock(spec=["type"])]
    _patch_client(mocker, message)
    with pytest.raises(llm.LLMError):
        llm.complete("prompt", model="m", max_tokens=1)


def test_complete_raises_llm_error_on_empty_content(mocker):
    message = MagicMock()
    message.content = []
    _patch_client(mocker, message)
    with pytest.raises(llm.LLMError):
        llm.complete("prompt", model="m", max_tokens=1)


@pytest.mark.parametrize(
    "raw, expected",
    [
        ('[{"a": 1}]', '[{"a": 1}]'),
        ('  [1, 2]  \n', "[1, 2]"),
        ('```json\n[1, 2]\n```', "[1, 2]"),
        ('```\n{"a": 1}\n```', '{"a": 1}'),
        ('```json\n[1, 2]\n```\n', "[1, 2]"),
    ],
)
def test_strip_code_fence(raw, expected):
    assert llm.strip_code_fence(raw) == expected
