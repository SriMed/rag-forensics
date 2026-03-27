"""Tests for services/generator.py — mocks Anthropic SDK entirely."""
import pytest
from unittest.mock import MagicMock
from models import RetrievedChunk

CHUNKS = [
    RetrievedChunk(chunk_id="c1", text="The sky is blue due to Rayleigh scattering.", score=0.9),
    RetrievedChunk(chunk_id="c2", text="Sunlight is composed of multiple wavelengths.", score=0.8),
]


def _make_anthropic_mock(mocker, text: str = "Generated answer."):
    mock_message = MagicMock()
    mock_message.content = [MagicMock(text=text)]
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_message
    mocker.patch("services.generator.anthropic.Anthropic", return_value=mock_client)
    return mock_client


def test_generate_answer_returns_non_empty_string(mocker):
    _make_anthropic_mock(mocker, text="The sky appears blue.")
    from services.generator import generate_answer
    result = generate_answer("Why is the sky blue?", CHUNKS)
    assert isinstance(result, str)
    assert len(result) > 0


def test_generate_answer_returns_text_from_api(mocker):
    expected = "Because of light scattering."
    _make_anthropic_mock(mocker, text=expected)
    from services.generator import generate_answer
    result = generate_answer("Why is the sky blue?", CHUNKS)
    assert result == expected


def test_generate_answer_calls_api_once(mocker):
    mock_client = _make_anthropic_mock(mocker)
    from services.generator import generate_answer
    generate_answer("Why is the sky blue?", CHUNKS)
    mock_client.messages.create.assert_called_once()


def test_generation_prompt_constant_is_non_empty():
    from prompts.generation_prompts import GENERATION_SYSTEM_PROMPT
    assert isinstance(GENERATION_SYSTEM_PROMPT, str)
    assert len(GENERATION_SYSTEM_PROMPT) > 0


def test_build_generation_prompt_includes_question():
    from prompts.generation_prompts import build_generation_prompt
    prompt = build_generation_prompt("Why is the sky blue?", CHUNKS)
    assert "Why is the sky blue?" in prompt


def test_build_generation_prompt_includes_chunk_text():
    from prompts.generation_prompts import build_generation_prompt
    prompt = build_generation_prompt("Why is the sky blue?", CHUNKS)
    assert "Rayleigh scattering" in prompt
    assert "multiple wavelengths" in prompt


def test_generate_answer_propagates_api_exception(mocker):
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = Exception("API error")
    mocker.patch("services.generator.anthropic.Anthropic", return_value=mock_client)
    from services.generator import generate_answer
    with pytest.raises(Exception, match="API error"):
        generate_answer("Why is the sky blue?", CHUNKS)
