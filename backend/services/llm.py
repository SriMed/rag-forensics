"""Single entry point for Anthropic calls: bounded timeouts, text extraction, one error type."""

import anthropic

_TIMEOUT_SECONDS = 60.0
_MAX_RETRIES = 2


class LLMError(Exception):
    """An LLM request failed or returned no usable text."""


def complete(
    prompt: str,
    *,
    model: str,
    max_tokens: int,
    system: str | None = None,
    **request_options,
) -> str:
    """Send one user prompt and return the first text block of the reply."""
    client = anthropic.Anthropic(timeout=_TIMEOUT_SECONDS, max_retries=_MAX_RETRIES)
    request = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
        **request_options,
    }
    if system is not None:
        request["system"] = system
    try:
        response = client.messages.create(**request)
    except anthropic.APIError as exc:
        raise LLMError(f"Anthropic request failed: {type(exc).__name__}") from exc
    for block in response.content:
        text = getattr(block, "text", None)
        if isinstance(text, str):
            return text
    raise LLMError("Anthropic response contained no text block")


def strip_code_fence(raw: str) -> str:
    """Remove a surrounding markdown code fence (``` or ```json) from a model reply."""
    text = raw.strip()
    if text.startswith("```"):
        text = text.split("\n", 1)[1] if "\n" in text else ""
        text = text.rsplit("```", 1)[0].strip()
    return text
