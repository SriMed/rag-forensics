import logging
import anthropic
from models import RetrievedChunk
from prompts.generation_prompts import GENERATION_SYSTEM_PROMPT, build_generation_prompt

logger = logging.getLogger(__name__)
_LLM_MODEL = "claude-haiku-4-5-20251001"


def generate_answer(question: str, chunks: list[RetrievedChunk]) -> str:
    logger.debug("generating answer for question=%r using %d chunks", question[:80], len(chunks))
    client = anthropic.Anthropic()
    prompt = build_generation_prompt(question, chunks)
    response = client.messages.create(
        model=_LLM_MODEL,
        max_tokens=1024,
        system=GENERATION_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    answer = response.content[0].text
    logger.debug("answer generated: %d chars, usage=%s", len(answer), response.usage)
    return answer
