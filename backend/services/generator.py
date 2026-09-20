import logging

from config import CLAUDE_HAIKU
from models import RetrievedChunk
from prompts.generation_prompts import GENERATION_SYSTEM_PROMPT, build_generation_prompt
from services.llm import complete

logger = logging.getLogger(__name__)


def generate_answer(question: str, chunks: list[RetrievedChunk]) -> str:
    logger.debug("generating answer for question=%r using %d chunks", question[:80], len(chunks))
    answer = complete(
        build_generation_prompt(question, chunks),
        model=CLAUDE_HAIKU,
        max_tokens=1024,
        system=GENERATION_SYSTEM_PROMPT,
    )
    logger.debug("answer generated: %d chars", len(answer))
    return answer
