import logging
import os

from dotenv import load_dotenv

load_dotenv()

# LOG_LEVEL controls verbosity only; unknown values fall back to INFO. Third-party loggers below
# stay at WARNING regardless, so SDK request payloads are never logged.
_LOG_LEVEL = logging.getLevelNamesMapping().get(os.environ.get("LOG_LEVEL", "INFO").strip().upper(), logging.INFO)

logging.basicConfig(
    level=_LOG_LEVEL,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
# Suppress SDK request payloads even when the host configures root logging at DEBUG.
for _noisy in ("anthropic", "httpx", "httpcore", "langchain", "chromadb", "ragas", "sentence_transformers"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from routers import analyze, example, health

app = FastAPI(title="RAG Forensics API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)
app.include_router(example.router)
app.include_router(analyze.router)
