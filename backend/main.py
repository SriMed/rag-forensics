import logging
from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s %(levelname)s %(name)s — %(message)s",
)
# Keep noisy third-party loggers at WARNING
for _noisy in ("httpx", "httpcore", "langchain", "chromadb", "ragas", "sentence_transformers"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from routers import example, analyze

app = FastAPI(title="RAG Forensics API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(example.router)
app.include_router(analyze.router)
