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
