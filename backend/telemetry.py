"""Turn off dependency usage telemetry unless the operator explicitly opts in.

Call before importing chromadb, ragas, sentence_transformers, or datasets:
huggingface_hub reads its switch at import time.
"""
import os

_OFF = {
    "ANONYMIZED_TELEMETRY": "False",  # Chroma
    "RAGAS_DO_NOT_TRACK": "true",  # RAGAS
    "HF_HUB_DISABLE_TELEMETRY": "1",  # Hugging Face Hub
}


def disable_dependency_telemetry() -> None:
    for name, value in _OFF.items():
        os.environ.setdefault(name, value)
