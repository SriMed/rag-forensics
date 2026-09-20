"""Log-safe description of an exception."""

import os
import traceback

ERROR_DETAILS_ENV = "RAG_FORENSICS_LOG_ERROR_DETAILS"
_TRUTHY = {"1", "true", "yes"}


def failure_detail(exc: BaseException) -> str:
    """Exception type and stack frames; the message only when explicitly opted in.

    Messages and chained causes can embed caller content (validation and provider errors often echo
    their input), so by default neither is included. Setting RAG_FORENSICS_LOG_ERROR_DETAILS=1 adds
    the exception's own message for local debugging; chained causes stay excluded.
    """
    frames = "".join(traceback.format_tb(exc.__traceback__))
    label = type(exc).__name__
    if os.environ.get(ERROR_DETAILS_ENV, "").strip().lower() in _TRUTHY:
        label = f"{label}: {exc}"
    return f"{label}\n{frames}".rstrip("\n")
