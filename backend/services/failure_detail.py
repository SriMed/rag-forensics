"""Log-safe description of an exception."""

import traceback


def failure_detail(exc: BaseException) -> str:
    """Exception type and stack frames only.

    The message and any chained cause can embed caller content (validation errors and provider
    errors often echo their input), so neither is included.
    """
    frames = "".join(traceback.format_tb(exc.__traceback__))
    return f"{type(exc).__name__}\n{frames}".rstrip("\n")
