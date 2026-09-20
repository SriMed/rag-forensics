from services.failure_detail import failure_detail

# Built at runtime so the marker never appears verbatim in this file's source lines,
# which tracebacks print.
CALLER = "synthetic-" + "caller-content"
CAUSE = "synthetic-" + "cause-content"


def _raises_with_cause():
    try:
        raise ValueError(CAUSE)
    except ValueError as cause:
        raise RuntimeError(CALLER) from cause


def test_failure_detail_has_type_and_frames_but_no_messages():
    try:
        _raises_with_cause()
    except RuntimeError as exc:
        detail = failure_detail(exc)
    assert detail.startswith("RuntimeError")
    assert "_raises_with_cause" in detail
    assert CALLER not in detail
    assert CAUSE not in detail


def test_failure_detail_without_traceback_is_just_the_type():
    assert failure_detail(KeyError(CALLER)) == "KeyError"
