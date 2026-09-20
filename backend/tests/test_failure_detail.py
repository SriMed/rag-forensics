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


def _failure():
    try:
        _raises_with_cause()
    except RuntimeError as exc:
        return exc


def test_error_details_opt_in_adds_the_exception_message(monkeypatch):
    monkeypatch.setenv("RAG_FORENSICS_LOG_ERROR_DETAILS", "1")
    detail = failure_detail(_failure())
    assert detail.startswith(f"RuntimeError: {CALLER}")
    assert "_raises_with_cause" in detail


def test_error_details_opt_in_accepts_common_truthy_spellings(monkeypatch):
    for value in ("true", "TRUE", "yes"):
        monkeypatch.setenv("RAG_FORENSICS_LOG_ERROR_DETAILS", value)
        assert CALLER in failure_detail(_failure())


def test_error_details_stay_off_for_falsey_values(monkeypatch):
    for value in ("", "0", "false", "no", "maybe"):
        monkeypatch.setenv("RAG_FORENSICS_LOG_ERROR_DETAILS", value)
        assert CALLER not in failure_detail(_failure())
