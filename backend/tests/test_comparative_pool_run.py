"""Tests for the issue #30 full-candidate-pool runner (no live calls; every system is mocked)."""

import json
from types import SimpleNamespace

import benchmark.comparative_diagnostics as cd
from evals.comparative_diagnostics.v1 import run_pool


def _record(case_id: str) -> SimpleNamespace:
    return SimpleNamespace(example_id=case_id)


def _healthy(system: cd.SystemName, case_id: str) -> cd.SystemDiagnosticRecord:
    return cd.SystemDiagnosticRecord(
        system=system,
        native=cd.NativeSystemOutput(
            system=system, system_version="test", availability="healthy",
            raw_output={"case": case_id}, error=None,
        ),
        suspected_component=None, supporting_observation=None, evidence_attribution=[],
        method=None, reliability=None, causal_strength_language=None,
        proposed_intervention=None, no_equivalent_fields=[],
    )


def _fake_runner(system: cd.SystemName, calls: list, fail_on: str | None = None):
    def run(records):
        ids = [r.example_id for r in records]
        calls.append((system, ids))
        if fail_on in ids:
            raise RuntimeError("subprocess exited 1")
        return {i: _healthy(system, i) for i in ids}

    return run


def _runners(calls, ragchecker_fail_on=None) -> dict[cd.SystemName, run_pool.Runner]:
    return {
        "rag_forensics": _fake_runner("rag_forensics", calls),
        "ragchecker": _fake_runner("ragchecker", calls, fail_on=ragchecker_fail_on),
        "ragvue": _fake_runner("ragvue", calls),
    }


def test_runs_every_case_through_every_system_in_batches(tmp_path):
    calls: list = []
    out = tmp_path / "pool-run.json"
    records = [_record(f"c{i}") for i in range(5)]

    run_pool.run_pool(records, out, runners=_runners(calls), batch_size=2, metadata={"k": "v"})

    doc = json.loads(out.read_text())
    assert set(doc["cases"]) == {f"c{i}" for i in range(5)}
    for case in doc["cases"].values():
        assert set(case) == {"rag_forensics", "ragchecker", "ragvue"}
    assert [ids for system, ids in calls if system == "ragchecker"] == [["c0", "c1"], ["c2", "c3"], ["c4"]]
    assert doc["metadata"] == {"k": "v"}


def test_resume_skips_cases_already_recorded(tmp_path):
    out = tmp_path / "pool-run.json"
    records = [_record(f"c{i}") for i in range(4)]
    run_pool.run_pool(records[:2], out, runners=_runners([]), batch_size=2, metadata={})

    calls: list = []
    run_pool.run_pool(records, out, runners=_runners(calls), batch_size=2, metadata={})

    assert all(ids == ["c2", "c3"] for _system, ids in calls)
    assert set(json.loads(out.read_text())["cases"]) == {"c0", "c1", "c2", "c3"}


def test_batch_failure_is_recorded_as_failed_not_raised(tmp_path):
    out = tmp_path / "pool-run.json"
    records = [_record(f"c{i}") for i in range(4)]

    run_pool.run_pool(
        records, out, runners=_runners([], ragchecker_fail_on="c3"), batch_size=2, metadata={},
    )

    cases = json.loads(out.read_text())["cases"]
    assert cases["c0"]["ragchecker"]["native"]["availability"] == "healthy"
    for case_id in ("c2", "c3"):
        native = cases[case_id]["ragchecker"]["native"]
        assert native["availability"] == "failed"
        assert "subprocess exited 1" in native["error"]
        assert cases[case_id]["ragvue"]["native"]["availability"] == "healthy"


def test_failed_cases_are_retried_on_resume(tmp_path):
    out = tmp_path / "pool-run.json"
    records = [_record("c0"), _record("c1")]
    run_pool.run_pool(records, out, runners=_runners([], ragchecker_fail_on="c1"), batch_size=2, metadata={})

    calls: list = []
    run_pool.run_pool(records, out, runners=_runners(calls), batch_size=2, metadata={})

    assert ("ragchecker", ["c0", "c1"]) in calls
    assert ("ragvue", ["c0", "c1"]) not in calls
    cases = json.loads(out.read_text())["cases"]
    assert cases["c1"]["ragchecker"]["native"]["availability"] == "healthy"


def test_seed_from_copies_only_the_named_systems(tmp_path):
    source = tmp_path / "v1.json"
    run_pool.run_pool([_record("c0"), _record("c1")], source, runners=_runners([]), batch_size=2, metadata={"v": 1})
    out = tmp_path / "v2.json"

    run_pool.seed_from(out, source, systems=["ragchecker", "ragvue"])

    cases = json.loads(out.read_text())["cases"]
    assert set(cases) == {"c0", "c1"}
    assert all(set(c) == {"ragchecker", "ragvue"} for c in cases.values())


def test_seeded_systems_are_not_rerun(tmp_path):
    source = tmp_path / "v1.json"
    records = [_record("c0"), _record("c1")]
    run_pool.run_pool(records, source, runners=_runners([]), batch_size=2, metadata={})
    out = tmp_path / "v2.json"
    run_pool.seed_from(out, source, systems=["ragchecker", "ragvue"])

    calls: list = []
    run_pool.run_pool(records, out, runners=_runners(calls), batch_size=2, metadata={})

    assert {system for system, _ in calls} == {"rag_forensics"}


def test_unscored_rag_forensics_requests_declare_scores_unavailable(mocker):
    from models import RetrievedChunk

    record = SimpleNamespace(
        example_id="c0", question="Q?", response="A.",
        chunks=[RetrievedChunk(chunk_id="d0", text="t", score=1.0)],
    )
    analyze = mocker.patch("routers.analyze.analyze_custom")
    analyze.return_value.verdict_signals = []
    analyze.return_value.model_dump.return_value = {}

    run_pool.run_rag_forensics_unscored([record])

    request = analyze.call_args.args[0]
    assert request.score_semantics == "unavailable"
    assert [c.score for c in request.chunks] == [None]
