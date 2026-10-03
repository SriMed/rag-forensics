# Joint evidence verification (issue #32)

Tests whether giving the claim verifier evidence sentences together, rather than one at a time, reduces false "unsupported" judgments, and whether any gain survives similarity-based evidence selection. The design, controls, and interpretation rules are frozen in [`v1-techqa/protocol.md`](v1-techqa/protocol.md); ADR-055 records why the primary condition adds neighboring sentences to the annotated evidence.

## Public synthetic fixture

[`fixture/fixture.json`](fixture/fixture.json) holds invented, TechQA-shaped inputs with a claim review that accepts the deterministic claims, so it contains no human judgment. It exercises every condition and contrast with the pinned embedding model and verifier:

```bash
cd backend
poetry run python -m benchmark.joint_evidence_cli run-fixture \
  --fixture evals/joint_evidence/fixture/fixture.json \
  --output /tmp/joint-evidence-fixture.json
```

Fixture outputs demonstrate the pipeline only; they are not evidence about the verifier.

## TechQA pilot

The pilot reads the private frozen claim and residual reviews from #29. The runner refuses any file whose sha256 differs from the protocol, and any population other than 39 supported and 330 unsupported sentences:

```bash
cd backend
poetry run python -m benchmark.joint_evidence_cli run \
  --claims evals/decomposition_evidence/v1-techqa/claim-review.json \
  --residual-review evals/decomposition_evidence/v1-techqa/residual-review.json \
  --output evals/joint_evidence/v1-techqa/results.json
```

`v1-techqa/results.json` contains the private reviewed claims and is kept out of git; only aggregate results are published. Supported-set and subgroup results cannot be reproduced without the private reviews. Unsupported-set results depend only on public data and the deterministic decomposer.
