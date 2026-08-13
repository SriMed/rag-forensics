# Source-metadata generation contract v2

This version evaluates issue #27's source-aware completeness contract against the exact production
model. It reuses the frozen v1 case inputs and compares the pre-#27 prompt with the implemented
contract. [`production-results.json`](production-results.json) preserves all 24 raw outputs from
the exact `claude-haiku-4-5-20251001` model: three pairs, complete and truncated variants, two
conditions, and two repetitions.

## Reviewed result

| Variant | Pre-#27 prompt | Source-metadata contract |
| --- | --- | --- |
| Complete evidence | 6/6 useful | 6/6 useful |
| Truncated evidence disclosed | 4/6 | 6/6 |
| Held-back phrase strictly avoided | 4/6 | 4/6 |

The contract improved disclosure without reducing reviewed complete-evidence usefulness. It did
not provide lexical enforcement: in both CovidQA contract runs the model changed the visible
fragment `may be a risk` into `may be a risk factor`. It did not invent the held-back object
`for HAdV-55 infection in young adults`, and both responses explicitly stated that the object was
unavailable because the sentence was truncated.

The FinQA condition abstained appropriately under both prompts. The TechQA baseline copied the
visible fragment without disclosing incompleteness; the contract disclosed truncation in both
runs. These are reviewed case-level observations, not reliability estimates.

## Response-level decision

No deterministic response rejection is implemented. Without the hidden source continuation, the
runtime cannot distinguish a plausible completion from a supported paraphrase or safely identify
the missing semantic object. The prompt contract and response metadata make the limitation
inspectable; systems requiring strict non-completion need a source-aware comparison boundary or a
separately evaluated verifier.

Run from `backend/`:

```bash
poetry run python -m evals.truncated_evidence.run_production_comparison \
  --repetitions 2 --output evals/truncated_evidence/v2/production-results.json
```

An unavailable CLI/model call is preserved with `status="unavailable"`; it is not scored as a
failure or converted into an empty answer.
