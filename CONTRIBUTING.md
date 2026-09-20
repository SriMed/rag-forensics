# Contributing to RAG Forensics

Start with the [local setup guide](docs/reference/local-setup.md). Use the committed Poetry and npm
lock files. Keep changes focused on one problem and explain the resulting behavior and evidence
in the pull request.

Python 3.13 is the primary backend version. CI also runs Python 3.11 to protect the declared
minimum; keep Python syntax and APIs compatible with that minimum. Ruff and mypy therefore retain
their Python 3.11 targets.

## Tests and checks

For behavior changes, add a test at the relevant public interface, run it to confirm it fails for
the intended reason, then make the smallest implementation change that passes. Use existing test
files when they cover that interface. Pure refactors should preserve behavior under the existing
tests. Do not commit failing tests. If three focused attempts do not resolve a failure, report the
failure and what you tried before continuing.

Keep automated tests offline: mock model-provider calls, dataset downloads, and embedding-model
downloads. Temporary local database tests may use isolated storage; never modify a developer's
corpus. Live verification is an explicit manual step, not part of the unit suite.

From `backend/`:

```bash
poetry check
poetry run ruff check .
poetry run pytest -q
poetry run mypy .
```

Mypy currently has known errors and is informational in CI. Avoid adding new errors. Run focused
tests while developing, then the relevant suite before handing off the change.

From `frontend/`:

```bash
npm run lint
npm run typecheck
npm test -- --runInBand
npm run build
```

Keep frontend API types aligned with `backend/models.py`. Preserve unavailable/error states rather
than converting failed evaluations to zero-valued scores.

## Research artifacts and documentation

Treat frozen evaluation inputs, human-review records, protocols, and committed results as
historical evidence. Do not overwrite them to make a new implementation look consistent with an
old result. Read the relevant evaluation README before changing research code; use a new version
or separately identified run for changed methods and record model/data revisions and commands.
Held-out results must not guide candidate tuning.

Use [CONTEXT.md](CONTEXT.md) for domain terminology and the [documentation guide](docs/README.md)
to find the canonical page for a change. Update current contracts and examples when behavior
changes. Architectural decisions belong in new entries in [ADR.md](ADR.md); preserve earlier
entries and explicitly supersede them when necessary.

Do not commit API keys, environment files, generated indexes, model caches, or private review
artifacts. Do not log caller prompts, answers, retrieved passages, or credentials in routine logs.
Bootstrap and live smoke checks require an explicit decision to run because they download data
or make paid model calls. Document which checks you actually ran and any remaining limitations.
