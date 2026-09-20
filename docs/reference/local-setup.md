# Local setup and verification

This is the native setup path for running RAG Forensics beside your own RAG system. The primary
workflow is `POST /analyze/custom`; the browser demo additionally requires a bundled RAGBench
index. Local execution still uses Anthropic for model judgments. Smoke checks establish that the
software runs and returns its contract, not that its diagnostic hypotheses are correct.

## Install and start the backend

Prerequisites: Python 3.13, Poetry 2.2.1, and an Anthropic API key. The frontend uses Node.js 22
and npm. Python 3.13 is the primary version; CI also tests the declared Python 3.11 minimum.
Install these tools before running the
commands below; no global Python application dependencies are needed.

From the repository root:

```bash
cd backend
poetry env use python3.13
poetry install
cp .env.example .env
```

Replace `your_key_here` in `backend/.env` with your key. Keep this ignored file private. Then:

```bash
poetry run uvicorn main:app --host 127.0.0.1 --port 8000
```

Run backend commands from `backend/`: the embedded corpus path is relative to that directory.
Poetry installs the locked dependencies in application mode; there is no installable root Python
package or wheel. Do not regenerate the lock to follow this setup.

The declared Python range remains `>=3.11,<3.15`. Python 3.14 is not the recommended setup:
the locked LangChain/Pydantic stack warns about its Pydantic V1 compatibility layer there.
See [ADR-047](../../ADR.md#adr-047-python-313-as-the-primary-runtime-with-minimum-version-ci).

The first analysis may download the sentence-transformer model and NLTK `punkt_tab` data. Allow
network access to obtain those assets and enough disk space for caches. A populated RAGBench
index is not needed for custom analysis.

## Verify the API

In another terminal, from `backend/`:

```bash
poetry run python -m scripts.smoke_local
```

This checks `/health` and `/ready`, makes no model calls, and prints available corpus domains.
An empty domain list is valid. `/ready` checks whether the key is nonblank, not whether Anthropic
accepts it; the example placeholder is also nonblank. Readiness does not validate local model
downloads, corpus contents, or a complete analysis.

For an explicit live check:

```bash
poetry run python -m scripts.smoke_local --analyze
```

This sends [`backend/examples/custom-analysis.json`](../../backend/examples/custom-analysis.json)
to `/analyze/custom`, validates the response schema, and rejects unavailable RAGAS scores and
failed forensics. It uses
the current API contract and exits nonzero on failure. This is a paid model call path; allow time
for downloads and several model requests. The client's 600-second HTTP timeout does not cancel
backend work or impose a total analysis deadline. `--base-url` overrides the default
`http://127.0.0.1:8000`.

The question, answer, and retrieved passages can reach Anthropic through the project prompts and
the RAGAS `ChatAnthropic` adapter. Full custom analysis has no offline-only switch. Use the default
health/readiness check or the mocked test suite when you need verification without model calls.
See [API integration](api-integration.md) to map your own RAG trace to the request.

## Optional corpus and frontend demo

Stop the backend before bootstrap. From `backend/`, explicitly run:

```bash
poetry run python scripts/bootstrap_data.py
```

Bootstrap downloads the TechQA, FinQA, and CovidQA training splits and embeds their document
chunks. It needs no Anthropic key, but can take substantial time, memory, and disk space.
Each domain is built in a staging collection before replacing the existing domain. Repeat runs
replace the domain rather than append duplicates. Fallback example IDs use a stable SHA-256 of
the question, namespaced by domain; IDs supplied by the dataset are retained.

Embedding, insertion, and empty-input failures preserve the previous collection. Promotion keeps
a backup until the replacement has its public name, and attempts rollback on a rename failure.
Domains are replaced independently: a failure in a later domain does not undo completed domains.
Do not run concurrent bootstraps or serve requests during replacement. Chroma collection renames
are not an atomic transaction: forced termination or a rollback failure can leave `*-staging-*`
or `*-backup-*` collections. Preserve those collections for recovery; the backup can be renamed
to the missing domain through Chroma's collection API with the backend stopped. Do not delete the
whole data directory as routine recovery.

Restart the backend with the same command and working directory. From the repository root in a
separate terminal:

```bash
cd frontend
npm ci
cp .env.local.example .env.local
npm run dev -- --hostname 127.0.0.1
```

Open `http://localhost:3000`. That browser origin is explicitly allowed by backend CORS; visiting
`http://127.0.0.1:3000` is a different origin. `NEXT_PUBLIC_API_URL` defaults to
`http://localhost:8000`; set it to `http://127.0.0.1:8000` if localhost resolves to IPv6 on your
machine. Restart the frontend after changing its environment. Load a domain example and run
analysis. The frontend uses `POST /example` followed by `POST /analyze`.

## Persistence and configuration

| Setting or location | Purpose |
|---|---|
| `backend/.env`: `ANTHROPIC_API_KEY` | Required for analysis; not needed for bootstrap, liveness, or mocked tests |
| `backend/.env`: `LOG_LEVEL` | Optional log verbosity (`DEBUG`, `INFO`, `WARNING`, ...); defaults to `INFO`, unknown values fall back to `INFO` |
| `backend/.env`: `RAG_FORENSICS_LOG_ERROR_DETAILS` | Optional; `1` adds exception messages to failure logs for local debugging. Messages can contain your question, answer, or chunk text, so leave it off when logs are shared |
| `frontend/.env.local`: `NEXT_PUBLIC_API_URL` | Public backend URL; never put a provider key here |
| `backend/data/chroma/` | Embedded Chroma index; ignored by Git and reused across restarts |
| Hugging Face cache (normally `~/.cache/huggingface/`) | Downloaded dataset/model files; `HF_HOME` can relocate the cache |
| `~/.cache/chroma/onnx_models/` | Chroma's default query-embedding model for the bundled demo |
| NLTK data (normally `~/nltk_data/`) | Sentence-tokenizer data; `NLTK_DATA` adds a lookup location |

Ordinary restarts do not delete these files or require bootstrap. Changing the working directory,
cache environment, or user account can make existing data appear absent. Copy the Chroma directory
only with the backend and bootstrap stopped.

Direct project model calls and the RAGAS adapter use a 60-second request timeout and two SDK
retries. RAGAS also has a 60-second evaluation-task timeout and one configured attempt at its
outer retry layer; dependency-owned parsing may make further calls. These are not an end-to-end
analysis deadline. See [the RAGAS contract](ragas-prompt-audit.md) and
[ADR-046](../../ADR.md#adr-046-application-setup-separate-ragas-boundary-and-staged-corpus-replacement).

## Checks and troubleshooting

```bash
# From backend/
poetry check
poetry run ruff check .
poetry run pytest -q

# From frontend/ in another terminal
npm run lint
npm run typecheck
npm test -- --runInBand
npm run build
```

- Connection refused: start the backend on the URL used by the smoke command.
- Missing-key readiness failure: edit `backend/.env` and restart. An invalid nonblank key needs
  a live analysis to detect; inspect backend logs if evaluation is unavailable.
- Model/tokenizer download failure: check network access and cache permissions. `/ready` does
  not test those dependencies.
- Empty corpus domain list: custom analysis can still run; explicitly bootstrap for the demo.
- Generic HTTP 500: inspect backend logs for the failure type and stack location; API responses
  intentionally omit details, and logs omit exception messages by default. To see the message while
  debugging locally, set `RAG_FORENSICS_LOG_ERROR_DETAILS=1` and restart (see the caution above).

The legacy notebooks in `smoke_tests/` are historical records with obsolete response fields, not
the supported smoke path. The maintained command above and its mocked tests replace that role.

Application dependencies are locked, but the bundled dataset and sentence-transformer downloads
still use unpinned upstream revisions. The locked dependencies and offline test suite have been
verified in a fresh Python 3.13.11 virtual environment on macOS. This did not establish clean-machine
model downloads or live-provider behavior. A clean-machine live run and complete restart/recovery
verification remain release checks under issue #12; passing mocked tests does not establish them.
