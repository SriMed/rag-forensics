#!/usr/bin/env bash
# Creates isolated virtual environments for RAGChecker and RAGVue.
#
# These are never added to backend/pyproject.toml: RAGChecker's `refchecker` dependency pins
# `anthropic<0.30`, which conflicts with this project's own `anthropic>=0.86,<0.87` pin (see
# ADR-044 and ../README.md). Run this script once from anywhere; the resulting `.venv-*`
# directories are gitignored and can be deleted and recreated at any time.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

python3 -m venv .venv-ragchecker
.venv-ragchecker/bin/pip install --quiet --upgrade pip
.venv-ragchecker/bin/pip install --quiet ragchecker

python3 -m venv .venv-ragvue
.venv-ragvue/bin/pip install --quiet --upgrade pip
# The "anthropic" extra is required to use Claude as the judge backend (RAGVUE_JUDGE_PROVIDER=anthropic);
# without it this project's Anthropic-only convention (no OPENAI_API_KEY) cannot be satisfied.
.venv-ragvue/bin/pip install --quiet "ragvue[anthropic]"
# RAGVue declares only `anthropic>=0.40`, which resolves to whatever is latest — verified live to be
# broken for RAGVue's Anthropic-judge call path (`Messages.create() got an unexpected keyword
# argument 'temperature'` on anthropic 1.6.0). Pinning to this project's own tested version fixes it.
.venv-ragvue/bin/pip install --quiet "anthropic>=0.86,<0.87"

echo "RAGChecker env: $(pwd)/.venv-ragchecker"
echo "RAGVue env:     $(pwd)/.venv-ragvue"
