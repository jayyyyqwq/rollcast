#!/usr/bin/env bash
# One command to check and run rollcast.
#   ./run.sh                 # checks, then the default demo
#   SKIP_CHECKS=1 ./run.sh   # demo only
set -euo pipefail
cd "$(dirname "$0")"

if [ "${SKIP_CHECKS:-0}" != "1" ]; then
  echo "== checks"
  uv run --locked ruff check .
  uv run --locked pytest -q
fi

echo "== run"
uv run --locked rollcast demo
