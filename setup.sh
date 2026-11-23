#!/usr/bin/env bash
# Install everything rollcast needs and make sure it builds.
set -euo pipefail
cd "$(dirname "$0")"

command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }
echo "installing dependencies"
uv sync --locked

echo "setup done. next: ./run.sh"
