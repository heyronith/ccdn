#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
if [ -x .venv/bin/python ]; then
  exec .venv/bin/python -m scripts.run_cycle2c "$@"
fi
exec python3 -m scripts.run_cycle2c "$@"
