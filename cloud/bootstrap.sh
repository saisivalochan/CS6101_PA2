#!/usr/bin/env bash
# Run inside the Ubuntu VM after uploading this project.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if command -v sudo >/dev/null; then
  sudo apt-get update
  sudo apt-get install -y python3-venv python3-dev g++ ca-certificates curl
else
  apt-get update
  apt-get install -y python3-venv python3-dev g++ ca-certificates curl
fi
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
mkdir -p results
.venv/bin/python -m pip freeze > results/environment.lock.txt
.venv/bin/python scripts/build_core.py
.venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m pa2.cli core-smoke
