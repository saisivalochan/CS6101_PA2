#!/usr/bin/env bash
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
mkdir -p results
date -u > results/cloud_start.txt
trap 'date -u > results/cloud_finished.txt; sync' EXIT
bash cloud/bootstrap.sh
.venv/bin/python -m pa2.cli run --config configs/main.json --max-hours 24
.venv/bin/python -m pa2.cli report --config configs/main.json --draft
echo 'Required experiments finished. Retrieve artifacts, supply submission details, and generate the final report.'
