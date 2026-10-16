#!/usr/bin/env bash
# Retrieve from Cloud Shell; does not delete the VM or its disk.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
PA2_PROJECT=${1:?Usage: bash cloud/fetch.sh PROJECT_ID}
[[ "$PA2_PROJECT" =~ ^[a-z][a-z0-9-]{4,61}[a-z0-9]$ ]] || exit 2
gcloud compute ssh cs6101-pa2 --project="$PA2_PROJECT" --zone=us-central1-a \
  --command='cd cs6101-pa2 && if test -d submission; then sudo tar -czf /tmp/pa2-results.tar.gz results submission; else sudo tar -czf /tmp/pa2-results.tar.gz results; fi && sudo chmod 644 /tmp/pa2-results.tar.gz'
mkdir -p cloud-download
gcloud compute scp cs6101-pa2:/tmp/pa2-results.tar.gz cloud-download/pa2-results.tar.gz \
  --project="$PA2_PROJECT" --zone=us-central1-a
tar -tzf cloud-download/pa2-results.tar.gz > cloud-download/archive_contents.txt
echo 'Saved cloud-download/pa2-results.tar.gz. Download it from Cloud Shell and verify the results locally.'
echo 'For resumability, leave data/indices on the VM until final reporting and audit complete.'
