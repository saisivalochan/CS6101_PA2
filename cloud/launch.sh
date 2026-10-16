#!/usr/bin/env bash
# Run from an authenticated Google Cloud Shell after uploading this source tree.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [[ $# -ne 2 ]]; then
  echo 'Usage: bash cloud/launch.sh PROJECT_ID VERIFIED_COMPUTE_HOURLY_USD' >&2
  echo 'Verify the on-demand n2-standard-16 us-central1 price before running.' >&2
  exit 2
fi
PA2_PROJECT=$1
PA2_RATE=$2
[[ "$PA2_PROJECT" =~ ^[a-z][a-z0-9-]{4,61}[a-z0-9]$ ]] || { echo 'Invalid project ID' >&2; exit 2; }
command -v gcloud >/dev/null || { echo 'Use authenticated Google Cloud Shell (gcloud is required)' >&2; exit 2; }
gcloud projects describe "$PA2_PROJECT" --format='value(projectId)'
PA2_EXISTING=$(gcloud compute instances list --project="$PA2_PROJECT" --filter='labels.assignment=cs6101-pa2' --format='value(name)')
[[ -z "$PA2_EXISTING" ]] || { echo "Existing assignment VM found: $PA2_EXISTING. Resume or retrieve it; do not create another." >&2; exit 2; }
if [[ -f cloud/runtime.json ]]; then
  echo 'A prior launch receipt exists. Retrieve/clean up that run and reconcile spend before another launch.' >&2
  exit 2
fi
python3 - "$PA2_PROJECT" "$PA2_RATE" <<'PY'
import datetime,json,pathlib,sys,time
project,rate=sys.argv[1],float(sys.argv[2])
if not 0<rate<=35:raise SystemExit('Hourly rate must be positive and <=US$35')
hours=min(24,35/rate)
seconds=int(hours*3600)
deadline=int(time.time())+seconds
value={'project':project,'zone':'us-central1-a','instance':'cs6101-pa2','machine_type':'n2-standard-16',
       'verified_compute_hourly_usd':rate,'compute_allowance_usd':35,'other_cost_reserve_usd':15,
       'budget_ceiling_usd':50,'created_epoch':int(time.time()),'termination_epoch':deadline,
       'termination_time':datetime.datetime.fromtimestamp(deadline,datetime.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
       'note':'User-verified on-demand rate; storage/network/taxes are additional. Alerts are not a hard spend cap.'}
pathlib.Path('cloud/runtime.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(value,indent=2))
PY
PA2_DEADLINE=$(python3 -c "import json; print(json.load(open('cloud/runtime.json'))['termination_time'])")
gcloud compute instances create cs6101-pa2 --project="$PA2_PROJECT" --zone=us-central1-a \
  --machine-type=n2-standard-16 --image-family=ubuntu-2404-lts-amd64 --image-project=ubuntu-os-cloud \
  --boot-disk-size=150GB --boot-disk-type=pd-balanced --labels=assignment=cs6101-pa2 \
  --metadata=enable-oslogin=TRUE --no-service-account --no-scopes --no-restart-on-failure \
  --termination-time="$PA2_DEADLINE" --instance-termination-action=STOP
PA2_ARCHIVE=$(mktemp /tmp/cs6101-pa2-source.XXXXXX.tar.gz)
tar --exclude=.git --exclude=.agents --exclude=.codex --exclude=.aws --exclude=.venv \
  --exclude=__pycache__ --exclude=build --exclude=data --exclude=indices --exclude=results \
  --exclude=submission --exclude=submission_details.json --exclude=llm_transcripts \
  --exclude='*.tar.gz' --exclude='*.zip' --exclude=cloud-download \
  -czf "$PA2_ARCHIVE" .
gcloud compute scp "$PA2_ARCHIVE" cs6101-pa2:pa2-source.tar.gz --project="$PA2_PROJECT" --zone=us-central1-a
rm -- "$PA2_ARCHIVE"
gcloud compute ssh cs6101-pa2 --project="$PA2_PROJECT" --zone=us-central1-a \
  --command='mkdir -p cs6101-pa2 && tar -xzf pa2-source.tar.gz -C cs6101-pa2 && cd cs6101-pa2 && sudo systemd-run --unit=cs6101-pa2 --working-directory="$PWD" /bin/bash cloud/run_vm.sh'
echo 'Monitor: gcloud compute ssh cs6101-pa2 --project=YOUR_PROJECT --zone=us-central1-a --command="sudo journalctl -u cs6101-pa2 -f"'
echo 'After completion run cloud/fetch.sh. The VM stops automatically; its disk and IP can still incur charges.'
