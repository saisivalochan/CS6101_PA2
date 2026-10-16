# A: Environment and cloud
Read AGENTS.md and docs/requirements.md first. Budget US$50; one n2-standard-16
CPU VM, Ubuntu 24.04, 150 GB pd-balanced disk in us-central1. Authenticate through
gcloud; project ID is supplied externally. Quote/check cost before paid operations.
Runtime <=24h and <=remaining US$35 compute allowance / conservative hourly rate.
Reserve US$15. Runtime caps are not billing caps; persistent disk/IP charges remain.
Inputs: source archive, cloud project. Outputs: setup logs, dependency versions,
machine metadata, resource receipts, copied results. Core builds locally with
Python stdlib + g++; cloud bootstrap installs data/FAISS/report dependencies.
Validation: build core, unittest suite, end-to-end synthetic smoke test, cloud
preflight and artifact retrieval. Never log credentials. Preserve results before
cleanup. Cloud access and package downloads are external execution dependencies.
