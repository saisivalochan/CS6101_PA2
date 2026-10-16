# CS6101 PA2 — Vamana and FAISS

This project implements the compulsory assignment in `ProblemStatement.pdf`:
SIFT1M, Quora, NQ, FAISS exact/HNSW, original C++ Vamana, the SIFT100K study,
full comparisons, and report/submission generation. Extra credit is excluded.

**Local execution status (2026-10-16):** all full datasets and 48 required
operating points pass the checksum-verifying experiment audit. The test suite passes
without skips. The isolated NQ rebuild is complete; its graph matches the original
after normalizing only the elapsed-time header, and its five query settings were
remeasured before the final tests. Original evidence is preserved. The report is
`results/report_draft.pdf`, with all eight required plots and local LLM transcript
snapshots; compact measurements are in `results/metrics.csv`. No paid operations
were used. Final packaging still requires roll numbers, genuine shareable links
for every LLM chat, and confirmation that disclosure is complete. Synthetic test
results are explicitly labeled and are never used as assignment measurements.

## Read first

`AGENTS.md` describes the working rules. `docs/requirements.md` maps the visible PDF
requirements to deliverables. `docs/parts/` gives each implementation part its own
inputs, outputs, decisions, and checks. `docs/progress.md` records the current state.
Page 7 requires full LLM documentation and shareable links; the final report and
packaging checks enforce this requirement.

## Local core validation (no Python packages needed)

Requires Linux, Python >=3.10 and g++ with C++17 support:

```bash
python3 scripts/build_core.py
python3 -m unittest discover -s tests -v
python3 -m pa2.cli core-smoke
```

The shared library uses a C ABI through Python's standard-library ctypes. It needs
no Python development headers. Each `Vamana` instance owns mutable search buffers
and must be used sequentially. Corpus arrays stay alive for the instance lifetime.
Use separate instances for independent concurrent work.

## Full environment

On an Ubuntu 24.04 CPU machine:

```bash
bash cloud/bootstrap.sh
source .venv/bin/activate
python -m pa2.cli run --config configs/main.json --max-hours 24
```

Bootstrap installs system/Python dependencies, builds the core, and runs tests.
It records exact resolved package versions in `results/environment.lock.txt`.
For another machine, install the project with `python -m pip install -e .`, then
build the core. Reproduce the recorded package versions with
`python -m pip install -r results/environment.lock.txt` after adjusting/removing
the editable-project path line, followed by `python -m pip install --no-deps -e .`.

Run from the repository root. Paths in JSON configs resolve relative to the
working directory. Artifact manifests contain checksums and preparation settings;
incompatible indices and checkpoints are rejected rather than reused.

The run command performs:

1. Download/prepare SIFT1M and build Vamana at 100K, 250K and 500K points.
2. Stream Quora/NQ, fit separate corpus-only PCA transforms, prepare search vectors.
3. Estimate full-build scaling and stop if full builds appear infeasible within
   75% of remaining runtime (leaving room for baselines/evaluation).
4. Save exact FAISS ground truth; build/time FAISS exact and five HNSW settings.
5. Evaluate three SIFT100K construction configurations at five search settings.
6. Build one main Vamana graph per full dataset; evaluate five search settings.
7. Audit all required experimental evidence.

All construction and timing tasks use fresh processes. FAISS/BLAS/OpenMP are limited
to one thread. Timings run without competing tasks. If scaling fails, inspect the
logs, optimize the implementation and rerun scaling; do not silently reduce the
full datasets or substitute development results. Full run time is not guaranteed
by the VM specification or budget.

## Google Cloud (US$50 ceiling)

No cloud login or service-account secret belongs in this repository. Use a normal
authenticated Cloud Shell session. Enable Compute Engine for the selected project
and confirm quota for 16 vCPUs, a default VPC with SSH access, and 150 GB disk.

Upload the project (or `pa2-source.tar.gz`) into Cloud Shell and enter its root.
Create that source-only upload bundle with `python3 scripts/source_bundle.py`.
Extract it with `tar -xzf pa2-source.tar.gz`, then `cd CS6101_PA2`.
Verify the current **on-demand n2-standard-16 us-central1** compute price in the
Google Cloud calculator/console, then pass that USD hourly rate:

```bash
bash cloud/launch.sh YOUR_PROJECT_ID VERIFIED_COMPUTE_HOURLY_USD
```

The script creates one `n2-standard-16` VM with 150 GB pd-balanced storage, no
service account, and an absolute automatic-stop timestamp. Compute allowance is
US$35; US$15 is reserved for other charges. Maximum runtime is min(24h, US$35/rate).
The script refuses another launch if the assignment VM or prior launch receipt
exists. Reconcile actual project spending before any new run. A runtime bound and
budget alerts do **not** guarantee an exact billing cap; storage and external IPs
can continue billing after a stop. No GPU is used.

`cloud/runtime.json` records the instance, project, assumed rate and deadline.
Monitor the VM from Cloud Shell:

```bash
gcloud compute ssh cs6101-pa2 --project=YOUR_PROJECT_ID --zone=us-central1-a \
  --command='sudo journalctl -u cs6101-pa2 -f'
```

This repository does not contain an authenticated account or a fixed project ID.
Local sandbox DNS restrictions cannot be solved by pasting credentials into chat.
Run cloud commands in authenticated Cloud Shell or a normal networked terminal.

## Resuming and individual tasks

Run the same `run` command again on the **same machine** after a failed task.
Compatible completed artifacts are reused. Vamana checkpoints preserve the graph,
RNG, permutation, and position every 15 minutes and at pass boundaries. A restarted
process can redo work since its last checkpoint. Recheck the remaining monetary
allowance before restarting a stopped cloud VM; a previous absolute stop timestamp
may need updating. Do not start a new 24-hour allocation automatically.

Logs are in `results/logs/`; graph progress is in `results/progress/`.
For example, individual tasks are:

```bash
python -m pa2.cli prepare --dataset sift1m --sift-archive /path/to/sift.tar.gz
python -m pa2.cli worker --task build --dataset sift100k --method vamana \
  --params '{"R":64,"L_build":100,"alpha":1.2}'
python -m pa2.cli worker --task evaluate --dataset sift100k --method vamana \
  --params '{"R":64,"L_build":100,"alpha":1.2}' --setting 100
python -m pa2.cli audit --quick
```

Prepare the SIFT100K subset and its exact truth before that evaluation. The full
workflow handles these dependencies automatically. `--force` preparation creates
new artifacts; move any incompatible indices/results aside explicitly afterward.

## Data and metric details

- SIFT1M: official TexMex `sift.tar.gz`; raw L2 vectors, supplied top-10 truth.
  An existing archive can be supplied if FTP downloads are unavailable.
- Quora/NQ: `CohereLabs/beir-embed-english-v3`, corpus `train`, queries `test`.
  Pin the dataset revision on first use. Stream only `_id` and `emb`, storing raw
  float32 embeddings on disk. Never load the whole 1024-dimensional corpus into RAM.
- Per IR dataset, sample 100,000 corpus rows uniformly without replacement with
  seed 42; randomized sklearn PCA, 128 components; apply the same transform to
  corpus/queries, cast float32, then L2-normalize. Save sample IDs and variance.
- SIFT100K uses a seeded uniform SIFT1M subset and newly computed exact neighbors.
- Use `IndexFlatL2`, HNSW `M=32/efConstruction=200`, five efSearch values.
  Vamana uses `R=64/L_build=100/alpha=1.2`; study variants change R to 32 or alpha to 1.
- GreedySearch counts expanded vertices and unique query-to-database distances,
  including the entry. It returns all expanded vertices for construction.
- RobustPrune compares `alpha² * squared_L2 <= squared_L2`, preserving the supplied
  Euclidean rule. Entry is nearest to the float64 corpus mean. Two shuffled passes
  use alpha=1 then chosen alpha. All adjacency IDs and degree bounds are validated.
- Recall uses every query. Individual-query timing uses the same seeded 1000-query
  subset when available, 50 warmup queries, three repetitions, and the median run
  by mean latency. QPS is reciprocal mean latency. Raw timing observations are saved.
- Report native build time separately from serialization/session time. Peak build
  RSS includes resident mapped pages, native allocations, and Python/FAISS overhead.

## Report and final submission

On the experiment machine, inspect completeness and generate an honest draft:

```bash
python -m pa2.cli audit
python -m pa2.cli report --details submission_details.draft.json --draft
```

Local Codex conversations for this workspace can be exported with
`python3 scripts/export_llm_chats.py`. This saves visible user/assistant messages
and tool activity under `llm_transcripts/`, excluding internal reasoning and
system/developer instructions. Inline images are decoded into checksum-named
attachments, displayed in the report appendix and included in final packaging;
their binary encodings do not become unreadable transcript text. Refresh snapshots
after finishing the conversations.
The export cannot discover other-product chats or create shareable URLs; include
those chats separately and provide genuine accessible links for every used chat.

The prepared `submission_details.draft.json` already references the three exported
local sessions. Complete it and save it as `submission_details.json`, adding any
other chats used. Alternatively, copy the example file. Fill in one or
two roll numbers, a description/shareable URL/full transcript path for every LLM
chat used, and confirm `llm_documentation_complete` only when all are included.
Transcript paths are relative to the details file. Shareable URLs must be genuine;
the tool checks their shape, not their public accessibility. Check accessibility
yourself before submission. All full transcripts are appended to the PDF.

If the experiments are on the VM, transfer these details and transcripts there
and generate the final report **before** retrieving only the compact results:

```bash
python -m pa2.cli report --details submission_details.json
python -m pa2.cli package --details submission_details.json
```

Final generation refuses missing experiments or disclosure details. Packaging
rechecks data/index hashes, current report evidence, and archive integrity.
The result is `submission/ROLL1_ROLL2.zip` (or `ROLL1.zip` for one member).
It includes implementation/experiment sources, configs, tests, result JSON/CSV,
figures, environment versions, transcripts, and the single report. Data, indices,
working context documents, environments and credentials are excluded by allowlist.

From Cloud Shell, retrieve the compact result/submission bundle:

```bash
bash cloud/fetch.sh YOUR_PROJECT_ID
```

Download `cloud-download/pa2-results.tar.gz` from Cloud Shell to your computer and
verify its contents. The fetch script requires the VM to be running. Keep large
data/indices remotely until final audit/report/package have finished. Once all
required outputs have verified local copies, delete the assignment-created VM and
its attached boot disk using the Google Cloud console. Confirm no assignment
resources remain billing. Only one team member should submit through the course
portal; this project does not perform that external submission.
